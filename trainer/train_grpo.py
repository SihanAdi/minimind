"""
GRPO:
对于同一个问题，模型生成N个不同的回答(例如N=4)，然后计算这N个回答的奖励分数。 
接着把这N个回答的平均奖励作为baseline，高于baseline的回答被鼓励，低于baseline的回答被抑制。 
用这种方式巧妙地避免了训练额外的critic网络

只要是RL都必须面对的正反样本这个原理性限制 - 退化组(Degenerate Groups)
假设某个问题略难，导致N个回答的奖励分数几乎一样（大部分情况是一样烂而不是一样好），那么这一组的学习信号就无限接近0。 
在MiniMind这种超小模型上，这个问题尤为明显，求解数学问题99.99%的情况下整组回答质量都很差，那么将无法学习。 
因此必须为模型指定合理的domain，即必须限制在能力边界内。
"""
import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import gc
from contextlib import nullcontext
import re
from contextlib import nullcontext
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoModel, AutoTokenizer
from trainer.trainer_utils import init_distributed_mode, setup_speed, lm_checkpoint, is_main_process, Logger, init_model, SkipBatchSampler, get_lr
from model.minimodel_model import MiniMindConfig, MiniMindForCausalLm
from dataset.lm_dataset import RLAIFDataset
import warnings
warnings.filterwarnings('ignore')


def calculate_rewards(prompts, responses, reward_model, reward_tokenizer):
    """整合所有奖励函数计算总奖励"""
    def reasoning_model_reward(rewards):
        # 格式奖励（仅针对训练推理模型时使用）
        pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
        pattern2 = r"^<think>\n.*?\n</think>\n\n<answer>\n.*?\n</answer>$"

        matches_pattern = [re.match(pattern, response, re.S) for response in responses]
        matches_pattern2 = [re.match(pattern2, response, re.S) for response in responses]

        format_rewards = []
        for match_pattern, match_pattern2 in zip(matches_pattern, matches_pattern2):
            if match_pattern:
                format_rewards.append(0.5)
            elif match_pattern2:
                format_rewards.append(0.5)
            else:
                format_rewards.append(0.0)
        rewards += torch.tensor(format_rewards, device=args.device)

        def mark_num(text):
            """
            标记奖励（防止严格奖励稀疏，仅针对训练推理模型时使用）
            在强化学习中，模型需要通过试错来学习，但如果只有最终答案正确/错误这一个奖励信号：
                模型很难知道中间推理步骤做得好不好
                学习效率低，收敛慢
                容易陷入局部最优
            引导结构化思考;防止滥用;中间反馈
            """
            reward = 0
            if text.count("<think>") == 1:
                reward += 0.25
            if text.count("</think>") == 1:
                reward += 0.25
            if text.count("<answer>") == 1:
                reward += 0.25
            if text.count("</answer>") == 1:
                reward += 0.25
            return reward
        
        mark_rewards = [mark_num(response) for response in responses]
        rewards += torch.tensor(mark_rewards, device=args.device)
        return rewards

    rewards = torch.zeros(len(responses), device=args.device)

    if args.reasoning == 1:
        rewards = reasoning_model_reward(rewards)

    # 使用reward model计算整个response的奖励
    with torch.no_grad():
        reward_model_scores = []
        batch_size = len(prompts)
        scale = 3.0
        for i in range(batch_size):
            for j in range(args.num_generations):
                response_idx = i * args.num_generations + j
                response = responses[response_idx]
                prompt = prompts[i]

                pattern = r"<\|im_start\|>(system|user|assistant)\s+(.*?)<\|im_end\|>"
                matches = re.findall(pattern, prompt, re.DOTALL)
                messages = [{"role": role, "content": content.strip()} for role, content in matches]

                tmp_chat = messages + [{"role": "assistant", "content": response}]
                score = reward_model.get_score(reward_tokenizer, tmp_chat)
                score = max(min(score, scale), -scale) # 避免异常值影响，保证稳定性；控制奖励范围；防止模型为了获得极端高奖励而学习"作弊"策略

                if args.reasoning == 1:
                    answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
                    if answer_match:
                        answer_content = answer_match.group(1).strip()
                        # 对answer内容单独计算reward
                        tmp_chat = messages + [{"role": "assistant", "content": answer_content}]
                        answer_score = reward_model.get_score(reward_tokenizer, tmp_chat)
                        answer_score = max(min(answer_score, scale), -scale)
                        score = score * 0.4 + answer_score * 0.6
                reward_model_scores.append(score)

        reward_model_scores = torch.tensor(reward_model_scores, device=args.device)
        rewards += reward_model_scores

    return rewards


def train_epoch(epoch, loader, iters, ref_model, reward_model, reward_tokenizer, start_step=0, wandb=None):
    model.train()

    for step, batch in enumerate(loader, start=start_step + 1):
        prompts = batch["prompt"]

        prompt_inputs = tokenizer(
            prompts, 
            return_tensors="pt", 
            padding=True, 
            padding_side="left", 
            return_token_type_ids=False, 
            add_special_tokens=False
        ).to(args.device) # input_ids: [B, P], attention_mask: [B, P]
        if args.max_seq_len:
            prompt_inputs["input_ids"] = prompt_inputs["input_ids"][:, -args.max_seq_len:]
            prompt_inputs["attention_mask"] = prompt_inputs["attention_mask"][:, -args.max_seq_len:]
        
        with torch.no_grad():
            # DDP 模型需要使用 .module 访问 generate 方法
            model_for_gen = model.module if isinstance(model, DistributedDataParallel) else model
            outputs = model_for_gen.generate(
                input_ids=prompt_inputs.input_ids,
                attention_mask=prompt_inputs.attention_mask,
                do_sample=True,
                temperature=0.8,
                num_return_sequences=args.num_generations,
                max_new_tokens=args.max_gen_len,
                pad_token_id=tokenizer.pad_token_id,
            )  # [B*num_gen, P+R]

        completion_ids = outputs[:, prompt_inputs["input_ids"].size(1):] # [B*num_gen, R]

        def get_per_token_logps(cur_model, input_ids, n_keep):
            input_ids = input_ids.detach().clone() if input_ids.is_inference() else input_ids
            logits = cur_model(input_ids=input_ids, logits_to_keep=n_keep + 1).logits[:, :-1, :]
            per_token_logps = []
            for logits_row, ids_row in zip(logits, input_ids[:, -n_keep:]):
                ids_row = ids_row.detach().clone() if ids_row.is_reference() else ids_row
                per_token_logps.append(
                    torch.gather(
                        F.log_softmax(logits_row, dim=-1),
                        1,
                        ids_row.unsqueeze(1)
                    ).squeeze(1)
                )
            return torch.stack(per_token_logps)

        per_token_logps = get_per_token_logps(model, outputs, completion_ids.size(1))  # [B*num_gen, R]
        with torch.no_grad():
            ref_per_token_logps = get_per_token_logps(ref_model, outputs, completion_ids.size(1))  # [B*num_gen, R]

        completions = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
        rewards = calculate_rewards(prompts, completions, reward_model, reward_tokenizer).to(args.device) # [B*num_gen]

        # 计算优势项
        grouped_rewards = rewards.view(-1, args.num_generations) # [B, num_gen]
        mean_r = grouped_rewards.mean(dim=1).repeat_interleave(args.num_generations) # [B*num_gen]
        std_r = grouped_rewards.std(dim=1).repeat_interleave(args.num_generations) # [B*num_gen]
        advantages = torch.clamp((rewards - mean_r) / (std_r + 1e-4) , -10, 10)
        # 全局归一化: 稳定训练：防止梯度爆炸/消失; 批处理一致性：不同batch间的优势值具有可比性
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8) # [B*num_gen]

        is_eos = completion_ids == tokenizer.eos_token_id # [B*num_gen, R]
        eos_idx = torch.full((is_eos.size(0), ), is_eos.size(1), dtype=torch.long, device=args.device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        completion_mask = (torch.arange(eos_idx.size(1), device=args.device).expand(eos_idx.size(0), -1) <= eos_idx.unsqueeze(1)).int() # [B*num_gen, R]

        kl_div = ref_per_token_logps - per_token_logps
        """
        利用 Jensen 不等式和指数函数的性质, 近似计算 KL 散度
        """
        per_token_kl = torch.exp(kl_div) - kl_div - 1 # [B*num_gen, R]
        per_token_loss = -(torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1) - args.beta * per_token_kl) # [B*num_gen, R]
        loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean() / args.accumulation_steps
        loss.backward()
        
        if (step + 1) % args.accumulation_steps == 0:
            if args.grad_clip > 0:
                clip_grad_norm_(model.parameters(), args.grad_clip)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            
        if step % args.log_interval == 0 or step == iters:
            policy_loss_val = loss.item()
            avg_reward_val = rewards.mean().item()
            avg_len_val = completion_mask.sum(dim=1).float().mean().item()
            current_lr = optimizer.param_groups[0]['lr']

            Logger(f'Epoch: {epoch+1}, Step: {step}/{iters}, '
                   f'Actor Loss: {policy_loss_val:.6f}, Reward: {avg_reward_val:.6f}, '
                   f'Avg Response Len: {avg_len_val:.2f}, LR: {current_lr:.2e}')

            if wandb and is_main_process():
                wandb.log({
                    "policy_loss": policy_loss_val,
                    "reward": avg_reward_val,
                    "avg_response_len": avg_len_val,
                    "advantages_mean": advantages.mean().item(),
                    "learning_rate": current_lr
                })

        if (step % args.save_interval == 0 or step == iters + 1) and is_main_process():
            # 提高保存过程的稳定性
            model.eval()
            moe_suffix = "_moe" if lm_config.use_moe else ""
            checkpoint = f"{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth"
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            state_dict = {k: v.half().cpu() for k, v in state_dict.items()}
            torch.save(state_dict, checkpoint)
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer, 
                         epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints',
                         scheduler=scheduler)
            model.train()
            del state_dict

        del prompt_inputs, outputs, completion_ids, per_token_logps, ref_per_token_logps
        del completions, rewards, grouped_rewards, mean_r, std_r, advantages, completion_mask
        torch.cuda.empty_cache()
        gc.collect()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind GRPO")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument("--save_weight", type=str, default="grpo", help="模型权重保存前缀")
    parser.add_argument("--epochs", type=int, default=1, help="模型训练轮数")
    parser.add_argument("--batch_size", type=int, default=2, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=8e-8, help="初始学习率")
    parser.add_argument("--critic_learning_rate", type=float, default=8e-8, help="Critic学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=1, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累计步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=1, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=10, help="模型保存间隔")
    parser.add_argument("--hidden_size", type=int, default=512, help="隐藏层维度")
    parser.add_argument("--num_hidden_layers", type=int, default=8, help="隐藏层数量")
    parser.add_argument("--max_seq_len", type=int, default=66, help="Prompt最大长度")
    parser.add_argument("--max_gen_len", type=int, default=1536, help="生成文本最大长度")
    parser.add_argument("--use_moe", type=int, default=0, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument("--data_path", type=str, default="../dataset/rlaif-mini.jsonl", help="RLAIF数据路径")
    parser.add_argument("--num_generations", type=int, default=8, help="每个prompt生成的样本数")
    parser.add_argument("--beta", type=float, default=0.02, help="KL惩罚系数")
    parser.add_argument("--reasoning", type=int, default=1, choices=[0, 1], help='推理模型类型（0=普通模型，1=推理模型）')
    parser.add_argument("--reward_model_path", type=str, default="../../internlm2-1_8b-reward", help="Reward模型路径")
    parser.add_argument("--from_resume", type=int, default=0, choices=[0, 1], help="是否自动检测&继续训练（0=否，1=是）")
    # action="store_true" 用于创建一个命令行标志。这种参数的特点是：不需要 跟随一个值。它的存在与否，直接决定了参数的值为 True 或 False。
    # 在命令行中写了 --use_wandb，那么解析后 args.use_wandb 的值就设为 True
    parser.add_argument("--use_wandb", action="store_true", default="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-GRPO", help="wandb项目名")
    args = parser.parse_args()

    # ========== 1 ==========
    # 初始化环境
    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
        
    # 初始化随机种子，使用全局rank确保每个进程都有不同的随机种子
    setup_speed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    # ========== 2 ==========
    # 创建文件夹
    os.makedirs(args.save_dir, exist_ok=True)

    # 配置模型参数
    lm_config = MiniMindConfig(
        hidden_size=args.hidden_size, 
        num_hidden_layers=args.num_hidden_layers, 
        max_seq_len=args.max_seq_len + args.max_gen_len, 
        use_moe=bool(args.use_moe)
    )

    # 读取checkpoint数据
    checkpoint_data = lm_checkpoint(lm_config, args.save_weight, save_dir="../checkpoints") if args.from_resume == 1 else None

    # ========== 3 ==========
    # 设置混合精度
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    """
    autocast: 在正向传播中, 将模型权重、激活值和损失函数计算保持在 FP32
    将其他操作自动转换为 FP16：例如，矩阵乘法（torch.mm, torch.matmul）和卷积（torch.cudnn.conv2d）等计算密集型操作，会使用 FP16 来计算
    """
    autocast_context = nullcontext() if device_type == "cpu" else torch.autocast("cuda", dtype=dtype)

    # ========== 4 ==========
    # 配置wandb
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = checkpoint_data.get("wandb_id") if checkpoint_data else None
        resume = "must" if wandb_id else None
        wandb_run_name = f"MiniMind-GRPO-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
        wandb.init(
            project=args.wandb_project,
            name=wandb_run_name,
            id=wandb_id,
            resume=resume,
        )
    
    # ========== 5 ==========
    # 定义模型、数据、及优化器
    base_weight = "reason" if args.reasoning else "full_sft"
    # Policy model
    model, tokenizer = init_model(lm_config, base_weight, args.device)
    # Reference model
    ref_model, _ = init_model(lm_config, base_weight, device=args.device)
    ref_model = ref_model.eval().requires_grad_(False)
    # Reward model
    reward_model = AutoModel.from_pretrained(args.reward_model_path, torch_dtype=torch.float16, trust_remote_code=True)
    reward_model = reward_model.to(args.device).eval().requires_grad_(False)
    reward_tokenizer = AutoTokenizer.from_pretrained(args.reward_model_path, trust_remote_code=True)

    # 数据和优化器
    train_ds = RLAIFDataset(args.data_path, tokenizer, args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    tmp_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler)
    iters = len(tmp_loader)
    total_optimizer_steps = (iters // args.accumulation_steps) * args.epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_optimizer_steps, eta_min=args.learning_rate / 10)

    # ========== 6 ==========
    # 从checkpoint恢复模型状态
    start_epoch, start_step = 0, 0
    if checkpoint_data:
        model.load_state_dict(checkpoint_data["model"])
        optimizer.load_state_dict(checkpoint_data["optimizer"])
        scheduler.load_state_dict(checkpoint_data["scheduler"])

        start_epoch = checkpoint_data["epoch"]
        start_step = checkpoint_data.get("step", 0)
        
    # ========== 7 ==========
    # DDP包装模型
    """
    在PyTorch DDP中，所有模型的参数和缓冲区默认都会在多个GPU/进程间同步
    RoPE（Rotary Positional Embedding）的位置编码参数。它们需要被排除是因为确定性计算，避免不必要的通信开销
    """
    if dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])

    
    # ========== 8 ==========
    # 开始训练
    for epoch in range(start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if epoch == start_epoch and start_step > 0:
            batch_sampler = SkipBatchSampler(
                train_sampler or range(len(train_ds)),
                args.batch_size,
                start_step + 1
            )
            loader = DataLoader(
                train_ds, 
                batch_sampler=batch_sampler,
                num_workers=args.num_workers,
                pin_memory=True
            )
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(epoch, loader, len(loader) + start_step + 1, ref_model, reward_model, reward_tokenizer, start_step, wandb)
        else:
            loader = DataLoader(
                train_ds, 
                batch_size=args.batch_size, 
                shuffle=(train_sampler is None),
                sampler=train_sampler,
                num_workers=args.num_workers,
                pin_memory=True,
                drop_last=False
            )
            train_epoch(epoch, loader, len(loader), ref_model, reward_model, reward_tokenizer, 0, wandb)
                
                
# torchrun --nproc_per_node 1 train_grpo.py 
