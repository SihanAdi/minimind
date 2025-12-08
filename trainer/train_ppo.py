import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import time
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


class CriticModel(MiniMindForCausalLm):
    def __init__(self, config):
        super().__init__(config)
        self.value_head = nn.Linear(config.hidden_size, 1)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        outputs = self.model(input_ids, attention_mask, **kwargs)
        hidden_states = self.model.norm(outputs[0])
        values = self.value_head(hidden_states).squeeze(-1)
        return values


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
        reward_model_score = []
        for prompt, response in zip(prompts, responses):
            pattern = r"<\|im_start\|>(system|user|assistant)\s+(.*?)<\|im_end\|>"
            matches = re.findall(pattern, prompt, re.DOTALL)
            messages = [{"role": role, "content": content.strip()} for role, content in matches]

            tmp_chat = messages + [{"role": "assistant", "content": response}]
            score = reward_model.get_score(reward_tokenizer, tmp_chat)
            reward_model_score.append(score)

            scale = 3.0
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
            reward_model_score.append(score)

        reward_model_scores = torch.tensor(reward_model_score, device=args.device)
        rewards += reward_model_scores

    return rewards



def train_epoch(epoch, loader, iters, old_actor_model, ref_model, actor_scheduler, critic_scheduler, reward_model, reward_tokenizer, start_step=0, wandb=None):
    actor_model.train()
    critic_model.train()

    for step, batch in enumerate(loader, start=start_step + 1):
        prompts = batch["prompt"]

        encoded_prompts = tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True, max_length=args.max_seq_len
        ).to(args.device) # input_ids: [B, P], attention_mask: [B, P]

        prompt_lengths = torch.full(
            (encoded_prompts.input_ids.size(0),), 
            encoded_prompts.input_ids.size(1), 
            dtype=torch.long, device=encoded_prompts.input_ids.device
        ) # [B]
        
        with torch.no_grad():
            # DDP 模型需要使用 .module 访问 generate 方法
            model_for_gen = actor_model.module if isinstance(actor_model, DistributedDataParallel) else actor_model
            gen_out = model_for_gen.generate(
                input_ids=encoded_prompts.input_ids,
                attention_mask=encoded_prompts.attention_mask,
                do_sample=True,
                temperature=0.8,
                max_new_tokens=args.max_gen_len,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            ) # [B, P+R]

        response_text = [tokenizer.decode(gen_out[i, prompt_lengths[i]:], skip_special_tokens=True) for i in range(len(prompts))]
        rewards = calculate_rewards(prompts, response_text, reward_model, reward_tokenizer) # [B]
        
        # PPO算法的策略梯度更新，优势值 = 奖励值 - 状态值估计，表示当前行动比平均水平好多少
        full_mask = (gen_out != tokenizer.pad_token_id).long() # [B, P+R]
        values_seq = critic_model(input_ids=gen_out, attention_mask=full_mask) # 输出每个位置的预测价值：[B, P+R]
        # 找到序列的最后一个有效位置
        last_indices = (full_mask * torch.arange(full_mask.size(1), device=gen_out.device)).argmax(dim=1)
        # 提取序列结束时的价值
        values = values_seq[torch.arange(values_seq.size(0), device=values_seq.device), last_indices] # [B]
        advantages = rewards - values.detach() # [B]
        
        logits = actor_model(input_ids=gen_out, attention_mask=full_mask).logits # [B, P+R, V]
        labels = gen_out[:, 1:].clone() # [B, P+R-1]
        # 取logits的前P+R-1个位置（因为最后位置没有对应的标签）
        logp_tokens = F.log_softmax(logits[:, :-1], dim=-1).gather(dim=2, index=labels.unsqueeze(-1)).squeeze(-1) # [B, P+R-1]
        # 创建回复部分的掩码
        seq_len = gen_out.size(1) - 1
        resp_mask = torch.arange(seq_len, device=gen_out.device).unsqueeze(0) >= prompt_lengths.unsqueeze(-1)
        final_mask = resp_mask & (~labels.eq(tokenizer.pad_token_id)) # [B, P+R-1]
        actor_logp = (logp_tokens * final_mask).sum(dim=-1)  # [B]
        
        with torch.no_grad():
            old_logits = old_actor_model(gen_out, full_mask).logits
            old_logp_tokens = F.log_softmax(old_logits[:, :-1], dim=-1).gather(2, index=labels.unsqueeze(-1)).squeeze(-1)
            old_logp = (old_logp_tokens * final_mask).sum(dim=-1)
            
            ref_logits = ref_model(gen_out, full_mask).logits
            ref_logp_tokens = F.log_softmax(ref_logits[:, :-1], dim=-1).gather(2, index=labels.unsqueeze(-1)).squeeze(-1)
            ref_logp = (ref_logp_tokens * final_mask).sum(dim=-1)
            
        # 计算PPO算法的总损失
        # 用于监控新旧策略的变化
        kl = (actor_logp - old_logp).mean() # scalar
        # 用于KL惩罚，约束策略不要偏离参考模型
        kl_ref = (actor_logp - ref_logp).mean() # scalar
        # 计算重要性采样比率
        """
        ratio = π_new(a|s) / π_old(a|s) = exp(log(π_new) - log(π_old))
        ratio > 1: 新策略更偏好这个动作
        ratio < 1: 新策略不太喜欢这个动作
        """
        ratio = torch.exp(actor_logp - old_logp) # [B]
        # 计算裁剪前后的策略目标
        # 未裁剪 - 标准的策略梯度目标
        surr1 = ratio * advantages
        # 裁剪后
        surr2 = torch.clamp(ratio, 1.0 - args.clip_epsilon, 1.0 + args.clip_epsilon) * advantages
        
        # 策略损失
        policy_loss = -torch.min(surr1, surr2).mean()
        # 价值损失
        value_loss = F.mse_loss(values, rewards)
        # 总损失
        loss = policy_loss + args.vf_coef * value_loss + args.kl_coef * kl_ref
        loss.backward()
        
        if (step + 1) % args.accumulation_steps == 0:
            clip_grad_norm_(actor_model.parameters(), args.grad_clip)
            clip_grad_norm_(critic_model.parameters(), args.grad_clip)
            actor_optimizer.step()
            critic_optimizer.step()
            actor_scheduler.step()
            critic_scheduler.step()
            actor_optimizer.zero_grad(set_to_none=True)
            critic_optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            
        if is_main_process():
            response_ids = gen_out[:, encoded_prompts.shape[1]:]
            is_eos = (response_ids == tokenizer.eos_token_id)
            eos_indices = torch.argmax(is_eos.int(), dim=1)
            has_eos = is_eos.any(dim=1)
            """
            orch.where(condition, x, y)
            当condition为True时，返回x
            当condition为False时，返回y
            """
            lengths = torch.where(has_eos, eos_indices + 1, torch.tensor(response_ids.shape[1], device=is_eos.device))
            avg_length = lengths.float().mean()
            
            actor_loss_val = policy_loss.item()
            critic_loss_val = value_loss.item()
            reward_val = rewards.mean().item()
            kl_val = kl.item()
            kl_ref_val = kl_ref.item()
            avg_len_val = avg_length.item()
            actor_lr = actor_optimizer.param_groups[0]['lr']
            critic_lr = critic_optimizer.param_groups[0]['lr']

            if wandb is not None:
                wandb.log({
                    "actor_loss": actor_loss_val,
                    "critic_loss": critic_loss_val,
                    "reward": reward_val,
                    "kl": kl_val,
                    "kl_ref": kl_ref_val,
                    "avg_response_len": avg_len_val,
                    "actor_lr": actor_lr,
                })

            Logger(f"Epoch: {epoch+1}, Step: {step}/{iters}, "
                   f"Actor Loss: {actor_loss_val:.6f}, Critic Loss: {critic_loss_val:.6f}, "
                   f"Reward: {reward_val:.6f}, KL: {kl_val:.6f}, KL_ref: {kl_ref_val:.6f}, "
                   f"Avg Response Len: {avg_len_val:.2f}, Actor LR: {actor_lr:.2e}, Critic LR: {critic_lr:.2e}")

        if (step + 1) % args.update_old_actor_freq == 0:
            state_dict = actor_model.module.state_dict() if isinstance(actor_model, DistributedDataParallel) else actor_model.state_dict()
            old_actor_model.load_state_dict({k: v.detach().cpu() for k, v in state_dict.items()})
            old_actor_model.to(args.device)
                
        if (step % args.save_interval == 0 or step == iters + 1) and is_main_process():
            # 提高保存过程的稳定性
            actor_model.eval()
            moe_suffix = "_moe" if lm_config.use_moe else ""
            checkpoint = f"{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth"
            if isinstance(actor_model, torch.nn.parallel.DistributedDataParallel):
                state_dict = actor_model.module.state_dict()
            else:
                state_dict = actor_model.state_dict()
            state_dict = {k: v.half().cpu() for k, v in state_dict.items()}
            torch.save(state_dict, checkpoint)
            lm_checkpoint(lm_config, weight=args.save_weight, model=actor_model, optimizer=actor_optimizer, 
                         epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints',
                         scheduler=actor_scheduler, critic_model=critic_model, 
                         critic_optimizer=critic_optimizer, critic_scheduler=critic_scheduler)
            actor_model.train()
            del state_dict

        del encoded_prompts, gen_out, response_text, rewards, full_mask, values_seq, values, advantages
        del logits, labels, logp_tokens, final_mask, actor_logp, old_logits, old_logp, ref_logits, ref_logp
        del kl, kl_ref, ratio, surr1, surr2, policy_loss, value_loss, loss
        torch.cuda.empty_cache()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind PPO")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument("--save_weight", type=str, default="ppo_actor", help="模型权重保存前缀")
    parser.add_argument("--epochs", type=int, default=1, help="模型训练轮数")
    parser.add_argument("--batch_size", type=int, default=2, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=8e-8, help="Actor 初始学习率")
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
    parser.add_argument("--clip_epsilon", type=float, default=0.1, help="PPO裁剪参数")
    parser.add_argument("--vf_coef", type=float, default=0.5, help="Value function系数")
    parser.add_argument("--kl_coef", type=float, default=0.02, help="KL散度惩罚系数")
    parser.add_argument("--reasoning", type=int, default=1, choices=[0, 1], help='推理模型类型（0=普通模型，1=推理模型）')
    parser.add_argument("--update_old_actor_freq", type=int, default=4, help="更新old_actor_model的频率")
    parser.add_argument("--reward_model_path", type=str, default="../../internlm2-1_8b-reward", help="Reward模型路径")
    parser.add_argument("--from_resume", type=int, default=0, choices=[0, 1], help="是否自动检测&继续训练（0=否，1=是）")
    # action="store_true" 用于创建一个命令行标志。这种参数的特点是：不需要 跟随一个值。它的存在与否，直接决定了参数的值为 True 或 False。
    # 在命令行中写了 --use_wandb，那么解析后 args.use_wandb 的值就设为 True
    parser.add_argument("--use_wandb", action="store_true", default="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-PPO", help="wandb项目名")
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
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe))

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
        wandb_run_name = f"MiniMind-PPO-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
        wandb.init(
            project=args.wandb_project,
            name=wandb_run_name,
            id=wandb_id,
            resume=resume,
        )
    
    # ========== 5 ==========
    # 定义模型、数据、及优化器
    base_weight = "reason" if args.reasoning else "full_sft"
    # Actor model
    actor_model, tokenizer = init_model(lm_config, base_weight, args.device)
    tokenizer.padding_side = "left"
    # Old Actor model
    old_actor_model, _ = init_model(lm_config, base_weight, args.device)
    old_actor_model = old_actor_model.to(args.device).eval().requires_grad_(False)
    # Reference model
    ref_model, _ = init_model(lm_config, base_weight, device=args.device)
    ref_model = ref_model.eval().requires_grad_(False)
    # Critic model
    moe_suffix = "_moe" if lm_config.use_moe else ""
    checkpoint = f'{args.save_dir}/{base_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
    state_dict = torch.load(checkpoint, map_location=args.device)
    critic_model = CriticModel(lm_config)
    critic_model.load_state_dict(state_dict, strict=False)
    critic_model = critic_model.to(args.device)
    # Reward model
    reward_model = AutoModel.from_pretrained(args.reward_model_path, torch_dtype=torch.float16, trust_remote_code=True)
    reward_model = reward_model.to(args.device).eval().requires_grad_(False)
    reward_tokenizer = AutoTokenizer.from_pretrained(args.reward_model_path, trust_remote_code=True)

    train_ds = RLAIFDataset(args.data_path, tokenizer, args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    
    actor_optimizer = torch.optim.AdamW(actor_model.parameters(), lr=args.learning_rate)
    critic_optimizer = torch.optim.AdamW(critic_model.parameters(), lr=args.critic_learning_rate)

    tmp_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler)
    iters = len(tmp_loader)
    total_optimizer_steps = (iters // args.accumulation_steps) * args.epochs
    actor_scheduler = CosineAnnealingLR(actor_optimizer, T_max=total_optimizer_steps, eta_min=args.learning_rate / 10)
    critic_scheduler = CosineAnnealingLR(critic_optimizer, T_max=total_optimizer_steps, eta_min=args.critic_learning_rate / 10)

    # ========== 6 ==========
    # 从checkpoint恢复模型状态
    start_epoch, start_step = 0, 0
    if checkpoint_data:
        actor_model.load_state_dict(checkpoint_data["model"])
        critic_model.load_state_dict(checkpoint_data["critic_model"])

        actor_optimizer.load_state_dict(checkpoint_data["optimizer"])
        critic_optimizer.load_state_dict(checkpoint_data["critic_optimizer"])

        actor_scheduler.load_state_dict(checkpoint_data["scheduler"])
        critic_scheduler.load_state_dict(checkpoint_data["critic_scheduler"])

        start_epoch = checkpoint_data["epoch"]
        start_step = checkpoint_data.get("step", 0)
        
    # ========== 7 ==========
    # DDP包装模型
    """
    在PyTorch DDP中，所有模型的参数和缓冲区默认都会在多个GPU/进程间同步
    RoPE（Rotary Positional Embedding）的位置编码参数。它们需要被排除是因为确定性计算，避免不必要的通信开销
    """
    if dist.is_initialized():
        actor_model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        critic_model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}

        actor_model = DistributedDataParallel(actor_model, device_ids=[local_rank])
        critic_model = DistributedDataParallel(critic_model, device_ids=[local_rank])

    
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
            train_epoch(epoch, loader, len(loader) + start_step + 1, old_actor_model, ref_model, actor_scheduler, critic_scheduler, reward_model, reward_tokenizer, start_step, wandb)
        else:
            loader = DataLoader(
                train_ds, 
                batch_size=args.batch_size, 
                shuffle=(train_sampler is None),
                sampler=train_sampler,
                num_workers=args.num_workers,
                pin_memory=True
            )
            train_epoch(epoch, loader, len(loader), old_actor_model, ref_model, actor_scheduler, critic_scheduler, reward_model, reward_tokenizer, 0, wandb)
                
                
# torchrun --nproc_per_node 1 train_ppo.py 
