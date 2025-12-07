"""推理蒸馏"""
import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import time
from contextlib import nullcontext
import time
from contextlib import nullcontext
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from trainer.trainer_utils import init_distributed_mode, setup_speed, lm_checkpoint, is_main_process, Logger, init_model, SkipBatchSampler, get_lr
from model.minimodel_model import MiniMindConfig
from dataset.lm_dataset import SFTDataset
import warnings
warnings.filterwarnings('ignore')

def train_epoch(epoch, loader, iters, tokenizer, lm_config, start_step=0, wandb=None):
    """
    推理模型R1的回复模板是：
    <think>\n思考过程\n</think>\n<answer>\n最终回答\n</answer>
    
    通过增加标记位置token的损失惩罚解决蒸馏过程虽然和SFT一样，但实验结果是模型难以每次都符合模板规范的回复，即脱离思考和回复标签约束。 
    """
    # 思考标签占位符
    start_of_think_ids = tokenizer("<think>").input_ids
    end_of_think_ids = tokenizer("</think>").input_ids
    start_of_answer_ids = tokenizer("<answer>").input_ids
    end_of_answer_ids = tokenizer("</answer>").input_ids
    
    loss_function = nn.CrossEntropyLoss(reduction="none")
    start_time = time.time()
    for step, (X, Y, loss_mask) in enumerate(loader, start=start_step + 1):
        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device)
        
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        # 覆盖参数组原有的学习率设置
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
            
        with autocast_context:
            res = model(X)
            loss = loss_function(
                res.logits.view(-1, res.logits.size(-1)), # [batch size * seq len, vocb len]
                Y.view(-1) # [batch size * seq len]
            ).view(Y.size())
            
            # 特殊标签位置增加权重
            special_ids = torch.isin(Y.view(-1), torch.tensor(
                start_of_think_ids + end_of_think_ids + start_of_answer_ids + end_of_answer_ids
            ).to(args.device))
            
            loss_mask = loss_mask.view(-1)
            # 对思考标签增加10倍权重
            loss_mask[special_ids] = 10
            loss_mask = loss_mask.view(Y.size())
            loss = (loss * loss_mask).sum() / loss_mask.sum()
            loss += res.aux_loss
            loss /= args.accumulation_steps
            
        # 损失缩放和反向传播
        scaler.scale(loss).backward()
        
        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            
            scaler.step(optimizer)
            scaler.update()
            
            # 减少内存占用，提高效率
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            
        if (step % args.log_interval == 0 or step == iters + 1) and is_main_process():
            spend_time = time.time() - start_time
            current_loss = loss * args.accumulation_steps
            current_lr = optimizer.param_groups[-1]["lr"]
            
            eta_min = spend_time / (step + 1) * iters // 60 - spend_time // 60
            
            Logger(f'Epoch:[{epoch + 1} / {args.epochs}]({step} / {iters}) loss:{current_loss:.6f} lr:{current_lr:.12f} epoch_Time:{eta_min}min:')
            
            if wandb: 
                wandb.log({"loss": current_loss, "lr": current_lr, "epoch_Time": eta_min})
                
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
            lm_checkpoint(
                lm_config, 
                weight=args.save_weight, 
                model=model, 
                optimizer=optimizer,
                epoch=epoch,
                step=step,
                wandb=wandb,
                save_dir='../checkpoints',
                scaler=scaler
            )
            model.train()
            del state_dict

        del X, Y, loss_mask, res, loss
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind Reasoning Distillation")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument("--save_weight", type=str, default="reason", help="模型权重保存前缀")
    parser.add_argument("--epochs", type=int, default=1, help="模型训练轮数")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-6, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=1, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累计步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=100, help="模型保存间隔")
    parser.add_argument("--hidden_size", type=int, default=512, help="隐藏层维度")
    parser.add_argument("--num_hidden_layers", type=int, default=8, help="隐藏层数量")
    parser.add_argument("--max_seq_len", type=int, default=1024, help="训练数据最大截断长度")
    parser.add_argument("--use_moe", type=int, default=0, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument("--data_path", type=str, default="../dataset/r1_mix_1024.jsonl", help="推理蒸馏数据路径")
    parser.add_argument("--from_weight", type=str, default="dpo", help="基于哪个模型权重开始训练，为none则从头开始训练")
    parser.add_argument("--from_resume", type=int, default=0, choices=[0, 1], help="是否自动检测&继续训练（0=否，1=是）")
    # action="store_true" 用于创建一个命令行标志。这种参数的特点是：不需要 跟随一个值。它的存在与否，直接决定了参数的值为 True 或 False。
    # 在命令行中写了 --use_wandb，那么解析后 args.use_wandb 的值就设为 True
    parser.add_argument("--use_wandb", action="store_true", default="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Reasoning", help="wandb项目名")
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
        wandb_run_name = f"MiniMind-Reasoning-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
        wandb.init(
            project=args.wandb_project,
            name=wandb_run_name,
            id=wandb_id,
            resume=resume,
        )
    
    # ========== 5 ==========
    # 定义模型、数据、及优化器
    model, tokenizer = init_model(lm_config, args.from_weight, args.device)
    train_ds = SFTDataset(args.data_path, tokenizer, args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    """
    通过自动缩放损失值来解决使用半精度（FP16）训练时的数值稳定性问题
    在训练的后期，梯度值可能小于 FP16 能表示的最小值
    过程：
        放大梯度：在前向传播后将损失值乘以一个缩放因子
        反向传播：计算放大后的梯度
        缩小梯度：将梯度除以相同的缩放因子
    """
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # ========== 6 ==========
    # 从checkpoint恢复模型状态
    start_epoch, start_step = 0, 0
    if checkpoint_data:
        model.load_state_dict(checkpoint_data["model"])
        optimizer.load_state_dict(checkpoint_data["optimizer"])
        scaler.load_state_dict(checkpoint_data["scaler"])
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
            train_epoch(epoch, loader, len(loader) + start_step + 1, tokenizer, lm_config, start_step, wandb)
        else:
            loader = DataLoader(
                train_ds, 
                batch_size=args.batch_size, 
                shuffle=(train_sampler is None),
                sampler=train_sampler,
                num_workers=args.num_workers,
                pin_memory=True
            )
            train_epoch(epoch, loader, len(loader), tokenizer, lm_config, 0, wandb)
                
                
# torchrun --nproc_per_node 1 train_distill_reason.py 
