import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind Pretrain")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument("--save_weight", type=str, default="pretrain", help="模型权重保存前缀")
    parser.add_argument("--epochs", type=int, default=1, help="模型训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=1, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=8, help="梯度累计步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=100, help="模型保存间隔")
    parser.add_argument("--hidden_size", type=int, default=512, help="隐藏层维度")
    parser.add_argument("--num_hidden_layers", type=int, default=8, help="隐藏层数量")
    parser.add_argument("--max_seq_len", type=int, default=512, help="训练数据最大截断长度")
    parser.add_argument("--use_moe", type=int, default=0, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument("--data_path", type=str, default="../dataset/pretrain_hq.jsonl", help="预训练数据路径")
    parser.add_argument("--from_weight", type=str, default="none", help="基于哪个模型权重开始训练，为none则从头开始训练")
    parser.add_argument("--from_resume", type=int, default=0, choices=[0, 1], help="是否自动检测&继续训练（0=否，1=是）")
    # action="store_true" 用于创建一个命令行标志。这种参数的特点是：不需要 跟随一个值。它的存在与否，直接决定了参数的值为 True 或 False。
    # 在命令行中写了 --use_wandb，那么解析后 args.use_wandb 的值就设为 True
    parser.add_argument("--use_wandb", action="store_true", default="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Pretrain", help="wandb项目名")
    args = parser.parse_args()
    
    
    