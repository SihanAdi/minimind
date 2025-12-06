import os
import random
import math
import numpy as np

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer
from torch.utils.data import Sampler

from model.minimodel_model import MiniMindForCausalLm


def get_lr(current_step, total_step, lr):
    """
    余弦退火学习率调度函数，结合了固定基础学习率和余弦退火调整
    固定基础部分: lr / 10 作为学习率的下限，防止学习率降为0
    余弦退火调整: 随着训练进行，余弦值从 1 平滑下降到 -1, lr 平滑下降到 0
    """
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_step))


def is_main_process():
    if dist.is_initialized() and dist.get_rank() != 0:
        return False
    return True


def Logger(msg):
    if is_main_process():
        print(msg, flush=True)


def init_distributed_mode():
    # 通过torchrun --nproc_per_node=2 / python -m torch.distributed.launch --nproc_per_node=2 启动会自动初始化 RANK=0, LOCAL_RANK=0, WORLD_SIZE=2
    if int(os.environ.get("RANK", -1)) == -1:
        # 非 DDP 模式
        return 0
    
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def setup_speed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True    # 强制CuDNN使用确定性算法
    torch.backends.cudnn.benchmark = False    # 禁用CuDNN的自动算法选择优化


def lm_checkpoint(
    lm_config,
    weight="full_sft",
    model=None,
    optimizer=None,
    epoch=0,
    step=0,
    wandb=None,
    save_dir="../checkpoints",
    **kwargs,
):
    """
    在分布式训练中 GPU 数量变化时调整 step
        - step：表示模型已经处理了多少个batch，每个step对应一次梯度更新
        - 在数据并行训练中，每个 step 处理的总样本数 = batch_size * world_size（world_size表示参与训练的GPU数量）
    """
    os.makedirs(save_dir, exist_ok=True)
    moe_path = "_moe" if lm_config.use_moe else ""
    checkpoint_path = f"{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}.pth"
    resume_path = f"{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}_resume.pth"

    if model is None:
        if os.path.exists(resume_path):
            checkpoint_data = torch.load(resume_path, map_location="cpu")
            saved_world_size = checkpoint_data.get("world_size", 1)
            cur_world_size = dist.get_world_size() if dist.is_initialized() else 1
            if saved_world_size != cur_world_size:
                checkpoint_data["step"] = int(checkpoint_data["step"]) * saved_world_size // cur_world_size
                Logger(f'GPU数量变化({saved_world_size} → {cur_world_size})，step已自动转换为{checkpoint_data["step"]}')
            return checkpoint_data
        return None
    else:
        state_dict = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()

        # 原子性保存：防止在保存过程中发生意外导致原始检查点文件损坏
        cpk_tmp_path = checkpoint_path + ".tmp"
        torch.save(
            {k: v.half() for k, v in state_dict.items()}, # value: float16存储
            cpk_tmp_path
        )
        os.replace(cpk_tmp_path, checkpoint_path)

        # 处理不同版本W&B API兼容性，获取W&B运行id（每个实验的唯一标识符， 用于恢复中断的实验；对比实验；模型版本管理）
        if wandb:
            if hasattr(wandb, "get_run"):
                run = wandb.get_run()
                wandb_id = getattr(run, "id", None) if run else None
            else:
                wandb_id = getattr(wandb, "id", None)
        else:
            wandb_id = None

        resume_data = {
            "model": state_dict,
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "epoch": epoch,
            "step": step,
            "wandb_id": wandb_id,
            "world_size": dist.get_world_size() if dist.is_initialized() else 1,
        }

        for k, v in kwargs.items():
            if v is not None:
                if hasattr(v, "state_dict"):
                    if isinstance(v, DDP):
                        resume_data[k] = v.module.state_dict()
                    else:
                        resume_data[k] = v.state_dict()
                else:
                    resume_data[k] = v
        
        resume_tmp_path = resume_path + ".tmp"
        torch.save(resume_data, resume_tmp_path)
        os.replace(resume_tmp_path, resume_path)
        

def init_model(
    lm_config,
    from_weight='pretrain',
    tokenizer_path='../model_weights',
    save_dir='../out',
    device='cuda'
):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = MiniMindForCausalLm(lm_config)
    
    if from_weight != 'none':
        moe_suffix = "_moe" if lm_config.use_moe else ""
        weight_path = f"{save_dir}/{from_weight}_{lm_config.hidden_size}{moe_suffix}.pth"
        weights = torch.load(weight_path, map_location=device)
        model.load_state_dict(weights, strict=False)
    
    Logger(f"所加载Model可训练的参数: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万")
    return model.to(device), tokenizer

class SkipBatchSampler(Sampler):
    def __init__(self, sampler, batch_size, skip_batches=0):
        super().__init__()
        self.sampler = sampler
        self.batch_size = batch_size
        self.skip_batches = skip_batches
        
    def __iter__(self):
        batch = []
        skipped = 0
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                if skipped < self.skip_batches:
                    skipped += 1
                    batch = []
                    continue
                yield batch
                batch = []
        if len(batch) > 0 and skipped >= self.skip_batches:
            yield batch
    
    def __len__(self):
        total_batches = (len(self.sampler) + self.batch_size - 1) // self.batch_size
        return max(0, total_batches - self.skip_batches)