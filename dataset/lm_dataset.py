import json
import random
import re

import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import os
from transformers import AutoTokenizer

"""
TOKENIZERS_PARALLELISM: 控制 Hugging Face tokenizers 库并行
禁用 tokenizers 库的并行处理，强制其使用单线程模式
    - 避免与多进程训练的冲突
    - 在资源受限的环境中，禁用并行可以减少内存使用
"""
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self.load_data(data_path)
    
    def load_data(self, data_path):
        samples = []
        with open(data_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                data = json.loads(line.strip())
                samples.append(data)
        return samples

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        sample = self.samples[index]
        
        encoding = self.tokenizer(
            str(sample["text"]),
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        input_ids = encoding.input_ids.squeeze()
        loss_mask = (input_ids != self.tokenizer.pad_token_id)
        
        X = torch.tensor(input_ids[:-1], dtype=torch.long)
        Y = torch.tensor(input_ids[1:], dtype=torch.long)
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)
        
        return X, Y, loss_mask
        
        
if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("../model_weights")
    pretrain_dataset = PretrainDataset("./pretrain_hq.jsonl", tokenizer)
    print(pretrain_dataset[0])
    