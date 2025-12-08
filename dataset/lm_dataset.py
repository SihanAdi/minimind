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
            return_tensors="pt",
        )
        input_ids = encoding.input_ids.squeeze()
        loss_mask = input_ids != self.tokenizer.pad_token_id

        X = torch.tensor(input_ids[:-1], dtype=torch.long)
        Y = torch.tensor(input_ids[1:], dtype=torch.long)
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)

        return X, Y, loss_mask


class SFTDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self.load_data(jsonl_path)
        self.bos_id = tokenizer(f"{tokenizer.bos_token}assistant", add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f"{tokenizer.eos_token}", add_special_tokens=False).input_ids
        
    def load_data(self, path):
        samples = []
        with open(path, "r", encoding='utf-8') as f:
            for i, row in enumerate(f, 1):
                data = json.loads(row.strip())
                samples.append(data)
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def _create_chat_template(self, conversations):
        tools = conversations[0]["functions"] if (
            conversations and conversations[0]["role"] == "system" and conversations[0].get("functions")
        ) else None
        messages = conversations.copy()
        """
        apply_chat_template 参数：
        messages
        tokenize
        add_generation_prompt：是否在末尾添加生成提示
        max_length / truncation
        return_tensors
        padding
        """
        return self.tokenizer.apply_chat_template(
            messages,
            tokenizer=False,
            add_generation_prompt=False,
            tools=tools
        )
        
    def _generate_loss_mask(self, input_ids):
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i: i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end: end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start + 1, min(self.max_length, end + len(self.eos_id) + 1)):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask
    
    def __getitem__(self, index):
        sample = self.samples[index]
        
        # 构建对话提示
        prompt = self._create_chat_template(sample["conversations"])
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
        
        # 生成损失掩码
        loss_mask = self._generate_loss_mask(input_ids)
        
        X = torch.tensor(input_ids[:-1], dtype=torch.long)
        Y = torch.tensor(input_ids[1:], dtype=torch.long)
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)

        return X, Y, loss_mask
        

class DPODataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=4096):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.bos_id = tokenizer(f"{tokenizer.bos_token}assistant", add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f"{tokenizer.eos_token}", add_special_tokens=False).input_ids
        self.padding = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self.samples = self.load_data(jsonl_path)
        
    def load_data(self, path):
        samples = []
        with open(path, "r", encoding='utf-8') as f:
            for i, row in enumerate(f, 1):
                data = json.loads(row.strip())
                samples.append(data)
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def _generate_loss_mask(self, input_ids):
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i: i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end: end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start + 1, min(self.max_length, end + len(self.eos_id) + 1)):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask
    
    def __getitem__(self, index):
        item = self.samples[index]
        chosen = item["chosen"]
        rejected = item["rejected"]
        
        chosen_prompt = self.tokenizer.apply_chat_template(
            chosen, tokenizer=False, add_generation_prompt=False
        )
        rejected_prompt = self.tokenizer.apply_chat_template(
            rejected, tokenizer=False, add_generation_prompt=False
        )
        
        chosen_embedding = self.tokenizer(
            chosen_prompt, truncation=True, max_length=self.max_length, padding="max_length"
        )
        rejected_embedding = self.tokenizer(
            rejected_prompt, truncation=True, max_length=self.max_length, padding="max_length"
        )
        
        chosen_input_ids = chosen_embedding["input_ids"]
        chosen_loss_mask = self._generate_loss_mask(chosen_input_ids)
        rejected_input_ids = chosen_embedding["input_ids"]
        rejected_loss_mask = self._generate_loss_mask(rejected_input_ids)
        
        x_chosen = torch.tensor(chosen_input_ids[:-1], dtype=torch.long)
        y_chosen = torch.tensor(chosen_input_ids[1:], dtype=torch.long)
        mask_chosen = torch.tensor(chosen_loss_mask[1:], dtype=torch.long)
        x_rejected = torch.tensor(rejected_input_ids[:-1], dtype=torch.long)
        y_rejected = torch.tensor(rejected_input_ids[1:], dtype=torch.long)
        mask_rejected = torch.tensor(rejected_loss_mask[1:], dtype=torch.long)
        
        return {
            'x_chosen': x_chosen,
            'y_chosen': y_chosen,
            'mask_chosen': mask_chosen,
            'x_rejected': x_rejected,
            'y_rejected': y_rejected,
            'mask_rejected': mask_rejected
        }
        

class RLAIFDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self.load_data(jsonl_path)
        self.bos_id = tokenizer(f"{tokenizer.bos_token}assistant", add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f"{tokenizer.eos_token}", add_special_tokens=False).input_ids

    def load_data(self, path):
        samples = []
        with open(path, "r", encoding='utf-8') as f:
            for i, row in enumerate(f, 1):
                data = json.loads(row.strip())
                samples.append(data)
        return samples

    def __len__(self):
        return len(self.samples)

    def _create_chat_template(self, conversations):
        messages = []
        answer = ""

        for i, msg in enumerate(conversations):
            role = "user" if i % 2 == 0 else "assistant"
            messages.append({"role": role, "content": msg["content"]})

            answer = msg["content"]

        return self.tokenizer.apply_chat_template(
            messages[:-1], tokenizer=False, add_generation_prompt=True
        ), answer

    def __getitem__(self, index):
        sample = self.samples[index]
        prompt, answer = self._create_chat_template(sample["conversations"])

        return {
            "prompt": prompt,
            "answer": answer
        }


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("../model_weights")
    pretrain_dataset = PretrainDataset("./pretrain_hq.jsonl", tokenizer)
    print(pretrain_dataset[0])
