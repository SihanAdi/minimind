import json
import random
import re

import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import os

"""
TOKENIZERS_PARALLELISM: 控制 Hugging Face tokenizers 库并行
禁用 tokenizers 库的并行处理，强制其使用单线程模式
    - 避免与多进程训练的冲突
    - 在资源受限的环境中，禁用并行可以减少内存使用
"""
os.environ["TOKENIZERS_PARALLELISM"] = "false"