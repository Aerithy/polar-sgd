import os
import datetime
import argparse
import logging
import numpy as np

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    BertModel,
    get_linear_schedule_with_warmup,
)
from transformers.models.bert.modeling_bert import (
    BertLayer,
    BertEmbeddings,
    BertPooler,
    BertEncoder,
)
from datasets import load_dataset, load_from_disk
from tqdm import tqdm
from utils.buffer import TensorBuffer

from polar_trainer import PolarTrainer, process_group_setup

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model", type=str, default="bert-base-uncased", help="model name"
)
parser.add_argument(
    "--pretrained", type=bool, default=False, help="use pretrained model"
)
parser.add_argument(
    "--data_path",
    type=str,
    default="data/glue/sst2",
    help="path to the training data",
)
parser.add_argument(
    "--model_path",
    type=str,
    default="models/bert-base-uncased",
    help="path to the model",
)
parser.add_argument(
    "--tokenizer_path",
    type=str,
    default="tokenizer/bert-base-uncased",
    help="path to the tokenizer",
)
parser.add_argument(
    "--num_labels", type=int, default=2, help="classification kinds"
)
parser.add_argument(
    "--max_length", type=int, default=128, help="max length of the input sequence"
)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--lr", type=float, default=1e-9)
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--seed", type=int, default=54)
parser.add_argument("--local_steps", type=int, default=2, help="local steps")
parser.add_argument("--using_hook", type=bool, default=False)
parser.add_argument("--single_node", type=bool, default=True)

args = parser.parse_args()

global_group, inter_group, local_group = process_group_setup()

config = AutoConfig.from_pretrained(
    "models/bert-base-uncased", torch_dtype=torch.float32, num_labels=2
)
model = AutoModelForSequenceClassification.from_config(config)

trainer = PolarTrainer(args=args, inter_group=inter_group, local_group=local_group, model=model)

trainer.train()

def find_hooked_layers(model: torch.nn.Module):
    hooked_layers = []

    # 递归遍历所有子模块
    for name, layer in model.named_modules():
        has_hook = False
        hook_types = []
        
        # 检查三类钩子是否存在
        if hasattr(layer, '_forward_pre_hooks') and layer._forward_pre_hooks:
            has_hook = True
            hook_types.append("forward_pre_hook")
        if hasattr(layer, '_forward_hooks') and layer._forward_hooks:
            has_hook = True
            hook_types.append("forward_hook")
        if hasattr(layer, '_backward_hooks') and layer._backward_hooks:
            has_hook = True
            hook_types.append("backward_hook")
        
        # 记录带钩子的层信息
        if has_hook:
            hooked_layers.append((name, layer, hook_types))
    
    return hooked_layers