import os
import torch
import argparse
from polar_trainer import PolarTrainer, process_group_setup
from transformers import AutoConfig, AutoModelForSequenceClassification

# 设置环境变量确保单节点运行
os.environ["RANK"] = "0"
os.environ["LOCAL_RANK"] = "0"  
os.environ["WORLD_SIZE"] = "1"

# 创建参数
args = argparse.Namespace(
    model="bert-base-uncased",
    pretrained=False,
    data_path="data/glue/sst2",
    model_path="models/bert-base-uncased", 
    tokenizer_path="tokenizer/bert-base-uncased",
    num_labels=2,
    max_length=128,
    batch_size=8,  # 小batch size
    lr=1e-6,       # 很低的学习率
    epochs=1,      # 只训练1个epoch
    seed=42,
    local_steps=2,
    using_hook=False,  # 禁用hook
    single_node=True
)

print("设置分布式环境...")
global_group, inter_group, local_group = process_group_setup()

print("创建模型...")
config = AutoConfig.from_pretrained(
    args.model_path, torch_dtype=torch.float32, num_labels=args.num_labels
)
model = AutoModelForSequenceClassification.from_config(config)

print("初始化trainer...")
trainer = PolarTrainer(
    args=args, 
    inter_group=inter_group, 
    local_group=local_group, 
    model=model
)

print("开始训练...")
trainer._train()
