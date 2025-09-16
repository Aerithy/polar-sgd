import os
import datetime
import argparse
import logging
import numpy as np

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
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
from typing import List

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


torch.manual_seed(42)
np.random.seed(42)
# 如果使用CUDA，还需设置CUDA随机种子
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
device = torch.device(
    f"cuda:0"
) if torch.cuda.is_available() else torch.device("cpu")
writer = SummaryWriter(log_dir=f"./log/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")

pretrained = False

if pretrained:
    model = AutoModelForSequenceClassification.from_pretrained(
        "models/bert-base-uncased", torch_dtype=torch.float32, num_labels=2
    )
    model.to(device)
    print(next(model.parameters()).device)
else:
    config = AutoConfig.from_pretrained(
        "models/bert-base-uncased", torch_dtype=torch.float32, num_labels=2
    )
    model = AutoModelForSequenceClassification.from_config(config)
    model.to(device)
    print(next(model.parameters()).device)

tokenizer = AutoTokenizer.from_pretrained("tokenizer/bert-base-uncased")
dataset = load_from_disk("data/glue/sst2")

def tokenize_function(examples):
    return tokenizer(
        examples["sentence"],
        padding="max_length",
        truncation=True,
        max_length=128,
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True)
tokenized_dataset = tokenized_dataset.remove_columns(["sentence", "idx"])
tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
tokenized_dataset.set_format("torch")
tokenized_dataset = tokenized_dataset

print(next(model.parameters()).device)

train_dataloader = DataLoader(
    tokenized_dataset["train"],
    batch_size=32,
)
eval_dataloader = DataLoader(
    tokenized_dataset["validation"],
    batch_size=32,
)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-6, weight_decay=0.01, eps=1e-8, betas=(0.9, 0.999))
total_steps = len(train_dataloader) * 10
# scheduler = get_linear_schedule_with_warmup(
#     optimizer,
#     num_warmup_steps=0.1 * total_steps,
#     num_training_steps=total_steps,
# )
from torch.optim.lr_scheduler import CosineAnnealingLR
scheduler = CosineAnnealingLR(optimizer, T_max=50)


correct_predictions = 0
total_predictions = 0
# 训练循环
for epoch in range(10):
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}")
    for batch_idx, batch in enumerate(progress_bar):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        if torch.isnan(loss).any():
            logger.error(f"NaN loss detected at epoch {epoch+1}, batch {batch_idx}")
            # 可以选择跳过这个batch或者停止训练
            continue
        logits = outputs.logits
        loss.backward()
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)

        has_nan_grad = False
        for name, param in model.named_parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                logger.error(f"NaN gradient detected in {name} at epoch {epoch+1}, batch {batch_idx}")
                has_nan_grad = True
                break
        
        if has_nan_grad:
            # 跳过此次更新
            optimizer.zero_grad()
            continue

        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        _, predicted = torch.max(logits, dim=1)
        correct_predictions += (predicted == batch["labels"]).sum().item()
        total_predictions += batch["labels"].size(0)

        current_accuracy = correct_predictions / total_predictions

        avg_train_loss = total_loss / (batch_idx + 1)
        progress_bar.set_postfix({"loss": loss.item()})
    
    writer.add_scalar('Loss/train', avg_train_loss, epoch)
    writer.add_scalar('Learning Rate', scheduler.get_last_lr()[0], epoch)
    writer.add_scalar('Accuracy/train', current_accuracy, epoch)
    scheduler.step()
    avg_train_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch+1} Average Loss: {avg_train_loss:.4f}")