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
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from datasets import load_dataset, load_from_disk
from tqdm import tqdm

from tokenizer.tokenize_preprocess import tokenize_function
from utils.buffer import TensorBuffer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def _train(args: argparse.Namespace, inter_group, local_group):
    local_rank = dist.get_rank(group=local_group)
    if local_rank < 0:
        logger.error("local_rank is less than 0, check the local_group initialization.")
        return
    device = torch.device(f"cuda:{local_rank}")

    dataset = load_from_disk(args.data_path)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset = tokenized_dataset.remove_columns(["sentence", "idx"])
    tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
    tokenized_dataset.set_format("torch")

    train_sampler = DistributedSampler(
        tokenized_dataset["train"],
        num_replicas=dist.get_world_size(),
        rank=dist.get_rank(),
        shuffle=True,
    )

    eval_sampler = DistributedSampler(
        tokenized_dataset["validation"],
        num_replicas=dist.get_world_size(),
        rank=dist.get_rank(),
        shuffle=False,
    )

    train_dataloader = DataLoader(
        tokenized_dataset["train"], batch_size=args.batch_size, sampler=train_sampler
    )

    eval_dataloader = DataLoader(
        tokenized_dataset["validation"],
        batch_size=args.batch_size,
        sampler=eval_sampler,
    )

    if args.pretrained:
        model = AutoModelForSequenceClassification.from_pretrained(
            args.epochs, torch_dtype=torch.float16, num_labels=args.num_labels
        )
    else:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(
            args.model_path, torch_dtype=torch.float16, num_labels=args.num_labels
        )
        model = AutoModelForSequenceClassification.from_config(config)
        
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_dataloader) * args.epochs
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0.1 * total_steps, num_training_steps=total_steps
    )

    send_buffers = [
        torch.zeros_like(param) for param in model.parameters() if param.requires_grad
    ]
    print("send_buffers")
    LOCAL_STEPS = 1  # 每4个batch同步一次梯度
    current_local_step = 0  # 当前本地步数计数器
    print(f"LOCAL_STEPS: {LOCAL_STEPS}")
    # 训练循环
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}")

        for batch_idx, batch in enumerate(progress_bar):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            current_local_step += 1
            need_sync = (current_local_step % LOCAL_STEPS == 0) or (batch_idx + 1 == len(train_dataloader))

            if need_sync:
                grad_vec = [
                    parameter.grad
                    for parameter in model.parameters()
                    if parameter.requires_grad
                ]
                for grad, send_buffer in zip(grad_vec, send_buffers):
                    send_buffer[:] = grad
                    
                tensor_buffer = TensorBuffer(send_buffers)
                flat_buffer = tensor_buffer.buffer
                dist.all_reduce(flat_buffer)
                tensor_buffer.buffer = flat_buffer
                grads = tensor_buffer.deflatten()

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            else:
                optimizer.step()
                optimizer.zero_grad()    
            
            total_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})

        avg_train_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1} Average Loss: {avg_train_loss:.4f}")

def process_group_setup():
    # init the global process group
    rank = os.environ["RANK"]
    local_rank = os.environ["LOCAL_RANK"]
    world_size = os.environ["WORLD_SIZE"]
    rank = int(rank)
    local_rank = int(local_rank)
    world_size = int(world_size)

    print(f"rank: {rank}, local_rank: {local_rank}, world_size: {world_size}")

    torch.cuda.set_device(local_rank)
    global_group = dist.init_process_group(
        backend="nccl",
        init_method="env://",
        rank=rank,
        world_size=world_size,
    )

    # init the local process group
    local_world_size = os.environ["LOCAL_WORLD_SIZE"]
    local_world_size = int(local_world_size)
    node_id = rank // local_world_size

    local_ranks = list(
        range(node_id * local_world_size, (node_id + 1) * local_world_size)
    )
    local_group = dist.new_group(ranks=local_ranks)

    print(f"local_groups: {local_ranks}")

    # init the inter-node process group
    inter_ranks = list(range(0, world_size, local_world_size))
    inter_group = dist.new_group(ranks=inter_ranks)

    print(f"inter_groups: {inter_ranks}")

    # torch.cuda.set_device(local_rank)
    return global_group, inter_group, local_group


if __name__ == "__main__":
    parser = argparse.Namespace()
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
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local_steps", type=int, default=4, help="local steps")

    logger.info("Setting up process groups...")
    global_group, inter_group, local_group = process_group_setup()

    logger.info("Starting training...")
    train(args=parser.parse_args(), inter_group=inter_group, local_group=local_group)
