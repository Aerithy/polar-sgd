import os
import argparse
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import math
from typing import Dict, Any

# -----------------------------
# 导入你提供的模型代码
# -----------------------------
from psgd.models.llama.llama_nn import LlamaConfig, MyLlamaForCausalLM  # 替换为你的实际文件名，如 model.py

# -----------------------------
# 数据集与 Tokenization
# -----------------------------

class TokenizedDataset(Dataset):
    def __init__(self, dataset, tokenizer, seq_length=2048):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.seq_length = seq_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        text = self.dataset[idx]["text"]
        tokens = self.tokenizer(
            text,
            truncation=True,
            max_length=self.seq_length + 1,  # +1 for shifting
            padding=False,
            return_tensors=None
        )["input_ids"]

        # Ensure we have at least 2 tokens
        if len(tokens) < 2:
            tokens = [self.tokenizer.bos_token_id, self.tokenizer.eos_token_id]

        # Pad or truncate to seq_length + 1
        if len(tokens) > self.seq_length + 1:
            tokens = tokens[:self.seq_length + 1]
        else:
            tokens = tokens + [self.tokenizer.pad_token_id] * (self.seq_length + 1 - len(tokens))

        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        labels = torch.tensor(tokens[1:], dtype=torch.long)
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask
        }

def get_dataloader(
    dataset_name: str = "wikitext",
    dataset_config: str = "wikitext-2-raw-v1",
    tokenizer_name: str = "meta-llama/Llama-2-7b-hf",  # 或使用 "hf-internal-testing/llama-tokenizer" 如果无权限
    seq_length: int = 1024,
    batch_size: int = 1,
    num_workers: int = 2,
    split: str = "train",
    use_auth_token: bool = False,
):
    # Load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            use_fast=False,
            trust_remote_code=False,
            use_auth_token=use_auth_token
        )
    except OSError:
        print("⚠️ Cannot load official LLaMA tokenizer. Using a compatible one.")
        tokenizer = AutoTokenizer.from_pretrained(
            "hf-internal-testing/llama-tokenizer",
            use_fast=False
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    dataset = load_dataset(dataset_name, dataset_config, split=split)

    # Tokenize
    tokenized_dataset = TokenizedDataset(dataset, tokenizer, seq_length=seq_length)

    # Distributed sampler
    sampler = None
    if dist.is_initialized():
        sampler = torch.utils.data.distributed.DistributedSampler(
            tokenized_dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=True
        )

    dataloader = DataLoader(
        tokenized_dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    return dataloader, tokenizer

# -----------------------------
# 训练函数
# -----------------------------

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--single", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seq_length", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dataset", type=str, default="wikitext")
    parser.add_argument("--dataset_config", type=str, default="wikitext-2-raw-v1")
    parser.add_argument("--tokenizer", type=str, default="hf-internal-testing/llama-tokenizer")
    parser.add_argument("--use_auth_token", action="store_true")
    parser.add_argument("--output_dir", type=str, default="./llama7b_checkpoints")
    args = parser.parse_args()

    # Setup DDP
    single = args.single
    if single == 0:
        local_rank = dist.get_rank()
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        world_size = dist.get_world_size()
        rank = dist.get_rank()
    else:
        rank = 0
        world_size = 1

    device = torch.device(f"cuda:{local_rank}" if single == 0 else "cuda" if torch.cuda.is_available() else "cpu")

    # Config
    config = LlamaConfig(
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        rope_theta=10000.0,
        pad_token_id=0,
        tie_word_embeddings=True
    )

    # Model
    model = MyLlamaForCausalLM(config)
    model.to(device)

    if single == 0:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Data
    dataloader, tokenizer = get_dataloader(
        dataset_name=args.dataset,
        dataset_config=args.dataset_config,
        tokenizer_name=args.tokenizer,
        seq_length=args.seq_length,
        batch_size=args.batch_size,
        use_auth_token=args.use_auth_token,
        split="train"
    )

    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"✅ Training on {args.dataset} with seq_len={args.seq_length}, batch_size={args.batch_size}")
        print(f"Model params: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")

    # Training loop
    model.train()
    global_step = 0
    for epoch in range(args.epochs):
        if dist.is_initialized():
            dataloader.sampler.set_epoch(epoch)

        if rank == 0:
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        else:
            pbar = dataloader

        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs #.loss

            loss.backward()
            optimizer.step()

            global_step += 1

            if rank == 0:
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
                if global_step % 100 == 0:
                    print(f"Step {global_step}, Loss: {loss.item():.4f}")

        # Save checkpoint (only rank 0)
        if rank == 0:
            ckpt_path = os.path.join(args.output_dir, f"llama7b_epoch{epoch+1}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
            }, ckpt_path)
            print(f"✅ Saved checkpoint to {ckpt_path}")

    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    train()