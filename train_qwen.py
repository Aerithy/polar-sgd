#!/usr/bin/env python3
"""
FSDP pretraining baseline for decoder-only LMs.

Notes:
- This script assumes a single-node multi-GPU setup.
- It is a baseline reference, not a full production pipeline.
"""

import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass
from typing import Iterable, Iterator, List, Optional

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, IterableDataset
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy, size_based_auto_wrap_policy

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


@dataclass
class TrainConfig:
    model_name: str
    tokenizer_name: str
    dataset_name_or_path: str
    dataset_config: Optional[str]
    text_field: str
    seq_len: int
    per_device_batch_size: int
    grad_accum_steps: int
    lr: float
    warmup_ratio: float
    max_tokens: int
    max_steps: int
    weight_decay: float
    beta1: float
    beta2: float
    clip_norm: float
    seed: int
    log_interval: int
    save_dir: str
    save_interval: int
    num_workers: int
    use_flash_attn: bool
    fp16: bool
    bf16: bool
    activation_checkpointing: bool
    hf_download_retries: int
    hf_download_retry_backoff: float


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="FSDP pretraining baseline for Qwen models")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--tokenizer-name", default=None)
    parser.add_argument("--dataset-name-or-path", required=True)
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.02)
    parser.add_argument("--max-tokens", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--clip-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--save-dir", default="checkpoints")
    parser.add_argument("--save-interval", type=int, default=1000)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--use-flash-attn", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--activation-checkpointing", action="store_true")

    # Hugging Face download resilience (helps when multiple ranks start at once)
    parser.add_argument(
        "--hf-download-retries",
        type=int,
        default=int(os.environ.get("HF_DOWNLOAD_RETRIES", "3")),
        help="Retries for Hugging Face model/tokenizer downloads.",
    )
    parser.add_argument(
        "--hf-download-retry-backoff",
        type=float,
        default=float(os.environ.get("HF_DOWNLOAD_RETRY_BACKOFF", "2.0")),
        help="Base backoff seconds between download retries (exponential).",
    )

    args = parser.parse_args()
    dataset_config = args.dataset_config or None

    return TrainConfig(
        model_name=args.model_name,
        tokenizer_name=args.tokenizer_name or args.model_name,
        dataset_name_or_path=args.dataset_name_or_path,
        dataset_config=dataset_config,
        text_field=args.text_field,
        seq_len=args.seq_len,
        per_device_batch_size=args.per_device_batch_size,
        grad_accum_steps=args.grad_accum_steps,
        lr=args.lr,
        warmup_ratio=args.warmup_ratio,
        max_tokens=args.max_tokens,
        max_steps=args.max_steps,
        weight_decay=args.weight_decay,
        beta1=args.beta1,
        beta2=args.beta2,
        clip_norm=args.clip_norm,
        seed=args.seed,
        log_interval=args.log_interval,
        save_dir=args.save_dir,
        save_interval=args.save_interval,
        num_workers=args.num_workers,
        use_flash_attn=args.use_flash_attn,
        fp16=args.fp16,
        bf16=args.bf16,
        activation_checkpointing=args.activation_checkpointing,
        hf_download_retries=args.hf_download_retries,
        hf_download_retry_backoff=args.hf_download_retry_backoff,
    )


class StreamingTokenDataset(IterableDataset):
    def __init__(self, dataset_iter: Iterable[dict], tokenizer, text_field: str, seq_len: int):
        self.dataset_iter = dataset_iter
        self.tokenizer = tokenizer
        self.text_field = text_field
        self.seq_len = seq_len

    def __iter__(self) -> Iterator[dict]:
        buffer: List[int] = []
        for sample in self.dataset_iter:
            text = sample.get(self.text_field, "")
            if not text:
                continue
            tokens = self.tokenizer(text, add_special_tokens=False)["input_ids"]
            buffer.extend(tokens)
            while len(buffer) >= self.seq_len + 1:
                chunk = buffer[: self.seq_len + 1]
                buffer = buffer[self.seq_len + 1 :]
                input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
                labels = torch.tensor(chunk[1:], dtype=torch.long)
                yield {"input_ids": input_ids, "labels": labels}


def setup_dist():
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    return local_rank


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_auto_wrap_policy(model):
    # Prefer transformer layer class if detected.
    layer_cls = None
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        if model.model.layers:
            layer_cls = model.model.layers[0].__class__
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        if model.transformer.h:
            layer_cls = model.transformer.h[0].__class__

    if layer_cls is not None:
        return transformer_auto_wrap_policy({layer_cls})

    return size_based_auto_wrap_policy(min_num_params=1_000_000)


def build_model(cfg: TrainConfig):
    attn_impl = "flash_attention_2" if cfg.use_flash_attn else None
    kwargs = {
        "torch_dtype": torch.bfloat16 if cfg.bf16 else (torch.float16 if cfg.fp16 else None),
        "attn_implementation": attn_impl,
        "trust_remote_code": True,
    }
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name, **kwargs)

    if cfg.activation_checkpointing:
        model.gradient_checkpointing_enable()

    return model


def _retry(fn, *, retries: int, backoff: float, what: str):
    """Simple exponential backoff retry wrapper."""
    last_err: Optional[BaseException] = None
    for attempt in range(retries + 1):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            last_err = e
            if attempt >= retries:
                break
            sleep_s = backoff * (2**attempt)
            # Avoid log spam from all ranks by printing only on rank0 when possible.
            if dist.is_available() and dist.is_initialized():
                if dist.get_rank() == 0:
                    print(f"[hf] {what} failed (attempt {attempt+1}/{retries+1}): {e}. Retrying in {sleep_s:.1f}s")
            else:
                print(f"[hf] {what} failed (attempt {attempt+1}/{retries+1}): {e}. Retrying in {sleep_s:.1f}s")
            time.sleep(sleep_s)
    assert last_err is not None
    raise last_err


def build_tokenizer(cfg: TrainConfig):
    """Build tokenizer in a distributed-safe way.

    When launching with torchrun, multiple ranks can concurrently try to download
    the same tokenizer files from Hugging Face Hub, which is prone to transient
    network errors (IncompleteRead / Response ended prematurely).

    Strategy:
    - Let rank0 download first (with retries), then barrier.
    - Other ranks wait, then load from cache/local disk.
    """

    def _load():
        return AutoTokenizer.from_pretrained(cfg.tokenizer_name, trust_remote_code=True)

    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        if rank == 0:
            tokenizer = _retry(
                _load,
                retries=cfg.hf_download_retries,
                backoff=cfg.hf_download_retry_backoff,
                what=f"AutoTokenizer.from_pretrained({cfg.tokenizer_name})",
            )
        dist.barrier()
        if rank != 0:
            tokenizer = _retry(
                _load,
                retries=cfg.hf_download_retries,
                backoff=cfg.hf_download_retry_backoff,
                what=f"AutoTokenizer.from_pretrained({cfg.tokenizer_name})",
            )
    else:
        tokenizer = _retry(
            _load,
            retries=cfg.hf_download_retries,
            backoff=cfg.hf_download_retry_backoff,
            what=f"AutoTokenizer.from_pretrained({cfg.tokenizer_name})",
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def get_dataset(cfg: TrainConfig, tokenizer):
    ds_kwargs = {}
    if cfg.dataset_config:
        ds_kwargs["name"] = cfg.dataset_config
    dataset = load_dataset(cfg.dataset_name_or_path, **ds_kwargs, split="train", streaming=True)
    return StreamingTokenDataset(dataset, tokenizer, cfg.text_field, cfg.seq_len)


def save_checkpoint(model: FSDP, optimizer: AdamW, step: int, save_dir: str):
    if dist.get_rank() != 0:
        return
    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, f"step_{step}")
    os.makedirs(ckpt_path, exist_ok=True)
    model_state = model.state_dict()
    torch.save(model_state, os.path.join(ckpt_path, "model.pt"))
    torch.save(optimizer.state_dict(), os.path.join(ckpt_path, "optim.pt"))


def main():
    cfg = parse_args()
    local_rank = setup_dist()
    set_seed(cfg.seed + dist.get_rank())

    torch.backends.cuda.matmul.allow_tf32 = True

    tokenizer = build_tokenizer(cfg)
    dataset = get_dataset(cfg, tokenizer)
    loader = DataLoader(
        dataset,
        batch_size=cfg.per_device_batch_size,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    model = build_model(cfg)
    wrap_policy = get_auto_wrap_policy(model)

    mixed_precision = None
    if cfg.bf16 or cfg.fp16:
        dtype = torch.bfloat16 if cfg.bf16 else torch.float16
        mixed_precision = MixedPrecision(
            param_dtype=dtype,
            reduce_dtype=dtype,
            buffer_dtype=dtype,
        )

    model = FSDP(
        model,
        auto_wrap_policy=wrap_policy,
        mixed_precision=mixed_precision,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=torch.cuda.current_device(),
    )

    optimizer = AdamW(
        model.parameters(),
        lr=cfg.lr,
        betas=(cfg.beta1, cfg.beta2),
        weight_decay=cfg.weight_decay,
    )

    world_size = dist.get_world_size()
    tokens_per_step = cfg.seq_len * cfg.per_device_batch_size * cfg.grad_accum_steps * world_size
    max_steps = cfg.max_steps
    if max_steps <= 0:
        if cfg.max_tokens <= 0:
            raise ValueError("Either --max-steps or --max-tokens must be set")
        max_steps = max(1, cfg.max_tokens // tokens_per_step)

    warmup_steps = max(1, int(max_steps * cfg.warmup_ratio))

    def lr_lambda(step: int):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, max_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    model.train()
    step = 0
    optimizer.zero_grad(set_to_none=True)

    for batch in loader:
        input_ids = batch["input_ids"].cuda(non_blocking=True)
        labels = batch["labels"].cuda(non_blocking=True)

        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss / cfg.grad_accum_steps
        loss.backward()

        if (step + 1) % cfg.grad_accum_steps == 0:
            clip_grad_norm_(model.parameters(), cfg.clip_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        if dist.get_rank() == 0 and step % cfg.log_interval == 0:
            lr = scheduler.get_last_lr()[0]
            print(json.dumps({"step": step, "loss": loss.item(), "lr": lr}))

        if step > 0 and step % cfg.save_interval == 0:
            save_checkpoint(model, optimizer, step, cfg.save_dir)

        step += 1
        if step >= max_steps:
            break

    if dist.get_rank() == 0:
        print("Training complete")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()