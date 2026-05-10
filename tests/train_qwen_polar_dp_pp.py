#!/usr/bin/env python3
"""
Polar-SGD pretraining for Qwen models with DP+PP parallelism.
"""

from ast import mod
from psgd.parallelism.polar.wrapper import PolarParallel

import os
import argparse
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.pipelining import PipelineStage, Schedule1F1B
from torch.distributed.device_mesh import init_device_mesh
from torch.utils.data import DataLoader, IterableDataset
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from typing import Iterable, Iterator, List, Optional


# -----------------------------
# Dataset (from train_qwen.py)
# -----------------------------
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
                attention_mask = torch.ones_like(input_ids)
                yield {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


def get_dataloader(
    pp_size: int,
    dataset_name_or_path: str = "wikitext",
    dataset_config: Optional[str] = None,
    tokenizer_name: str = "Qwen/Qwen-7B",
    seq_length: int = 1024,
    batch_size: int = 1,
    num_workers: int = 2,
    text_field: str = "text",
    use_auth_token: bool = False,
):
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        use_fast=False,
        trust_remote_code=True,
        use_auth_token=use_auth_token
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load streaming dataset
    ds_kwargs = {}
    if dataset_config:
        ds_kwargs["name"] = dataset_config
    dataset = load_dataset(dataset_name_or_path, **ds_kwargs, split="train", streaming=True)
    tokenized_dataset = StreamingTokenDataset(dataset, tokenizer, text_field, seq_length)

    dataloader = DataLoader(
        tokenized_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    return dataloader, tokenizer


# -----------------------------
# Qwen Model Partitioning
# -----------------------------
def partition_qwen_model(model, stage_idx: int, num_stages: int):
    """
    Partition Qwen model for pipeline parallelism.
    - Keep only layers assigned to this stage
    - Remove unused components (embeddings, lm_head, etc.)
    """
    config = model.config
    num_layers = config.num_hidden_layers

    # Flexible layer assignment (supports non-divisible cases)
    layers_per_stage = num_layers // num_stages
    remainder = num_layers % num_stages
    start_layer = stage_idx * layers_per_stage + min(stage_idx, remainder)
    end_layer = start_layer + layers_per_stage + (1 if stage_idx < remainder else 0)

    # Remove layers not assigned to this stage
    # Qwen uses model.layers (ModuleList)
    layers_to_keep = list(range(start_layer, end_layer))
    new_layers = torch.nn.ModuleList([
        model.model.layers[i] for i in layers_to_keep
    ])
    model.model.layers = new_layers

    # If current stage has no layers, add an Identity
    if len(model.model.layers) == 0:
        model.model.layers = torch.nn.ModuleList([torch.nn.Identity()])

    # Stage 0: keep embed_tokens, remove lm_head and final_norm
    if stage_idx == 0:
        model.lm_head = None
        if hasattr(model.model, 'final_norm'):
            model.model.final_norm = None
    # Last stage: keep lm_head and final_norm, remove embed_tokens
    elif stage_idx == num_stages - 1:
        model.model.embed_tokens = None
    # Middle stages: remove all non-layer components
    else:
        model.model.embed_tokens = None
        if hasattr(model.model, 'final_norm'):
            model.model.final_norm = None
        model.lm_head = None

    assigned_layers = list(range(start_layer, end_layer))
    print(f"[partition] Stage {stage_idx}: assigned layers {assigned_layers}")
    
    return model


def build_qwen_model(model_name: str, bf16: bool = False, fp16: bool = False):
    """Build Qwen model with appropriate dtype."""
    attn_impl = "flash_attention_2" if torch.cuda.is_available() else None
    kwargs = {
        "torch_dtype": torch.bfloat16 if bf16 else (torch.float16 if fp16 else None),
        "attn_implementation": attn_impl,
        "trust_remote_code": True,
    }
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    return model


# -----------------------------
# Main Training Loop
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Polar-SGD pretraining for Qwen models")
    
    # Model and tokenizer
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen-7B")
    parser.add_argument("--tokenizer-name", type=str, default=None)
    
    # Dataset
    parser.add_argument("--dataset-name-or-path", type=str, required=True)
    parser.add_argument("--dataset-config", type=str, default=None)
    parser.add_argument("--text-field", type=str, default="text")
    
    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seq_length", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-4)
    
    # Parallelism
    parser.add_argument("--pp_size", type=int, default=1)
    parser.add_argument("--micro_batches", type=int, default=1)
    parser.add_argument("--comm_timing", type=int, default=-1)
    parser.add_argument("--using_polar", type=bool, default=True)
    
    # Polar hooks
    parser.add_argument(
        "--polar_hook",
        type=str,
        default="momentum",
        choices=["io", "momentum", "gpipe", "ef_only", "scaling_only", "none"],
        help=(
            "Which POLAR gradient prediction hook to use: "
            "'momentum' (no scaling, EMA momentum extrapolation), "
            "'io' (IO-optimized scaling hook), "
            "'gpipe' (legacy scaling hook), "
            "'ef_only' (error feedback only), "
            "'scaling_only' (scaling only), "
            "or 'none' (no scaling, no error feedback)."
        ),
    )
    parser.add_argument(
        "--polar_beta",
        type=float,
        default=0.9,
        help="EMA momentum beta for polar_hook=momentum.",
    )
    
    # Training limits
    parser.add_argument(
        "--max_steps",
        type=int,
        default=500,
        help="Maximum training steps (batches) to run; default 500.",
    )
    
    # Baseline mode
    parser.add_argument(
        "--baseline_mode",
        type=str,
        default="manual",
        choices=["manual", "ddp"],
        help=(
            "Baseline training mode for DP+PP: 'manual' does explicit DP "
            "gradient all-reduce after backward; 'ddp' wraps each stage with "
            "DDP (may OOM in pipeline scenarios)."
        ),
    )
    
    # Local-SGD arguments
    parser.add_argument(
        "--use_local_sgd",
        action="store_true",
        help="Enable Local-SGD mode (sync parameters every N steps)"
    )
    parser.add_argument(
        "--local_sgd_steps",
        type=int,
        default=10,
        help="Synchronize parameters every N steps in Local-SGD mode"
    )
    
    # Mixed precision
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")

    args = parser.parse_args()
    
    # Set tokenizer name if not provided
    if args.tokenizer_name is None:
        args.tokenizer_name = args.model_name

    # Initialize distributed
    dist.init_process_group(backend="nccl", init_method="env://")
    world_size = dist.get_world_size()
    
    pp_size = args.pp_size
    assert world_size % pp_size == 0, f"world_size {world_size} must be divisible by PP_SIZE {pp_size}"
    dp_size = world_size // pp_size
    device_mesh = init_device_mesh("cuda", (dp_size, pp_size), mesh_dim_names=("dp", "pp"))
    dp_mesh = device_mesh["dp"]
    pp_mesh = device_mesh["pp"]

    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    # Build and partition Qwen model
    model = build_qwen_model(args.model_name, args.bf16, args.fp16)
    stage_idx = pp_mesh.get_local_rank()
    print(f"Stage index: {stage_idx} / {pp_size}")
    
    # Partition model for pipeline parallelism
    stage_model = partition_qwen_model(model, stage_idx, pp_size)
    
    dp_rank = dp_mesh.get_local_rank()
    print(f"DP rank: {dp_rank} / {dp_size}")
    
    # Get dataloader
    dataloader, tokenizer = get_dataloader(
        pp_size=pp_size,
        dataset_name_or_path=args.dataset_name_or_path,
        dataset_config=args.dataset_config,
        tokenizer_name=args.tokenizer_name,
        seq_length=args.seq_length,
        batch_size=args.batch_size,
        text_field=args.text_field,
    )

    def loss_fn(output, target):
        """LM loss function."""
        shift_logits = output[..., :-1, :].contiguous()
        shift_labels = target[..., 1:].contiguous()
        return F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=tokenizer.pad_token_id,
        )
    
    trainer = PolarParallel(
        args=args,
        device_mesh=device_mesh,
        micro_batches=args.micro_batches,
        loss_fn=loss_fn,
        stage_model=stage_model,
        dataloader=dataloader,
        comm_timing=args.comm_timing,
        use_local_sgd=args.use_local_sgd,
        local_sgd_steps=args.local_sgd_steps,
        baseline_mode=args.baseline_mode,
    )

    trainer.train()


if __name__ == "__main__":
    main()