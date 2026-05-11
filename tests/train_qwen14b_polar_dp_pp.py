#!/usr/bin/env python3
"""
Polar-SGD pretraining for Qwen2.5 models with DP+PP parallelism.
Based on train_qwen.py and polar-sgd framework.
"""

from psgd.parallelism.polar.wrapper import PolarParallel

import os
import argparse
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.pipelining import Schedule1F1B
from torch.distributed.device_mesh import init_device_mesh
from torch.utils.data import DataLoader, IterableDataset
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from typing import Iterable, Iterator, List, Optional
from dataclasses import dataclass

# -----------------------------
# Training Configuration (aligned with train_qwen.py)
# -----------------------------
@dataclass
class TrainConfig:
    model_name: str = "Qwen/Qwen2.5-14B-Instruct"
    tokenizer_name: str = None
    dataset_name_or_path: str = "HuggingFaceFW/fineweb"
    dataset_config: Optional[str] = None
    text_field: str = "text"
    seq_len: int = 4096
    per_device_batch_size: int = 2
    grad_accum_steps: int = 4
    lr: float = 2.0e-4
    warmup_ratio: float = 0.02
    max_tokens: int = 0
    max_steps: int = 10
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    clip_norm: float = 1.0
    log_interval: int = 10
    save_interval: int = 1000
    save_dir: str = "checkpoints/qwen2_5_14b_instruct"
    num_workers: int = 2
    use_flash_attn: bool = True
    bf16: bool = True
    fp16: bool = False
    activation_checkpointing: bool = True


# -----------------------------
# Streaming Dataset (from train_qwen.py)
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
    cfg: TrainConfig,
    tokenizer,
    pp_size: int,
):
    """Build streaming dataloader aligned with train_qwen.py."""
    ds_kwargs = {}
    if cfg.dataset_config:
        ds_kwargs["name"] = cfg.dataset_config
    
    # Use streaming dataset as in train_qwen.py
    dataset = load_dataset(cfg.dataset_name_or_path, **ds_kwargs, split="train", streaming=True)
    tokenized_dataset = StreamingTokenDataset(dataset, tokenizer, cfg.text_field, cfg.seq_len)

    dataloader = DataLoader(
        tokenized_dataset,
        batch_size=cfg.per_device_batch_size,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True
    )
    return dataloader


# -----------------------------
# Qwen Model Partitioning for Pipeline Parallelism
# -----------------------------
def partition_qwen_model(
    model,
    stage_idx: int,
    num_stages: int,
    tp_mesh=None,
    enable_sp: bool = False,
):
    """
    Partition Qwen model for pipeline parallelism.
    - Keep only layers assigned to this stage
    - Remove unused components (embeddings, lm_head, etc.)
    - Add custom forward method to handle partitioned model
    """
    config = model.config
    num_layers = config.num_hidden_layers

    # Flexible layer assignment (supports non-divisible cases)
    layers_per_stage = num_layers // num_stages
    remainder = num_layers % num_stages
    start_layer = stage_idx * layers_per_stage + min(stage_idx, remainder)
    end_layer = start_layer + layers_per_stage + (1 if stage_idx < remainder else 0)

    # Qwen2 uses model.layers (ModuleList)
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
        if hasattr(model, 'norm'):
            model.norm = None
    # Last stage: keep lm_head and norm, remove embed_tokens
    elif stage_idx == num_stages - 1:
        model.model.embed_tokens = None
    # Middle stages: remove all non-layer components
    else:
        model.model.embed_tokens = None
        if hasattr(model.model, 'final_norm'):
            model.model.final_norm = None
        if hasattr(model, 'norm'):
            model.norm = None
        model.lm_head = None

    # Save reference to original model for position embeddings
    original_model = model.model
    rope_theta = getattr(config, 'rope_theta', 10000.0)
    max_position_embeddings = getattr(config, 'max_position_embeddings', 4096)
    
    # Custom forward method for partitioned model with optional SP and proper RoPE handling
    def custom_forward(input_ids_or_hidden, attention_mask=None):
        if model.model.embed_tokens is not None:
            # Stage 0: input is token IDs
            hidden_states = model.model.embed_tokens(input_ids_or_hidden)
        else:
            # Stage 1+: input is hidden states
            hidden_states = input_ids_or_hidden

        tp_size = 1
        tp_rank = 0
        tp_group = None
        if enable_sp and tp_mesh is not None:
            tp_size = tp_mesh.size()
            tp_rank = tp_mesh.get_local_rank()
            tp_group = tp_mesh.get_group()
            if tp_size > 1:
                seq_len = hidden_states.shape[1]
                if seq_len % tp_size != 0:
                    raise ValueError(
                        f"SP requires seq_len divisible by tp_size: {seq_len} % {tp_size} != 0"
                    )
                local_len = seq_len // tp_size
                hidden_states = hidden_states[
                    :, tp_rank * local_len : (tp_rank + 1) * local_len
                ].contiguous()

        def _gather_seq(local_hidden):
            if tp_size == 1:
                return local_hidden
            gathered = [torch.empty_like(local_hidden) for _ in range(tp_size)]
            dist.all_gather(gathered, local_hidden, group=tp_group)
            return torch.cat(gathered, dim=1)

        def _build_position_embeddings(full_hidden):
            seq_length = full_hidden.shape[1]
            batch_size = full_hidden.shape[0]
            position_ids = (
                torch.arange(seq_length, device=full_hidden.device)
                .unsqueeze(0)
                .repeat(batch_size, 1)
            )
            if hasattr(model.model, 'rotary_emb') and model.model.rotary_emb is not None:
                return model.model.rotary_emb(full_hidden, position_ids)
            if hasattr(original_model, 'rotary_emb') and original_model.rotary_emb is not None:
                return original_model.rotary_emb(full_hidden, position_ids)
            from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding
            rotary_emb = Qwen2RotaryEmbedding(
                config.hidden_size // config.num_attention_heads,
                rope_theta=rope_theta,
                max_position_embeddings=max_position_embeddings
            ).to(full_hidden.device)
            return rotary_emb(full_hidden, position_ids)

        # Pass through layers
        for layer in model.model.layers:
            if isinstance(layer, torch.nn.Identity):
                hidden_states = layer(hidden_states)
            else:
                if tp_size > 1:
                    hidden_states = _gather_seq(hidden_states)
                position_embeddings = _build_position_embeddings(hidden_states)
                # Pass position_embeddings to Qwen2 layers
                layer_outputs = layer(
                    hidden_states, 
                    attention_mask=attention_mask,
                    position_embeddings=position_embeddings
                )
                # Qwen2 layers return a tuple (hidden_states, attention_weights)
                if isinstance(layer_outputs, tuple):
                    hidden_states = layer_outputs[0]
                else:
                    hidden_states = layer_outputs
                if tp_size > 1:
                    seq_len = hidden_states.shape[1]
                    local_len = seq_len // tp_size
                    hidden_states = hidden_states[
                        :, tp_rank * local_len : (tp_rank + 1) * local_len
                    ].contiguous()

        # Apply final norm if present
        if (hasattr(model, 'norm') and model.norm is not None) or (
            hasattr(model.model, 'final_norm') and model.model.final_norm is not None
        ):
            if tp_size > 1:
                hidden_states = _gather_seq(hidden_states)
            if hasattr(model, 'norm') and model.norm is not None:
                hidden_states = model.norm(hidden_states)
            elif hasattr(model.model, 'final_norm') and model.model.final_norm is not None:
                hidden_states = model.model.final_norm(hidden_states)

        # Apply lm_head if present
        if model.lm_head is not None:
            if tp_size > 1:
                hidden_states = _gather_seq(hidden_states)
            return model.lm_head(hidden_states)
        else:
            if tp_size > 1:
                hidden_states = _gather_seq(hidden_states)
            return hidden_states

    # Replace forward method
    model.forward = custom_forward

    assigned_layers = list(range(start_layer, end_layer))
    print(f"[partition] Stage {stage_idx}: assigned layers {assigned_layers}")
    print(f"[partition] Stage {stage_idx}: lm_head={model.lm_head is not None}, "
          f"embed_tokens={model.model.embed_tokens is not None}")
    
    return model


def build_qwen_model(cfg: TrainConfig):
    """Build Qwen model with configuration from train_qwen.py."""
    attn_impl = "flash_attention_2" if cfg.use_flash_attn else None
    kwargs = {
        "torch_dtype": torch.bfloat16 if cfg.bf16 else (torch.float16 if cfg.fp16 else None),
        "attn_implementation": attn_impl,
        "trust_remote_code": True,
    }
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name, **kwargs)

    if cfg.activation_checkpointing:
        # Disable KV cache for gradient checkpointing correctness.
        model.config.use_cache = False
        model.gradient_checkpointing_enable()

    return model


def build_tokenizer(cfg: TrainConfig):
    """Build tokenizer aligned with train_qwen.py."""
    tokenizer_name = cfg.tokenizer_name or cfg.model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return tokenizer


def apply_tensor_parallel(stage_model, tp_mesh):
    """Apply tensor parallelism to Qwen2 transformer blocks within a PP stage."""
    if tp_mesh.size() == 1:
        return stage_model

    try:
        from torch.distributed.tensor.parallel import (
            parallelize_module,
            ColwiseParallel,
            RowwiseParallel,
        )
    except Exception as exc:  # pragma: no cover - import depends on torch build
        raise RuntimeError(
            "Tensor parallel requires torch.distributed.tensor.parallel. "
            "Please use a torch build with TP support."
        ) from exc

    for layer in getattr(stage_model.model, "layers", []):
        if isinstance(layer, torch.nn.Identity):
            continue

        plan = {}
        attn = getattr(layer, "self_attn", None)
        if attn is not None:
            if hasattr(attn, "q_proj"):
                plan["self_attn.q_proj"] = ColwiseParallel()
            if hasattr(attn, "k_proj"):
                plan["self_attn.k_proj"] = ColwiseParallel()
            if hasattr(attn, "v_proj"):
                plan["self_attn.v_proj"] = ColwiseParallel()
            if hasattr(attn, "o_proj"):
                plan["self_attn.o_proj"] = RowwiseParallel()

        mlp = getattr(layer, "mlp", None)
        if mlp is not None:
            if hasattr(mlp, "gate_proj"):
                plan["mlp.gate_proj"] = ColwiseParallel()
            if hasattr(mlp, "up_proj"):
                plan["mlp.up_proj"] = ColwiseParallel()
            if hasattr(mlp, "down_proj"):
                plan["mlp.down_proj"] = RowwiseParallel()

        if plan:
            parallelize_module(layer, tp_mesh, plan)

    return stage_model


# -----------------------------
# Main Training Loop
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Polar-SGD pretraining for Qwen2.5 models")
    
    # Model and tokenizer
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-14B-Instruct")
    parser.add_argument("--tokenizer-name", type=str, default=None)
    
    # Dataset
    parser.add_argument("--dataset-name-or-path", type=str, default="HuggingFaceFW/fineweb")
    parser.add_argument("--dataset-config", type=str, default=None)
    parser.add_argument("--text-field", type=str, default="text")
    
    # Training hyperparameters (from config)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2.0e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.02)
    parser.add_argument("--max-tokens", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--clip-norm", type=float, default=1.0)
    
    # Logging and saving
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--save-interval", type=int, default=1000)
    parser.add_argument("--save-dir", type=str, default="checkpoints/qwen2_5_14b_instruct")
    
    # Data loader
    parser.add_argument("--num-workers", type=int, default=2)
    
    # Mixed precision and optimization
    parser.add_argument("--use-flash-attn", action="store_true", default=True)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--fp16", action="store_true", default=False)
    parser.add_argument("--activation-checkpointing", action="store_true", default=True)
    
    # Parallelism
    parser.add_argument("--pp-size", type=int, default=1)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--enable-sp", action="store_true", default=False)
    parser.add_argument("--micro-batches", type=int, default=1)
    parser.add_argument("--comm-timing", type=int, default=-1)
    parser.add_argument("--using-polar", type=bool, default=True)
    
    # Polar hooks
    parser.add_argument(
        "--polar-hook",
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
        "--polar-beta",
        type=float,
        default=0.9,
        help="EMA momentum beta for polar_hook=momentum.",
    )
    
    # Baseline mode
    parser.add_argument(
        "--baseline-mode",
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
        "--use-local-sgd",
        action="store_true",
        help="Enable Local-SGD mode (sync parameters every N steps)"
    )
    parser.add_argument(
        "--local-sgd-steps",
        type=int,
        default=10,
        help="Synchronize parameters every N steps in Local-SGD mode"
    )

    args = parser.parse_args()
    
    # Create config object aligned with train_qwen.py
    cfg = TrainConfig(
        model_name=args.model_name,
        tokenizer_name=args.tokenizer_name,
        dataset_name_or_path=args.dataset_name_or_path,
        dataset_config=args.dataset_config,
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
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        save_dir=args.save_dir,
        num_workers=args.num_workers,
        use_flash_attn=args.use_flash_attn,
        bf16=args.bf16,
        fp16=args.fp16,
        activation_checkpointing=args.activation_checkpointing,
    )

    # Initialize distributed
    dist.init_process_group(backend="nccl", init_method="env://")
    world_size = dist.get_world_size()
    
    pp_size = args.pp_size
    tp_size = args.tp_size
    denom = pp_size * tp_size
    assert world_size % denom == 0, (
        f"world_size {world_size} must be divisible by PP_SIZE*TP_SIZE={denom}"
    )
    dp_size = world_size // denom
    device_mesh = init_device_mesh(
        "cuda",
        (dp_size, pp_size, tp_size),
        mesh_dim_names=("dp", "pp", "tp"),
    )
    dp_mesh = device_mesh["dp"]
    pp_mesh = device_mesh["pp"]
    tp_mesh = device_mesh["tp"]

    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    
    # Enable TF32 for better performance
    torch.backends.cuda.matmul.allow_tf32 = True

    # Build tokenizer
    tokenizer = build_tokenizer(cfg)
    
    # Build and partition Qwen model
    model = build_qwen_model(cfg)
    stage_idx = pp_mesh.get_local_rank()
    print(f"Stage index: {stage_idx} / {pp_size}")
    
    # Partition model for pipeline parallelism
    stage_model = partition_qwen_model(
        model,
        stage_idx,
        pp_size,
        tp_mesh=tp_mesh,
        enable_sp=args.enable_sp,
    )
    stage_model = apply_tensor_parallel(stage_model, tp_mesh)
    
    dp_rank = dp_mesh.get_local_rank()
    print(f"DP rank: {dp_rank} / {dp_size}")
    tp_rank = tp_mesh.get_local_rank()
    print(f"TP rank: {tp_rank} / {tp_size}")
    
    # Get dataloader
    dataloader = get_dataloader(cfg, tokenizer, pp_size)

    def loss_fn(output, target):
        """LM loss function with padding mask."""
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
    from dataclasses import dataclass
    main()