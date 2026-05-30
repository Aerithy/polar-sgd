#!/usr/bin/env python3
"""
Polar-SGD pretraining for Qwen2.5 models with DP+PP parallelism.
Based on train_qwen.py and polar-sgd framework.
"""

from psgd.parallelism.polar.wrapper import PolarParallel

import os
import argparse
import sys
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.pipelining import Schedule1F1B
from torch.distributed.device_mesh import init_device_mesh
from torch.utils.data import DataLoader, IterableDataset
from datasets import load_dataset
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from pathlib import Path
from typing import Iterable, Iterator, List, Optional
from dataclasses import dataclass

# -----------------------------
# Training Configuration (aligned with train_qwen.py)
# -----------------------------


def import_bitscom():
    try:
        import bitscom

        return bitscom
    except ImportError:
        repo_root = Path(__file__).resolve().parents[2]
        bitscom_python = repo_root / "bitscom" / "python"
        if bitscom_python.exists():
            sys.path.insert(0, str(bitscom_python))
        import bitscom

        return bitscom


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
    init_from_pretrained: bool = False


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
                labels = input_ids.clone()
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
    debug_nan_steps: int = 0,
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

    # Stage 0: keep embed_tokens, remove lm_head and final norm
    if stage_idx == 0:
        model.lm_head = None
        if hasattr(model.model, 'norm'):
            model.model.norm = None
        if hasattr(model.model, 'final_norm'):
            model.model.final_norm = None
    # Last stage: keep lm_head and norm, remove embed_tokens
    elif stage_idx == num_stages - 1:
        model.model.embed_tokens = None
    # Middle stages: remove all non-layer components
    else:
        model.model.embed_tokens = None
        if hasattr(model.model, 'norm'):
            model.model.norm = None
        if hasattr(model.model, 'final_norm'):
            model.model.final_norm = None
        model.lm_head = None

    # Save reference to original model for position embeddings
    original_model = model.model
    rope_theta = getattr(config, 'rope_theta', 10000.0)
    max_position_embeddings = getattr(config, 'max_position_embeddings', 4096)
    # The model is later materialized with to_empty(); HF RoPE buffers created
    # on meta can become uninitialized device buffers. Rebuild RoPE lazily.
    if hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb = None

    def get_fallback_rotary_embedding(device):
        rotary_emb = getattr(model.model, "_polar_rotary_emb", None)
        if rotary_emb is not None:
            return rotary_emb.to(device)

        import inspect
        from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding

        signature = inspect.signature(Qwen2RotaryEmbedding.__init__)
        params = signature.parameters
        kwargs = {}
        if "config" in params:
            kwargs["config"] = config
            if "device" in params:
                kwargs["device"] = device
            rotary_emb = Qwen2RotaryEmbedding(**kwargs)
        else:
            head_dim = config.hidden_size // config.num_attention_heads
            if "max_position_embeddings" in params:
                kwargs["max_position_embeddings"] = max_position_embeddings
            if "rope_theta" in params:
                kwargs["rope_theta"] = rope_theta
            elif "base" in params:
                kwargs["base"] = rope_theta
            if "device" in params:
                kwargs["device"] = device
            rotary_emb = Qwen2RotaryEmbedding(head_dim, **kwargs)

        rotary_emb = rotary_emb.to(device)
        model.model._polar_rotary_emb = rotary_emb
        return rotary_emb

    def apply_rotary_embedding(rotary_emb, hidden_states, position_ids, seq_length):
        import inspect

        signature = inspect.signature(rotary_emb.forward)
        params = signature.parameters
        if "position_ids" in params:
            return rotary_emb(hidden_states, position_ids)
        if "seq_len" in params:
            return rotary_emb(hidden_states, seq_len=seq_length)
        return rotary_emb(hidden_states, position_ids)

    def layer_attention_mask(attention_mask):
        if attention_mask is None:
            return None
        if attention_mask.dim() == 2 and bool(attention_mask.all().item()):
            return None
        return attention_mask
    
    # Custom forward method for partitioned model with proper RoPE handling
    def custom_forward(input_ids_or_hidden, attention_mask=None):
        debug_forward = int(debug_nan_steps or 0) > 0
        debug_count = int(getattr(model, "_polar_forward_debug_count", 0))
        if debug_forward and debug_count < int(debug_nan_steps):
            x = input_ids_or_hidden
            x_float = x.detach().float() if torch.is_floating_point(x) else x.detach()
            print(
                f"[debug_forward][stage {stage_idx}] enter "
                f"shape={tuple(x.shape)} dtype={x.dtype} "
                f"finite={bool(torch.isfinite(x_float).all().item())} "
                f"min={float(x_float.min().item()) if x.numel() else 0.0:.6g} "
                f"max={float(x_float.max().item()) if x.numel() else 0.0:.6g}",
                flush=True,
            )

        if model.model.embed_tokens is not None:
            # Stage 0: input is token IDs
            hidden_states = model.model.embed_tokens(input_ids_or_hidden)
        else:
            # Stage 1+: input is hidden states
            hidden_states = input_ids_or_hidden
            if torch.is_floating_point(hidden_states) and not hidden_states.requires_grad:
                hidden_states.requires_grad_(True)

        # Get sequence length for position embeddings
        seq_length = hidden_states.shape[1]
        
        # Handle position embeddings for Qwen2 RoPE
        position_embeddings = None
        
        # Create position_ids tensor with correct shape [batch_size, seq_length]
        batch_size = hidden_states.shape[0]
        position_ids = torch.arange(seq_length, device=hidden_states.device).unsqueeze(0).repeat(batch_size, 1)
        
        if hasattr(model.model, 'rotary_emb') and model.model.rotary_emb is not None:
            position_embeddings = apply_rotary_embedding(
                model.model.rotary_emb, hidden_states, position_ids, seq_length
            )
        elif hasattr(original_model, 'rotary_emb') and original_model.rotary_emb is not None:
            position_embeddings = apply_rotary_embedding(
                original_model.rotary_emb, hidden_states, position_ids, seq_length
            )
        else:
            rotary_emb = get_fallback_rotary_embedding(hidden_states.device)
            position_embeddings = apply_rotary_embedding(
                rotary_emb, hidden_states, position_ids, seq_length
            )

        decoder_attention_mask = layer_attention_mask(attention_mask)

        # Pass through layers
        for layer_idx, layer in enumerate(model.model.layers):
            if isinstance(layer, torch.nn.Identity):
                hidden_states = layer(hidden_states)
            else:
                # Pass position_embeddings to Qwen2 layers
                layer_outputs = layer(
                    hidden_states, 
                    attention_mask=decoder_attention_mask,
                    position_embeddings=position_embeddings
                )
                # Qwen2 layers return a tuple (hidden_states, attention_weights)
                if isinstance(layer_outputs, tuple):
                    hidden_states = layer_outputs[0]
                else:
                    hidden_states = layer_outputs
                if debug_forward and debug_count < int(debug_nan_steps):
                    hs_float = hidden_states.detach().float()
                    if not bool(torch.isfinite(hs_float).all().item()):
                        print(
                            f"[debug_forward][stage {stage_idx}] nonfinite after "
                            f"local_layer={layer_idx}",
                            flush=True,
                        )
                        break

        # Apply final norm if present
        if hasattr(model.model, 'norm') and model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
        elif hasattr(model.model, 'final_norm') and model.model.final_norm is not None:
            hidden_states = model.model.final_norm(hidden_states)

        # Apply lm_head if present
        if model.lm_head is not None:
            output = model.lm_head(hidden_states)
        else:
            output = hidden_states

        if debug_forward and debug_count < int(debug_nan_steps):
            out_float = output.detach().float()
            print(
                f"[debug_forward][stage {stage_idx}] exit "
                f"shape={tuple(output.shape)} dtype={output.dtype} "
                f"finite={bool(torch.isfinite(out_float).all().item())} "
                f"min={float(out_float.min().item()) if output.numel() else 0.0:.6g} "
                f"max={float(out_float.max().item()) if output.numel() else 0.0:.6g}",
                flush=True,
            )
            model._polar_forward_debug_count = debug_count + 1
        return output

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
    if cfg.init_from_pretrained:
        model = AutoModelForCausalLM.from_pretrained(cfg.model_name, **kwargs)
    else:
        config = AutoConfig.from_pretrained(cfg.model_name, trust_remote_code=True)
        if attn_impl is not None:
            config._attn_implementation = attn_impl
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(
                config,
                torch_dtype=kwargs["torch_dtype"],
                trust_remote_code=True,
            )

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


def vocab_parallel_lm_loss(
    output,
    target,
    ignore_index: int,
    tp_mesh,
    debug: bool = False,
):
    """Causal LM loss for either full logits or TP vocab-sharded DTensor logits."""
    shift_labels = target[..., 1:].contiguous()

    if not hasattr(output, "to_local"):
        shift_logits = output[..., :-1, :].contiguous()
        return F.cross_entropy(
            shift_logits.float().view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=ignore_index,
        )

    if tp_mesh is None:
        raise RuntimeError("DTensor logits require a TP mesh for vocab-parallel loss.")

    local_logits = output.to_local()[..., :-1, :].float().contiguous()
    flat_logits = local_logits.view(-1, local_logits.size(-1))
    flat_labels = shift_labels.view(-1)

    tp_rank = int(tp_mesh.get_local_rank())
    tp_size = int(tp_mesh.size())
    global_vocab_size = int(output.size(-1))
    base = global_vocab_size // tp_size
    remainder = global_vocab_size % tp_size
    vocab_start = tp_rank * base + min(tp_rank, remainder)
    vocab_end = vocab_start + base + (1 if tp_rank < remainder else 0)

    valid = flat_labels.ne(ignore_index)
    in_local_vocab = valid & flat_labels.ge(vocab_start) & flat_labels.lt(vocab_end)
    local_target = (flat_labels - vocab_start).clamp(
        min=0,
        max=max(vocab_end - vocab_start - 1, 0),
    )

    local_max = flat_logits.max(dim=-1).values
    global_max = local_max.clone()
    dist.all_reduce(global_max, op=dist.ReduceOp.MAX, group=tp_mesh.get_group())

    exp_sum = torch.exp(flat_logits - global_max.unsqueeze(-1)).sum(dim=-1)
    dist.all_reduce(exp_sum, op=dist.ReduceOp.SUM, group=tp_mesh.get_group())

    target_logits = torch.zeros_like(global_max)
    if in_local_vocab.any():
        target_logits[in_local_vocab] = flat_logits[
            in_local_vocab,
            local_target[in_local_vocab],
        ]
    dist.all_reduce(target_logits, op=dist.ReduceOp.SUM, group=tp_mesh.get_group())

    losses = torch.log(exp_sum.clamp_min(1e-20)) + global_max - target_logits
    if debug:
        rank = dist.get_rank()
        local_finite = bool(torch.isfinite(flat_logits).all().item())
        losses_finite = bool(torch.isfinite(losses).all().item())
        print(
            f"[debug_loss][rank {rank}] "
            f"dtensor={hasattr(output, 'to_local')} "
            f"logits_finite={local_finite} losses_finite={losses_finite} "
            f"logits_min={float(flat_logits.nan_to_num().min().item()):.6g} "
            f"logits_max={float(flat_logits.nan_to_num().max().item()):.6g} "
            f"global_max_min={float(global_max.nan_to_num().min().item()):.6g} "
            f"global_max_max={float(global_max.nan_to_num().max().item()):.6g} "
            f"exp_sum_min={float(exp_sum.nan_to_num().min().item()):.6g} "
            f"exp_sum_max={float(exp_sum.nan_to_num().max().item()):.6g} "
            f"valid={int(valid.sum().item())} "
            f"local_targets={int(in_local_vocab.sum().item())} "
            f"vocab_range=[{vocab_start},{vocab_end})",
            flush=True,
        )
    if valid.any():
        return losses[valid].mean()
    return losses.sum() * 0.0


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
    parser.add_argument(
        "--debug-nan-steps",
        type=int,
        default=0,
        help="Print parameter and batch finite/range checks for the first N steps.",
    )
    
    # Data loader
    parser.add_argument("--num-workers", type=int, default=2)
    
    # Mixed precision and optimization
    parser.add_argument("--use-flash-attn", action="store_true", default=True)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--fp16", action="store_true", default=False)
    parser.add_argument(
        "--activation-checkpointing",
        action="store_true",
        default=False,
        help=(
            "Enable HF gradient checkpointing. Disabled by default for PP "
            "debugging because checkpoint requires PP boundary activations to "
            "carry requires_grad."
        ),
    )
    parser.add_argument(
        "--init-from-pretrained",
        action="store_true",
        default=False,
        help=(
            "Load pretrained weights before partitioning. By default this "
            "script builds from config because PolarParallel initializes the "
            "partitioned stage on device."
        ),
    )
    
    # Parallelism
    parser.add_argument("--pp-size", type=int, default=1)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--micro-batches", type=int, default=1)
    parser.add_argument("--comm-timing", type=int, default=-1)
    parser.add_argument("--using-polar", type=bool, default=True)
    
    # Polar hooks
    parser.add_argument(
        "--polar-hook",
        type=str,
        default="momentum",
        choices=["io", "momentum", "gpipe", "ef_only", "ef_lowmem", "scaling_only", "none"],
        help=(
            "Which POLAR gradient prediction hook to use: "
            "'momentum' (no scaling, EMA momentum extrapolation), "
            "'io' (IO-optimized scaling hook), "
            "'gpipe' (legacy scaling hook), "
            "'ef_only' (error feedback only), "
            "'ef_lowmem' (error feedback only without grads_pred buffers), "
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
    parser.add_argument(
        "--polar-bucket-numel",
        type=int,
        default=4_000_000,
        help="Maximum elements per ef_lowmem POLAR DP communication bucket.",
    )
    parser.add_argument(
        "--polar-max-inflight-buckets",
        type=int,
        default=1,
        help="Maximum ef_lowmem buckets kept alive at the same time.",
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

    # Communication backend for POLAR DP all-reduce.
    parser.add_argument(
        "--method",
        type=str,
        default="bitscom",
        choices=["none", "bitscom"],
        help="Use dense torch.distributed DP communication or bitscom LowBitGroup.",
    )
    parser.add_argument("--bitwidth", type=int, default=4)
    parser.add_argument("--simulate-quantization", action="store_true")
    parser.add_argument("--stochastic-rounding", action="store_true")

    args = parser.parse_args()

    if args.pp_size <= 0 or args.tp_size <= 0:
        raise ValueError("--pp-size and --tp-size must be positive")
    if args.micro_batches < args.pp_size:
        raise ValueError(
            f"--micro-batches ({args.micro_batches}) must be >= "
            f"--pp-size ({args.pp_size}) for pipeline parallel training."
        )
    if args.per_device_batch_size < args.micro_batches:
        raise ValueError(
            f"--per-device-batch-size ({args.per_device_batch_size}) must be >= "
            f"--micro-batches ({args.micro_batches}) because pipeline "
            "microbatching splits the batch dimension."
        )
    if args.per_device_batch_size % args.micro_batches != 0:
        raise ValueError(
            f"--per-device-batch-size ({args.per_device_batch_size}) must be "
            f"divisible by --micro-batches ({args.micro_batches})."
        )
    if args.comm_timing != -1 and not (0 <= args.comm_timing < args.micro_batches):
        raise ValueError(
            f"--comm-timing must be -1 or in [0, {args.micro_batches - 1}], "
            f"got {args.comm_timing}."
        )
    if args.polar_bucket_numel <= 0:
        raise ValueError("--polar-bucket-numel must be positive")
    if args.polar_max_inflight_buckets <= 0:
        raise ValueError("--polar-max-inflight-buckets must be positive")

    bitscom_module = None
    if args.method == "bitscom":
        bitscom_module = import_bitscom()
        bitscom_module.init(bitwidth=args.bitwidth)
    
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
        init_from_pretrained=args.init_from_pretrained,
    )

    # Initialize distributed
    dist.init_process_group(backend="nccl", init_method="env://")
    world_size = dist.get_world_size()
    
    pp_size = args.pp_size
    tp_size = args.tp_size
    model_parallel_size = pp_size * tp_size
    assert world_size % model_parallel_size == 0, (
        f"world_size {world_size} must be divisible by "
        f"PP_SIZE * TP_SIZE ({pp_size} * {tp_size})"
    )
    dp_size = world_size // model_parallel_size
    if tp_size > 1:
        device_mesh = init_device_mesh(
            "cuda",
            (dp_size, pp_size, tp_size),
            mesh_dim_names=("dp", "pp", "tp"),
        )
    else:
        device_mesh = init_device_mesh(
            "cuda",
            (dp_size, pp_size),
            mesh_dim_names=("dp", "pp"),
        )
    dp_mesh = device_mesh["dp"]
    pp_mesh = device_mesh["pp"]

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
    tp_rank = device_mesh["tp"].get_local_rank() if tp_size > 1 else 0
    print(f"Stage index: {stage_idx} / {pp_size}; TP rank: {tp_rank} / {tp_size}")
    
    # Partition model for pipeline parallelism
    stage_model = partition_qwen_model(
        model,
        stage_idx,
        pp_size,
        debug_nan_steps=args.debug_nan_steps,
    )
    
    dp_rank = dp_mesh.get_local_rank()
    print(f"DP rank: {dp_rank} / {dp_size}")

    lowbit_group = None
    if args.method == "bitscom":
        lowbit_group = bitscom_module.LowBitGroup(
            bitwidth=args.bitwidth,
            process_group=dp_mesh.get_group(),
            simulate_quantization=args.simulate_quantization,
            stochastic_rounding=args.stochastic_rounding,
        )
        if dist.get_rank() == 0:
            print(
                "[bitscom] enabled for POLAR DP communication: "
                f"bitwidth={args.bitwidth} "
                f"simulate_quantization={args.simulate_quantization} "
                f"stochastic_rounding={args.stochastic_rounding}",
                flush=True,
            )
    
    # Get dataloader
    dataloader = get_dataloader(cfg, tokenizer, pp_size)

    def loss_fn(output, target):
        """LM loss function with padding mask."""
        return vocab_parallel_lm_loss(
            output,
            target,
            ignore_index=tokenizer.pad_token_id,
            tp_mesh=device_mesh["tp"] if tp_size > 1 else None,
            debug=int(args.debug_nan_steps or 0) > 0,
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
    trainer.lowbit_group = lowbit_group

    trainer.train()


if __name__ == "__main__":
    from dataclasses import dataclass
    main()
