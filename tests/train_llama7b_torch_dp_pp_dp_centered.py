import os
import time
import argparse
from typing import List, Tuple

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.pipelining import PipelineStage, ScheduleGPipe

from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

from psgd.models.llama.llama_nn import LlamaConfig
from psgd.models.llama.partition_model import partition_llama_model


class TokenizedDataset(Dataset):
    def __init__(self, dataset, tokenizer, seq_length=2048, text_field="text"):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.text_field = text_field

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        text = self.dataset[idx][self.text_field]
        tokens = self.tokenizer(
            text,
            truncation=True,
            max_length=self.seq_length + 1,
            padding=False,
            return_tensors=None,
        )["input_ids"]

        if len(tokens) < 2:
            tokens = [self.tokenizer.bos_token_id, self.tokenizer.eos_token_id]

        if len(tokens) > self.seq_length + 1:
            tokens = tokens[: self.seq_length + 1]
        else:
            tokens = tokens + [self.tokenizer.pad_token_id] * (self.seq_length + 1 - len(tokens))

        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        labels = torch.tensor(tokens[1:], dtype=torch.long)
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


def get_tokenizer(tokenizer_name: str, use_auth_token: bool = False):
    try:
        tok = AutoTokenizer.from_pretrained(
            tokenizer_name,
            use_fast=False,
            trust_remote_code=False,
            use_auth_token=use_auth_token,
        )
    except OSError:
        tok = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer", use_fast=False)

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def build_dataset(dataset_name: str, dataset_config: str, split: str):
    if dataset_name == "c4":
        if not dataset_config:
            dataset_config = "en"
        ds = load_dataset("allenai/c4", dataset_config, split=split, streaming=False)
        text_field = "text"
    else:
        ds = load_dataset(dataset_name, dataset_config, split=split)
        text_field = "text"
    return ds, text_field


def make_pp_dp_groups_dp_centered(
    world_size: int,
    nnodes: int,
    local_world_size: int,
) -> Tuple[dist.ProcessGroup, dist.ProcessGroup, int, int, int, int]:
    """DP-centered mapping.

    Assumptions:
      - global ranks are node-contiguous: node_id = rank // local_world_size
      - PP group is per-node (ranks [0-7], [8-15], ...)
      - DP group is cross-node at same local_rank (ranks [0,8,16,...], [1,9,17,...], ...)

    Returns:
      dp_group, pp_group, dp_size, pp_size, dp_rank, pp_rank
    """
    assert world_size == nnodes * local_world_size, (
        f"world_size ({world_size}) must equal nnodes ({nnodes}) "
        f"* local_world_size ({local_world_size})"
    )
    pp_size = local_world_size
    dp_size = nnodes

    rank = dist.get_rank()
    node_id = rank // local_world_size
    local_rank = rank % local_world_size

    # PP group: all ranks within same node (e.g., [0-7] for node0, [8-15] for node1)
    pp_ranks = list(range(node_id * local_world_size, (node_id + 1) * local_world_size))
    pp_group = dist.new_group(ranks=pp_ranks)

    # DP group: all ranks with same local_rank across nodes (e.g., [0,8,16,...] for local_rank=0)
    dp_ranks = [local_rank + i * local_world_size for i in range(nnodes)]
    # Ensure all ranks exist before creating group
    assert all(r < world_size for r in dp_ranks), (
        f"Invalid dp_ranks {dp_ranks} for world_size {world_size}"
    )
    dp_group = dist.new_group(ranks=dp_ranks)

    dp_rank = node_id
    pp_rank = local_rank
    
    # Debug: print group membership on rank 0 of each PP group
    if local_rank == 0:
        print(f"[Rank {rank}] Node {node_id}: PP group={pp_ranks}, DP group={dp_ranks}")
    
    return dp_group, pp_group, dp_size, pp_size, dp_rank, pp_rank


def throughput_log(
    tag: str,
    samples: int,
    elapsed_s: float,
    rank0_only: bool = True,
):
    if (not rank0_only) or dist.get_rank() == 0:
        tput = samples / max(elapsed_s, 1e-9)
        print(f"[{tag}] samples={samples} elapsed_s={elapsed_s:.4f} throughput(samples/s)={tput:.2f}")


def _sync_grads_manual(module: torch.nn.Module, dp_group: dist.ProcessGroup, dp_size: int):
    """All-reduce gradients across dp_group (SUM then /dp_size)."""
    for p in module.parameters():
        if p.grad is None:
            continue
        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM, group=dp_group)
        p.grad.div_(dp_size)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=50, help="Only run N optimizer steps (after warmup).")
    parser.add_argument("--warmup_steps", type=int, default=5)

    parser.add_argument("--batch_size", type=int, default=1, help="GLOBAL batch size (per DP replica).")
    parser.add_argument("--seq_length", type=int, default=1024)
    parser.add_argument("--micro_batches", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)

    parser.add_argument("--dataset", type=str, default="wikitext")
    parser.add_argument("--dataset_config", type=str, default="wikitext-103-raw-v1")
    parser.add_argument("--tokenizer", type=str, default="hf-internal-testing/llama-tokenizer")
    parser.add_argument("--use_auth_token", action="store_true")

    # topology
    parser.add_argument("--nnodes", type=int, required=True)
    parser.add_argument("--nproc_per_node", type=int, required=True)

    parser.add_argument(
        "--dp_sync",
        type=str,
        default="manual",
        choices=["manual", "ddp"],
        help="How to synchronize gradients across dp_group: manual all_reduce or DDP wrapper.",
    )

    args = parser.parse_args()

    dist.init_process_group(backend="nccl", init_method="env://")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    dp_group, pp_group, dp_size, pp_size, dp_rank, pp_rank = make_pp_dp_groups_dp_centered(
        world_size=world_size,
        nnodes=args.nnodes,
        local_world_size=args.nproc_per_node,
    )

    if rank == 0:
        print(
            f"DP-centered mapping: world_size={world_size}, nnodes={args.nnodes}, "
            f"nproc_per_node={args.nproc_per_node}, dp_size={dp_size}, pp_size={pp_size}"
        )

    # model config (llama7b-ish)
    config = LlamaConfig(
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        rope_theta=10000.0,
        pad_token_id=0,
        tie_word_embeddings=False,
    )

    stage_model = partition_llama_model(config, stage_idx=pp_rank, num_stages=pp_size)
    stage_model.to_empty(device=device, recurse=True)
    stage_model.apply(lambda m: m.reset_parameters() if hasattr(m, "reset_parameters") else None)

    stage = PipelineStage(
        stage_model,
        stage_index=pp_rank,
        num_stages=pp_size,
        device=device,
        group=pp_group,
    )

    # DP sync option
    dp_wrapped_module = stage.submod
    if args.dp_sync == "ddp":
        from torch.nn.parallel import DistributedDataParallel as DDP

        dp_wrapped_module = DDP(
            stage.submod,
            process_group=dp_group,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            gradient_as_bucket_view=True,
            find_unused_parameters=False,
        )

        stage = PipelineStage(
            dp_wrapped_module,
            stage_index=pp_rank,
            num_stages=pp_size,
            device=device,
            group=pp_group,
        )

    optimizer = torch.optim.AdamW(stage.submod.parameters(), lr=args.lr)

    # data: only first stage loads dataset and feeds pipeline
    tok = get_tokenizer(args.tokenizer, use_auth_token=args.use_auth_token)
    ds, text_field = build_dataset(args.dataset, args.dataset_config, split="train")
    tokenized = TokenizedDataset(ds, tok, seq_length=args.seq_length, text_field=text_field)

    # DP rank here is node_id; each dp replica owns disjoint data.
    if stage.is_first:
        sampler = DistributedSampler(
            tokenized,
            num_replicas=dp_size,
            rank=dp_rank,
            shuffle=True,
            drop_last=True,
        )
        dataloader = DataLoader(
            tokenized,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=2,
            pin_memory=True,
            drop_last=True,
        )
    else:
        dataloader = None

    def loss_fn(output, target):
        shift_logits = output[..., :-1, :].contiguous()
        shift_labels = target[..., 1:].contiguous()
        return F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=tok.pad_token_id,
        )

    schedule = ScheduleGPipe(stage, n_microbatches=args.micro_batches, loss_fn=loss_fn)

    # Train loop: measure steady-state throughput using global barrier.
    # Each step consumes batch_size samples per DP replica => global_samples_per_step = batch_size * dp_size
    global_samples_per_step = args.batch_size * dp_size

    def barrier_all():
        dist.barrier()

    # epoch/iter handling
    step = 0
    warmup = args.warmup_steps
    measure_steps = args.steps
    measured_start = None

    if stage.is_first:
        it = iter(dataloader)
        pbar = tqdm(total=warmup + measure_steps, disable=(rank != 0))
    else:
        it = None
        pbar = None

    barrier_all()

    while step < warmup + measure_steps:
        if stage.is_first:
            batch = next(it)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        else:
            input_ids = None
            attention_mask = None

        if stage.is_last:
            if stage.is_first:
                labels = batch["labels"].to(device, non_blocking=True)
            else:
                labels = None
        else:
            labels = None

        optimizer.zero_grad(set_to_none=True)

        if stage.is_first:
            schedule.step(input_ids, attention_mask=attention_mask)
        elif stage.is_last:
            losses: List[torch.Tensor] = []
            schedule.step(target=labels, losses=losses, attention_mask=attention_mask)
            loss = torch.stack(losses).mean()
            loss.backward()
        else:
            schedule.step(attention_mask=attention_mask)

        # If using manual DP, all-reduce after backward and before optimizer step.
        if args.dp_sync == "manual":
            _sync_grads_manual(stage.submod, dp_group=dp_group, dp_size=dp_size)

        optimizer.step()

        step += 1

        if step == warmup:
            barrier_all()
            torch.cuda.synchronize()
            measured_start = time.time()

        if pbar:
            pbar.update(1)

    barrier_all()
    torch.cuda.synchronize()
    measured_end = time.time()

    # measured window
    if measured_start is None:
        measured_start = measured_end

    elapsed = measured_end - measured_start
    total_samples = global_samples_per_step * measure_steps

    # use rank0 to log
    throughput_log(tag="dp_centered", samples=total_samples, elapsed_s=elapsed, rank0_only=True)


if __name__ == "__main__":
    main()
