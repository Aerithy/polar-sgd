#!/usr/bin/env bash
set -euo pipefail

# Single-node DP-centered test (no cross-node DP)
# 1 node, 8 GPUs => dp_size=1, pp_size=8

export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1

DP_SYNC=${DP_SYNC:-manual}

torchrun \
  --nproc_per_node=8 \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=localhost \
  --master_port=29500 \
  tests/train_llama7b_torch_dp_pp_dp_centered.py \
  --dp_sync ${DP_SYNC} \
  --nnodes 1 \
  --nproc_per_node 8 \
  --batch_size 16 \
  --seq_length 512 \
  --micro_batches 8 \
  --lr 1e-4 \
  --warmup_steps 2 \
  --steps 5 \
  --dataset wikitext \
  --dataset_config wikitext-103-raw-v1 \
  --tokenizer hf-internal-testing/llama-tokenizer
