#!/usr/bin/env bash
set -euo pipefail

# Node 0 of 2. Each node uses 16 GPUs as 8 PP stages x 2 TP ranks.
# Across nodes, ranks with the same local (PP, TP) coordinates form POLAR DP.
# Conservative 32GB smoke-test defaults: low-memory EF + bitscom, batch 1, seq 256.

MASTER_ADDR="${MASTER_ADDR:-10.48.95.29}"
MASTER_PORT="${MASTER_PORT:-11234}"
NNODES="${NNODES:-2}"
NPROC_PER_NODE="${NPROC_PER_NODE:-16}"
NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-bond0}"
NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"

export NCCL_SOCKET_IFNAME
export NCCL_IB_DISABLE
export PYTHONPATH="../bitscom/python:${PYTHONPATH:-}"

torchrun \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --nnodes="${NNODES}" \
  --node_rank=0 \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  tests/train_qwen14b_polar_dp_pp.py \
  --model-name Qwen/Qwen2.5-14B-Instruct \
  --pp-size 8 \
  --tp-size 2 \
  --micro-batches 1 \
  --comm-timing 0 \
  --max-steps 10 \
  --per-device-batch-size 1 \
  --seq-len 256 \
  --lr 2e-4 \
  --dataset-name-or-path HuggingFaceFW/fineweb \
  --text-field text \
  --polar-hook ef_lowmem \
  --method bitscom \
  --bitwidth 4
