#!/usr/bin/env bash
set -euo pipefail

# Node 1 of 2. Keep MASTER_ADDR/MASTER_PORT identical to node 0.
# Conservative 32GB smoke-test defaults: low-memory EF + bitscom, seq 256.

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
  --node_rank=1 \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  tests/train_qwen14b_polar_dp_pp.py \
  --model-name Qwen/Qwen2.5-14B-Instruct \
  --pp-size 8 \
  --tp-size 2 \
  --micro-batches 8 \
  --comm-timing 0 \
  --max-steps 10 \
  --per-device-batch-size 8 \
  --seq-len 256 \
  --lr 2e-4 \
  --dataset-name-or-path HuggingFaceFW/fineweb \
  --text-field text \
  --polar-hook ef_lowmem \
  --polar-bucket-numel 4000000 \
  --polar-max-inflight-buckets 1 \
  --method bitscom \
  --bitwidth 4
