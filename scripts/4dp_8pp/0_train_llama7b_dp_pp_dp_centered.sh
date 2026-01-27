#!/usr/bin/env bash
set -euo pipefail

# DP-centered (node-local PP):
# - 2 nodes, 8 GPUs/node
# - PP groups: [0-7] and [8-15]
# - DP replicas: one per node

export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=bond0

MASTER_ADDR=${MASTER_ADDR:-10.48.95.29}
MASTER_PORT=${MASTER_PORT:-11234}

NNODES=${NNODES:-2}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
NODE_RANK=${NODE_RANK:-0}

DP_SYNC=${DP_SYNC:-manual}  # manual | ddp

torchrun \
  --nproc_per_node=${NPROC_PER_NODE} \
  --nnodes=${NNODES} \
  --node_rank=${NODE_RANK} \
  --master_addr=${MASTER_ADDR} \
  --master_port=${MASTER_PORT} \
  tests/train_llama7b_torch_dp_pp_dp_centered.py \
  --dp_sync ${DP_SYNC} \
  --nnodes ${NNODES} \
  --nproc_per_node ${NPROC_PER_NODE} \
  --batch_size 128 \
  --seq_length 512 \
  --micro_batches 32 \
  --lr 5e-4 \
  --warmup_steps 5 \
  --steps 50 \
  --dataset wikitext \
  --dataset_config wikitext-103-raw-v1 \
  --tokenizer hf-internal-testing/llama-tokenizer
