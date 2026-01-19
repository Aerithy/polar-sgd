#!/usr/bin/env bash
set -euo pipefail

# Local-SGD training with parameter sync every 10 steps
# This reduces communication frequency compared to gradient sync every step

export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=bond0

MASTER_ADDR=${MASTER_ADDR:-10.48.95.29}
MASTER_PORT=${MASTER_PORT:-11234}

NNODES=${NNODES:-2}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
NODE_RANK=${NODE_RANK:-0}

# Local-SGD configuration
LOCAL_SGD_STEPS=${LOCAL_SGD_STEPS:-8}  # Sync every N steps

torchrun \
  --nproc_per_node=${NPROC_PER_NODE} \
  --nnodes=${NNODES} \
  --node_rank=${NODE_RANK} \
  --master_addr=${MASTER_ADDR} \
  --master_port=${MASTER_PORT} \
  tests/train_llama7b_polar_dp_pp.py \
  --pp_size 8 \
  --epochs 1 \
  --batch_size 128 \
  --seq_length 512 \
  --lr 1e-4 \
  --dataset wikitext \
  --dataset_config wikitext-103-raw-v1 \
  --tokenizer hf-internal-testing/llama-tokenizer \
  --output_dir ./checkpoints \
  --micro_batches 32 \
  --use_local_sgd \
  --local_sgd_steps ${LOCAL_SGD_STEPS}
