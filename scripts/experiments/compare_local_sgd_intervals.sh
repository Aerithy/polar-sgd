#!/usr/bin/env bash
set -euo pipefail

# Comparison experiment: Different Local-SGD sync intervals
# Run this script to compare:
# - Standard training (sync every step, baseline)
# - Local-SGD with sync every 5 steps
# - Local-SGD with sync every 10 steps
# - Local-SGD with sync every 20 steps

export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=bond0

MASTER_ADDR=${MASTER_ADDR:-10.48.95.29}
MASTER_PORT=${MASTER_PORT:-11234}
NNODES=${NNODES:-2}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
NODE_RANK=${NODE_RANK:-0}

BASE_CMD="torchrun \
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
  --micro_batches 32"

echo "======================================"
echo "Experiment 1: Baseline (no Local-SGD)"
echo "======================================"
$BASE_CMD

echo ""
echo "======================================"
echo "Experiment 2: Local-SGD sync every 5 steps"
echo "======================================"
$BASE_CMD --use_local_sgd --local_sgd_steps 5

echo ""
echo "======================================"
echo "Experiment 3: Local-SGD sync every 10 steps"
echo "======================================"
$BASE_CMD --use_local_sgd --local_sgd_steps 10

echo ""
echo "======================================"
echo "Experiment 4: Local-SGD sync every 20 steps"
echo "======================================"
$BASE_CMD --use_local_sgd --local_sgd_steps 20

echo ""
echo "======================================"
echo "All experiments completed!"
echo "Check ./log/ for tensorboard logs to compare convergence and throughput"
echo "======================================"
