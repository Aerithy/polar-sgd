#!/usr/bin/env bash
set -euo pipefail

# Baseline DP+PP training (tests/train_llama7b_ddp_dp_pp.py) with periodic
# validation (loss/ppl). Validation is computed on last pipeline stage.

export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-eth01}

MASTER_ADDR=${MASTER_ADDR:-10.82.120.26}
MASTER_PORT=${MASTER_PORT:-11234}

NNODES=${NNODES:-4}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
NODE_RANK=${NODE_RANK:-0}

PP_SIZE=${PP_SIZE:-8}
EPOCHS=${EPOCHS:-1}
BATCH_SIZE=${BATCH_SIZE:-128}
SEQ_LENGTH=${SEQ_LENGTH:-512}
LR=${LR:-5e-4}
DATASET=${DATASET:-wikitext}
DATASET_CONFIG=${DATASET_CONFIG:-wikitext-103-raw-v1}
TOKENIZER=${TOKENIZER:-hf-internal-testing/llama-tokenizer}
OUTPUT_DIR=${OUTPUT_DIR:-./checkpoints}
MICRO_BATCHES=${MICRO_BATCHES:-32}

BASELINE_MODE=${BASELINE_MODE:-ddp}      # ddp | manual
USING_POLAR=${USING_POLAR:-False}

MAX_STEPS=${MAX_STEPS:-500}

# Validation
EVAL_SPLIT=${EVAL_SPLIT:-}                 # e.g. validation
TRAIN_VAL_RATIO=${TRAIN_VAL_RATIO:-0.0}    # e.g. 0.01
EVAL_INTERVAL=${EVAL_INTERVAL:-50}
EVAL_MAX_BATCHES=${EVAL_MAX_BATCHES:-20}

torchrun \
  --nproc_per_node=${NPROC_PER_NODE} \
  --nnodes=${NNODES} \
  --node_rank=${NODE_RANK} \
  --master_addr=${MASTER_ADDR} \
  --master_port=${MASTER_PORT} \
  tests/train_llama7b_ddp_dp_pp.py \
  --pp_size ${PP_SIZE} \
  --epochs ${EPOCHS} \
  --batch_size ${BATCH_SIZE} \
  --seq_length ${SEQ_LENGTH} \
  --lr ${LR} \
  --dataset ${DATASET} \
  --dataset_config ${DATASET_CONFIG} \
  --tokenizer ${TOKENIZER} \
  --output_dir ${OUTPUT_DIR} \
  --micro_batches ${MICRO_BATCHES} \
  --baseline_mode ${BASELINE_MODE} \
  --using_polar ${USING_POLAR} \
  --max_steps ${MAX_STEPS} \
  --eval_split "${EVAL_SPLIT}" \
  --train_val_ratio ${TRAIN_VAL_RATIO} \
  --eval_interval ${EVAL_INTERVAL} \
  --eval_max_batches ${EVAL_MAX_BATCHES}
