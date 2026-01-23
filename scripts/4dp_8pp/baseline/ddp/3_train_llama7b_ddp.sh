#!/usr/bin/env bash
set -euo pipefail

# Baseline DP+PP training using PolarParallel._train()
# - baseline_mode=ddp: wraps each pipeline stage with DDP (may OOM)

export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-eth01}

MASTER_ADDR=${MASTER_ADDR:-10.82.123.23}
MASTER_PORT=${MASTER_PORT:-11234}

NNODES=${NNODES:-4}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
NODE_RANK=${NODE_RANK:-3}

PP_SIZE=${PP_SIZE:-8}
EPOCHS=${EPOCHS:-1}
BATCH_SIZE=${BATCH_SIZE:-256}
SEQ_LENGTH=${SEQ_LENGTH:-512}
LR=${LR:-1e-4}
DATASET=${DATASET:-wikitext}
DATASET_CONFIG=${DATASET_CONFIG:-wikitext-103-raw-v1}
TOKENIZER=${TOKENIZER:-hf-internal-testing/llama-tokenizer}
OUTPUT_DIR=${OUTPUT_DIR:-./checkpoints}
MICRO_BATCHES=${MICRO_BATCHES:-32}

BASELINE_MODE=${BASELINE_MODE:-ddp}      # ddp | manual
USING_POLAR=${USING_POLAR:-False}

USE_LOCAL_SGD=${USE_LOCAL_SGD:-0}        # 0/1
LOCAL_SGD_STEPS=${LOCAL_SGD_STEPS:-1}

EXTRA_ARGS=()
if [[ "${USE_LOCAL_SGD}" == "1" ]]; then
  EXTRA_ARGS+=(--use_local_sgd --local_sgd_steps "${LOCAL_SGD_STEPS}")
fi

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
  "${EXTRA_ARGS[@]}"
