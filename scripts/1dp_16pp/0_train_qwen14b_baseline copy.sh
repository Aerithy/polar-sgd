# 单机 8 GPU，4-way PP + 2-way DP
torchrun --nnodes=1 --nproc_per_node=16 train_qwen_polar_dp_pp.py \
    --model-name Qwen/Qwen-14B \
    --dataset-name-or-path wikitext \
    --dataset-config wikitext-2-raw-v1 \
    --seq_length 1024 \
    --batch_size 1 \
    --pp_size 16 \
    --micro_batches 16 \
    --comm_timing 1 \
    --lr 1e-4 \
    --max_steps 1000 \
    --bf16