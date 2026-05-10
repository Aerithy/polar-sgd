# 单机 16 GPU，16-way PP + 1-way DP
torchrun --nnodes=1 --nproc_per_node=16 train_qwen14b_polar_dp_pp.py \
    --pp-size 16 \
    --micro-batches 16 \
    --comm-timing 1 \
    --max-steps 10