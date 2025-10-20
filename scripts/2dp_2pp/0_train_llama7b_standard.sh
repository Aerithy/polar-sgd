NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=bond0 \
torchrun --nproc_per_node=2 --nnodes=2 --node_rank=0 --master_addr=10.82.252.14 --master_port=11234 \
tests/train_llama7b_polar_dp_pp.py                  \
--pp_size=2                                         \
--epochs 1                                          \
--batch_size 8                                      \
--seq_length 512                                    \
--lr 1e-4                                           \
--dataset wikitext                                  \
--dataset_config wikitext-2-raw-v1                  \
--tokenizer hf-internal-testing/llama-tokenizer     \
--output_dir ./checkpoints                          \
--micro_batches 4                                   \