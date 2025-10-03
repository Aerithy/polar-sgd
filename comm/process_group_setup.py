import os
import torch
import torch.distributed as dist

def process_group_setup():
    '''
    return global_group, inter_group, local_group
    '''
    # init the global process group
    rank = os.getenv("RANK", "0")
    local_rank = os.getenv("LOCAL_RANK", "0")
    world_size = os.getenv("WORLD_SIZE", "1")
    rank = int(rank)
    local_rank = int(local_rank)
    world_size = int(world_size)

    if world_size == 1:
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            global_group = dist.init_process_group(
                backend="nccl",
                init_method="tcp://127.0.0.1:23456",
                rank=rank,
                world_size=world_size,
            )
            local_group = global_group
            inter_group = global_group
            return global_group, inter_group, local_group
        else:
            global_group = dist.init_process_group(
                backend="gloo",
                init_method="tcp://127.0.0.1:23456",
                rank=rank,
                world_size=world_size,
            )
            local_group = global_group
            inter_group = global_group
            return global_group, inter_group, local_group

    print(f"rank: {rank}, local_rank: {local_rank}, world_size: {world_size}")

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    global_group = dist.init_process_group(
        backend="nccl",
        init_method="env://",
        rank=rank,
        world_size=world_size,
    )

    # init the local process group
    local_world_size = os.environ["LOCAL_WORLD_SIZE"]
    local_world_size = int(local_world_size)
    node_id = rank // local_world_size

    local_ranks = list(
        range(node_id * local_world_size, (node_id + 1) * local_world_size)
    )
    local_group = dist.new_group(ranks=local_ranks)

    print(f"local_groups: {local_ranks}")

    # init the inter-node process group
    inter_ranks = list(range(0, world_size, local_world_size))
    inter_group = dist.new_group(ranks=inter_ranks)

    print(f"inter_groups: {inter_ranks}")

    # torch.cuda.set_device(local_rank)
    return global_group, inter_group, local_group
