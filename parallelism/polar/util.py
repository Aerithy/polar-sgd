import torch

def split_model_by_split_spec(model, split_spec, device=None):
    """
    使用 split_spec 自动划分模型为多个 stage，并返回每个 stage 的模块列表
    """
    from torch.distributed.pipelining import pipeline

    # 构建 pipeline 仅用于分析（num_chunks=1）
    pipe = pipeline(
        model,
        split_spec=split_spec,
    )

    partitions = []
    for i, (name, submod) in enumerate(pipe.split_gm.named_children()):
        # 获取该 submod 包含的所有“叶子模块”（用于注册钩子）
        leaf_modules = []
        for n, m in submod.named_modules():
            # 只选“叶子模块”（有参数且无子模块，或为 Linear/Embedding 等）
            if len(list(m.children())) == 0 and len(list(m.parameters())) > 0:
                leaf_modules.append(m)
        partitions.append(leaf_modules)

    return partitions, pipe  # 返回分区 + pipeline 对象（可选）