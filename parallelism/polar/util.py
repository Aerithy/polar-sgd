import torch
from transformers import AutoTokenizer

@torch._dynamo.disable
def split_model_by_split_spec(model, split_spec, tokenizer, device=None):
    """
    使用 split_spec 自动划分模型为多个 stage，并返回每个 stage 的模块列表
    """
    from torch.distributed.pipelining import pipeline
    
    example_batch = tokenizer("This is a dummy input for tracing.", return_tensors="pt", padding=True).to(device)
    example_args = (example_batch['input_ids'].to(device),)
    example_kwargs = {'attention_mask': example_batch['attention_mask'].to(device)}
    
    model.config.use_cache = False
    
    # 构建 pipeline 仅用于分析（num_chunks=1）
    pipe = pipeline(
        model,
        # split_spec=split_spec,
        mb_args=example_args,
        # mb_kwargs=example_kwargs,
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
    
    model.config.use_cache = False

    return partitions, pipe  # 返回分区 + pipeline 对象（可选）

from torch.export import export

def split_model_by_export(model, split_spec, tokenizer, device=None):
    model.config.use_cache = False
    model.eval()  # 确保是推理模式

    example_batch = tokenizer("Hello world", return_tensors="pt")
    args = (example_batch['input_ids'].to(device),)
    kwargs = {'attention_mask': example_batch['attention_mask'].to(device)}

    # 导出为 ExportedProgram
    exported = export(model, args=args, kwargs=kwargs)

    # 获取 graph_module
    gm = exported.module()

    # 根据 split_spec 手动拆分（需要解析 split_spec 字典）
    # 示例：按模块名拆分
    partitions = []
    current_partition = []
    split_points = set(split_spec.keys())

    for name, node in gm.named_children():
        if name in split_points and split_spec[name] == SplitPoint.BEGINNING:
            if current_partition:
                partitions.append(current_partition)
                current_partition = []
        current_partition.append(node)

    if current_partition:
        partitions.append(current_partition)

    return partitions, gm