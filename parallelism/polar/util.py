import torch
from transformers import AutoTokenizer

# @torch._dynamo.disable
# def get_partitions_and_pipe(model, tokenizer, device=None):
#     """
#     使用 split_spec 自动划分模型为多个 stage，并返回每个 stage 的模块列表
#     """
#     from torch.distributed.pipelining import pipeline
    
#     was_training = model.training
#     model.eval()  # 确保是推理模式
    
#     dummy_text = ["This is a dummy input for tracing."] * 2
#     example_batch = tokenizer(
#         dummy_text, 
#         return_tensors="pt", 
#         padding="max_length",
#         max_length=512,
#         truncation=True,
#     ).to(device)
#     # obtain model parameter's data type
#     dtype = next(model.parameters()).dtype
    
#     example_args = (example_batch['input_ids'].to(device),)
#     example_kwargs = {
#         'attention_mask': example_batch['attention_mask'].to(device, dtype),
#         # 'use_cache': False,
#     }
#     need_restore = False
#     if hasattr(model, "export_mode"):
#         old_flag = getattr(model, "export_mode")
#         setattr(model, "export_mode", True)
#         need_restore = True
        
#     # 在 tracing 前，强制关闭 use_cache
#     if hasattr(model.config, 'use_cache'):
#         model.config.use_cache = False
    
#     # 构建 pipeline 仅用于分析（num_chunks=1）
#     pipe = pipeline(
#         model,
#         mb_args=example_args,
#         mb_kwargs=example_kwargs,
#     )
    
#     if need_restore:
#         setattr(model, "export_mode", old_flag)
#     if was_training:
#         model.train()

#     partitions = []
#     for i, (name, submod) in enumerate(pipe.split_gm.named_children()):
#         # 获取该 submod 包含的所有“叶子模块”（用于注册钩子）
#         leaf_modules = []
#         for n, m in submod.named_modules():
#             # 只选“叶子模块”（有参数且无子模块，或为 Linear/Embedding 等）
#             if len(list(m.children())) == 0 and len(list(m.parameters())) > 0:
#                 leaf_modules.append(m)
#         partitions.append(leaf_modules)
    
#     # model.config.use_cache = False

#     return partitions, pipe  # 返回分区 + pipeline 对象（可选）


# util.py
@torch._dynamo.disable
def get_partitions_and_pipe(model, tokenizer, device=None):
    """
    使用 split_spec 自动划分模型为多个 stage，并返回每个 stage 的模块列表
    """
    from torch.distributed.pipelining import pipeline
    
    was_training = model.training
    model.eval()  # 确保是推理模式
    
    dummy_text = ["This is a dummy input for tracing."] * 2
    example_batch = tokenizer(
        dummy_text, 
        return_tensors="pt", 
        padding="max_length",
        max_length=512,
        truncation=True,
    ).to(device)
    # obtain model parameter's data type
    dtype = next(model.parameters()).dtype
    
    example_args = (example_batch['input_ids'].to(device),)
    example_kwargs = {
        'attention_mask': example_batch['attention_mask'].to(device, dtype),
    }
    need_restore = False
    if hasattr(model, "export_mode"):
        old_flag = getattr(model, "export_mode")
        setattr(model, "export_mode", True)
        need_restore = True
        
    # 在 tracing 前，强制关闭 use_cache
    if hasattr(model.config, 'use_cache'):
        model.config.use_cache = False
    
    # 构建 pipeline 仅用于分析（num_chunks=1）
    pipe = pipeline(
        model,
        mb_args=example_args,
        mb_kwargs=example_kwargs,
    )
    
    if need_restore:
        setattr(model, "export_mode", old_flag)
    if was_training:
        model.train()

    partitions = []
    for i, (name, submod) in enumerate(pipe.split_gm.named_children()):
        # 修复：直接使用 submod 而不是提取叶子模块
        # 因为后续的 PolarCommHook 期望的是完整的子模块
        partitions.append([submod])  # 将子模块包装在列表中
        
        # 调试信息
        print(f"Partition {i}: {name}, type: {type(submod)}")
    
    return partitions, pipe