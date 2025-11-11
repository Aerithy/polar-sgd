import torch
from .llama_nn import LlamaConfig, MyLlamaForCausalLM

def partition_llama_model(config, stage_idx, num_stages):
    """
    Manually partition LLaMA model for pipeline parallelism.
    - Initialize on 'meta' to avoid OOM
    - Keep only layers assigned to this stage
    - Remove unused components (embeddings, lm_head, etc.)
    """
    with torch.device("meta"):
        model = MyLlamaForCausalLM(config)

    num_layers = config.num_hidden_layers
    assert num_layers % num_stages == 0, "num_layers must be divisible by num_stages"
    layers_per_stage = num_layers // num_stages

    start_layer = stage_idx * layers_per_stage
    end_layer = (stage_idx + 1) * layers_per_stage

    # 转换 layers 为 ModuleDict（保留 FQN）
    # layers_dict = {str(i): model.model.layers[i] for i in range(num_layers)}
    # model.model.layers = torch.nn.ModuleDict(layers_dict)

    # 删除不属于当前 stage 的层
    for i in list(model.model.transformers.keys()):
        if not (start_layer <= int(i) < end_layer):
            del model.model.transformers[i]

    # Stage 0: 保留 embed_tokens，移除 lm_head 和 final_norm
    if stage_idx == 0:
        model.lm_head = None
        model.model.final_norm = None
    # Last stage: 保留 lm_head 和 final_norm，移除 embed_tokens
    elif stage_idx == num_stages - 1:
        model.model.embed_tokens = None
    # 中间 stage: 移除所有非 layer 组件
    else:
        model.model.embed_tokens = None
        model.model.final_norm = None
        model.lm_head = None

    return model