# train_llama7b_manual_pp.py
from gettext import dpgettext
import os
import argparse
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.pipelining import PipelineStage, ScheduleGPipe
from torch.distributed.device_mesh import init_device_mesh
from torch.utils.data import DataLoader, DistributedSampler, Dataset
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

# -----------------------------
# 替换为你自己的模型定义
# -----------------------------
from psgd.models.llama.llama_nn import LlamaConfig, MyLlamaForCausalLM  # e.g., from model import ...

# -----------------------------
# Dataset
# -----------------------------
class TokenizedDataset(Dataset):
    def __init__(self, dataset, tokenizer, seq_length=2048):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.seq_length = seq_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        text = self.dataset[idx]["text"]
        tokens = self.tokenizer(
            text,
            truncation=True,
            max_length=self.seq_length + 1,  # +1 for shifting
            padding=False,
            return_tensors=None
        )["input_ids"]

        # Ensure we have at least 2 tokens
        if len(tokens) < 2:
            tokens = [self.tokenizer.bos_token_id, self.tokenizer.eos_token_id]

        # Pad or truncate to seq_length + 1
        if len(tokens) > self.seq_length + 1:
            tokens = tokens[:self.seq_length + 1]
        else:
            tokens = tokens + [self.tokenizer.pad_token_id] * (self.seq_length + 1 - len(tokens))

        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        labels = torch.tensor(tokens[1:], dtype=torch.long)
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask
        }

def get_dataloader(
    dataset_name: str = "wikitext",
    dataset_config: str = "wikitext-2-raw-v1",
    tokenizer_name: str = "meta-llama/Llama-2-7b-hf",  # 或使用 "hf-internal-testing/llama-tokenizer" 如果无权限
    seq_length: int = 1024,
    batch_size: int = 1,
    num_workers: int = 2,
    split: str = "train",
    use_auth_token: bool = False,
):
    # Load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            use_fast=False,
            trust_remote_code=False,
            use_auth_token=use_auth_token
        )
    except OSError:
        print("⚠️ Cannot load official LLaMA tokenizer. Using a compatible one.")
        tokenizer = AutoTokenizer.from_pretrained(
            "hf-internal-testing/llama-tokenizer",
            use_fast=False
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    dataset = load_dataset(dataset_name, dataset_config, split=split)

    # Tokenize
    tokenized_dataset = TokenizedDataset(dataset, tokenizer, seq_length=seq_length)

    # Distributed sampler
    # sampler = None
    # if dist.is_initialized():
    #     sampler = torch.utils.data.distributed.DistributedSampler(
    #         tokenized_dataset,
    #         num_replicas=dist.get_world_size(),
    #         rank=dist.get_rank(),
    #         shuffle=True
    #     )

    # dataloader = DataLoader(
    #     tokenized_dataset,
    #     batch_size=batch_size,
    #     sampler=sampler,
    #     shuffle=(sampler is None),
    #     num_workers=num_workers,
    #     pin_memory=True,
    #     drop_last=True
    # )
    return tokenized_dataset, tokenizer

# -----------------------------
# Manual Model Partitioning (Option 1)
# -----------------------------
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
    for i in list(model.model.layers.keys()):
        if not (start_layer <= int(i) < end_layer):
            del model.model.layers[i]

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

def check_pp_group_status(device_mesh: dist.device_mesh.DeviceMesh):
    # 初始化设备网格
    # 使用 (DP_SIZE, PP_SIZE) 但确保 PP 在同一机器内
    pp_mesh = device_mesh["pp"]
    
    # 验证 PP 组是否在同一机器
    pp_group = pp_mesh.get_group()
    pp_ranks = dist.get_process_group_ranks(pp_group)
    
    # 获取所有 PP 组进程的主机名
    hostnames = []
    for rank in pp_ranks:
        if rank == dist.get_rank():
            hostname = os.uname().nodename
        else:
            hostname = None
        hostname = dist.broadcast_object(hostname, src=rank, group=pp_group)
        hostnames.append(hostname)
    
    if dist.get_rank() in pp_ranks[0]:
        print(f"PP Group {pp_ranks}: Hostnames {set(hostnames)}")
    
    # 确保所有 PP 组进程在同一主机
    if len(set(hostnames)) > 1:
        raise RuntimeError(f"PP group spans multiple machines: {set(hostnames)}")

# -----------------------------
# Main Training Loop
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seq_length", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dataset", type=str, default="wikitext")
    parser.add_argument("--dataset_config", type=str, default="wikitext-2-raw-v1")
    parser.add_argument("--tokenizer", type=str, default="hf-internal-testing/llama-tokenizer")
    parser.add_argument("--use_auth_token", action="store_true")
    parser.add_argument("--output_dir", type=str, default="./llama7b_checkpoints")
    parser.add_argument("--pp_size", type=int, default=1)
    args = parser.parse_args()

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    PP_SIZE = args.pp_size
    
    assert world_size % PP_SIZE == 0, f"world_size {world_size} must be divisible by PP_SIZE {PP_SIZE}"
    dp_size = world_size // PP_SIZE
    device_mesh = init_device_mesh("cuda", (dp_size, PP_SIZE), mesh_dim_names=("dp", "pp"))
    dp_mesh = device_mesh["dp"]
    pp_mesh = device_mesh["pp"]
    
    

    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    config = LlamaConfig(
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=4096,
        num_hidden_layers=32,
        num_attention_heads=32,
        rope_theta=10000.0,
        pad_token_id=0,
        tie_word_embeddings=True,
    )

    # ✅ 手动分区
    stage_idx = pp_mesh.get_local_rank()
    # stage_idx = pp_mesh.get_coordinate()
    # stage_idx = pp_mesh.get_rank()
    stage_model = partition_llama_model(config, stage_idx, PP_SIZE)
    stage_model.to_empty(device=device, recurse=True)
    stage_model.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)

    # ✅ 构建 PipelineStage
    stage = PipelineStage(
        stage_model,
        stage_index=stage_idx,
        num_stages=PP_SIZE,
        device=device,
        group=pp_mesh.get_group(),
    )

    optimizer = torch.optim.AdamW(stage.submod.parameters(), lr=1e-4) # if stage.is_last else None
    
    dp_rank = dp_mesh.get_local_rank()
    dp_group_idx = dist.get_rank() // PP_SIZE
    # dp_rank = dp_mesh.get_rank()
    tokenized_dataset, tokenizer = get_dataloader(
        dataset_name=args.dataset,
        dataset_config=args.dataset_config,
        tokenizer_name=args.tokenizer,
        seq_length=args.seq_length,
        batch_size=args.batch_size,
        use_auth_token=args.use_auth_token,
        split="train"
    )
    sampler = DistributedSampler(
        tokenized_dataset.dataset,
        num_replicas=dp_size,
        rank=dp_group_idx,
        shuffle=True,
    )
    dataloader = DataLoader(
        tokenized_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=2,
        pin_memory=True,
        drop_last=True
    )
    # tokenized_dataset = DataLoader(
    #     tokenized_dataset.dataset,
    #     batch_size=args.batch_size,
    #     sampler=sampler,
    #     pin_memory=False,
    # )
    
    def loss_fn(output, target):
        shift_logits = output[..., :-1, :].contiguous()
        shift_labels = target[..., 1:].contiguous()
        return F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=0,
        )

    schedule = ScheduleGPipe(stage, n_microbatches=4, loss_fn=loss_fn)

    def hook(grad):
        dist.all_reduce(grad, op=dist.ReduceOp.SUM, group=dp_mesh.get_group())
        return grad
    
    # ... after creating stage ...
    
    dp_rank = dp_mesh.get_local_rank()
    dp_group = dp_mesh.get_group()
    pp_rank = pp_mesh.get_local_rank()
    pp_group = pp_mesh.get_group()
    dist.get_world_size(dp_group)
    print(f"rank: {rank}, dp_group rank: {dp_group.rank()}, dp_group size: {dp_group.size()}, dp_group local rank: {dp_rank}")
    print(f"rank: {rank}, pp_group rank: {pp_group.rank()}, pp_group size: {pp_group.size()}, pp_group local rank: {pp_rank}")

    # Register gradient hooks for DP sync
    # for param in stage.submod.parameters():
    #     if param.requires_grad:
    #         param.register_hook(
    #             lambda grad, group=dp_group: (
    #                 # print(f"rank: {dp_rank}/{rank} running all reduce on group: {group.world_size}"),
    #                 dist.all_reduce(grad, op=dist.ReduceOp.AVG, group=group),
    #                 grad
    #             )[-1]
    #         )

    # ... training loop (no manual all_reduce) ...
    global_step = 0
    if stage.is_last:
        pbar = tqdm(tokenized_dataset, desc=f"Epoch {args.epochs}")
    else:
        pbar = tokenized_dataset
    for batch in pbar:
        # print(f"rank: {rank} running pp on group: {pp_rank}")
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device) if stage.is_last else None
        attention_mask = batch["attention_mask"].to(device)

        if optimizer:
            optimizer.zero_grad()

        if stage.is_first:
            output = schedule.step(input_ids, attention_mask=attention_mask)
        elif stage.is_last:
            losses = []
            schedule.step(target=labels, losses=losses, attention_mask=attention_mask)  # target 传给 last stage 的 forward
            loss = torch.stack(losses).mean()
            
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            if global_step % 100 == 0:
                print(f"Step {global_step}, Loss: {loss.item():.4f}")
        else:
            schedule.step(attention_mask=attention_mask)
            
        # 在 optimizer.step() 之前
        grads = []
        for param in stage.submod.parameters():
            if param.requires_grad:
                # if param.grad is None:
                #     param.grad = torch.zeros_like(param)
                grads.append(param.grad)

        if grads:
            # 融合 all_reduce
            # print(f"rank: {rank} running all reduce on group: {dp_group.rank()}")
            dist.all_reduce_coalesced(grads, op=dist.ReduceOp.AVG, group=dp_group)
                
        optimizer.step()
        global_step += 1

    dist.destroy_process_group

if __name__ == "__main__":
    main()