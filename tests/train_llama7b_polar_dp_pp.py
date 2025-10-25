from psgd.parallelism.polar.wrapper import PolarParallel

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
    pp_size: int,
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
    sampler = None
    if dist.is_initialized():
        sampler = torch.utils.data.distributed.DistributedSampler(
            tokenized_dataset,
            num_replicas=dist.get_world_size() // pp_size,
            rank=dist.get_rank() // pp_size,
            shuffle=True
        )

    dataloader = DataLoader(
        tokenized_dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    return dataloader, tokenizer

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
        print(f"Model params: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")

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
    parser.add_argument("--micro_batches", type=int, default=1)
    args = parser.parse_args()

    dist.init_process_group(backend="nccl")
    world_size = dist.get_world_size()
    
    pp_size = args.pp_size
    
    
    
    assert world_size % pp_size == 0, f"world_size {world_size} must be divisible by PP_SIZE {pp_size}"
    dp_size = world_size // pp_size
    device_mesh = init_device_mesh("cuda", (dp_size, pp_size), mesh_dim_names=("dp", "pp"))
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
    print(f"Stage index: {stage_idx} / {pp_size}")
    # stage_idx = pp_mesh.get_rank()
    stage_model = partition_llama_model(config, stage_idx, pp_size)
    
    dp_rank = dp_mesh.get_local_rank()
    print(f"DP rank: {dp_rank} / {dp_size}")
    dataloader, tokenizer = get_dataloader(
        pp_size=pp_size,
        dataset_name=args.dataset,
        dataset_config=args.dataset_config,
        tokenizer_name=args.tokenizer,
        seq_length=args.seq_length,
        batch_size=args.batch_size,
        use_auth_token=args.use_auth_token,
        split="train"
    )
    # sampler = DistributedSampler(
    #     dataloader.dataset,
    #     num_replicas=dp_size,
    #     rank=dp_rank,
    #     shuffle=True,
    # )
    # dataloader = DataLoader(
    #     dataloader.dataset,
    #     batch_size=args.batch_size,
    #     # sampler=sampler,
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
    
    trainer = PolarParallel(
        args=args,
        device_mesh=device_mesh,
        micro_batches=args.micro_batches,
        loss_fn=loss_fn,
        stage_model=stage_model,
        dataloader=dataloader,
    )
    
    trainer.train()

if __name__ == "__main__":
    main()
    