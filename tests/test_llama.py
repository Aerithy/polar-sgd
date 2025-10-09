# 该代码块需要被保存在一个新的文件，例如 test_llama.py

import argparse
import os
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.pipelining import SplitPoint
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from psgd.models.llama.llama_nn import LlamaConfig, MyLlamaForCausalLM

# 假设你的项目结构如下，可以正确导入 PolarDataParallel
# ./
# ├── parallelism/
# │   └── polar/
# │       ├── wrapper.py
# │       └── ...
# └── test_llama.py
from psgd.parallelism.polar.wrapper import PolarDataParallel
from psgd.comm.process_group_setup import process_group_setup

def setup_distributed():
    """初始化分布式环境"""
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def create_parser():
    """创建命令行参数解析器"""
    parser = argparse.ArgumentParser(description="Llama-7b Test with Polar Parallelism")
    # 模型与数据参数
    parser.add_argument("--model_path", type=str, default="meta-llama/Llama-2-7b-hf", help="Path to the Llama model")
    parser.add_argument("--tokenizer_path", type=str, default="meta-llama/Llama-2-7b-hf", help="Path to the tokenizer")
    parser.add_argument("--dataset_name", type=str, default="wikitext", help="Dataset name from Hugging Face Hub")
    parser.add_argument("--dataset_config", type=str, default="wikitext-2-raw-v1", help="Dataset config name")

    # 训练参数
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4, help="Macro-batch size")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)

    # 并行化参数 (与 PolarDataParallel 保持一致)
    parser.add_argument("--micro_batches", type=int, default=4, help="Number of micro-batches for pipeline parallelism")
    parser.add_argument("--using_hook", action="store_true", default=True, help="Whether to use the custom Polar hook")
    parser.add_argument("--local_steps", type=int, default=4, help="Number of partitions for the model (PP degree)")
    
    # PolarDataParallel 内部使用的参数，我们也需要定义
    parser.add_argument("--pretrained", action="store_true", default=True) # 我们总是从预训练加载
    parser.add_argument("--num_labels", type=int, default=2) # 对于CausalLM，这个参数不会被使用，但wrapper需要它

    return parser.parse_args()


def main():
    args = create_parser()
    # setup_distributed()
    global_group, inter_group, local_group = process_group_setup()

    # 获取分布式通信组
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    
    # 假设所有进程在同一个 DP 组内，内部进行 PP
    # 在更复杂的场景下，这里可以创建不同的 inter_group 和 local_group
    # 但对于单节点的 DP+PP 测试，使用默认组即可
    # inter_group = dist.new_group(ranks=list(range(world_size)))
    # local_group = dist.new_group(ranks=list(range(world_size)))

    # --- 1. 为 Llama-7b 定义分割策略 (split_spec) ---
    # Llama-2-7b 有 32 个 Transformer 层 (从 0 到 31)，名为 'model.layers.i'
    # 如果 local_steps=4, 我们将其分为 4 个 stage
    # 每个 stage 包含 32 / 4 = 8 层
    # split_spec = {
    #     # 第 2 个 stage 从第 8 层开始
    #     # "model.layers.8": SplitPoint.BEGINNING,
    #     # 第 3 个 stage 从第 16 层开始
    #     "model.layers.16": SplitPoint.BEGINNING,
    #     # 第 4 个 stage 从第 24 层开始
    #     # "model.layers.24": SplitPoint.BEGINNING,
    # }
    def build_split_spec(num_layers: int, local_steps: int):
        if local_steps <= 1:
            return {}
        
        layers_per_stage = num_layers // local_steps
        # 在每个 stage 的末尾之后创建切分点
        split_indices = [layers_per_stage * (i + 1) for i in range(local_steps - 1)]
        
        return {f"model.layers.{i}": SplitPoint.BEGINNING for i in split_indices}

    # 使用命令行参数 local_steps 动态生成 split_spec
    # 假设模型有32层，后面会用真实配置更新
    # split_spec = build_split_spec(num_layers=32, local_steps=args.local_steps)
    # if rank == 0:
    #     print(f"Dynamically generated split_spec for {args.local_steps} stages: {split_spec}")
    
    
    # --- 2. 加载模型和 Tokenizer ---
    # 关键：使用 AutoModelForCausalLM 而不是 AutoModelForSequenceClassification
    # model = AutoModelForCausalLM.from_pretrained(
    #     args.model_path,
    #     torch_dtype=torch.bfloat16, # 使用 bfloat16 以节省,
    #     attn_implementation="eager",
    # )
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    # 关键：Llama2 tokenizer 没有默认的 pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    llama_config = LlamaConfig(
        vocab_size=len(tokenizer),
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,        # 7B 约 32 层
        num_attention_heads=32,
        rope_theta=10000.0,
        pad_token_id=tokenizer.pad_token_id,
        tie_word_embeddings=True,
    )
    
    model = MyLlamaForCausalLM(llama_config)    # .to(dtype=torch.bfloat16)
    
    def get_split_indices(num_layers: int, local_steps: int):
        if local_steps <= 1:
            return []
        
        layers_per_stage = num_layers // local_steps
        return [layers_per_stage * (i + 1) - 1 for i in range(local_steps - 1)]
    
    split_indices = get_split_indices(llama_config.num_hidden_layers, args.local_steps)
    model.model.set_split_points(split_indices)
    if rank == 0:
        print(f"Manually inserting split points after layers: {split_indices}")

    # 确保模型也知道新的 pad_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    # --- 3. 加载和处理数据集 ---
    raw_datasets = load_dataset(args.dataset_name, args.dataset_config)

    def tokenize_function(examples):
        # 关键：与原始 wrapper.py 中的 tokenize_function 不同
        # 我们处理的是 "text" 字段，用于语言模型任务
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        )

    # 过滤掉空行
    raw_datasets = raw_datasets.filter(lambda example: len(example['text']) > 0)
    
    # 对齐 `wrapper.py` 的数据处理逻辑，让它认为这是一个分类任务
    # 这是适配现有 wrapper 的一个 workaround
    def format_for_wrapper(example):
        example['labels'] = example['input_ids'][:] # 在CausalLM中，labels就是input_ids
        return example
    
    if rank == 0:
        print("Tokenizing dataset...")
    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True, remove_columns=["text"])
    tokenized_datasets = tokenized_datasets.map(format_for_wrapper, batched=True)
    tokenized_datasets.set_format("torch")
    
    train_sampler = DistributedSampler(tokenized_datasets["train"], num_replicas=world_size, rank=rank, shuffle=True)
    eval_sampler = DistributedSampler(tokenized_datasets["validation"], num_replicas=world_size, rank=rank, shuffle=False)
    train_dataloader = DataLoader(tokenized_datasets["train"], sampler=train_sampler, batch_size=args.batch_size)
    eval_dataloader = DataLoader(tokenized_datasets["validation"], sampler=eval_sampler, batch_size=args.batch_size)
    # --- 4. 实例化并运行 PolarDataParallel ---
    # 注意：我们不再让 Wrapper 内部加载模型和数据，而是直接将创建好的对象传入
    if rank == 0:
        print("Initializing PolarDataParallel...")
        
    # # 我们需要模拟一个假的 dataset 对象，因为它在 wrapper 内部被引用
    # # 这是对当前 wrapper 实现的一个小妥协
    # class MockDataset:
    #     def __init__(self, tokenized_data):
    #         self.train = tokenized_data['train']
    #         self.validation = tokenized_data['validation']
    #         self.test = tokenized_data['test']
    #     def __getitem__(self, key):
    #         return getattr(self, key)
    
    # # Monkey-patch wrapper.py 内部的数据加载和处理逻辑，因为我们已经在外部完成了
    # # 这是一个更健壮的方案，避免修改 wrapper.py 内部逻辑
    # def new_init(self, args, inter_group, local_group, model, device, tokenizer, tokenized_dataset, train_dataloader, eval_dataloader):
    #     # 直接调用原始__init__，但跳过模型、tokenizer和数据的加载
    #     original_init = PolarDataParallel.__original_init__
        
    #     # 模拟wrapper内部加载过程
    #     self.args = args
    #     self.tokenizer = tokenizer
    #     self.model = model.to(device)
    #     self.device = device
    #     self.tokenized_dataset = tokenized_dataset
    #     self.dataset = MockDataset(tokenized_dataset)
        
    #     # 调用原始 __init__ 的剩余部分
    #     original_init(self, args, inter_group, local_group, model=self.model, device=device, 
    #                   tokenizer=self.tokenizer, train_dataloader=train_dataloader, eval_dataloader=eval_dataloader)

    # # 保存原始的 init 方法
    # PolarDataParallel.__original_init__ = PolarDataParallel.__init__
    # # 用我们的新 init 替换它
    # PolarDataParallel.__init__ = new_init


    polar_wrapper = PolarDataParallel(
        args=args,
        inter_group=inter_group,
        local_group=local_group,
        model=model,
        tokenizer=tokenizer,
        # tokenized_dataset=tokenized_datasets, # 传入已处理好的数据
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        device=torch.device(f"cuda:{dist.get_rank(group=local_group)}")
    )

    if rank == 0:
        print("Starting training...")
    polar_wrapper.train()

    # 恢复原始的__init__
    PolarDataParallel.__init__ = PolarDataParallel.__original_init__

if __name__ == "__main__":
    main()