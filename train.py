import os
import datetime
import argparse
import logging
import numpy as np

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    BertModel,
    get_linear_schedule_with_warmup,
)
from transformers.models.bert.modeling_bert import (
    BertLayer,
    BertEmbeddings,
    BertPooler,
    BertEncoder,
)
from datasets import load_dataset, load_from_disk
from tqdm import tqdm
from utils.buffer import TensorBuffer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class PolarCommHook:
    def __init__(self, rank, partitions, manage_hook_handle, inter_group, local_group, comm_work, grads, tensor_buffer):
        self.rank = rank
        self.partitions = partitions
        self.manage_hook_handle = manage_hook_handle
        self.inter_group = inter_group
        self.local_group = local_group
        self.comm_work = comm_work
        self.grads = grads
        self.tensor_buffer = tensor_buffer
        
    def __call__(self, module, grad_input, grad_output):
        # If this is not the first partitions. Then remove the hook of the previous backward partitions. 
        if self.rank:
            self.manage_hook_handle[self.rank - 1].remove()
            self.manage_hook_handle[self.rank - 1] = None
        
        # Register a hook to the last layer of this partitions, which won't be called in the current iteration
        # This hook will be called in the next iteration, which is used to remove the current hook. 
        # So the next iteration won't call the current hook again. 
        hook = PolarManageHook(
            inter_group=self.inter_group,
            local_group=self.local_group,
            partitions=self.partitions,
            send_buffer=self.partitions[self.rank],
        )
        self.manage_hook_handle[self.rank] = self.partitions[self.rank][-1].register_full_backward_hook(hook)
            
        self.grads = [p.grad.copy() for p in self.partitions[self.rank]]   
        self.tensor_buffer = TensorBuffer(self.grads)
        self.comm_work = dist.all_reduce(self.tensor_buffer.buffer, async_op=True, group=self.inter_group)
    
class PolarManageHook:
    def __init__(self, rank, partitions, comm_hook_handle, inter_group, local_group, comm_work, grads, tensor_buffer):
        self.rank = rank
        self.partitions = partitions
        self.comm_hook_handle = comm_hook_handle
        self.inter_group = inter_group
        self.local_group = local_group
        self.comm_work = comm_work
        self.grads = grads
        self.tensor_buffer = tensor_buffer
        
    def __call__(self, module, grad_input, grad_output):
        if self.comm_hook_handle[self.rank] is not None:
            self.comm_hook_handle[self.rank].remove()
            self.comm_hook_handle[self.rank] = None
        
        self.comm_work.wait()
        self.comm_work = None
        
        # self.grads[self.rank] = self.tensor_buffer[self.rank]

class GradientCollector:
    def __init__(self, inter_group, local_group, partitions, send_buffer):
        """__init__

        Args:
            inter_group (_type_): Distributed group for inter-node communication
            local_group (_type_): Distributed group for intra-node communication
            partition (_type_): Model partition to which this hook is attached. i.e. hook does not modify the model parameters either the gradients.
            send_buffer (_type_): Send buffer for gradients, size of this buffer should be equal to the partition's size, everything received from all_reduce operation will be an in-place operation..
        """
        self.inter_group = inter_group
        self.local_group = local_group
        self.partitions = partitions
        self.comm_hook_handle = [None for _ in range(len(self.partitions))]
        self.comm_work = None
        self.manage_hook_handle = [None for _ in range(len(self.partitions))]
        self.send_buffer = send_buffer
        self.grads = [[p.grad.copy() for p in partition] for partition in self.partitions]
        self.tensor_buffers = [TensorBuffer(grad) for grad in self.grads]
        
    def register_hook(self, rank):
        self.comm_hook_handle[rank] = self.partitions[rank][0].register_full_backward_hook(
            PolarCommHook(
                rank=rank,
                partitions=self.partitions,
                manage_hook_handle=self.manage_hook_handle,
                inter_group=self.inter_group,
                local_group=self.local_group,
                comm_work=self.comm_work,
            )
        )
        
    def synchronize(self):
        if self.comm_work is not None:
            self.comm_work.wait()
            self.comm_work = None
        if self.comm_hook_handle is not None:
            self.comm_hook_handle.remove()
            self.comm_hook_handle = None
        

class PolarTrainer:
    def __init__(
        self,
        args: argparse.Namespace,
        inter_group,
        local_group,
        model=None,
        device=None,
        tokenizer=None,
    ):
        self.local_group = local_group
        self.inter_group = inter_group
        self.device = device or torch.device(
            f"cuda:{dist.get_rank(group=self.local_group)}"
        )
        if args.pretrained:
            self.model = model or AutoModelForSequenceClassification.from_pretrained(
                args.model_path, torch_dtype=torch.float16, num_labels=args.num_labels
            )
            self.model.to(device)
        else:
            config = AutoConfig.from_pretrained(
                args.model_path, torch_dtype=torch.float16, num_labels=args.num_labels
            )
            self.model = model or AutoModelForSequenceClassification.from_config(config)
            self.model.to(device)

        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(args.tokenizer_path)
        self.dataset = load_from_disk(args.data_path)

        def tokenize_function(examples):
            return tokenizer(
                examples["sentence"],
                padding="max_length",
                truncation=True,
                max_length=args.max_length,
            )

        tokenized_dataset = self.dataset.map(tokenize_function, batched=True)
        tokenized_dataset = tokenized_dataset.remove_columns(["sentence", "idx"])
        tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
        tokenized_dataset.set_format("torch")
        self.tokenized_dataset = tokenized_dataset

        self.model_partitions = None
        if args.model == "bert-base-uncased":
            self.model_partitions = self.split_bert_based_model_into_partitions(
                self.model, self.args.local_steps
            )
        else:
            raise NotImplementedError(
                "Only BERT-based models are supported for partitioning at the moment."
            )

        train_sampler = DistributedSampler(
            tokenized_dataset["train"],
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=True,
        )

        eval_sampler = DistributedSampler(
            tokenized_dataset["validation"],
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=False,
        )

        self.train_dataloader = DataLoader(
            tokenized_dataset["train"],
            batch_size=args.batch_size,
            sampler=train_sampler,
        )

        self.eval_dataloader = DataLoader(
            tokenized_dataset["validation"],
            batch_size=args.batch_size,
            sampler=eval_sampler,
        )

        self.optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        total_steps = len(self.train_dataloader) * args.epochs

        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=0.1 * total_steps,
            num_training_steps=total_steps,
        )
        
        self.grad_partitions_bucket = []

    def get_bert_all_layers(self, module):
        for name, child in module.named_children():
            if isinstance(child, BertModel):
                yield from self.get_bert_all_layers(child)
            if isinstance(child, BertEncoder):
                yield from self.get_bert_all_layers(child)
            if isinstance(child, torch.nn.ModuleList):
                yield from self.get_bert_all_layers(child)
            if isinstance(child, BertEmbeddings):
                yield child
            if isinstance(child, BertLayer):
                yield child
            if isinstance(child, BertPooler):
                yield child
            if isinstance(child, torch.nn.Linear):
                yield child
            if isinstance(child, torch.nn.Dropout):
                yield child

    def split_bert_based_model_into_partitions(self, num_partitions):
        # TODO: the module list need to reverse. 
        modules = list(self.get_bert_all_layers(self.model))
        layer_param_sizes = []

        for layer in modules:
            layer_param_sizes.append(
                sum(p.numel() for p in layer.parameters() if p.requires_grad)
            )

        total_params = sum(layer_param_sizes)
        target_size = total_params / num_partitions

        partitions = []
        i = 0
        while True:
            partitions_params_size = 0
            partition = []
            while partitions_params_size < target_size and i < len(layer_param_sizes):
                partitions_params_size += layer_param_sizes[i]
                partition.append(modules[i])
                i += 1

            if i == len(layer_param_sizes):
                partitions.append(partition)
                break

            if abs(partitions_params_size - target_size) < abs(
                partitions_params_size + layer_param_sizes[i] - target_size
            ):
                partitions.append(partition)
            else:
                partitions.append(partition.append(modules[i]))
                i += 1

        return partitions

    def train(self, args):
        # send_buffers = [
        #     torch.zeros_like(param)
        #     for param in self.model.parameters()
        #     if param.requires_grad
        # ]
        print("send_buffers")
        LOCAL_STEPS = 1  # 每4个batch同步一次梯度
        current_local_step = 0  # 当前本地步数计数器
        print(f"LOCAL_STEPS: {args.local_steps}")
        # 训练循环
        for epoch in range(args.epochs):
            self.model.train()
            total_loss = 0
            progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}")
            send_buffers = [
                torch.zeros_like(param)
                for param in self.model_partitions[0][0].parameters()
            ]
            for batch_idx, batch in enumerate(progress_bar):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs.loss
                loss.backward()
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                part_rank = current_local_step % args.local_steps
                send_buffer = [torch.zeros_like(param) for param in self.model_partitions[part_rank]]
                
                
                
                current_local_step += 1
                need_sync = (current_local_step % args.local_steps == 0) or (
                    batch_idx + 1 == len(self.train_dataloader)
                )

                if need_sync:
                    grad_vec = [
                        parameter.grad
                        for parameter in self.model.parameters()
                        if parameter.requires_grad
                    ]
                    for grad, send_buffer in zip(grad_vec, send_buffers):
                        send_buffer[:] = grad

                    tensor_buffer = TensorBuffer(send_buffers)
                    flat_buffer = tensor_buffer.buffer
                    dist.all_reduce(flat_buffer)
                    tensor_buffer.buffer = flat_buffer
                    grads = tensor_buffer.deflatten()

                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                else:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                total_loss += loss.item()
                progress_bar.set_postfix({"loss": loss.item()})

            avg_train_loss = total_loss / len(self.train_dataloader)
            print(f"Epoch {epoch+1} Average Loss: {avg_train_loss:.4f}")


def _train(args: argparse.Namespace, inter_group, local_group):
    local_rank = dist.get_rank(group=local_group)
    if local_rank < 0:
        logger.error("local_rank is less than 0, check the local_group initialization.")
        return
    device = torch.device(f"cuda:{local_rank}")

    dataset = load_from_disk(args.data_path)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    def tokenize_function(examples):
        return tokenizer(
            examples["sentence"],
            padding="max_length",
            truncation=True,
            max_length=args.max_length,
        )

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset = tokenized_dataset.remove_columns(["sentence", "idx"])
    tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
    tokenized_dataset.set_format("torch")

    train_sampler = DistributedSampler(
        tokenized_dataset["train"],
        num_replicas=dist.get_world_size(),
        rank=dist.get_rank(),
        shuffle=True,
    )

    eval_sampler = DistributedSampler(
        tokenized_dataset["validation"],
        num_replicas=dist.get_world_size(),
        rank=dist.get_rank(),
        shuffle=False,
    )

    train_dataloader = DataLoader(
        tokenized_dataset["train"], batch_size=args.batch_size, sampler=train_sampler
    )

    eval_dataloader = DataLoader(
        tokenized_dataset["validation"],
        batch_size=args.batch_size,
        sampler=eval_sampler,
    )

    if args.pretrained:
        model = AutoModelForSequenceClassification.from_pretrained(
            args.epochs, torch_dtype=torch.float16, num_labels=args.num_labels
        )
    else:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(
            args.model_path, torch_dtype=torch.float16, num_labels=args.num_labels
        )
        model = AutoModelForSequenceClassification.from_config(config)

    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_dataloader) * args.epochs

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0.1 * total_steps, num_training_steps=total_steps
    )

    send_buffers = [
        torch.zeros_like(param) for param in model.parameters() if param.requires_grad
    ]
    print("send_buffers")
    LOCAL_STEPS = 1  # 每4个batch同步一次梯度
    current_local_step = 0  # 当前本地步数计数器
    print(f"LOCAL_STEPS: {args.local_steps}")
    # 训练循环
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}")

        for batch_idx, batch in enumerate(progress_bar):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            current_local_step += 1
            need_sync = (current_local_step % args.local_steps == 0) or (
                batch_idx + 1 == len(train_dataloader)
            )

            if need_sync:
                grad_vec = [
                    parameter.grad
                    for parameter in model.parameters()
                    if parameter.requires_grad
                ]
                for grad, send_buffer in zip(grad_vec, send_buffers):
                    send_buffer[:] = grad

                tensor_buffer = TensorBuffer(send_buffers)
                flat_buffer = tensor_buffer.buffer
                dist.all_reduce(flat_buffer)
                tensor_buffer.buffer = flat_buffer
                grads = tensor_buffer.deflatten()

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            else:
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})

        avg_train_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1} Average Loss: {avg_train_loss:.4f}")


def process_group_setup():
    # init the global process group
    rank = os.environ["RANK"]
    local_rank = os.environ["LOCAL_RANK"]
    world_size = os.environ["WORLD_SIZE"]
    rank = int(rank)
    local_rank = int(local_rank)
    world_size = int(world_size)

    print(f"rank: {rank}, local_rank: {local_rank}, world_size: {world_size}")

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="bert-base-uncased", help="model name"
    )
    parser.add_argument(
        "--pretrained", type=bool, default=False, help="use pretrained model"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/glue/sst2",
        help="path to the training data",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/bert-base-uncased",
        help="path to the model",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="tokenizer/bert-base-uncased",
        help="path to the tokenizer",
    )
    parser.add_argument(
        "--num_labels", type=int, default=2, help="classification kinds"
    )
    parser.add_argument(
        "--max_length", type=int, default=128, help="max length of the input sequence"
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local_steps", type=int, default=4, help="local steps")

    logger.info("Setting up process groups...")
    global_group, inter_group, local_group = process_group_setup()

    logger.info("Starting training...")
    _train(args=parser.parse_args(), inter_group=inter_group, local_group=local_group)
