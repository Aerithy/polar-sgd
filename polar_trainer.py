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
from typing import List

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class PolarCommHook:
    def __init__(
        self,
        partition_id: int,
        partitions: List[List[torch.nn.Module]],
        inter_group: torch.distributed.ProcessGroup,
        local_group: torch.distributed.ProcessGroup,
        comm_works: List[torch.distributed.Work],
        grads: List[List[torch.Tensor]],
        grads_pred: List[List[torch.Tensor]],
        # tensor_buffer,
        errors: List[List[torch.Tensor]],
    ):
        """
        rank (int): Communication rank of the partitions.
        partitions (list): List of model partitions. Each partition is a list of layers.
        grads (list): List of gradients. These gradients are used to accumulate the parameter.grads.
        iterations (int): Currently batch iterations.
        """
        self.partition_id = partition_id
        self.partitions = partitions
        self.inter_group = inter_group
        self.local_group = local_group
        self.comm_works = comm_works
        self.grads = grads
        self.grads_pred = grads_pred
        self.flatten_grad_pred = None
        # self.tensor_buffer = tensor_buffer
        self.iterations = 0
        self.errors = errors

    def __call__(self, module, grad_input, grad_output):
        # If this is not the first partitions. Then remove the hook of the previous backward partitions.
        # For every iteration, self.grads is the accumulation of the previous partitions' gradients.
        # print("iterations: ", self.iterations)
        device = self.partitions[self.partition_id][0].parameters().__next__().device
        for i in range(len(self.partitions[self.partition_id])):
            # self.grads[self.rank][i] += self.partitions[self.rank][i].grad
            for j, param in enumerate(self.partitions[self.partition_id][i].parameters()):
                if param.grad is not None:
                    # print(param.grad)
                    if self.grads[self.partition_id][i][j] is None:
                        self.grads[self.partition_id][i][j] = torch.zeros_like(param.grad)
                        # print(self.grads[self.rank][i][j])
                    self.grads[self.partition_id][i][j].add_(param.grad)
                    # print(self.grads[self.rank][i][j])
                
        # print(self.grads[self.rank])

        """_summary_
        doing broadcast, node [rank] will broadcast the [rank]th partition's gradients to all other nodes
         0   1   2   3 
        {0}  1   2   3 iter_0
        """
        if self.partition_id == self.iterations:
            scale = len(self.partitions) / (self.partition_id + 1)
            self.grads_pred[self.iterations] = [
                [
                    g * scale + e if e is not None and g is not None else torch.zeros(1,device=device)
                    for g, e in zip(layer_g, layer_e)
                ]
                for layer_g, layer_e in zip(self.grads[self.iterations], self.errors[self.iterations])
            ]
            (
                self.flattened_grad_pred,
                self.num_tensors_per_list,
                self.tensor_sizes,
                self.original_shapes,
            ) = self.flatten_nested_tensor_list(self.grads_pred[self.iterations])

            comm_handle = dist.broadcast(
                self.flattened_grad_pred, src=self.iterations, async_op=True, group=self.inter_group
            )
            self.comm_works[self.iterations] = comm_handle

        if self.iterations == len(self.partitions) - 1:
            self.errors[self.partition_id] = [
                [
                    g - p if g is not None and p is not None else torch.zeros(1, device=device)
                    for g, p in zip(layer_g, layer_p) 
                ]
                for layer_g, layer_p in zip(self.grads[self.partition_id], self.grads_pred[self.partition_id])
            ]
            self.comm_works[self.partition_id].wait()
            self.grads_pred[self.partition_id] = self.unflatten_nested_tensor_list(
                self.flattened_grad_pred,
                self.num_tensors_per_list,
                self.tensor_sizes,
                self.original_shapes,
            )
            
            all_reduce_fix = [
                [
                    g - p if g is not None and p is not None else torch.zeros(1, device=device)
                    for g, p in zip(layer_g, layer_p)
                ]
                for layer_g, layer_p in zip(self.grads[self.partition_id], self.grads_pred[self.partition_id])
            ]
            
            for i in range(len(self.partitions[self.partition_id])):
                for p, p_pred in zip(self.partitions[self.partition_id][i].parameters(), all_reduce_fix[i]):
                    if p.grad is not None:
                        p_pred = p_pred.to(p.grad.dtype)
                        p.grad = p.grad - p_pred

        self.iterations += 1
        self.iterations %= len(self.partitions)

    def flatten_nested_tensor_list(
        self,
        nested_list: List[List[torch.Tensor]],
    ) -> tuple[torch.Tensor, List[int], List[int], List[List[torch.Size]]]:
        # 压平内层 List[torch.Tensor] 并记录结构 flatten each sublist of tensor (List[torch.Tensor]) and record structure of them
        flattened_tensors = []
        tensor_sizes = []  # 存储每个 Tensor 的元素数量 store the number of elements in each tensor
        num_tensors_per_list = []  # 存储每个子列表的 Tensor 数量 store the number of tensors in each sublist

        for sublist in nested_list:
            num_tensors_per_list.append(len(sublist))  # 记录当前子列表的 Tensor 数量 record the number of tensors in current sublist
            for tensor in sublist:
                flattened = tensor.flatten() if tensor is not None else None # 压平当前 Tensor flatten the current Tensor
                tensor_sizes.append(flattened.numel())  # 记录元素数量 record the number of elements
                flattened_tensors.append(flattened)

        # 将所有 Tensor 拼接成一个 1D Tensor cat all tensor in to a 1D Tensor
        final_flattened = torch.cat(flattened_tensors)

        original_shapes = [[t.shape for t in sublist] for sublist in nested_list]

        return final_flattened, num_tensors_per_list, tensor_sizes, original_shapes

    def unflatten_nested_tensor_list(
        self,
        flattened: torch.Tensor,
        num_tensors_per_list: List[int],
        tensor_sizes: List[int],
        original_shapes: List[List[tuple]],  # 需额外传入原始形状
    ) -> List[List[torch.Tensor]]:
        # Step 1: 按 tensor_sizes 切分 1D Tensor
        split_tensors = torch.split(flattened, tensor_sizes)

        # Step 2: 恢复每个 Tensor 的原始形状
        restored_tensors = []
        idx = 0
        for shape_group in original_shapes:
            current_group = []
            for shape in shape_group:
                tensor = split_tensors[idx].view(shape)  # 恢复形状
                current_group.append(tensor)
                idx += 1
            restored_tensors.append(current_group)

        return restored_tensors


class NativePolarGradientCollector:
    """
    methods:
        register_hook(rank): Register hook to the first layer of the partition.
        synchronize(): Synchronize the gradients across all partitions.
    """

    def __init__(
        self, inter_group, local_group, partitions: List[List[torch.nn.Module]]
    ):
        """__init__

        Args:
            inter_group (_type_): Distributed group for inter-node communication
            local_group (_type_): Distributed group for intra-node communication
            partition (_type_): Model partition to which this hook is attached. i.e. hook does not modify the model parameters either the gradients.
            send_buffer (_type_): Send buffer for gradients, size of this buffer should be equal to the partition's size, everything received from all_reduce operation will be an in-place operation..
        """
        self.inter_group = inter_group
        self.local_group = local_group
        self.partitions = partitions[::-1]
        self.comm_hook_handle = [None for _ in range(len(self.partitions))]
        self.comm_works = [None for _ in range(len(self.partitions))]
        # self.send_buffer = send_buffer
        self.grads_accumulation = [
            [
                [None for p in layer.parameters()]
                for layer in partition
            ]
            for partition in self.partitions
        ]
        self.grads_pred = [
            [
                [None for p in layer.parameters()]
                for layer in partition
            ]
            for partition in self.partitions
        ]
        # self.tensor_buffers = [TensorBuffer(grad) for grad in self.grads]
        self.errors = [
            [
                [None for p in layer.parameters()]
                for layer in partition
            ]
            for partition in self.partitions
        ]
        # print(self.grads_accumulation[0][0][0].device)
        # print(self.grads_pred[0][0][0].device)
        # print(self.errors[0][0][0].device)

    def register_hook(self):
        """Register hook to the first layer of the partition.

        # Architecture

        ```
        model = [
            partition[N - 1] = {
                layer[0 * M]        ==> register hook here, if rank = N - 1.
                ...
                layer[1 * M - 1]
            }
            partition[0] = {
                layer[(N - 1) * M]  ==> register hook here, if rank = 0.
                ...
                layer[N * M - 1]
            }
        ]
        ```

        Args:
            rank (int): Rank of the partition.

        """
        for rank in range(len(self.partitions)):
            self.comm_hook_handle[rank] = self.partitions[rank][
                0
            ].register_full_backward_hook(
                PolarCommHook(
                    partition_id=rank,
                    partitions=self.partitions,
                    inter_group=self.inter_group,
                    local_group=self.local_group,
                    comm_works=self.comm_works,
                    grads=self.grads_accumulation,
                    grads_pred=self.grads_pred,
                    errors=self.errors,
                )
            )

    def synchronize(self):
        if self.comm_work is not None:
            for work in self.comm_works:
                if work is not None:
                    work.wait()
                    work = None

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
        self.args = args
        self.local_group = local_group
        self.inter_group = inter_group
        self.device = torch.device(
            f"cuda:{dist.get_rank(group=self.local_group)}"
        ) if torch.cuda.is_available() else torch.device("cpu")
        if args.pretrained:
            self.model = model or AutoModelForSequenceClassification.from_pretrained(
                args.model_path, torch_dtype=torch.float16, num_labels=args.num_labels
            )
            self.model.to(self.device)
            print(next(self.model.parameters()).device)
        else:
            config = AutoConfig.from_pretrained(
                args.model_path, torch_dtype=torch.float16, num_labels=args.num_labels
            )
            self.model = model or AutoModelForSequenceClassification.from_config(config)
            self.model.to(self.device)
            print(next(self.model.parameters()).device)

        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(args.tokenizer_path)
        self.dataset = load_from_disk(args.data_path)

        def tokenize_function(examples):
            return self.tokenizer(
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

        print(next(self.model.parameters()).device)
        self.model_partitions = None
        if args.model == "bert-base-uncased":
            self.model_partitions = self.split_bert_based_model_into_partitions(
                args.local_steps
            )
        else:
            raise NotImplementedError(
                "Only BERT-based models are supported for partitioning at the moment."
            )
        print(next(self.model_partitions[0][0].parameters()).device)

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
        self.optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
        total_steps = len(self.train_dataloader) * args.epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=0.1 * total_steps,
            num_training_steps=total_steps,
        )
        self.grad_partitions_bucket = []
        self.gradient_collector = NativePolarGradientCollector(
            inter_group=self.inter_group if not args.single_node else self.local_group,
            local_group=self.local_group,
            partitions=self.model_partitions,
        )
        if args.using_hook:
            self.gradient_collector.register_hook()

    def get_bert_all_layers(self, module: torch.nn.Module):
        # 遍历模块的所有子模块
        for child in module.modules():
            if isinstance(child, (BertModel, BertEncoder, torch.nn.ModuleList)):
                continue  # 跳过容器类模块，避免重复处理
            if isinstance(child, (BertEmbeddings, BertLayer, BertPooler, torch.nn.Linear, torch.nn.Dropout)):
                yield child

    def split_bert_based_model_into_partitions(self, num_partitions: int):
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

    def train(self):
        # send_buffers = [
        #     torch.zeros_like(param)
        #     for param in self.model.parameters()
        #     if param.requires_grad
        # ]
        print("send_buffers")
        LOCAL_STEPS = 1  # 每4个batch同步一次梯度
        current_local_step = 0  # 当前本地步数计数器
        print(f"LOCAL_STEPS: {self.args.local_steps}")
        # 训练循环
        for epoch in range(self.args.epochs):
            self.model.train()
            total_loss = 0
            progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}")
            for batch_idx, batch in enumerate(progress_bar):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs.loss
                loss.backward()
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                part_rank = current_local_step % self.args.local_steps

                current_local_step += 1

                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

                total_loss += loss.item()
                progress_bar.set_postfix({"loss": loss.item()})

            avg_train_loss = total_loss / len(self.train_dataloader)
            print(f"Epoch {epoch+1} Average Loss: {avg_train_loss:.4f}")
            
    def _train(self):
        # send_buffers = [
        #     torch.zeros_like(param)
        #     for param in self.model.parameters()
        #     if param.requires_grad
        # ]
        print("send_buffers")
        LOCAL_STEPS = 1  # 每4个batch同步一次梯度
        current_local_step = 0  # 当前本地步数计数器
        print(f"LOCAL_STEPS: {self.args.local_steps}")
        # 训练循环
        for epoch in range(self.args.epochs):
            self.model.train()
            total_loss = 0
            progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}")
            for batch_idx, batch in enumerate(progress_bar):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs.loss
                loss.backward()
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                part_rank = current_local_step % self.args.local_steps

                current_local_step += 1

                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

                total_loss += loss.item()
                progress_bar.set_postfix({"loss": loss.item()})

            avg_train_loss = total_loss / len(self.train_dataloader)
            print(f"Epoch {epoch+1} Average Loss: {avg_train_loss:.4f}")


def process_group_setup():
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
    # _train(args=parser.parse_args(), inter_group=inter_group, local_group=local_group)
