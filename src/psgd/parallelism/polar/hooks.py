import os
import datetime
import argparse
import logging
import numpy as np

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    BertModel,
    get_linear_schedule_with_warmup,
)
import transformers
from transformers.models.bert.modeling_bert import (
    BertLayer,
    BertEmbeddings,
    BertPooler,
    BertEncoder,
)
from datasets import load_dataset, load_from_disk
from tqdm import tqdm
from psgd.utils.buffer import TensorBuffer
from typing import List, Tuple

from .util import get_partitions_and_pipe

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class GpipeHook:
    """Consider Gpipe automatically adapt gradient accumulation mechanism, we do not need to accumulate gradient manually
    """
    def __init__(
        self,
        device_mesh: dist.device_mesh.DeviceMesh,
        model: torch.nn.Module,
        grads: List[torch.Tensor],
        grads_pred: List[torch.Tensor],
        errors: List[torch.Tensor],
        micro_batch_size: int,
    ):
        self.device_mesh = device_mesh
        self.model = model
        self.pp_mesh = device_mesh["pp"]
        self.dp_mesh = device_mesh["dp"]
        self.pp_group = self.pp_mesh.get_group()
        self.dp_group = self.dp_mesh.get_group()
        self.pp_local_rank = self.device_mesh.get_local_rank("pp")
        self.dp_local_rank = self.device_mesh.get_local_rank("dp")
        self.pp_size = self.pp_mesh.size()
        self.dp_size = self.dp_mesh.size()
        self.grads = grads
        self.grads_pred = grads_pred
        self.errors = errors
        self.micro_batch_counter = 0
        self.micro_batch_size = micro_batch_size
        self.offset = 0
        self.comm_handle = None

    def __call__(self, *args, **kwds):
        # self.micro_batch_counter += 1
        device = next(self.model.parameters())
        # device = dist.
        logger.debug(f"[hook:call] pid={self.pp_mesh.get_local_rank()}, mb_counter={self.micro_batch_counter}/{self.micro_batch_size}, iter={self.iterations}")
        
        # 0 1|2 3|4 5|6 7| micro_batch_size / self.pp_size  micro_batch_count / (micro_batch_size / self.pp_size)
        
        if self.micro_batch_counter == (self.pp_local_rank + 1) * (self.micro_batch_size / self.pp_size) - 1:
            scale = self.micro_batch_size / (self.micro_batch_counter + 1)
            self.grads_pred = [
                [
                    g * scale + e if e is not None and g is not None else torch.zeros(1,device=device)
                    for g, e in zip(layer_g, layer_e)
                ]
                for layer_g, layer_e in zip(self.grads, self.errors)
            ]
            (
                self.flattened_grad_pred,
                self.num_tensors_per_list,
                self.tensor_sizes,
                self.original_shapes,
            ) = self.flatten_nested_tensor_list(self.grads_pred)
            self.comm_handle = dist.all_reduce(
                self.flattened_grad_pred, group=self.dp_group, async_op=True
            )
            self.micro_batch_counter += 1
        elif self.micro_batch_counter == self.micro_batch_size:
            self.comm_handle.wait()
            self.errors = self.grads - self.grads_pred
            self.grads_pred = self.unflatten_nested_tensor_list(
                self.flattened_grad_pred,
                self.num_tensors_per_list,
                self.tensor_sizes,
                self.original_shapes,
            )
            for param, grad_pred in zip(self.model.parameters(), self.grads_pred):
                param.grad = grad_pred
            
            self.micro_batch_counter = 0
            
    def flatten_nested_tensor_list(
        self,
        nested_list: List[List[torch.Tensor]],
    ) -> Tuple[torch.Tensor, List[int], List[int], List[List[tuple]]]:
        '''
        Args:
            nested_list (List[List[torch.Tensor]]): A nested list of tensors to be flattened
        Returns:
            final_flattened (torch.Tensor): A single 1D tensor containing all elements from the nested list
            num_tensors_per_list (List[int]): Number of tensors in each sublist
            tensor_sizes (List[int]): Number of elements in each tensor
            original_shapes (List[List[tuple]]): Original shapes of each tensor in the nested list
        '''
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
        '''
        Args:
            flattened (torch.Tensor): A single 1D tensor containing all elements from the nested list
            num_tensors_per_list (List[int]): Number of tensors in each sublist
            tensor_sizes (List[int]): Number of elements in each tensor
            original_shapes (List[List[tuple]]): Original shapes of each tensor in the nested list
        Returns:
            restored_tensors (List[List[torch.Tensor]]): The nested list of tensors restored to their original shapes    
        '''
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

class PolarCommHook:
    def __init__(
        self,
        partition_id: int,
        partitions: List[List[torch.nn.Module]],
        num_chunks: int,
        inter_group: torch.distributed.ProcessGroup,
        local_group: torch.distributed.ProcessGroup,
        comm_works: List[torch.distributed.Work],
        grads: List[List[torch.Tensor]],
        grads_pred: List[List[torch.Tensor]],
        errors: List[List[torch.Tensor]],
        stream: torch.cuda.Stream,
    ):
        """
        rank (int): Communication rank of the partitions.
        partitions (list): List of model partitions. Each partition is a list of layers.
        grads (list): List of gradients. These gradients are used to accumulate the parameter.grads.
        iterations (int): Currently batch iterations.
        """
        self.partition_id = partition_id
        self.partitions = partitions
        self.num_chunks = num_chunks
        self.inter_group = inter_group
        self.local_group = local_group
        self.comm_works = comm_works
        self.grads = grads
        self.grads_pred = grads_pred
        self.flatten_grad_pred = None
        self.iterations = 0
        self.micro_batch_counter = 0
        self.errors = errors
        self.stream = stream
        self.comm_start_event = torch.cuda.Event(enable_timing=False)
        self.comm_end_event = torch.cuda.Event(enable_timing=False)
        self.offset = 0

    def __call__(self, module, grad_input, grad_output):
        self.micro_batch_counter += 1
        logger.debug(f"[hook:call] pid={self.partition_id}, mb_counter={self.micro_batch_counter}/{self.num_chunks}, iter={self.iterations}")
        if self.micro_batch_counter < self.num_chunks:
            return
        self.micro_batch_counter = 0
        # If this is not the first partitions. Then remove the hook of the previous backward partitions.
        # For every iteration, self.grads is the accumulation of the previous partitions' gradients.
        device = self.partitions[self.partition_id][0].parameters().__next__().device
        for i in range(len(self.partitions[self.partition_id])):
            for j, param in enumerate(self.partitions[self.partition_id][i].parameters()):
                if param.grad is not None:
                    if self.grads[self.partition_id][i][j] is None:
                        self.grads[self.partition_id][i][j] = torch.zeros_like(param.grad)
                    self.grads[self.partition_id][i][j].add_(param.grad)

        """_summary_
        doing broadcast, node [rank] will broadcast the [rank]th partition's gradients to all other nodes
         0   1   2   3 
        {0}  1   2   3  iter_0
         0  {1}  2   3  iter_1
         0   1  {2}  3  iter_2
         0   1   2  {3} iter_3
        """
        if self.partition_id == (self.iterations + self.offset) % len(self.partitions):
            scale = len(self.partitions) / (self.partition_id + 1)
            self.grads_pred[self.partition_id] = [
                [
                    g * scale + e if e is not None and g is not None else torch.zeros(1,device=device)
                    for g, e in zip(layer_g, layer_e)
                ]
                for layer_g, layer_e in zip(self.grads[self.partition_id], self.errors[self.partition_id])
            ]
            (
                self.flattened_grad_pred,
                self.num_tensors_per_list,
                self.tensor_sizes,
                self.original_shapes,
            ) = self.flatten_nested_tensor_list(self.grads_pred[self.partition_id])
            
            comm_handle = dist.broadcast(
                self.flattened_grad_pred, src=self.partition_id % dist.get_world_size(self.inter_group), async_op=True, group=self.inter_group
            )
            self.comm_works[self.partition_id] = comm_handle

        if self.iterations == len(self.partitions) - 1:
            self.errors[self.partition_id] = [
                [
                    g - p if g is not None and p is not None else torch.zeros(1, device=device)
                    for g, p in zip(layer_g, layer_p) 
                ]
                for layer_g, layer_p in zip(self.grads[self.partition_id], self.grads_pred[self.partition_id])
            ]
            self.comm_works[self.partition_id].wait()
            # self.comm_end_event.wait(stream=torch.cuda.default_stream())
            # self.comm_end_event.synchronize()
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
            
            self.offset = (self.offset + 1) % len(self.partitions)

        self.iterations += 1
        self.iterations %= len(self.partitions)

    def flatten_nested_tensor_list(
        self,
        nested_list: List[List[torch.Tensor]],
    ) -> Tuple[torch.Tensor, List[int], List[int], List[List[tuple]]]:
        '''
        Args:
            nested_list (List[List[torch.Tensor]]): A nested list of tensors to be flattened
        Returns:
            final_flattened (torch.Tensor): A single 1D tensor containing all elements from the nested list
            num_tensors_per_list (List[int]): Number of tensors in each sublist
            tensor_sizes (List[int]): Number of elements in each tensor
            original_shapes (List[List[tuple]]): Original shapes of each tensor in the nested list
        '''
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
        '''
        Args:
            flattened (torch.Tensor): A single 1D tensor containing all elements from the nested list
            num_tensors_per_list (List[int]): Number of tensors in each sublist
            tensor_sizes (List[int]): Number of elements in each tensor
            original_shapes (List[List[tuple]]): Original shapes of each tensor in the nested list
        Returns:
            restored_tensors (List[List[torch.Tensor]]): The nested list of tensors restored to their original shapes    
        '''
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