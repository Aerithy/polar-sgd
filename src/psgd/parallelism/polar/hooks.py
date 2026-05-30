import logging
import os
import time
from dataclasses import dataclass

import torch
import torch.distributed as dist
from typing import List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def _polar_hook_debug_enabled() -> bool:
    return os.environ.get("POLAR_HOOK_DEBUG", "0").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _polar_hook_debug(message: str) -> None:
    if not _polar_hook_debug_enabled():
        return
    try:
        rank = dist.get_rank()
    except Exception:
        rank = -1
    print(
        f"[polar-hook-debug rank={rank} t={time.time():.6f}] {message}",
        flush=True,
    )


def _trace_explain_enabled() -> bool:
    return os.environ.get("TRACE_EXPLAIN", "0").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _trace_evidence(component: str, action: str, message: str) -> None:
    if not _trace_explain_enabled():
        return
    try:
        rank = dist.get_rank()
    except Exception:
        rank = -1
    print(
        f"[trace-evidence rank={rank}] component={component} "
        f"action={action} {message}",
        flush=True,
    )


def _all_reduce_flat_async(
    tensor: torch.Tensor,
    *,
    group,
    lowbit_group=None,
):
    if lowbit_group is not None:
        _trace_evidence(
            "POLAR",
            "handoff_to_bitscom_lowbit",
            "meaning='POLAR produced one flat predicted DP-gradient buffer "
            "and now calls bitscom LowBitGroup.all_reduce instead of dense "
            "torch.distributed all_reduce' "
            f"numel={int(tensor.numel())} dtype={tensor.dtype} "
            f"device={tensor.device} bitwidth={getattr(lowbit_group, 'bitwidth', 'unknown')}",
        )
        if not getattr(_all_reduce_flat_async, "_logged_lowbit", False):
            logger.info(
                "[polar-hook] using LowBitGroup for flat DP all-reduce: "
                "bitwidth=%s numel=%s dtype=%s device=%s",
                getattr(lowbit_group, "bitwidth", "unknown"),
                int(tensor.numel()),
                tensor.dtype,
                tensor.device,
            )
            _all_reduce_flat_async._logged_lowbit = True
        _polar_hook_debug(
            "lowbit all_reduce start "
            f"numel={int(tensor.numel())} dtype={tensor.dtype} "
            f"bitwidth={getattr(lowbit_group, 'bitwidth', 'unknown')}"
        )
        work = lowbit_group.all_reduce(
            tensor,
            op=dist.ReduceOp.SUM,
            async_op=True,
        )
        _polar_hook_debug("lowbit all_reduce returned")
        return work
    _trace_evidence(
        "POLAR",
        "dense_dp_allreduce",
        "meaning='POLAR hook is using regular dense DP all-reduce for this "
        "buffer because no LowBitGroup was provided' "
        f"numel={int(tensor.numel())} dtype={tensor.dtype} device={tensor.device}",
    )
    _polar_hook_debug(
        "dense all_reduce async start "
        f"numel={int(tensor.numel())} dtype={tensor.dtype}"
    )
    work = dist.all_reduce(tensor, group=group, async_op=True)
    _polar_hook_debug("dense all_reduce async returned")
    return work


def _wait_and_average_flat(work, tensor: torch.Tensor, group) -> None:
    if work is not None:
        work.wait()
    world_size = dist.get_world_size(group)
    if world_size > 1:
        tensor.div_(world_size)


@dataclass(frozen=True)
class _BucketEntry:
    param_idx: int
    start: int
    length: int


@dataclass
class _InflightBucket:
    work: object
    buffer: torch.Tensor
    entries: List[_BucketEntry]


@dataclass
class _OffloadedBucket:
    buffer: torch.Tensor
    entries: List[_BucketEntry]


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
        comm_timing: int,
        lowbit_group=None,
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
        self.comm_timing = comm_timing
        self.lowbit_group = lowbit_group

    def __call__(self, *args, **kwds):
        device = next(self.model.parameters()).device
        for i, param in enumerate(self.model.parameters()):
            if param.grad is not None:
                # ✅ 累积梯度而不是直接赋值
                if self.grads[i] is None:
                    self.grads[i] = torch.zeros_like(param.grad)
                self.grads[i].add_(param.grad)
        
        # trigger_condition = self.micro_batch_counter == (self.pp_local_rank 
        # + 1) * (self.micro_batch_size / self.pp_size) - 1
        # trigger_condition = self.micro_batch_counter == self.pp_local_rank
        if self.comm_timing == -1:
            # trigger_batch = self.pp_local_rank + self.micro_batch_size / 2
            trigger_batch = self.micro_batch_size / 2
            trigger_condition = self.micro_batch_counter == trigger_batch
            logger.debug(f"""[Rank {dist.get_rank()}] 
                         PP{self.pp_local_rank} MB{self.micro_batch_counter}:
                         trigger={trigger_condition}""")
        else:
            trigger_batch = self.comm_timing
            trigger_condition = self.micro_batch_counter == trigger_batch
        if trigger_condition:
            scale = self.micro_batch_size / (self.micro_batch_counter + 1)
            # scale = 1.0
            grads_pred = []
            for g, e, p in zip(self.grads, self.errors, self.model.parameters()):
                if g is None:
                    grads_pred.append(None)
                    continue
                if e is None:
                    e = torch.zeros_like(g)
                grads_pred.append(g * scale + e)
            self.grads_pred = grads_pred
            (
                self.flattened_grad_pred,
                self.is_none_mask,
                self.tensor_numels,
                self.original_shapes,
            ) = self.flatten_tensor_list(self.grads_pred)
            if self.flattened_grad_pred.device != device:
                self.flattened_grad_pred = self.flattened_grad_pred.to(device)
                
            self.comm_handle = _all_reduce_flat_async(
                self.flattened_grad_pred,
                group=self.dp_group,
                lowbit_group=self.lowbit_group,
            )
            
            for i, e in enumerate(self.errors):
                if e is not None:
                    e.zero_()
        
        self.micro_batch_counter += 1
        
        if self.micro_batch_counter == self.micro_batch_size:
            _wait_and_average_flat(
                self.comm_handle,
                self.flattened_grad_pred,
                self.dp_group,
            )
            new_errors = []
            for g, p in zip(self.grads, self.grads_pred):
                if g is None or p is None:
                    new_errors.append(None)
                else:
                    new_errors.append(g - p)
            self.errors = new_errors
            self.grads_pred = self.unflatten_tensor_list(
                self.flattened_grad_pred,
                self.is_none_mask,
                self.tensor_numels,
                self.original_shapes,
            )
            for param, grad_pred in zip(self.model.parameters(), self.grads_pred):
                if grad_pred is None:
                    param.grad = None
                else:
                    param.grad = grad_pred.detach().clone()
            
            # ✅ 清零累积的梯度，为下一个周期做准备
            self.grads = [None for _ in self.grads]
            self.micro_batch_counter = 0
        
        # print(f"[Rank {dist.get_rank()}] hook finished, MB{self.micro_batch_counter}")
            
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
            tensor_sizes (List[int): Number of elements in each tensor
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
    
    def flatten_tensor_list(
        self,
        tensor_list: List[Optional[torch.Tensor]],
    ) -> Tuple[torch.Tensor, List[bool], List[int], List[torch.Size]]:
        """
        Flatten a list of tensors (some may be None) into a single 1D tensor.

        Args:
            tensor_list: List of tensors or None (e.g., [grad1, None, grad3, ...])

        Returns:
            flattened: Concatenated 1D tensor of all non-None tensors
            is_none: Boolean list indicating which entries were None
            tensor_numels: Number of elements for each non-None tensor (for splitting)
            original_shapes: Original shapes of non-None tensors
        """
        flattened_tensors = []
        is_none = []
        tensor_numels = []
        original_shapes = []

        for tensor in tensor_list:
            if tensor is None:
                is_none.append(True)
            else:
                is_none.append(False)
                flattened = tensor.reshape(-1)
                flattened_tensors.append(flattened)
                tensor_numels.append(flattened.numel())
                original_shapes.append(tensor.shape)

        if flattened_tensors:
            flattened = torch.cat(flattened_tensors)
        else:
            # Edge case: all None
            flattened = torch.tensor([], device=tensor_list[0].device if tensor_list and tensor_list[0] is not None else "cpu")

        return flattened, is_none, tensor_numels, original_shapes


    def unflatten_tensor_list(
        self,
        flattened: torch.Tensor,
        is_none: List[bool],
        tensor_numels: List[int],
        original_shapes: List[torch.Size],
    ) -> List[Optional[torch.Tensor]]:
        """
        Restore a flattened tensor back to the original list structure.

        Args:
            flattened: 1D tensor from flatten_tensor_list
            is_none: Boolean mask from flatten_tensor_list
            tensor_numels: Element counts of non-None tensors
            original_shapes: Shapes of non-None tensors

        Returns:
            restored: List[Optional[Tensor]] matching original structure
        """
        if len(tensor_numels) > 0:
            split_tensors = torch.split(flattened, tensor_numels)
        else:
            split_tensors = []

        restored = []
        none_idx = 0
        tensor_idx = 0

        for is_n in is_none:
            if is_n:
                restored.append(None)
            else:
                if tensor_idx < len(split_tensors):
                    restored.append(split_tensors[tensor_idx].view(original_shapes[tensor_idx]))
                    tensor_idx += 1
                else:
                    # Should not happen if inputs are consistent
                    restored.append(None)

        return restored

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

class PolarGpipeIoOptimHook:
    """POLAR gradient prediction hook with reduced IO/allocations.

    This hook preserves the exact high-level logic of `GpipeHook`:
      1) Accumulate true gradients across microbatches in `self.grads`.
      2) At a trigger microbatch, form predicted grads: g * scale + e.
      3) Flatten predicted grads and DP all-reduce them asynchronously.
      4) At end of the step, wait, update error feedback e := g - p,
         and replace param.grad with reduced predicted grads.

    IO optimizations vs `GpipeHook`:
      - Preallocate a single flat buffer (`self.flat_pred`) and reuse it.
      - Avoid torch.cat/torch.split allocations each step.
      - Avoid per-parameter `detach().clone()`: write into existing grad
        tensors via `copy_`.

    Notes:
      - This hook is meant to be registered exactly like `GpipeHook`:
        `module.register_full_backward_hook(PolarGpipeIoOptimHook(...))`
      - It does not change math; it changes memory movement/allocation only.
    """

    def __init__(
        self,
        device_mesh: dist.device_mesh.DeviceMesh,
        model: torch.nn.Module,
        grads: List[Optional[torch.Tensor]],
        grads_pred: List[Optional[torch.Tensor]],
        errors: List[Optional[torch.Tensor]],
        micro_batch_size: int,
        comm_timing: int,
        lowbit_group=None,
    ):
        self.device_mesh = device_mesh
        self.model = model

        self.pp_mesh = device_mesh["pp"]
        self.dp_mesh = device_mesh["dp"]
        self.dp_group = self.dp_mesh.get_group()

        self.pp_local_rank = self.device_mesh.get_local_rank("pp")
        self.micro_batch_size = micro_batch_size
        self.comm_timing = comm_timing

        # External state lists (kept for compatibility with existing wrapper)
        self.grads = grads
        self.grads_pred = grads_pred
        self.errors = errors

        self.micro_batch_counter = 0
        self.comm_handle: Optional[dist.Work] = None
        self.lowbit_group = lowbit_group

        # Build stable param list once
        self.params: List[torch.nn.Parameter] = [
            p for p in self.model.parameters()
        ]

        # Precompute flatten layout for non-None entries using param shapes.
        # We assume the model parameter set is stable.
        self.is_none_mask: List[bool] = []
        self.tensor_numels: List[int] = []
        self.original_shapes: List[torch.Size] = []
        total_numel = 0
        for p in self.params:
            # By construction, grads_pred entries correspond 1:1 with params.
            self.is_none_mask.append(False)
            n = p.numel()
            self.tensor_numels.append(n)
            self.original_shapes.append(p.shape)
            total_numel += int(n)

        dev = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype
        # One reusable flat buffer for predicted grads
        self.flat_pred = torch.empty(total_numel, device=dev, dtype=dtype)

    def _trigger_condition(self) -> bool:
        if self.comm_timing == -1:
            trigger_batch = self.micro_batch_size / 2
            return self.micro_batch_counter == trigger_batch
        return self.micro_batch_counter == self.comm_timing

    @torch.no_grad()
    def _pack_predicted_(self, scale: float):
        """Write predicted grads into self.flat_pred (no allocations)."""
        offset = 0
        for i, (p, g, e) in enumerate(
            zip(self.params, self.grads, self.errors)
        ):
            n = self.tensor_numels[i]
            if g is None:
                # Keep consistent with old logic: mark pred as None.
                self.grads_pred[i] = None
                self.flat_pred[offset: offset + n].zero_()
                offset += n
                continue

            if e is None:
                # Keep error tensor persistent to avoid allocs
                e = torch.zeros_like(g)
                self.errors[i] = e

            # p_pred = g * scale + e
            # Write into flat buffer directly.
            self.flat_pred[offset: offset + n].copy_(
                (g * scale + e).reshape(-1)
            )
            offset += n

            # Keep grads_pred logical view for compatibility/debug
            self.grads_pred[i] = g * scale + e

    @torch.no_grad()
    def _unpack_predicted_(self):
        """Scatter reduced predicted grads into param.grad (no clone)."""
        offset = 0
        for i, p in enumerate(self.params):
            n = self.tensor_numels[i]
            if self.grads_pred[i] is None:
                p.grad = None
                offset += n
                continue

            # Ensure grad tensor exists; then copy into it.
            if p.grad is None or p.grad.numel() != n:
                p.grad = torch.empty_like(p)
            p.grad.view(-1).copy_(self.flat_pred[offset: offset + n])
            offset += n

    def __call__(self, *args, **kwds):
        # Accumulate grads (same as GpipeHook)
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            if self.grads[i] is None:
                self.grads[i] = torch.zeros_like(p.grad)
            self.grads[i].add_(p.grad)

        if self._trigger_condition():
            scale = self.micro_batch_size / (self.micro_batch_counter + 1)
            _trace_evidence(
                "POLAR",
                "hook_trigger",
                "meaning='IO-optimized POLAR hook fires at the configured "
                "microbatch, forms predicted_grad = accumulated_grad * scale "
                "+ error_feedback, then starts async DP communication' "
                f"hook=PolarGpipeIoOptimHook pp_rank={self.pp_local_rank} "
                f"micro_batch={self.micro_batch_counter} "
                f"micro_batches={self.micro_batch_size} "
                f"comm_timing={self.comm_timing} scale={scale:.6f} "
                "uses_error_feedback=True uses_gradient_scaling=True",
            )
            _polar_hook_debug(
                "PolarGpipeIoOptimHook trigger "
                f"pp_rank={self.pp_local_rank} "
                f"micro_batch={self.micro_batch_counter} "
                f"micro_batches={self.micro_batch_size} "
                f"comm_timing={self.comm_timing} "
                f"scale={scale:.6f}"
            )
            self._pack_predicted_(scale=scale)

            # Async DP all-reduce on flat buffer
            self.comm_handle = _all_reduce_flat_async(
                self.flat_pred,
                group=self.dp_group,
                lowbit_group=self.lowbit_group,
            )

            # Clear error buffers (same semantic as old code)
            for e in self.errors:
                if e is not None:
                    e.zero_()

        self.micro_batch_counter += 1

        if self.micro_batch_counter == self.micro_batch_size:
            # Wait for DP reduction
            if self.comm_handle is not None:
                _trace_evidence(
                    "POLAR",
                    "wait_for_async_dp_reduce",
                    "meaning='Pipeline reached the end of this step; POLAR now "
                    "waits for the early DP communication result and averages "
                    "it across DP replicas before optimizer.step()' "
                    f"hook=PolarGpipeIoOptimHook pp_rank={self.pp_local_rank}",
                )
                _polar_hook_debug(
                    "PolarGpipeIoOptimHook wait start "
                    f"pp_rank={self.pp_local_rank}"
                )
                _wait_and_average_flat(
                    self.comm_handle,
                    self.flat_pred,
                    self.dp_group,
                )
                _polar_hook_debug(
                    "PolarGpipeIoOptimHook wait done "
                    f"pp_rank={self.pp_local_rank}"
                )

            # Update errors: e := g - p
            new_errors: List[Optional[torch.Tensor]] = []
            for g, p_pred in zip(self.grads, self.grads_pred):
                if g is None or p_pred is None:
                    new_errors.append(None)
                else:
                    # Keep persistent error tensor to avoid allocs
                    err = g - p_pred
                    new_errors.append(err)
            self.errors = new_errors
            _trace_evidence(
                "POLAR",
                "error_feedback_update",
                "meaning='POLAR stores residual error_feedback = true_grad - "
                "communicated_predicted_grad for the next step' "
                f"hook=PolarGpipeIoOptimHook pp_rank={self.pp_local_rank}",
            )

            # Scatter reduced predicted grads to param.grad without clone
            self._unpack_predicted_()

            # Reset for next step
            for i in range(len(self.grads)):
                self.grads[i] = None
            self.micro_batch_counter = 0
            self.comm_handle = None

class OneStepDelayHook:
    """One-step delayed SGD (no error compensation).

    Policy:
      - Step 0 (first full step): compute grads, DP all-reduce, store as
        `prev_flat`, but do NOT apply update (freeze).
      - Step t (t>=1):
          * Apply update using grads from step t-1 (`prev_flat`).
          * Compute current grads, DP all-reduce into `cur_flat`.
          * Swap: prev_flat <- cur_flat.

    Notes:
      - This is a *hook-driven* optimizer: it directly updates parameters.
        You must NOT call optimizer.step() in the training loop when this hook
        is enabled (otherwise you'd double-apply updates).
      - Works for pipeline parallel stages: hook triggers at end of each global
        step (after `micro_batch_size` microbatches).
      - DP sync is done with all-reduce(SUM) and divide by dp_size.
    """

    def __init__(
        self,
        device_mesh: dist.device_mesh.DeviceMesh,
        model: torch.nn.Module,
        micro_batch_size: int,
        lr: float = 1e-4,
        weight_decay: float = 0.0,
    ):
        self.device_mesh = device_mesh
        self.model = model

        self.dp_mesh = device_mesh["dp"]
        self.dp_group = self.dp_mesh.get_group()
        self.dp_size = self.dp_mesh.size()

        self.micro_batch_size = int(micro_batch_size)
        self.micro_batch_counter = 0

        self.lr = float(lr)
        self.weight_decay = float(weight_decay)

        # Stable parameter list
        self.params: List[torch.nn.Parameter] = [
            p for p in self.model.parameters()
        ]

        # Flatten layout
        self.numels: List[int] = [int(p.numel()) for p in self.params]
        self.shapes: List[torch.Size] = [p.shape for p in self.params]
        self.total_numel = int(sum(self.numels))

        dev = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype

        # Buffers
        self.cur_flat = torch.empty(self.total_numel, device=dev, dtype=dtype)
        self.prev_flat: Optional[torch.Tensor] = None

        self.step_idx = 0

    @torch.no_grad()
    def _apply_update_from_flat_(self, flat: torch.Tensor):
        """In-place SGD update using flattened grads."""
        offset = 0
        for p, n in zip(self.params, self.numels):
            g = flat[offset: offset + n].view(p.shape)
            if self.weight_decay != 0.0:
                # Decoupled weight decay (AdamW) would be p *= (1-lr*wd).
                # Here we do classic L2 regularization: g += wd * p
                g = g.add(p, alpha=self.weight_decay)
            p.add_(g, alpha=-self.lr)
            offset += n

    def __call__(self, *args, **kwds):
        # Accumulate grads across microbatches (same spirit as GpipeHook)
        for p in self.params:
            if p.grad is None:
                continue
            # no accumulation buffer: rely on autograd accumulation;
            # just leave p.grad as is.

        self.micro_batch_counter += 1

        if self.micro_batch_counter < self.micro_batch_size:
            return

        # End of a global step
        self.micro_batch_counter = 0

        # Apply update using previous step grads (freeze at step0)
        if self.prev_flat is not None:
            self._apply_update_from_flat_(self.prev_flat)

        # Build current flat grad buffer and DP all-reduce
        self._flatten_grads_(self.cur_flat)

        if self.dp_size > 1:
            dist.all_reduce(
                self.cur_flat,
                op=dist.ReduceOp.SUM,
                group=self.dp_group,
            )
            self.cur_flat.div_(self.dp_size)

        # Save for next step
        if self.prev_flat is None:
            self.prev_flat = torch.empty_like(self.cur_flat)
        self.prev_flat.copy_(self.cur_flat)

        # Clear grads so next step doesn't accidentally reuse them
        for p in self.params:
            p.grad = None

        self.step_idx += 1

class PolarGpipeMomentumExtrapHook:
    """POLAR hook without scaling; uses momentum/trend extrapolation.

    Compared to `PolarGpipeIoOptimHook` / `GpipeHook`:
      - NO microbatch scaling factor (no `g * scale`).
      - Prediction uses an EMA momentum buffer `m`:

            m_t = beta * m_{t-1} + (1 - beta) * g_t
            pred = m_t + e

        where `g_t` is the accumulated gradient over the whole step.

    Communication:
      - At the trigger microbatch, we all-reduce the *current prediction*
        asynchronously (still uses the IO-optimized flat buffer path).
      - At end of step, we wait, update error feedback `e := g - pred`,
        and write the reduced prediction to `param.grad`.

    Notes:
      - This hook is compatible with the existing wrapper lists:
        `grads`, `grads_pred`, `errors`.
      - If you use this hook, you still call `optimizer.step()` as usual
        (same as other POLAR hooks).
    """

    def __init__(
        self,
        device_mesh: dist.device_mesh.DeviceMesh,
        model: torch.nn.Module,
        grads: List[Optional[torch.Tensor]],
        grads_pred: List[Optional[torch.Tensor]],
        errors: List[Optional[torch.Tensor]],
        micro_batch_size: int,
        comm_timing: int,
        beta: float = 0.9,
        lowbit_group=None,
    ):
        self.device_mesh = device_mesh
        self.model = model

        self.pp_mesh = device_mesh["pp"]
        self.dp_mesh = device_mesh["dp"]
        self.dp_group = self.dp_mesh.get_group()

        self.pp_local_rank = self.device_mesh.get_local_rank("pp")
        self.micro_batch_size = int(micro_batch_size)
        self.comm_timing = int(comm_timing)

        self.beta = float(beta)

        # External state lists
        self.grads = grads
        self.grads_pred = grads_pred
        self.errors = errors

        self.micro_batch_counter = 0
        self.comm_handle: Optional[dist.Work] = None
        self.lowbit_group = lowbit_group

        # Stable param list
        self.params: List[torch.nn.Parameter] = [
            p for p in self.model.parameters()
        ]

        # Momentum buffers (per-parameter)
        self.momentum: List[Optional[torch.Tensor]] = [
            None for _ in self.params
        ]

        # Precompute flatten layout
        self.tensor_numels: List[int] = [int(p.numel()) for p in self.params]
        self.original_shapes: List[torch.Size] = [p.shape for p in self.params]
        total_numel = int(sum(self.tensor_numels))

        dev = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype
        self.flat_pred = torch.empty(total_numel, device=dev, dtype=dtype)

    def _trigger_condition(self) -> bool:
        if self.comm_timing == -1:
            trigger_batch = self.micro_batch_size / 2
            return self.micro_batch_counter == trigger_batch
        return self.micro_batch_counter == self.comm_timing

    @torch.no_grad()
    def _pack_predicted_(self):
        """Write predicted grads (momentum extrapolated) into flat buffer."""
        offset = 0
        for i, (p, g, e) in enumerate(
            zip(self.params, self.grads, self.errors)
        ):
            n = self.tensor_numels[i]
            if g is None:
                self.grads_pred[i] = None
                self.flat_pred[offset: offset + n].zero_()
                offset += n
                continue

            if self.momentum[i] is None:
                self.momentum[i] = torch.zeros_like(g)

            # m = beta*m + (1-beta)*g
            m = self.momentum[i]
            m.mul_(self.beta).add_(g, alpha=(1.0 - self.beta))

            if e is None:
                e = torch.zeros_like(g)
                self.errors[i] = e

            pred = m + e
            self.grads_pred[i] = pred
            self.flat_pred[offset: offset + n].copy_(pred.reshape(-1))
            offset += n

    @torch.no_grad()
    def _unpack_predicted_(self):
        offset = 0
        for i, p in enumerate(self.params):
            n = self.tensor_numels[i]
            if self.grads_pred[i] is None:
                p.grad = None
                offset += n
                continue
            if p.grad is None or p.grad.numel() != n:
                p.grad = torch.empty_like(p)
            p.grad.view(-1).copy_(self.flat_pred[offset: offset + n])
            offset += n

    def __call__(self, *args, **kwds):
        # Accumulate raw grads across microbatches
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            if self.grads[i] is None:
                self.grads[i] = torch.zeros_like(p.grad)
            self.grads[i].add_(p.grad)

        if self._trigger_condition():
            # No scaling here; prediction uses momentum/EMA.
            self._pack_predicted_()
            self.comm_handle = _all_reduce_flat_async(
                self.flat_pred,
                group=self.dp_group,
                lowbit_group=self.lowbit_group,
            )

            # Clear errors (same semantic as existing POLAR code)
            for e in self.errors:
                if e is not None:
                    e.zero_()

        self.micro_batch_counter += 1

        if self.micro_batch_counter == self.micro_batch_size:
            if self.comm_handle is not None:
                _wait_and_average_flat(
                    self.comm_handle,
                    self.flat_pred,
                    self.dp_group,
                )

            # Update error feedback: e := g - pred
            new_errors: List[Optional[torch.Tensor]] = []
            for g, p_pred in zip(self.grads, self.grads_pred):
                if g is None or p_pred is None:
                    new_errors.append(None)
                else:
                    new_errors.append(g - p_pred)
            self.errors = new_errors

            # Write reduced predicted grads to param.grad
            self._unpack_predicted_()

            # Reset for next step
            for i in range(len(self.grads)):
                self.grads[i] = None
            self.micro_batch_counter = 0
            self.comm_handle = None

class PolarGpipeErrorFeedbackOnlyHook:
    """Ablation: error-feedback only (no scaling).

    Prediction at trigger microbatch:
        pred = g + e

    End of step:
        e := g - pred
        param.grad := allreduced(pred)

    This keeps the same async DP all-reduce behavior as other POLAR hooks.
    """

    def __init__(
        self,
        device_mesh: dist.device_mesh.DeviceMesh,
        model: torch.nn.Module,
        grads: List[Optional[torch.Tensor]],
        grads_pred: List[Optional[torch.Tensor]],
        errors: List[Optional[torch.Tensor]],
        micro_batch_size: int,
        comm_timing: int,
        lowbit_group=None,
    ):
        self.device_mesh = device_mesh
        self.model = model

        self.dp_mesh = device_mesh["dp"]
        self.dp_group = self.dp_mesh.get_group()

        self.micro_batch_size = int(micro_batch_size)
        self.comm_timing = int(comm_timing)

        self.grads = grads
        self.grads_pred = grads_pred
        self.errors = errors

        self.micro_batch_counter = 0
        self.comm_handle: Optional[dist.Work] = None
        self.lowbit_group = lowbit_group

        self.params: List[torch.nn.Parameter] = [
            p for p in self.model.parameters()
        ]
        # Flatten layout + reusable flat buffer
        self.tensor_numels: List[int] = [int(p.numel()) for p in self.params]
        total_numel = int(sum(self.tensor_numels))
        dev = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype
        self.flat_pred = torch.empty(total_numel, device=dev, dtype=dtype)

    def _trigger_condition(self) -> bool:
        if self.comm_timing == -1:
            trigger_batch = self.micro_batch_size / 2
            return self.micro_batch_counter == trigger_batch
        return self.micro_batch_counter == self.comm_timing

    @torch.no_grad()
    def _pack_predicted_(self):
        offset = 0
        for i, (p, g, e) in enumerate(
            zip(self.params, self.grads, self.errors)
        ):
            n = self.tensor_numels[i]
            if g is None:
                self.grads_pred[i] = None
                self.flat_pred[offset: offset + n].zero_()
                offset += n
                continue
            if e is None:
                e = torch.zeros_like(g)
                self.errors[i] = e
            pred = g + e
            self.grads_pred[i] = pred
            self.flat_pred[offset: offset + n].copy_(pred.reshape(-1))
            offset += n

    @torch.no_grad()
    def _unpack_predicted_(self):
        offset = 0
        for i, p in enumerate(self.params):
            n = self.tensor_numels[i]
            if self.grads_pred[i] is None:
                p.grad = None
                offset += n
                continue
            if p.grad is None or p.grad.numel() != n:
                p.grad = torch.empty_like(p)
            p.grad.view(-1).copy_(self.flat_pred[offset: offset + n])
            offset += n

    def __call__(self, *args, **kwds):
        # Accumulate grads (same as GpipeHook)
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            if self.grads[i] is None:
                self.grads[i] = torch.zeros_like(p.grad)
            self.grads[i].add_(p.grad)

        if self._trigger_condition():
            self._pack_predicted_()
            self.comm_handle = _all_reduce_flat_async(
                self.flat_pred,
                group=self.dp_group,
                lowbit_group=self.lowbit_group,
            )

        self.micro_batch_counter += 1

        if self.micro_batch_counter == self.micro_batch_size:
            if self.comm_handle is not None:
                _wait_and_average_flat(
                    self.comm_handle,
                    self.flat_pred,
                    self.dp_group,
                )

            new_errors: List[Optional[torch.Tensor]] = []
            for g, p_pred in zip(self.grads, self.grads_pred):
                if g is None or p_pred is None:
                    new_errors.append(None)
                else:
                    new_errors.append(g - p_pred)
            self.errors = new_errors

            self._unpack_predicted_()

            for i in range(len(self.grads)):
                self.grads[i] = None
            self.micro_batch_counter = 0
            self.comm_handle = None


class PolarGpipeLowMemoryErrorFeedbackHook:
    """Bucketed error-feedback only, without persistent grads_pred buffers.

    Memory policy:
      - Do not allocate/maintain `grads` or `grads_pred`.
      - Reuse autograd's accumulated `param.grad` as the true gradient.
      - Never build a full-stage flat buffer. Instead, build deterministic
        buckets and all-reduce at most `max_inflight_buckets` bucket buffers.

    At the trigger microbatch, send `current_grad + error` bucket-by-bucket and
    offload reduced predictions to CPU. At step end, update residual
    `error = final_grad - sent` and write the reduced prediction back to
    `param.grad`. This restores POLAR's prediction/error-feedback semantics
    without keeping a full-stage prediction buffer on GPU.
    """

    def __init__(
        self,
        device_mesh: dist.device_mesh.DeviceMesh,
        model: torch.nn.Module,
        errors: List[Optional[torch.Tensor]],
        micro_batch_size: int,
        comm_timing: int,
        lowbit_group=None,
        bucket_numel: int = 4_000_000,
        max_inflight_buckets: int = 1,
    ):
        self.device_mesh = device_mesh
        self.model = model

        self.dp_mesh = device_mesh["dp"]
        self.dp_group = self.dp_mesh.get_group()

        self.micro_batch_size = int(micro_batch_size)
        self.comm_timing = int(comm_timing)
        self.errors = errors
        self.micro_batch_counter = 0
        self.comm_handle: Optional[dist.Work] = None
        self.lowbit_group = lowbit_group
        self.bucket_numel = max(1, int(bucket_numel))
        self.max_inflight_buckets = max(1, int(max_inflight_buckets))

        self.params: List[torch.nn.Parameter] = [
            p for p in self.model.parameters()
        ]
        first_local_param = self._local_tensor(self.params[0].data)
        self.tensor_numels: List[int] = [
            int(self._local_tensor(p.data).numel()) for p in self.params
        ]
        dev = first_local_param.device
        dtype = first_local_param.dtype
        self.device = dev
        self.dtype = dtype
        self.buckets = self._build_buckets()
        self._logged_bucket_schema = False
        self._logged_predict_trigger = False
        self.offloaded_pred_buckets: Optional[List[_OffloadedBucket]] = None

    @staticmethod
    def _local_tensor(tensor: torch.Tensor) -> torch.Tensor:
        if hasattr(tensor, "to_local"):
            return tensor.to_local()
        return tensor

    def _new_grad_like_param(self, param: torch.nn.Parameter) -> torch.Tensor:
        data = param.data
        if hasattr(data, "to_local"):
            return torch.empty_like(data)
        return torch.empty_like(param)

    def _local_grad(self, param: torch.nn.Parameter) -> Optional[torch.Tensor]:
        if param.grad is None:
            return None
        return self._local_tensor(param.grad)

    def _ensure_local_grad_(self, param: torch.nn.Parameter, param_idx: int) -> torch.Tensor:
        grad = self._local_grad(param)
        if grad is None or int(grad.numel()) != self.tensor_numels[param_idx]:
            param.grad = self._new_grad_like_param(param)
            grad = self._local_tensor(param.grad)
        return grad

    def _trigger_condition(self) -> bool:
        if self.comm_timing == -1:
            trigger_batch = self.micro_batch_size / 2
            return self.micro_batch_counter == trigger_batch
        return self.micro_batch_counter == self.comm_timing

    def _build_buckets(self) -> List[List[_BucketEntry]]:
        buckets: List[List[_BucketEntry]] = []
        current: List[_BucketEntry] = []
        current_numel = 0

        for param_idx, numel in enumerate(self.tensor_numels):
            start = 0
            remaining = numel
            while remaining > 0:
                room = self.bucket_numel - current_numel
                if room <= 0:
                    buckets.append(current)
                    current = []
                    current_numel = 0
                    room = self.bucket_numel

                take = min(remaining, room)
                current.append(_BucketEntry(param_idx, start, take))
                current_numel += take
                start += take
                remaining -= take

                if current_numel == self.bucket_numel:
                    buckets.append(current)
                    current = []
                    current_numel = 0

        if current:
            buckets.append(current)
        return buckets

    def _log_bucket_schema_once(self) -> None:
        if self._logged_bucket_schema:
            return
        self._logged_bucket_schema = True
        total_numel = int(sum(self.tensor_numels))
        logger.info(
            "[polar-hook] ef_lowmem bucketed DP all-reduce: "
            "total_numel=%s bucket_numel=%s buckets=%s max_inflight=%s "
            "dtype=%s device=%s lowbit=%s",
            total_numel,
            self.bucket_numel,
            len(self.buckets),
            self.max_inflight_buckets,
            self.dtype,
            self.device,
            self.lowbit_group is not None,
        )

    @torch.no_grad()
    def _pack_bucket_and_update_error_(self, entries: List[_BucketEntry]) -> torch.Tensor:
        bucket_numel = int(sum(entry.length for entry in entries))
        buffer = torch.empty(bucket_numel, device=self.device, dtype=self.dtype)

        offset = 0
        for entry in entries:
            p = self.params[entry.param_idx]
            err = self.errors[entry.param_idx]
            dst = buffer[offset: offset + entry.length]
            grad = self._local_grad(p)

            if grad is None:
                if err is None:
                    dst.zero_()
                else:
                    err_flat = err.reshape(-1)
                    dst.copy_(err_flat[entry.start: entry.start + entry.length])
                    err_flat[entry.start: entry.start + entry.length].neg_()
            else:
                grad_flat = grad.reshape(-1)
                dst.copy_(grad_flat[entry.start: entry.start + entry.length])
                if err is not None:
                    err_flat = err.reshape(-1)
                    dst.add_(err_flat[entry.start: entry.start + entry.length])
                    err_flat[entry.start: entry.start + entry.length].neg_()

            offset += entry.length

        return buffer

    @torch.no_grad()
    def _pack_pred_bucket_and_seed_error_(self, entries: List[_BucketEntry]) -> torch.Tensor:
        bucket_numel = int(sum(entry.length for entry in entries))
        buffer = torch.empty(bucket_numel, device=self.device, dtype=self.dtype)

        offset = 0
        for entry in entries:
            p = self.params[entry.param_idx]
            err = self.errors[entry.param_idx]
            dst = buffer[offset: offset + entry.length]
            grad = self._local_grad(p)

            if grad is None and err is None:
                dst.zero_()
                offset += entry.length
                continue

            if err is None:
                err = torch.empty_like(self._local_tensor(p.data))
                self.errors[entry.param_idx] = err

            err_flat = err.reshape(-1)
            if grad is None:
                dst.copy_(err_flat[entry.start: entry.start + entry.length])
            else:
                grad_flat = grad.reshape(-1)
                dst.copy_(grad_flat[entry.start: entry.start + entry.length])
                dst.add_(err_flat[entry.start: entry.start + entry.length])

            # Seed residual with -sent. Later microbatches continue to
            # accumulate into param.grad; at step end we add final local grad.
            err_flat[entry.start: entry.start + entry.length].copy_(dst)
            err_flat[entry.start: entry.start + entry.length].neg_()
            offset += entry.length

        return buffer

    @torch.no_grad()
    def _finish_bucket_(self, inflight: _InflightBucket) -> None:
        _wait_and_average_flat(inflight.work, inflight.buffer, self.dp_group)

        offset = 0
        for entry in inflight.entries:
            p = self.params[entry.param_idx]
            grad = self._ensure_local_grad_(p, entry.param_idx)
            grad_flat = grad.reshape(-1)
            grad_flat[entry.start: entry.start + entry.length].copy_(
                inflight.buffer[offset: offset + entry.length]
            )
            offset += entry.length

    @torch.no_grad()
    def _bucketed_all_reduce_(self) -> None:
        self._log_bucket_schema_once()
        inflight: List[_InflightBucket] = []

        for entries in self.buckets:
            buffer = self._pack_bucket_and_update_error_(entries)
            work = _all_reduce_flat_async(
                buffer,
                group=self.dp_group,
                lowbit_group=self.lowbit_group,
            )
            inflight.append(_InflightBucket(work=work, buffer=buffer, entries=entries))

            if len(inflight) >= self.max_inflight_buckets:
                self._finish_bucket_(inflight.pop(0))

        while inflight:
            self._finish_bucket_(inflight.pop(0))

    @torch.no_grad()
    def _bucketed_predict_to_cpu_(self) -> None:
        self._log_bucket_schema_once()
        if not self._logged_predict_trigger:
            logger.info(
                "[polar-hook] ef_lowmem POLAR trigger: bucketed predicted "
                "DP gradients via %s, comm_timing=%s, micro_batches=%s, "
                "buckets=%s, bucket_numel=%s",
                "bitscom" if self.lowbit_group is not None else "dense",
                self.comm_timing,
                self.micro_batch_size,
                len(self.buckets),
                self.bucket_numel,
            )
            self._logged_predict_trigger = True
        self.offloaded_pred_buckets = []

        for entries in self.buckets:
            buffer = self._pack_pred_bucket_and_seed_error_(entries)
            work = _all_reduce_flat_async(
                buffer,
                group=self.dp_group,
                lowbit_group=self.lowbit_group,
            )
            _wait_and_average_flat(work, buffer, self.dp_group)
            self.offloaded_pred_buckets.append(
                _OffloadedBucket(
                    buffer=buffer.detach().to("cpu"),
                    entries=entries,
                )
            )
            del buffer

    @torch.no_grad()
    def _finish_offloaded_prediction_(self) -> None:
        if self.offloaded_pred_buckets is None:
            return

        for offloaded in self.offloaded_pred_buckets:
            device_buffer = offloaded.buffer.to(self.device)
            offset = 0
            for entry in offloaded.entries:
                p = self.params[entry.param_idx]
                err = self.errors[entry.param_idx]
                grad = self._local_grad(p)
                if err is not None and grad is not None:
                    err_flat = err.reshape(-1)
                    grad_flat = grad.reshape(-1)
                    err_flat[entry.start: entry.start + entry.length].add_(
                        grad_flat[entry.start: entry.start + entry.length]
                    )

                out_grad = self._ensure_local_grad_(p, entry.param_idx)
                out_grad.reshape(-1)[entry.start: entry.start + entry.length].copy_(
                    device_buffer[offset: offset + entry.length]
                )
                offset += entry.length

        self.offloaded_pred_buckets = None

    def __call__(self, *args, **kwds):
        if self._trigger_condition() and self.offloaded_pred_buckets is None:
            _trace_evidence(
                "POLAR",
                "ef_lowmem_bucketed_predict",
                "meaning='ef_lowmem sends bucketed predicted gradients at "
                "the configured POLAR trigger and offloads reduced buckets "
                "to CPU until the pipeline step finishes' "
                f"bucket_numel={self.bucket_numel} "
                f"max_inflight={self.max_inflight_buckets}",
            )
            self._bucketed_predict_to_cpu_()

        self.micro_batch_counter += 1

        if self.micro_batch_counter == self.micro_batch_size:
            if self.offloaded_pred_buckets is None:
                self._bucketed_all_reduce_()
            else:
                self._finish_offloaded_prediction_()
            self.micro_batch_counter = 0
            self.comm_handle = None

    def reset_state(self) -> None:
        self.offloaded_pred_buckets = None
        self.micro_batch_counter = 0
        self.comm_handle = None


class PolarGpipeScalingOnlyHook:
    """Ablation: gradient scaling only (no error-feedback).

    Prediction at trigger microbatch:
        pred = g * scale

    End of step:
        (no error update)
        param.grad := allreduced(pred)
    """

    def __init__(
        self,
        device_mesh: dist.device_mesh.DeviceMesh,
        model: torch.nn.Module,
        grads: List[Optional[torch.Tensor]],
        grads_pred: List[Optional[torch.Tensor]],
        errors: List[Optional[torch.Tensor]],
        micro_batch_size: int,
        comm_timing: int,
        lowbit_group=None,
    ):
        self.device_mesh = device_mesh
        self.model = model

        self.dp_mesh = device_mesh["dp"]
        self.dp_group = self.dp_mesh.get_group()

        self.micro_batch_size = int(micro_batch_size)
        self.comm_timing = int(comm_timing)

        self.grads = grads
        self.grads_pred = grads_pred
        self.errors = errors

        self.micro_batch_counter = 0
        self.comm_handle: Optional[dist.Work] = None
        self.lowbit_group = lowbit_group

        self.params: List[torch.nn.Parameter] = [
            p for p in self.model.parameters()
        ]
        # Flatten layout + reusable flat buffer
        self.tensor_numels: List[int] = [int(p.numel()) for p in self.params]
        total_numel = int(sum(self.tensor_numels))
        dev = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype
        self.flat_pred = torch.empty(total_numel, device=dev, dtype=dtype)

    def _trigger_condition(self) -> bool:
        if self.comm_timing == -1:
            trigger_batch = self.micro_batch_size / 2
            return self.micro_batch_counter == trigger_batch
        return self.micro_batch_counter == self.comm_timing

    @torch.no_grad()
    def _pack_predicted_(self, scale: float):
        offset = 0
        for i, g in enumerate(self.grads):
            n = self.tensor_numels[i]
            if g is None:
                self.grads_pred[i] = None
                self.flat_pred[offset: offset + n].zero_()
                offset += n
                continue
            pred = g * scale
            self.grads_pred[i] = pred
            self.flat_pred[offset: offset + n].copy_(pred.reshape(-1))
            offset += n

    @torch.no_grad()
    def _unpack_predicted_(self):
        offset = 0
        for i, p in enumerate(self.params):
            n = self.tensor_numels[i]
            if self.grads_pred[i] is None:
                p.grad = None
                offset += n
                continue
            if p.grad is None or p.grad.numel() != n:
                p.grad = torch.empty_like(p)
            p.grad.view(-1).copy_(self.flat_pred[offset: offset + n])
            offset += n

    def __call__(self, *args, **kwds):
        # Accumulate grads (same as GpipeHook)
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            if self.grads[i] is None:
                self.grads[i] = torch.zeros_like(p.grad)
            self.grads[i].add_(p.grad)

        if self._trigger_condition():
            scale = self.micro_batch_size / (self.micro_batch_counter + 1)
            self._pack_predicted_(scale=scale)
            self.comm_handle = _all_reduce_flat_async(
                self.flat_pred,
                group=self.dp_group,
                lowbit_group=self.lowbit_group,
            )

        self.micro_batch_counter += 1

        if self.micro_batch_counter == self.micro_batch_size:
            if self.comm_handle is not None:
                _wait_and_average_flat(
                    self.comm_handle,
                    self.flat_pred,
                    self.dp_group,
                )

            # No error-feedback update.
            for i in range(len(self.errors)):
                self.errors[i] = None

            self._unpack_predicted_()

            for i in range(len(self.grads)):
                self.grads[i] = None
            self.micro_batch_counter = 0
            self.comm_handle = None


class PolarGpipeNothingHook:
    """Ablation: neither scaling nor error-feedback.

    Prediction at trigger microbatch:
        pred = g

    End of step:
        param.grad := allreduced(pred)

    This isolates the effect of early/async DP all-reduce timing only.
    """

    def __init__(
        self,
        device_mesh: dist.device_mesh.DeviceMesh,
        model: torch.nn.Module,
        grads: List[Optional[torch.Tensor]],
        grads_pred: List[Optional[torch.Tensor]],
        errors: List[Optional[torch.Tensor]],
        micro_batch_size: int,
        comm_timing: int,
        lowbit_group=None,
    ):
        self.device_mesh = device_mesh
        self.model = model

        self.dp_mesh = device_mesh["dp"]
        self.dp_group = self.dp_mesh.get_group()

        self.micro_batch_size = int(micro_batch_size)
        self.comm_timing = int(comm_timing)

        self.grads = grads
        self.grads_pred = grads_pred
        self.errors = errors

        self.micro_batch_counter = 0
        self.comm_handle: Optional[dist.Work] = None
        self.lowbit_group = lowbit_group

        self.params: List[torch.nn.Parameter] = [
            p for p in self.model.parameters()
        ]
        # Flatten layout + reusable flat buffer
        self.tensor_numels: List[int] = [int(p.numel()) for p in self.params]
        total_numel = int(sum(self.tensor_numels))
        dev = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype
        self.flat_pred = torch.empty(total_numel, device=dev, dtype=dtype)

    def _trigger_condition(self) -> bool:
        if self.comm_timing == -1:
            trigger_batch = self.micro_batch_size / 2
            return self.micro_batch_counter == trigger_batch
        return self.micro_batch_counter == self.comm_timing

    @torch.no_grad()
    def _pack_predicted_(self):
        offset = 0
        for i, g in enumerate(self.grads):
            n = self.tensor_numels[i]
            if g is None:
                self.grads_pred[i] = None
                self.flat_pred[offset: offset + n].zero_()
                offset += n
                continue
            self.grads_pred[i] = g
            self.flat_pred[offset: offset + n].copy_(g.reshape(-1))
            offset += n

    @torch.no_grad()
    def _unpack_predicted_(self):
        offset = 0
        for i, p in enumerate(self.params):
            n = self.tensor_numels[i]
            if self.grads_pred[i] is None:
                p.grad = None
                offset += n
                continue
            if p.grad is None or p.grad.numel() != n:
                p.grad = torch.empty_like(p)
            p.grad.view(-1).copy_(self.flat_pred[offset: offset + n])
            offset += n

    def __call__(self, *args, **kwds):
        # Accumulate grads (same as GpipeHook)
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            if self.grads[i] is None:
                self.grads[i] = torch.zeros_like(p.grad)
            self.grads[i].add_(p.grad)

        if self._trigger_condition():
            self._pack_predicted_()
            self.comm_handle = _all_reduce_flat_async(
                self.flat_pred,
                group=self.dp_group,
                lowbit_group=self.lowbit_group,
            )

        self.micro_batch_counter += 1

        if self.micro_batch_counter == self.micro_batch_size:
            if self.comm_handle is not None:
                _wait_and_average_flat(
                    self.comm_handle,
                    self.flat_pred,
                    self.dp_group,
                )

            # No error-feedback.
            for i in range(len(self.errors)):
                self.errors[i] = None

            self._unpack_predicted_()

            for i in range(len(self.grads)):
                self.grads[i] = None
            self.micro_batch_counter = 0
            self.comm_handle = None
