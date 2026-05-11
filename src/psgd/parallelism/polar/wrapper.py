import os
import datetime
import argparse
import logging
# from turtle import back
import numpy as np

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
# from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torch.distributed.device_mesh import (
    # init_device_mesh,
    DeviceMesh
)
from torch.distributed.pipelining import (
    SplitPoint,
    pipeline,
    ScheduleGPipe,
    PipelineStage,
    Schedule1F1B
)
from transformers import (
    # AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    # BertModel,
    # get_linear_schedule_with_warmup,
)
import transformers
# from datasets import load_dataset, load_from_disk
from tqdm import tqdm
# from psgd.utils.buffer import TensorBuffer
from typing import List  # Tuple

from .util import get_partitions_and_pipe
from .hooks import PolarCommHook, GpipeHook

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class NativePolarGradientCollector:
    """
    methods:
        register_hook(rank): Register hook to the first layer of the partition.
        synchronize(): Synchronize the gradients across all partitions.
    """

    def __init__(
        self, inter_group, local_group,
        partitions: List[List[torch.nn.Module]], num_chunks: int
    ):
        """__init__

        Args:
            inter_group: Distributed group for inter-node communication
            local_group: Distributed group for intra-node communication
            partition: Model partition to which this hook is attached. i.e.
                hook does not modify the model parameters either the gradients.
            send_buffer: Send buffer for gradients, size of this buffer should
                be equal to the partition's size, everything received from
                all_reduce operation will be an in-place operation.
        """
        self.inter_group = inter_group
        self.local_group = local_group
        self.partitions = partitions
        self.num_chunks = num_chunks
        self.comm_hook_handle = [None for _ in range(len(self.partitions))]
        self.comm_works = [None for _ in range(len(self.partitions))]
        print("Partitions structure:", [type(p) for p in self.partitions])
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
        self.errors = [
            [
                [None for p in layer.parameters()]
                for layer in partition
            ]
            for partition in self.partitions
        ]

        print("Partitions structure:", [type(p) for p in self.partitions])
        # 注入：分区统计与示例层类型
        try:
            sizes = [len(p) for p in self.partitions]
            head_types = [[
                    type(m).__name__ for m in p[:3]
                ] for p in self.partitions]
            total_params = [sum(p_.numel() for m in p for p_ in m.parameters())
                            for p in self.partitions]
            logger.info(f"[collector:init] partitions_sizes={sizes}, "
                        f"partitions_total_params={total_params}, "
                        f"head_layer_types={head_types}")
            for idx, p in enumerate(self.partitions):
                if len(p) == 0:
                    logger.warning(f"[collector:init] EMPTY partition after "
                                   f"reverse at idx={idx}. Check split points "
                                   f"and stage assignment.")
        except Exception as e:
            logger.exception(f"[collector:init] partition "
                             f"introspection failed: {e}")

    def register_hook(self):
        """Register hook to the first layer of the partition.

        hook is registered to the first layer of each partition.
        For example, if there are N partitions, and M layers in each partition,
        then the hook will be registered to the following layers:

        Args:
            rank (int): Rank of the partition.

        """
        comm_stream = torch.cuda.Stream()
        for rank in range(len(self.partitions)):
            self.comm_hook_handle[rank] = self.partitions[rank][
                0
            ].register_full_backward_hook(
                PolarCommHook(
                    partition_id=rank,
                    partitions=self.partitions,
                    num_chunks=self.num_chunks,
                    inter_group=self.inter_group,
                    local_group=self.local_group,
                    comm_works=self.comm_works,
                    grads=self.grads_accumulation,
                    grads_pred=self.grads_pred,
                    errors=self.errors,
                    stream=comm_stream,
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


class PolarParallel:
    def __init__(
        self,
        args: argparse.Namespace,
        device_mesh: DeviceMesh,
        micro_batches: int,     # you may call it the local step size
        stage_model,
        loss_fn,
        dataloader: DataLoader,
        comm_timing: int,
        eval_dataloader: DataLoader | None = None,
        eval_interval: int = 50,
        eval_max_batches: int = 20,
        optimizer="adamw",
        use_local_sgd: bool = False,
        local_sgd_steps: int = 1,
        baseline_mode: str = "manual",
    ):
        """__init__: initialize the PolarParallel

        Args:
            args (argparse.Namespace): args from user argparse
            device_mesh: DeviceMesh for DP and PP
            micro_batches (int): micro_batches for pipeline parallel
            stage_model: partitioned model for this stage
            loss_fn: loss function
            dataloader (DataLoader): training datasets
            eval_dataloader (DataLoader): optional evaluation dataloader
            comm_timing (int): communication timing parameter
            optimizer (str): optimizer type
            use_local_sgd (bool): enable Local-SGD mode
            local_sgd_steps (int): synchronize parameters every N steps
            baseline_mode (str): baseline training mode: "manual"
                (manual DP grad all-reduce) or "ddp" (wrap stage with DDP;
                may OOM with pipeline + large models).
        """
        os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
        self.args = args
        self.device_mesh = device_mesh
        self.dp_mesh = self.device_mesh["dp"]
        self.pp_mesh = self.device_mesh["pp"]
        self.micro_batches = micro_batches
        self.comm_timing = comm_timing

        self.baseline_mode = (baseline_mode or "manual").lower()
        if self.baseline_mode not in ("manual", "ddp"):
            raise ValueError(
                f"Unsupported baseline_mode={baseline_mode!r}. "
                f"Use 'manual' or 'ddp'."
            )
        if self.baseline_mode == "ddp":
            logger.warning(
                "baseline_mode='ddp' wraps each pipeline stage with DDP. "
                "This can increase memory usage and may OOM "
                "for large models/microbatches. "
                "Prefer baseline_mode='manual' for a robust baseline."
            )

        # Local-SGD settings
        self.use_local_sgd = use_local_sgd
        self.local_sgd_steps = local_sgd_steps
        self.local_step_counter = 0

        local_rank = int(os.environ["LOCAL_RANK"])
        self.device = torch.device(f"cuda:{local_rank}")
        self.datetime = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
        if self.use_local_sgd:
            log_dir = (
                f"./log/local_sgd"
                f"/{self.args.dataset_config}/{optimizer}"
                f"/{self.local_sgd_steps}"
                f"/{self.datetime}-{self.dp_mesh.size()}-{self.pp_mesh.size()}"
                f"/{self.dp_mesh.get_local_rank()}/tb_scalars"
            )
        else:
            log_dir = (
                f"./log/{self.args.using_polar}"
                f"/{self.args.dataset_config}/{optimizer}/{self.comm_timing}"
                f"/{self.datetime}-{self.dp_mesh.size()}-{self.pp_mesh.size()}"
                f"/{self.dp_mesh.get_local_rank()}/tb_scalars"
            )
        self.writer = SummaryWriter(log_dir=log_dir)
        self.tensorboard_trace_dir = log_dir.replace("tb_scalars", "tb_trace")

        stage_idx = self.pp_mesh.get_local_rank()
        self.stage_idx = stage_idx
        self.stage_model = stage_model
        self.stage_model.to_empty(device=self.device, recurse=True)
        self.stage_model.apply(
            lambda m: m.reset_parameters()
            if hasattr(m, 'reset_parameters') else None
        )

        self.stage = PipelineStage(
            self.stage_model,
            stage_index=stage_idx,
            num_stages=self.pp_mesh.size(),
            device=self.device,
            group=self.pp_mesh.get_group(),
        )

        # Only construct DDP wrapper when explicitly requested.
        self.ddp_model = None
        if self.baseline_mode == "ddp":
            from torch.nn.parallel import DistributedDataParallel as DDP
            self.ddp_model = DDP(
                self.stage.submod,
                process_group=self.dp_mesh.get_group(),
                gradient_as_bucket_view=True,
                broadcast_buffers=False,  # Pipeline does not require sync
            )

        # Use CLI lr if provided (scripts pass --lr). Fall back to a
        # reasonable optimizer-specific default.
        self.lr = float(
            getattr(
                self.args,
                "lr",
                5e-4 if optimizer == "adamw" else 1e-3,
            )
        )

        self.optimizer = None
        self.optimizer_name = optimizer
        if optimizer == "adamw":
            self.optimizer = torch.optim.AdamW(
                self.stage.submod.parameters(), lr=self.lr, foreach=False
            )
        elif optimizer == "sgd":
            self.optimizer = torch.optim.SGD(
                self.stage.submod.parameters(), lr=self.lr, foreach=False
            )

        # dp_rank = self.dp_mesh.get_local_rank()

        self.dataloader = dataloader
        self.eval_dataloader = eval_dataloader
        self.eval_interval = int(eval_interval) if eval_interval is not None else 0
        self.eval_max_batches = int(eval_max_batches) if eval_max_batches is not None else 0

        self.schedule = Schedule1F1B(
            self.stage,
            n_microbatches=micro_batches,
            loss_fn=loss_fn
        )

        logger.info(
            f"[PolarParallel:init] optimizer={optimizer} lr={self.lr} "
            f"baseline_mode={self.baseline_mode} use_local_sgd={self.use_local_sgd}"
        )

        self.errors = [None for param in self.stage.submod.parameters()]
        self.gradients = [param.grad for param in self.stage.submod.parameters()]
        self.grads_pred = [None for param in self.stage.submod.parameters()]

        print(f"Rank {dist.get_rank()}: Stage {self.stage_idx}, Model layers: {len(self.stage_model.model.layers)}")

    def _sync_parameters_local_sgd(self):
        """
        Synchronize model parameters across DP group (for Local-SGD).
        Average parameters across all DP replicas.
        """
        dp_group = self.dp_mesh.get_group()
        dp_size = self.dp_mesh.size()
        
        logger.info(
            f"[Rank {dist.get_rank()}] Local-SGD parameter sync at step {self.local_step_counter}"
        )
        
        with torch.no_grad():
            for param in self.stage.submod.parameters():
                # All-reduce parameters (SUM) then average
                dist.all_reduce(param.data, op=dist.ReduceOp.SUM, group=dp_group)
                param.data.div_(dp_size)

    def _allreduce_dp_grads_(self):
        """All-reduce grads across the DP group (SUM then average).

        Intended for baseline_mode='manual' (no DDP).
        """
        dp_group = self.dp_mesh.get_group()
        dp_size = self.dp_mesh.size()
        if dp_size == 1:
            return
        for p in self.stage.submod.parameters():
            if p.grad is None:
                continue
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM, group=dp_group)
            p.grad.div_(dp_size)

    @torch.no_grad()
    def _evaluate_val_loss_ppl(self):
        """Validation loss/perplexity (LM) on eval_dataloader.

        Only last stage can compute the loss (it has logits). We aggregate
        across DP by summing loss*ntokens and ntokens.
        """
        if self.eval_dataloader is None:
            return None, None

        self.stage.submod.eval()

        total_loss_times_tokens = torch.tensor(
            0.0, device=self.device, dtype=torch.float32
        )
        total_tokens = torch.tensor(0, device=self.device, dtype=torch.long)

        is_last = self.stage.is_last

        for bidx, batch in enumerate(self.eval_dataloader):
            if self.eval_max_batches and bidx >= self.eval_max_batches:
                break

            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device) if is_last else None

            if self.stage.is_first:
                self.schedule.step(input_ids, attention_mask=attention_mask)
            elif is_last:
                out = self.stage.submod(
                    input_ids,
                    attention_mask=attention_mask,
                )
                logits = out.logits

                # LM loss, ignore pad (0)
                shift_logits = logits[..., :-1, :]
                shift_labels = labels[..., 1:]
                valid = shift_labels.ne(0)
                n_tokens = valid.sum()
                if n_tokens.item() > 0:
                    import torch.nn.functional as F

                    loss = F.cross_entropy(
                        shift_logits.reshape(-1, shift_logits.size(-1)),
                        shift_labels.reshape(-1),
                        ignore_index=0,
                        reduction="mean",
                    )
                    total_loss_times_tokens += loss * n_tokens.float()
                    total_tokens += n_tokens
            else:
                self.schedule.step(attention_mask=attention_mask)

        if self.dp_mesh.size() > 1:
            dist.all_reduce(
                total_loss_times_tokens,
                op=dist.ReduceOp.SUM,
                group=self.dp_mesh.get_group(),
            )
            dist.all_reduce(
                total_tokens,
                op=dist.ReduceOp.SUM,
                group=self.dp_mesh.get_group(),
            )

        avg_loss = (total_loss_times_tokens / total_tokens.clamp_min(1)).item()
        ppl = float(torch.exp(torch.tensor(avg_loss)).item())

        self.stage.submod.train()
        return avg_loss, ppl

    def train(self):
        # Register hook only if not using Local-SGD (Polar gradient prediction)
        if not self.use_local_sgd:
            polar_hook = getattr(self.args, "polar_hook", "io")
            polar_beta = float(getattr(self.args, "polar_beta", 0.9))

            if polar_hook == "momentum":
                from .hooks import PolarGpipeMomentumExtrapHook

                self.stage.submod.register_full_backward_hook(
                    PolarGpipeMomentumExtrapHook(
                        device_mesh=self.device_mesh,
                        model=self.stage.submod,
                        grads=self.gradients,
                        grads_pred=self.grads_pred,
                        errors=self.errors,
                        micro_batch_size=self.micro_batches,
                        comm_timing=self.comm_timing,
                        beta=polar_beta,
                    )
                )
            elif polar_hook == "io":
                from .hooks import PolarGpipeIoOptimHook

                self.stage.submod.register_full_backward_hook(
                    PolarGpipeIoOptimHook(
                        device_mesh=self.device_mesh,
                        model=self.stage.submod,
                        grads=self.gradients,
                        grads_pred=self.grads_pred,
                        errors=self.errors,
                        micro_batch_size=self.micro_batches,
                        comm_timing=self.comm_timing,
                    )
                )
            elif polar_hook == "ef_only":
                from .hooks import PolarGpipeErrorFeedbackOnlyHook

                self.stage.submod.register_full_backward_hook(
                    PolarGpipeErrorFeedbackOnlyHook(
                        device_mesh=self.device_mesh,
                        model=self.stage.submod,
                        grads=self.gradients,
                        grads_pred=self.grads_pred,
                        errors=self.errors,
                        micro_batch_size=self.micro_batches,
                        comm_timing=self.comm_timing,
                    )
                )
            elif polar_hook == "scaling_only":
                from .hooks import PolarGpipeScalingOnlyHook

                self.stage.submod.register_full_backward_hook(
                    PolarGpipeScalingOnlyHook(
                        device_mesh=self.device_mesh,
                        model=self.stage.submod,
                        grads=self.gradients,
                        grads_pred=self.grads_pred,
                        errors=self.errors,
                        micro_batch_size=self.micro_batches,
                        comm_timing=self.comm_timing,
                    )
                )
            elif polar_hook == "none":
                from .hooks import PolarGpipeNothingHook

                self.stage.submod.register_full_backward_hook(
                    PolarGpipeNothingHook(
                        device_mesh=self.device_mesh,
                        model=self.stage.submod,
                        grads=self.gradients,
                        grads_pred=self.grads_pred,
                        errors=self.errors,
                        micro_batch_size=self.micro_batches,
                        comm_timing=self.comm_timing,
                    )
                )
            else:
                # Legacy scaling hook
                self.stage.submod.register_full_backward_hook(
                    GpipeHook(
                        device_mesh=self.device_mesh,
                        model=self.stage.submod,
                        grads=self.gradients,
                        grads_pred=self.grads_pred,
                        errors=self.errors,
                        micro_batch_size=self.micro_batches,
                        comm_timing=self.comm_timing,
                    )
                )

        global_step = 0
        if self.stage.is_last:
            pbar = tqdm(self.dataloader)
        else:
            pbar = self.dataloader

        max_steps = getattr(self.args, "max_steps", None)

        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            profile_memory=True,
            record_shapes=True,
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(self.tensorboard_trace_dir),
            with_stack=True,
            acc_events=True,
        ) as prof:
            for batch_idx, batch in enumerate(pbar):
                if max_steps is not None and batch_idx >= int(max_steps):
                    break
                input_ids = batch["input_ids"].to(self.device)
                labels = (
                    batch["labels"].to(self.device)
                    if self.stage.is_last
                    else None
                )
                attention_mask = batch["attention_mask"].to(self.device)

                if self.optimizer:
                    self.optimizer.zero_grad()

                if self.stage.is_first:
                    self.schedule.step(
                        input_ids, attention_mask=attention_mask
                    )
                elif self.stage.is_last:
                    losses = []
                    self.schedule.step(
                        target=labels, losses=losses,
                        attention_mask=attention_mask
                    )
                    loss = torch.stack(losses).mean()

                    pbar.set_postfix({"loss": f"{loss.item():.4f}"})
                    if global_step % 100 == 0:
                        print(f"Step {global_step}, Loss: {loss.item():.4f}")
                else:
                    self.schedule.step(attention_mask=attention_mask)

                if self.optimizer and global_step == 0:
                    try:
                        summary = torch.cuda.memory_summary(
                            device=self.device,
                            abbreviated=True,
                        )
                        print(
                            f"[mem] rank={dist.get_rank()} stage={self.stage_idx} "
                            "before optimizer.step()\n"
                            f"{summary}"
                        )
                    except Exception as exc:
                        print(
                            f"[mem] rank={dist.get_rank()} stage={self.stage_idx} "
                            f"before optimizer.step() failed: {exc}"
                        )

                self.optimizer.step()

                if self.optimizer and global_step == 0:
                    try:
                        summary = torch.cuda.memory_summary(
                            device=self.device,
                            abbreviated=True,
                        )
                        print(
                            f"[mem] rank={dist.get_rank()} stage={self.stage_idx} "
                            "after optimizer.step()\n"
                            f"{summary}"
                        )
                    except Exception as exc:
                        print(
                            f"[mem] rank={dist.get_rank()} stage={self.stage_idx} "
                            f"after optimizer.step() failed: {exc}"
                        )
                self.local_step_counter += 1
                global_step += 1

                # Local-SGD: sync parameters every N steps
                if self.use_local_sgd and (
                    self.local_step_counter % self.local_sgd_steps == 0
                ):
                    self._sync_parameters_local_sgd()

                # Optional eval
                if (
                    self.eval_dataloader is not None
                    and self.eval_interval > 0
                    and (global_step % self.eval_interval == 0)
                ):
                    val_loss, val_ppl = self._evaluate_val_loss_ppl()
                    if self.stage.is_last and val_loss is not None:
                        self.writer.add_scalar('Loss/val', val_loss, global_step)
                        self.writer.add_scalar('Perplexity/val', val_ppl, global_step)
                        print(
                            f"[val] step={global_step} "
                            f"loss={val_loss:.4f} ppl={val_ppl:.2f}"
                        )

                prof.step()

                if self.stage.is_last:
                    avg_train_loss = loss
                    self.writer.add_scalar(
                        'Loss/train', avg_train_loss, batch_idx
                    )

    def train_test(self):
        self.stage.submod.register_full_backward_hook(GpipeHook(
            device_mesh=self.device_mesh,
            model=self.stage.submod,
            grads=self.gradients,
            grads_pred=self.grads_pred,
            errors=self.errors,
            micro_batch_size=self.micro_batches,
        ))
        global_step = 0
        if self.stage.is_last:
            pbar = tqdm(self.dataloader)
        else:
            pbar = self.dataloader
            
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            profile_memory=True,
            record_shapes=True,
            schedule=torch.profiler.schedule(wait=1, warmup=1, 
                                             active=3, repeat=1),
            # on_trace_ready=torch.profiler.tensorboard_trace_handler(f"./log/{self.datetime}-{self.dp_mesh.size()}-{self.pp_mesh.size()}"),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                self.tensorboard_trace_dir
            ),
            with_stack=True,
            acc_events=True,
        ) as prof:
            for batch_idx, batch in enumerate(pbar):
                input_ids = batch["input_ids"].to(self.device)
                if self.stage.is_last:
                    labels = batch["labels"].to(self.device)
                else:
                    labels = None
                # attention_mask = batch["attention_mask"].to(self.device)
                # attention_mask currently unused in this legacy path
                _attention_mask = batch["attention_mask"].to(self.device)
                del _attention_mask

                if self.optimizer:
                    self.optimizer.zero_grad()

                if self.stage.is_first:
                    self.schedule.step(input_ids)
                elif self.stage.is_last:
                    losses = []
                    # target 传给 last stage 的 forward
                    self.schedule.step(target=labels, losses=losses)
                    loss = torch.stack(losses).mean()

                    pbar.set_postfix({"loss": f"{loss.item():.4f}"})
                    if global_step % 100 == 0:
                        print(f"Step {global_step}, Loss: {loss.item():.4f}")
                else:
                    self.schedule.step()

                self.optimizer.step()
                global_step += 1
                prof.step()

                if self.stage.is_last:
                    avg_train_loss = loss  # / len(self.train_dataloader)
                    self.writer.add_scalar(
                        'Loss/train',
                        avg_train_loss,
                        batch_idx
                    )

    def _train(self):
        """Baseline training (no gradient prediction).

        - baseline_mode='manual': bare pipeline + manual DP grad all-reduce.
        - baseline_mode='ddp': pipeline stage wrapped with DDP (may OOM).

        NOTE: Must use a schedule built from the same stage we intend to train.
        """
        # Select the module for this stage
        if self.baseline_mode == "ddp":
            if self.ddp_model is None:
                from torch.nn.parallel import DistributedDataParallel as DDP
                self.ddp_model = DDP(
                    self.stage.submod,
                    process_group=self.dp_mesh.get_group(),
                    gradient_as_bucket_view=True,
                    broadcast_buffers=False,
                )
            stage_mod = self.ddp_model
        else:
            stage_mod = self.stage.submod

        baseline_stage = PipelineStage(
            stage_mod,
            stage_index=self.stage_idx,
            num_stages=self.pp_mesh.size(),
            device=self.device,
            group=self.pp_mesh.get_group(),
        )

        # Optimizer over the actual module used
        if self.optimizer_name == "sgd":
            optimizer = torch.optim.SGD(stage_mod.parameters(), lr=self.lr)
        elif self.optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(stage_mod.parameters(), lr=self.lr)
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_name}")

        def loss_fn(output, target):
            """Memory-optimized LM loss (avoid contiguous huge temps)."""
            import torch.nn.functional as F
            logits = output[..., :-1, :]
            labels = target[..., 1:]
            return F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=0,
            )

        schedule = Schedule1F1B(
            baseline_stage,
            n_microbatches=self.micro_batches,
            loss_fn=loss_fn,
        )

        global_step = 0
        if baseline_stage.is_last:
            pbar = tqdm(self.dataloader)
        else:
            pbar = self.dataloader

        max_steps = getattr(self.args, "max_steps", None)

        # Baseline logging layout: 
        # ./log/baseline_{mode}/.../{dp_local_rank}/tb_{scalars,trace}
        baseline_root = (
            f"./log/baseline_{self.baseline_mode}"
            f"/{self.args.dataset_config}"
            f"/{self.optimizer_name}"
            f"/{self.comm_timing}"
            f"/{self.datetime}-{self.dp_mesh.size()}-{self.pp_mesh.size()}"
            f"/{self.dp_mesh.get_local_rank()}"
        )
        baseline_scalar_dir = f"{baseline_root}/tb_scalars"
        baseline_trace_dir = f"{baseline_root}/tb_trace"

        baseline_writer = None
        if baseline_stage.is_last:
            baseline_writer = SummaryWriter(log_dir=baseline_scalar_dir)

        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            profile_memory=True,
            record_shapes=True,
            schedule=torch.profiler.schedule(wait=1, warmup=1,
                                             active=3, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                baseline_trace_dir),
            with_stack=True,
            acc_events=True,
        ) as prof:
            for batch_idx, batch in enumerate(pbar):
                if max_steps is not None and batch_idx >= int(max_steps):
                    break
                input_ids = batch["input_ids"].to(self.device)
                labels = (
                    batch["labels"].to(self.device)
                    if baseline_stage.is_last
                    else None
                )
                attention_mask = batch["attention_mask"].to(self.device)

                optimizer.zero_grad(set_to_none=True)

                if baseline_stage.is_first:
                    schedule.step(input_ids, attention_mask=attention_mask)
                elif baseline_stage.is_last:
                    losses = []
                    schedule.step(
                        target=labels,
                        losses=losses,
                        attention_mask=attention_mask,
                    )
                    loss = torch.stack(losses).mean()
                    pbar.set_postfix({"loss": f"{loss.item():.4f}"})
                    if global_step % 100 == 0:
                        print(f"Step {global_step}, Loss: {loss.item():.4f}")
                else:
                    schedule.step(attention_mask=attention_mask)

                # In manual baseline, DP-sync gradients explicitly.
                if self.baseline_mode == "manual":
                    self._allreduce_dp_grads_()

                optimizer.step()
                global_step += 1

                if baseline_stage.is_last and baseline_writer is not None:
                    baseline_writer.add_scalar('Loss/train', loss, batch_idx)

                prof.step()

        # Ensure scalars are flushed to disk.
        if baseline_writer is not None:
            baseline_writer.flush()
            baseline_writer.close()


class PolarDataParallel:
    def __init__(
        self,
        args: argparse.Namespace,
        inter_group: torch.distributed.ProcessGroup,
        local_group: torch.distributed.ProcessGroup,
        model: torch.nn.Module = None,
        device: torch.device = None,
        tokenizer: transformers.PreTrainedTokenizer = None,
        train_dataloader: DataLoader = None,
        eval_dataloader: DataLoader = None,
    ):
        '''
        Args:
            args (argparse.Namespace):
                Command line arguments containing training configurations.
            inter_group (torch.distributed.ProcessGroup):
                Process group for inter-node communication.
            local_group (torch.distributed.ProcessGroup):
                Process group for intra-node communication.
            model (torch.nn.Module, optional): Predefined model.
                If None, a model will be created based on args.
            device (torch.device, optional): Device to run the model on.
                If None, it will be set based on availability of CUDA.
            tokenizer (transformers.PreTrainedTokenizer): Predefined tokenizer.
                If None, a tokenizer will be created based on args.
        '''
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        # 如果使用CUDA，还需设置CUDA随机种子
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        
        # 获取命令行参数
        self.args = args
        # 设置通信组及设备
        self.local_group = local_group
        self.inter_group = inter_group
        self.datetime = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.device = torch.device(
            f"cuda:{dist.get_rank(group=self.local_group)}"
        ) if torch.cuda.is_available() else torch.device("cpu")
        self.writer = SummaryWriter(log_dir=f"./log/{self.datetime}-{args.using_hook}-{args.local_steps}")
        
        self.tokenizer = tokenizer
        if args.pretrained:
            self.model = model or AutoModelForSequenceClassification.from_pretrained(
                args.model_path, torch_dtype=torch.float32, num_labels=args.num_labels
            )
            self.model.to(self.device)
            print(next(self.model.parameters()).device)
        else:
            config = AutoConfig.from_pretrained(
                args.model_path, torch_dtype=torch.float32, num_labels=args.num_labels
            )
            self.model = model or AutoModelForSequenceClassification.from_config(config)
            self.model.to(self.device)
            print(next(self.model.parameters()).device)

        print(next(self.model.parameters()).device)
        
        if hasattr(self.model, "model") and hasattr(
            self.model.model, "_attn_implementation_internal"
        ):
            logger.info("Forcing eager attention implementation for tracing.")
            self.model.model._attn_implementation_internal = "eager"
            
        self.model_partitions, self.pipe_model = get_partitions_and_pipe(
            model=self.model, tokenizer=tokenizer, device=self.device
        )
        # self.model_partition, self.pipe_model = split_model_by_export(
        #     model=self.model,
        #     split_spec=split_spec,
        #     tokenizer=tokenizer,
        #     device=self.device,
        # )
        print("Model partitions created:", [len(p) for p in self.model_partitions])
        # <<< refactored code for splitting model into partitions <<<

        # self.pipeline_model = self._create_pipeline_model(split_spec)
        print(f"Rank {dist.get_rank()}: Building stage for index {dist.get_rank(local_group)}")
        stage = self.pipe_model.build_stage(
            stage_index=dist.get_rank(local_group),
            device=self.device,
            group=self.local_group,
        )
        if stage is None:
            raise ValueError(f"Stage {dist.get_rank(local_group)} is None - check split configuration")
        
        self.pipeline_schedule = ScheduleGPipe(
            stage=stage,
            n_microbatches=self.args.micro_batches,
        )
        
        if train_dataloader is None or eval_dataloader is None:
            raise ValueError("train_dataloader and eval_dataloader must be provided.")
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-6, weight_decay=0.01, eps=1e-8, betas=(0.9, 0.999))
        total_steps = len(self.train_dataloader) * args.epochs
        
        from torch.optim.lr_scheduler import CosineAnnealingLR
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=50)
        if self.args.using_hook:
            self.grad_partitions_bucket = []
            self.gradient_collector = NativePolarGradientCollector(
                inter_group=self.inter_group,
                local_group=self.local_group,
                partitions=self.model_partitions,
                num_chunks=self.args.micro_batches,
            )
        if self.args.using_hook:
            self.gradient_collector.register_hook()
                
    def split_model_into_partitions(self, num_partitions: int): 
        """
        Usage: Split the model into `num_partitions` partitions based on its architecture.
        Args:
            num_partitions (int): Number of partitions to split the model into.
        Returns:
            List[List[torch.nn.Module]]: List of partitions, each partition is a list of layers.
        """
        if hasattr(self.model, 'bert'): # BERT-based model
            layer_prefix = "bert.encoder.layer."
            total_layers = len([
                n for n, _ in self.model.named_modules() if n.startswith(layer_prefix) and n.count(".") == layer_prefix.count(".") + 1
            ])
        elif hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"): # GPT-based model
            layer_prefix = "transformer.h."
            total_layers = len([
                n for n, _ in self.model.named_modules() if n.startswith(layer_prefix) and n.count(".") == layer_prefix.count(".") + 1
            ])
        elif hasattr(self.model, "model") and hasattr(self.model.model, "layers"): # LlaMa-based model
            layer_prefix = "model.layers."
            total_layers = len([
                n for n, _ in self.model.named_modules() if n.startswith(layer_prefix) and n.count(".") == layer_prefix.count(".") + 1
            ])
        else:
            raise NotImplementedError("Model architecture not supported for partitioning.")
        
        layers_per_stage = total_layers // num_partitions
        split_spec = {}
        for i in range(num_partitions):
            layer_idx = i * layers_per_stage
            split_spec[f"{layer_prefix}{layer_idx}"] = SplitPoint.BEGINNING
            
        self.model_partitions, _ = get_partitions_and_pipe(
            self.model, tokenizer, device=self.device
        )
        
    def _create_pipeline_model(self, split_spec):
        """
        Create a pipeline model using torch.distributed.pipeline.sync.Pipe.
        Args:
            split_spec (dict): Split specification for the pipeline.
        Returns:
            torch.distributed.pipeline.sync.Pipe: Pipeline model.
        """
        pp_rank = dist.get_rank(group=self.local_group)
        
        # for tracing the model, we need a dummy input
        example_batch = self.tokenizer(
            "This is a dummy input for tracing the model.",
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.args.max_length,
        )
        example_batch.pop("token_type_ids", None)  # We don't need to use token_type_ids while tracing
        example_args = (example_batch['input_ids'],)
        example_kwargs = {'attention_mask': example_batch['attention_mask']}

        print(f'Tracing the model on rank {pp_rank} with example input on device {self.device}...')

        # Create the Pipeline model
        pipeline_model = pipeline(
            self.model,
            mb_args=example_args,
            mb_kwargs=example_kwargs,
            split_spec=split_spec,
        )

        assert pipeline_model is not None, "Pipeline model creation failed."
        print(f'Pipeline model created successfully on rank {pp_rank}. Current process holds stages: {pipeline_model.split_gm}')
        return pipeline_model

    def train(self):
        if not self.args.using_hook:
            self._train()
            return

        pp_rank = dist.get_rank(group=self.local_group)
        last_pp_rank = dist.get_world_size(self.local_group) - 1

        current_local_step = 0  # 当前本地步数计数器
        print(f"LOCAL_STEPS: {self.args.local_steps}")
        correct_predictions = 0
        total_predictions = 0

        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            profile_memory=True,
            record_shapes=True,
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(f"./log/{self.datetime}-{self.args.using_hook}-{self.args.local_steps}"),
            with_stack=True,
            acc_events=True,
        ) as prof:
            for epoch in range(self.args.epochs):
                self.model.train()
                total_loss = 0
                progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}")
                for batch_idx, batch in enumerate(progress_bar):
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    
                    outputs = self.pipeline_schedule.step(
                        batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        # labels=batch['labels'],
                    )
                    
                    if pp_rank == last_pp_rank:
                        loss = outputs.loss
                        logits = outputs.logits
                        loss.backward()
                        
                        total_loss += loss.item()
                        _, predicted = torch.max(logits, dim=1)
                        correct_predictions += (predicted == batch["labels"]).sum().item()
                        total_predictions += batch["labels"].size(0)
                        current_accuracy = correct_predictions / total_predictions
                        avg_train_loss = total_loss / (batch_idx + 1)
                        progress_bar.set_postfix({"loss": loss.item()})
                        
                        current_local_step += 1     # Update local step counter

                        # 记录当前 step 到 profiler
                        if current_local_step % self.args.local_steps == 0:
                            prof.step()

                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                if pp_rank == last_pp_rank:
                    avg_train_loss = total_loss / len(self.train_dataloader)
                    self.writer.add_scalar('Loss/train', avg_train_loss, epoch)
                    self.writer.add_scalar('Learning Rate', self.scheduler.get_last_lr()[0], epoch)
                    self.writer.add_scalar('Accuracy/train', current_accuracy, epoch)
                    print(f"Epoch {epoch+1} Average Loss: {avg_train_loss:.4f}")
                    
                self.scheduler.step()
