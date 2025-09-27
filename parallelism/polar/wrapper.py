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
from torch.distributed.pipelining import SplitPoint, pipeline, ScheduleGPipe
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
from utils.buffer import TensorBuffer
from typing import List, Tuple

from .util import get_partitions_and_pipe, split_model_by_export
from .hooks import PolarCommHook

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
        self, inter_group, local_group, partitions: List[List[torch.nn.Module]], num_chunks: int
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

    def register_hook(self):
        """Register hook to the first layer of the partition.

        hook is registered to the first layer of each partition.
        For example, if there are N partitions, and M layers in each partition,
        then the hook will be registered to the following layers:

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


class PolarDataParallel:
    def __init__(
        self,
        args: argparse.Namespace,
        inter_group: torch.distributed.ProcessGroup,
        local_group: torch.distributed.ProcessGroup,
        model: torch.nn.Module = None,
        # split_spec: dict = None,
        device: torch.device = None,
        tokenizer: transformers.PreTrainedTokenizer = None,
        train_dataloader: DataLoader = None,
        eval_dataloader: DataLoader = None,
    ):
        '''
        Args:
            args (argparse.Namespace): Command line arguments containing training configurations.
            inter_group (torch.distributed.ProcessGroup): Process group for inter-node communication.
            local_group (torch.distributed.ProcessGroup): Process group for intra-node communication.
            model (torch.nn.Module, optional): Predefined model. If None, a model will be created based on args. Defaults to None.
            device (torch.device, optional): Device to run the model on. If None, it will be set based on availability of CUDA. Defaults to None.
            tokenizer (transformers.PreTrainedTokenizer, optional): Predefined tokenizer. If None, a tokenizer will be created based on args. Defaults to None.
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
        stage = self.pipe_model.build_stage(
            stage_index=dist.get_rank(local_group),
            device=self.device,
            group=self.local_group,
        )
        
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
                inter_group=self.inter_group if not self.args.single_node else self.local_group,
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
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(f"./log/{self.datetime}-{self.args.using_hook}-{self.args.local_steps}"),
            with_stack=True,
        ) as prof:
            for epoch in range(self.args.epochs):
                self.model.train()
                total_loss = 0
                progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}")
                for batch_idx, batch in enumerate(progress_bar):
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    
                    outputs = self.pipeline_schedule(
                        batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        labels=batch['labels'],
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
    