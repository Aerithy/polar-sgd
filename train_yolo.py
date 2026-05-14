import argparse
import logging
import os
from typing import List, Tuple

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torchvision.datasets import CocoDetection
from torchvision.transforms import Compose, Normalize, ToTensor

from polar_trainer import NativePolarGradientCollector, process_group_setup
from utils.seed import set_seed

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def collate_fn(batch: List[Tuple[torch.Tensor, dict]]):
    images, targets = zip(*batch)
    return list(images), list(targets)


def build_coco_dataset(data_root: str, year: str, split: str):
    transform = Compose(
        [
            ToTensor(),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    ann_file = os.path.join(
        data_root, "annotations", f"instances_{split}{year}.json"
    )
    image_root = os.path.join(data_root, f"{split}{year}")
    return CocoDetection(root=image_root, annFile=ann_file, transform=transform)


def create_yolo_model(weights: str, device: torch.device):
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "ultralytics is required for YOLO training. Install it with `pip install ultralytics`."
        ) from exc

    model = YOLO(weights)
    if hasattr(model, "model"):
        model.model.to(device)
        model.model.train()
    else:
        model.to(device)
        model.train()
    return model


def get_model_parameters(model):
    if hasattr(model, "model"):
        return model.model.parameters()
    return model.parameters()


def get_torch_model(model):
    return model.model if hasattr(model, "model") else model


def compute_loss(model, images, targets):
    if hasattr(model, "model") and hasattr(model.model, "loss"):
        preds = model.model(images)
        loss = model.model.loss(preds, targets)
        if isinstance(loss, (tuple, list)):
            loss = loss[0]
        return loss

    output = model(images, targets)
    if isinstance(output, (tuple, list)):
        return output[0]
    if isinstance(output, dict) and "loss" in output:
        return output["loss"]
    if torch.is_tensor(output):
        return output
    raise RuntimeError("Unsupported YOLO output format; adjust compute_loss().")


def get_leaf_layers(module: torch.nn.Module):
    for child in module.modules():
        if list(child.children()):
            continue
        if any(p.requires_grad for p in child.parameters(recurse=False)):
            yield child


def split_model_into_partitions(module: torch.nn.Module, num_partitions: int):
    layers = list(get_leaf_layers(module))
    if not layers:
        raise RuntimeError("Unable to find trainable layers for partitioning.")

    if num_partitions <= 1:
        return [layers]

    layer_param_sizes = [
        sum(p.numel() for p in layer.parameters() if p.requires_grad) for layer in layers
    ]
    total_params = sum(layer_param_sizes)
    target_size = total_params / num_partitions

    partitions = []
    idx = 0
    while idx < len(layer_param_sizes):
        partition = []
        partition_size = 0
        while partition_size < target_size and idx < len(layer_param_sizes):
            partition_size += layer_param_sizes[idx]
            partition.append(layers[idx])
            idx += 1

        if idx == len(layer_param_sizes):
            partitions.append(partition)
            break

        if abs(partition_size - target_size) < abs(
            partition_size + layer_param_sizes[idx] - target_size
        ):
            partitions.append(partition)
        else:
            partition.append(layers[idx])
            partitions.append(partition)
            idx += 1

    return partitions


def sync_gradients(parameters, world_size: int):
    if world_size == 1:
        return
    for param in parameters:
        if param.grad is not None:
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            param.grad /= world_size


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="./data/coco")
    parser.add_argument("--year", type=str, default="2017")
    parser.add_argument("--weights", type=str, default="yolov8n.pt")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--using_hook", type=bool, default=False)
    parser.add_argument("--local_steps", type=int, default=2)
    parser.add_argument("--clip_norm", type=float, default=0.5)
    args = parser.parse_args()

    set_seed(args.seed)

    global_group, inter_group, local_group = process_group_setup()
    world_size = dist.get_world_size(global_group) if dist.is_initialized() else 1

    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    logger.info("Initializing YOLO model...")
    model = create_yolo_model(args.weights, device)
    parameters = list(get_model_parameters(model))
    optimizer = torch.optim.AdamW(parameters, lr=args.lr)

    gradient_collector = None
    if args.using_hook:
        if not torch.cuda.is_available():
            raise RuntimeError("using_hook requires CUDA.")
        partitions = split_model_into_partitions(
            get_torch_model(model), args.local_steps
        )
        gradient_collector = NativePolarGradientCollector(
            inter_group=inter_group, local_group=local_group, partitions=partitions
        )
        gradient_collector.register_hook()

    train_dataset = build_coco_dataset(args.data_root, args.year, "train")
    sampler = DistributedSampler(train_dataset) if world_size > 1 else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )

    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        epoch_loss = 0.0
        step = -1
        for step, (images, targets) in enumerate(train_loader):
            images = [img.to(device) for img in images]
            targets = [
                {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in t.items()}
                for t in targets
            ]

            optimizer.zero_grad()
            loss = compute_loss(model, images, targets)
            loss.backward()
            if not args.using_hook:
                sync_gradients(parameters, world_size)
            torch.nn.utils.clip_grad_norm_(parameters, max_norm=args.clip_norm)
            optimizer.step()

            epoch_loss += loss.item()
            if (step + 1) % args.log_interval == 0:
                logger.info(
                    "Epoch %s Step %s/%s - loss: %.4f",
                    epoch + 1,
                    step + 1,
                    len(train_loader),
                    loss.item(),
                )

        steps = step + 1
        if steps > 0:
            avg_loss = epoch_loss / steps
            logger.info("Epoch %s finished. avg loss: %.4f", epoch + 1, avg_loss)
        else:
            logger.warning("Epoch %s finished with no training steps.", epoch + 1)


if __name__ == "__main__":
    main()
