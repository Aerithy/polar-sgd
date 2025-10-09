import torch
import torchvision
import torch.distributed as dist
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, DistributedSampler
# from ultralytics import YOLO

import models
from utils.logger import Logger
from utils.timer import Timer
from utils.metrics import AverageMeter

class COCO:
    """
    Instantiates a YOLOv8 model for object detection on the specified device.
    """
    def __init__(self, device: torch.device, timer: Timer, seed: int):
        """Initializes a COCO model.

        Args:
            device (torch.device): Device to run the model on.
            timer (Timer): A timer class to log the training process.
            architecture (str): model architecture to use (e.g., "YOLOv8").
            seed (int): Training seed.
        """
        self._device = device
        self._timer = timer
        self._seed = seed

        self._epoch = 0
        self._model = self._create_model()
        self._train_set, self._test_set = self._load_dataset()

        self.len_train_loader = None
        self.len_test_loader = None

        # YOLOv8 内部处理损失，无需手动设置 criterion
        self.parameters = [p for p in self._model.parameters()]

    def _load_dataset(self, data_root="./data/coco", year="2017"):
        """Load COCO dataset with annotations.

        Args:
            data_root (str): Path to COCO dataset root directory.
            year (str): Dataset year (e.g., "2017").

        Returns:
            tuple: (train_dataset, val_dataset)
        """
        from torchvision.datasets import CocoDetection
        from torchvision.transforms import Compose, ToTensor, Normalize

        train_transform = Compose([
            ToTensor(),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

        train_ann_file = f"{data_root}/annotations/instances_train{year}.json"
        val_ann_file = f"{data_root}/annotations/instances_val{year}.json"

        train_set = CocoDetection(
            root=f"{data_root}/train{year}",
            annFile=train_ann_file,
            transform=train_transform
        )

        val_set = CocoDetection(
            root=f"{data_root}/val{year}",
            annFile=val_ann_file,
            transform=train_transform
        )

        return train_set, val_set

    def _create_model(self):
        """Initialize YOLOv8 model with 80 classes (COCO categories)."""
        torch.random.manual_seed(self._seed)
        model = YOLO("yolov8n.pt")
        model.to(self._device)
        model.train()
        return model

    def train_dataloader(self, batch_size=32):
        """Distributed training dataloader for COCO."""
        train_sampler = DistributedSampler(self._train_set)
        train_sampler.set_epoch(self._epoch)

        train_loader = DataLoader(
            dataset=self._train_set,
            batch_size=batch_size,
            sampler=train_sampler,
            pin_memory=True,
            drop_last=True,
            num_workers=dist.get_world_size(),
            collate_fn=self._collate_fn  # 需自定义collate函数处理检测数据
        )

        self.len_train_loader = len(train_loader)

        for images, targets in train_loader:
            images = list(img.to(self._device) for img in images)
            targets = [{k: v.to(self._device) for k, v in t.items()} for t in targets]
            yield images, targets

        self._epoch += 1

    def test_dataloader(self, batch_size=32):
        """Distributed validation dataloader for COCO."""
        test_sampler = DistributedSampler(self._test_set)

        test_loader = DataLoader(
            dataset=self._test_set,
            batch_size=batch_size,
            sampler=test_sampler,
            pin_memory=True,
            drop_last=True,
            num_workers=dist.get_world_size(),
            collate_fn=self._collate_fn
        )

        self.len_test_loader = len(test_loader)

        for images, targets in test_loader:
            images = list(img.to(self._device) for img in images)
            targets = [{k: v.to(self._device) for k, v in t.items()} for t in targets]
            yield images, targets

    def _collate_fn(self, batch):
        """Custom collate function for detection data."""
        images, targets = zip(*batch)
        return list(images), list(targets)

    def batch_loss(self, batch):
        """Compute loss and metrics for a batch (inference mode)."""
        with torch.no_grad():
            images, targets = batch

            with self._timer("batch.forward", float(self._epoch)):
                # YOLOv8 的模型可能直接返回损失和预测结果
                loss, predictions = self._model(images, targets)

            with self._timer("batch.evaluate", float(self._epoch)):
                metrics = self.evaluate_predictions(predictions, targets)

        return loss.item(), metrics

    def batch_loss_with_gradients(self, batch):
        """Compute loss and gradients for a batch (training mode)."""
        self._model.zero_grad()
        images, targets = batch

        with self._timer("batch.forward", float(self._epoch)):
            loss, predictions = self._model(images, targets)

        with self._timer("batch.backward", float(self._epoch)):
            loss.backward()

        with self._timer("batch.evaluate", float(self._epoch)):
            metrics = self.evaluate_predictions(predictions, targets)

        grad_vec = [p.grad for p in self._model.parameters() if p.grad is not None]

        return loss.detach(), grad_vec, metrics

    def evaluate_predictions(self, predictions, targets):
        """Compute evaluation metrics for detection (e.g., mAP)."""
        # 使用 COCO API 或 YOLOv8 内置评估
        # 示例：假设 predictions 是模型输出的边界框和类别
        # 这里需要根据具体模型输出格式调整
        from ultralytics import YOLO
        results = self._model.val()  # 调用 YOLOv8 的验证函数
        return {
            "loss": results.loss,
            "mAP": results.metrics["map"],
            "mAP50": results.metrics["map50"]
        }

    def state_dict(self):
        return self._model.state_dict()

    def test(self, batch_size=32):
        """Evaluate model on validation set."""
        self._model.eval()
        test_loader = self.test_dataloader(batch_size=batch_size)

        # 使用 COCO API 计算指标
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        coco_gt = COCO(self._test_set.annFile)
        predictions = []

        for images, targets in test_loader:
            with torch.no_grad():
                outputs = self._model(images)
                # 将预测结果转换为 COCO 格式
                for img_idx, (img, target) in enumerate(zip(images, targets)):
                    boxes = outputs[img_idx]["boxes"].cpu().numpy()
                    scores = outputs[img_idx]["scores"].cpu().numpy()
                    classes = outputs[img_idx]["labels"].cpu().numpy()

                    for box, score, cls in zip(boxes, scores, classes):
                        pred = {
                            "image_id": target["image_id"].item(),
                            "category_id": int(cls),
                            "bbox": [box[0], box[1], box[2]-box[0], box[3]-box[1]],
                            "score": float(score)
                        }
                        predictions.append(pred)

        coco_dt = coco_gt.loadRes(predictions)
        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        return coco_eval.stats  # 返回 mAP 等指标

class CIFAR:
    """
    Instantiates a deep model of the specified architecture on the specified device.
    """
    def __init__(self, device: torch.device, timer: Timer, architecture: str, seed: int):
        """Initializes a CIFAR model.

        Args:
            device (torch.device): Device to run the model on.
            timer (Timer): A timer class to log the training process.
            architecture (str): model architecture to use.
            seed (int): Training seed.
        """
        self._device = device
        self._timer = timer
        self._architecture = architecture
        self._seed = seed

        self._epoch = 0
        self._model = self._create_model()
        self._train_set, self._test_set = self._load_dataset()

        self.len_train_loader = None
        self.len_aux_train_loader = None
        self.len_test_loader = None

        self._criterion = torch.nn.CrossEntropyLoss().to(self._device)
        self.parameters = [parameter for parameter in self._model.parameters()]

    def _load_dataset(self, data_path="./data"):
        """
        Args:
            data_path (str): The path to dataset. Defaults to "./data".

        Returns:
            _type_: ((torch.utils.data.Dataset, torch.utils.data.Dataset))
        """
        mean = (0.4914, 0.4822, 0.4465) # mean and std_dev of CIFAR10
        std_dev = (0.247, 0.243, 0.261)

        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std_dev),
            ]
        )

        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean, std_dev),
            ]
        )

        print(f'loading dataset from {data_path}')
        train_set = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform_train)
        test_set = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform_test)

        return train_set, test_set

    def _create_model(self):
        """Creates a model of the specified architecture.

        Returns:
            models._architecture: _description_
        """
        torch.random.manual_seed(self._seed)
        model = getattr(models, self._architecture)()
        print(f'moving model to {self._device}')
        model.to(self._device)
        model.train()

        return model

    def train_dataloader(self, batch_size=32):
        train_sampler = DistributedSampler(dataset=self._train_set)
        train_sampler.set_epoch(self._epoch)

        train_loader = DataLoader(
            dataset=self._train_set,
            batch_size=batch_size,
            sampler=train_sampler,
            pin_memory=True,
            drop_last=True,
            num_workers=dist.get_world_size(),
        )

        self.len_train_loader = len(train_loader)

        for imgs, labels in train_loader:
            imgs = imgs.to(self._device)
            labels = labels.to(self._device)

            yield imgs, labels

        self._epoch += 1

    def test_dataloader(self, batch_size=32):
        test_sampler = DistributedSampler(dataset=self._test_set)

        test_loader = DataLoader(
            dataset=self._test_set,
            batch_size=batch_size,
            sampler=test_sampler,
            pin_memory=True,
            drop_last=True,
            num_workers=dist.get_world_size(),
        )

        self.len_test_loader = len(test_loader)

        for imgs, labels in test_loader:
            imgs = imgs.to(self._device)
            labels = labels.to(self._device)

            yield imgs, labels

    def batch_loss(self, batch):
        with torch.no_grad():
            imgs, labels = batch

            with self._timer("batch.forward", float(self._epoch)):
                prediction = self._model(imgs)
                loss = self._criterion(prediction, labels)

            with self._timer("batch.evaluate", float(self._epoch)):
                metrics = self.evaluate_predictions(prediction, labels)

        return loss.item(), metrics

    def batch_loss_with_gradients(self, batch):
        self._model.zero_grad()
        imgs, labels = batch

        with self._timer("batch.forward", float(self._epoch)):
            prediction = self._model(imgs)
            loss = self._criterion(prediction, labels)

        with self._timer("batch.backward", float(self._epoch)):
            loss.backward()

        with self._timer("batch.evaluate", float(self._epoch)):
            metrics = self.evaluate_predictions(prediction, labels)

        grad_vec = [parameter.grad for parameter in self._model.parameters()]

        return loss.detach(), grad_vec, metrics

    def evaluate_predictions(self, pred_labels, true_labels):
        def accuracy(output, target, topk=(1,)):
            maxk = max(topk)
            batch_size = true_labels.size()[0]

            _, pred_topk = output.topk(maxk, 1, True, True)
            pred_topk = pred_topk.t()
            correct = pred_topk.eq(target.view(1, -1).expand_as(pred_topk))

            res = []
            for k in topk:
                correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(1 / batch_size))

            return res

        with torch.no_grad():
            cross_entropy_loss = self._criterion(pred_labels, true_labels)
            top1_accuracy, top5_accuracy = accuracy(pred_labels, true_labels, topk=(1, 5))

        return {
            "cross_entropy_loss": cross_entropy_loss.item(),
            "top1_accuracy": top1_accuracy.item(),
            "top5_accuracy": top5_accuracy.item(),
        }

    def state_dict(self):
        return self._model.state_dict()

    def test(self, batch_size=256):
        test_loader = self.test_dataloader(batch_size=batch_size)

        mean_metrics = AverageMeter(self._device)
        test_model = self._model
        test_model.eval()

        for i, batch in enumerate(test_loader):
            with torch.no_grad():
                imgs, labels = batch
                prediction = test_model(imgs)
                metrics = self.evaluate_predictions(prediction, labels)

            mean_metrics.add(metrics)

        mean_metrics.reduce()
        test_model.train()

        return mean_metrics
