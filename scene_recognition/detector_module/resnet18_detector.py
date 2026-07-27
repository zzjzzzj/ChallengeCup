"""ResNet18-FPN Faster R-CNN baseline for full-image object detection.

Unlike the crop classifier, this module receives an entire image and predicts
boxes, labels, and confidence scores. It therefore shares YOLO's input/output
contract and can be evaluated with mAP on the same held-out images.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import yaml
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.ops import MultiScaleRoIAlign
from torchvision.transforms.functional import pil_to_tensor


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNS = PROJECT_ROOT / "scene_recognition" / "detector_module" / "runs"
IOU_THRESHOLDS = tuple(round(value, 2) for value in np.arange(0.5, 0.96, 0.05))


def resolve_label_path(image_path: Path) -> Path:
    """Resolve both sibling-label and images/labels YOLO directory layouts."""

    sibling = image_path.with_suffix(".txt")
    if sibling.is_file():
        return sibling
    parts = list(image_path.parts)
    for index in range(len(parts) - 1, -1, -1):
        if parts[index] == "images":
            parts[index] = "labels"
            candidate = Path(*parts).with_suffix(".txt")
            if candidate.is_file():
                return candidate
            break
    return sibling


def parse_yolo_rows(label_path: Path, class_count: int) -> list[tuple[int, float, float, float, float]]:
    if not label_path.is_file():
        raise FileNotFoundError(f"标签文件不存在: {label_path}")

    rows: list[tuple[int, float, float, float, float]] = []
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"{label_path}:{line_number} 应包含5列，实际为{len(parts)}列")
        try:
            class_id = int(parts[0])
            x_center, y_center, width, height = (float(value) for value in parts[1:])
        except ValueError as exc:
            raise ValueError(f"{label_path}:{line_number} 包含非数值字段") from exc
        if not 0 <= class_id < class_count:
            raise ValueError(f"{label_path}:{line_number} 类别编号越界: {class_id}")
        if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 < width <= 1 and 0 < height <= 1):
            raise ValueError(f"{label_path}:{line_number} 包含非法归一化边界框")
        rows.append((class_id, x_center, y_center, width, height))
    return rows


def yolo_rows_to_target(
    rows: Iterable[tuple[int, float, float, float, float]], image_width: int, image_height: int
) -> dict[str, Tensor]:
    boxes: list[list[float]] = []
    labels: list[int] = []
    for class_id, x_center, y_center, width, height in rows:
        x1 = max(0.0, (x_center - width / 2) * image_width)
        y1 = max(0.0, (y_center - height / 2) * image_height)
        x2 = min(float(image_width), (x_center + width / 2) * image_width)
        y2 = min(float(image_height), (y_center + height / 2) * image_height)
        if x2 <= x1 or y2 <= y1:
            continue
        boxes.append([x1, y1, x2, y2])
        # torchvision detection reserves class 0 for background.
        labels.append(class_id + 1)
    return {
        "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        "labels": torch.tensor(labels, dtype=torch.int64),
    }


class YoloManifestDataset(Dataset):
    """Read full images listed by a YOLO/Ultralytics text manifest."""

    def __init__(self, manifest_path: Path, class_count: int) -> None:
        self.manifest_path = manifest_path.resolve()
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"图片清单不存在: {self.manifest_path}")
        self.class_count = class_count
        self.image_paths = [Path(line.strip()) for line in self.manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not self.image_paths:
            raise ValueError(f"图片清单为空: {self.manifest_path}")
        missing = next((path for path in self.image_paths if not path.is_file()), None)
        if missing is not None:
            raise FileNotFoundError(f"清单中的图片不存在: {missing}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[Tensor, dict[str, Tensor]]:
        image_path = self.image_paths[index]
        with Image.open(image_path) as image:
            rgb_image = image.convert("RGB")
            image_width, image_height = rgb_image.size
            tensor = pil_to_tensor(rgb_image).float().div_(255.0)
        label_path = resolve_label_path(image_path)
        target = yolo_rows_to_target(
            parse_yolo_rows(label_path, self.class_count), image_width, image_height
        )
        return tensor, target


def collate_detection_batch(batch: list[tuple[Tensor, dict[str, Tensor]]]) -> tuple[list[Tensor], list[dict[str, Tensor]]]:
    images, targets = zip(*batch)
    return list(images), list(targets)


def build_resnet18_detector(
    class_count: int,
    pretrained: bool,
    min_size: int = 640,
    max_size: int = 640,
) -> FasterRCNN:
    """Build a ResNet18-FPN Faster R-CNN that has the same I/O as YOLO."""

    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    backbone = resnet_fpn_backbone(
        backbone_name="resnet18",
        weights=weights,
        trainable_layers=5,
    )
    anchor_generator = AnchorGenerator(
        sizes=((8,), (16,), (32,), (64,), (128,)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 5,
    )
    roi_pooler = MultiScaleRoIAlign(
        featmap_names=["0", "1", "2", "3"], output_size=7, sampling_ratio=2
    )
    return FasterRCNN(
        backbone=backbone,
        num_classes=class_count + 1,
        min_size=min_size,
        max_size=max_size,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
        # Preserve low-confidence predictions for AP; precision/recall uses an explicit threshold.
        box_score_thresh=0.001,
        box_detections_per_img=200,
    )


def _box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((len(boxes1), len(boxes2)), dtype=torch.float32)
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    top_left = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    bottom_right = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    intersection = (bottom_right - top_left).clamp(min=0)
    intersection_area = intersection[..., 0] * intersection[..., 1]
    return intersection_area / (area1[:, None] + area2[None, :] - intersection_area).clamp(min=1e-9)


def _class_detections(
    predictions: list[dict[str, Tensor]],
    targets: list[dict[str, Tensor]],
    class_label: int,
    iou_threshold: float,
    confidence_threshold: float | None = None,
) -> tuple[list[tuple[float, bool]], int]:
    """Return score-sorted TP/FP flags and ground-truth support for one class."""

    records: list[tuple[float, int, Tensor]] = []
    ground_truth: dict[int, Tensor] = {}
    for image_index, (prediction, target) in enumerate(zip(predictions, targets, strict=True)):
        gt_mask = target["labels"] == class_label
        ground_truth[image_index] = target["boxes"][gt_mask]
        pred_mask = prediction["labels"] == class_label
        if confidence_threshold is not None:
            pred_mask &= prediction["scores"] >= confidence_threshold
        for box, score in zip(prediction["boxes"][pred_mask], prediction["scores"][pred_mask], strict=True):
            records.append((float(score), image_index, box))
    records.sort(key=lambda record: record[0], reverse=True)

    matched: dict[int, set[int]] = {image_index: set() for image_index in ground_truth}
    flags: list[tuple[float, bool]] = []
    for score, image_index, box in records:
        boxes = ground_truth[image_index]
        if not len(boxes):
            flags.append((score, False))
            continue
        ious = _box_iou(box.reshape(1, 4), boxes).flatten()
        best_iou, best_index = (float(ious.max()), int(ious.argmax()))
        is_true_positive = best_iou >= iou_threshold and best_index not in matched[image_index]
        if is_true_positive:
            matched[image_index].add(best_index)
        flags.append((score, is_true_positive))
    return flags, sum(len(boxes) for boxes in ground_truth.values())


def _average_precision(flags: list[tuple[float, bool]], ground_truth_count: int) -> float:
    if ground_truth_count == 0:
        return float("nan")
    if not flags:
        return 0.0
    true_positives = np.cumsum([is_tp for _score, is_tp in flags], dtype=float)
    false_positives = np.cumsum([not is_tp for _score, is_tp in flags], dtype=float)
    recall = true_positives / ground_truth_count
    precision = true_positives / np.maximum(true_positives + false_positives, 1e-12)
    # COCO-style 101-point interpolated AP. A perfect detection yields exactly 1.0.
    values = [float(precision[recall >= level].max()) if np.any(recall >= level) else 0.0 for level in np.linspace(0, 1, 101)]
    return float(np.mean(values))


def detection_metrics(
    predictions: list[dict[str, Tensor]],
    targets: list[dict[str, Tensor]],
    class_names: list[str],
    confidence_threshold: float = 0.25,
) -> dict:
    """Calculate P/R and COCO-style mAP from full-image detection outputs."""

    cpu_predictions = [
        {name: value.detach().cpu() for name, value in prediction.items()} for prediction in predictions
    ]
    cpu_targets = [{name: value.detach().cpu() for name, value in target.items()} for target in targets]
    per_class: dict[str, dict] = {}
    all_precision_true_positives = 0
    all_precision_false_positives = 0
    all_ground_truth = 0
    map75_values: list[float] = []

    for class_index, class_name in enumerate(class_names, start=1):
        ap_by_threshold: dict[float, float] = {}
        for threshold in IOU_THRESHOLDS:
            flags, support = _class_detections(cpu_predictions, cpu_targets, class_index, threshold)
            ap_by_threshold[threshold] = _average_precision(flags, support)
        map75_values.append(ap_by_threshold[0.75])
        operating_flags, support = _class_detections(
            cpu_predictions,
            cpu_targets,
            class_index,
            0.5,
            confidence_threshold=confidence_threshold,
        )
        true_positives = sum(is_tp for _score, is_tp in operating_flags)
        false_positives = len(operating_flags) - true_positives
        all_precision_true_positives += true_positives
        all_precision_false_positives += false_positives
        all_ground_truth += support
        per_class[class_name] = {
            "support": support,
            "precision": true_positives / (true_positives + false_positives) if (true_positives + false_positives) else 0.0,
            "recall": true_positives / support if support else 0.0,
            "map50": ap_by_threshold[0.5],
            "map75": ap_by_threshold[0.75],
            "map50_95": float(np.nanmean(list(ap_by_threshold.values()))),
        }

    map50_values = [row["map50"] for row in per_class.values() if not np.isnan(row["map50"])]
    map5095_values = [row["map50_95"] for row in per_class.values() if not np.isnan(row["map50_95"])]
    return {
        "confidence_threshold": confidence_threshold,
        "precision": all_precision_true_positives / (all_precision_true_positives + all_precision_false_positives)
        if (all_precision_true_positives + all_precision_false_positives)
        else 0.0,
        "recall": all_precision_true_positives / all_ground_truth if all_ground_truth else 0.0,
        "map50": float(np.mean(map50_values)) if map50_values else 0.0,
        "map50_95": float(np.mean(map5095_values)) if map5095_values else 0.0,
        "map75": float(np.nanmean(map75_values)) if map75_values else 0.0,
        "per_class": per_class,
    }


@torch.no_grad()
def evaluate_detector(
    model: FasterRCNN,
    loader: DataLoader,
    device: torch.device,
    class_names: list[str],
    confidence_threshold: float,
) -> dict:
    model.eval()
    predictions: list[dict[str, Tensor]] = []
    targets: list[dict[str, Tensor]] = []
    for images, batch_targets in loader:
        outputs = model([image.to(device) for image in images])
        predictions.extend(outputs)
        targets.extend(batch_targets)
    return detection_metrics(predictions, targets, class_names, confidence_threshold)


def _read_data_config(data_path: Path) -> tuple[dict, list[str]]:
    config = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    names = config["names"]
    class_names = [str(names[index] if index in names else names[str(index)]) for index in range(len(names))] if isinstance(names, dict) else [str(name) for name in names]
    return config, class_names


def _make_loader(dataset: Dataset, batch_size: int, shuffle: bool, workers: int, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_detection_batch,
        generator=generator,
    )


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


@dataclass(frozen=True)
class TrainingConfig:
    data: Path
    output: Path
    epochs: int
    patience: int
    batch_size: int
    workers: int
    device: str
    seed: int
    pretrained: bool
    learning_rate: float
    min_size: int
    max_size: int
    confidence_threshold: float


def train_detector(config: TrainingConfig) -> dict:
    _seed_everything(config.seed)
    data_config, class_names = _read_data_config(config.data)
    for split in ("train", "val", "test"):
        if split not in data_config:
            raise ValueError(f"数据配置缺少 {split} 划分: {config.data}")

    datasets = {
        split: YoloManifestDataset(Path(data_config[split]), len(class_names))
        for split in ("train", "val", "test")
    }
    loaders = {
        "train": _make_loader(datasets["train"], config.batch_size, True, config.workers, config.seed),
        "val": _make_loader(datasets["val"], config.batch_size, False, config.workers, config.seed),
        "test": _make_loader(datasets["test"], config.batch_size, False, config.workers, config.seed),
    }
    device = torch.device(config.device)
    model = build_resnet18_detector(
        class_count=len(class_names),
        pretrained=config.pretrained,
        min_size=config.min_size,
        max_size=config.max_size,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    config.output.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []
    best_key = (-1.0, -1.0)
    best_epoch = 0
    stale_epochs = 0
    started_at = time.perf_counter()

    for epoch in range(1, config.epochs + 1):
        model.train()
        losses: list[float] = []
        for images, targets in loaders["train"]:
            images = [image.to(device, non_blocking=True) for image in images]
            targets = [{name: value.to(device, non_blocking=True) for name, value in target.items()} for target in targets]
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                loss_dict = model(images, targets)
                loss = sum(loss_dict.values())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()

        val_metrics = evaluate_detector(
            model, loaders["val"], device, class_names, config.confidence_threshold
        )
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_map50": val_metrics["map50"],
            "val_map50_95": val_metrics["map50_95"],
        }
        history.append(row)
        key = (val_metrics["map50"], val_metrics["map50_95"])
        if key > best_key:
            best_key = key
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "class_names": class_names,
                    "configuration": config.__dict__,
                    "val_metrics": val_metrics,
                },
                config.output / "best.pt",
            )
        else:
            stale_epochs += 1
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if stale_epochs >= config.patience:
            print(f"验证集 mAP 连续 {config.patience} 轮未提升，提前停止。", flush=True)
            break

    torch.save({"epoch": history[-1]["epoch"], "model_state": model.state_dict()}, config.output / "last.pt")
    checkpoint = torch.load(config.output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    val_metrics = evaluate_detector(model, loaders["val"], device, class_names, config.confidence_threshold)
    test_metrics = evaluate_detector(model, loaders["test"], device, class_names, config.confidence_threshold)
    with (config.output / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    summary = {
        "model": "fasterrcnn_resnet18_fpn",
        "input": "完整图像",
        "output": "边界框、类别、置信度",
        "class_names": class_names,
        "configuration": {
            "data": str(config.data.resolve()),
            "pretrained": config.pretrained,
            "weight_init": "imagenet_pretrained" if config.pretrained else "random_scratch",
            "epochs_requested": config.epochs,
            "epochs_completed": len(history),
            "patience": config.patience,
            "batch_size": config.batch_size,
            "device": str(device),
            "seed": config.seed,
            "learning_rate": config.learning_rate,
            "min_size": config.min_size,
            "max_size": config.max_size,
        },
        "splits": {split: len(dataset) for split, dataset in datasets.items()},
        "best_epoch": best_epoch,
        "selection_rule": "验证集 mAP@0.5 为主，同分时 mAP@0.5:0.95 更高者优先",
        "validation": val_metrics,
        "test": test_metrics,
        "elapsed_minutes": round((time.perf_counter() - started_at) / 60, 2),
    }
    (config.output / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练 ResNet18-FPN Faster R-CNN 并输出检测 mAP")
    parser.add_argument("--data", type=Path, required=True, help="包含 train/val/test 清单的 YOLO 数据 YAML")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--min-size", type=int, default=640)
    parser.add_argument("--max-size", type=int, default=640)
    parser.add_argument("--confidence-threshold", type=float, default=0.25)
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_detector(
        TrainingConfig(
            data=args.data,
            output=args.output,
            epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            workers=args.workers,
            device=args.device,
            seed=args.seed,
            pretrained=not args.no_pretrained,
            learning_rate=args.learning_rate,
            min_size=args.min_size,
            max_size=args.max_size,
            confidence_threshold=args.confidence_threshold,
        )
    )


if __name__ == "__main__":
    main()
