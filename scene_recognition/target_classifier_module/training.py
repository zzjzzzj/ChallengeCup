from __future__ import annotations

import csv
import json
import random
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter, ImageOps
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from scene_recognition.target_classifier_module import CLASS_NAMES


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
VALID_AUGMENTATIONS = ("none", "flip", "rotate90", "invert", "open", "close")


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 12
    batch_size: int = 32
    image_size: int = 224
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    seed: int = 42
    pretrained: bool = True
    num_workers: int = 0
    augmentation: str = "none"
    device: str = "auto"
    freeze_backbone_epochs: int = 0


class SquarePad:
    """Pad an image to a square without changing the target aspect ratio."""

    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        side = max(width, height)
        left = (side - width) // 2
        top = (side - height) // 2
        right = side - width - left
        bottom = side - height - top
        return ImageOps.expand(image, border=(left, top, right, bottom), fill=0)


class Morphology:
    """Apply one grayscale-style morphology operation consistently to every channel."""

    def __init__(self, mode: str) -> None:
        if mode not in {"open", "close"}:
            raise ValueError(f"未知形态学操作: {mode}")
        self.mode = mode

    def __call__(self, image: Image.Image) -> Image.Image:
        if self.mode == "open":
            return image.filter(ImageFilter.MinFilter(3)).filter(ImageFilter.MaxFilter(3))
        return image.filter(ImageFilter.MaxFilter(3)).filter(ImageFilter.MinFilter(3))


def build_transforms(image_size: int, augmentation: str = "none"):
    if image_size <= 0:
        raise ValueError("image_size 必须为正数")
    if augmentation not in VALID_AUGMENTATIONS:
        raise ValueError(f"未知增广方式: {augmentation}")

    augmentation_steps = {
        "none": [],
        "flip": [transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()],
        "rotate90": [
            transforms.RandomChoice(
                [
                    transforms.Lambda(
                        lambda image, angle=angle: image.rotate(angle)
                    )
                    for angle in (0, 90, 180, 270)
                ]
            )
        ],
        "invert": [transforms.RandomInvert(p=1.0)],
        "open": [Morphology("open")],
        "close": [Morphology("close")],
    }[augmentation]
    common = [
        SquarePad(),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    return transforms.Compose([*augmentation_steps, *common]), transforms.Compose(common)


class TargetCropDataset(Dataset):
    """Read one split from the traceable target-crop manifest."""

    REQUIRED_FIELDS = {
        "crop_path",
        "source_image_path",
        "split",
        "sensor",
        "scene",
        "class_id",
        "class_name",
    }

    def __init__(self, manifest_path: Path, split: str, transform) -> None:
        with manifest_path.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            missing = self.REQUIRED_FIELDS - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"裁剪清单缺少字段: {', '.join(sorted(missing))}")
            self.rows = [row for row in reader if row["split"] == split]
        if not self.rows:
            raise ValueError(f"裁剪清单没有 {split} 样本: {manifest_path}")
        for row_number, row in enumerate(self.rows, start=1):
            try:
                class_id = int(row["class_id"])
            except ValueError as exc:
                raise ValueError(f"{split}第{row_number}条类别编号不是整数") from exc
            if not 0 <= class_id < len(CLASS_NAMES):
                raise ValueError(f"{split}第{row_number}条类别编号越界: {class_id}")
            expected_name = CLASS_NAMES[class_id]
            if row["class_name"] != expected_name:
                raise ValueError(
                    f"{split}第{row_number}条类别编号与名称不一致: "
                    f"{class_id}应为{expected_name}，实际为{row['class_name']}"
                )
            if not Path(row["crop_path"]).is_file():
                raise FileNotFoundError(f"裁剪图不存在: {row['crop_path']}")
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        crop_path = Path(row["crop_path"])
        with Image.open(crop_path) as opened:
            image = opened.convert("RGB")
        return (
            self.transform(image),
            int(row["class_id"]),
            row["sensor"],
            row["scene"],
            str(crop_path),
            row["source_image_path"],
        )


def compute_classification_metrics(
    true: list[int],
    predicted: list[int],
    sensors: list[str],
    scenes: list[str],
    class_names: list[str],
) -> dict:
    """Compute overall and context-sliced classification metrics."""

    lengths = {len(true), len(predicted), len(sensors), len(scenes)}
    if len(lengths) != 1 or not true:
        raise ValueError("真实标签、预测、模态和场景必须非空且长度一致")
    class_ids = list(range(len(class_names)))
    per_class = recall_score(
        true, predicted, labels=class_ids, average=None, zero_division=0
    )
    metrics = {
        "accuracy": float(accuracy_score(true, predicted)),
        "macro_f1": float(f1_score(true, predicted, average="macro", zero_division=0)),
        "macro_recall": float(
            recall_score(true, predicted, average="macro", zero_division=0)
        ),
        "per_class_recall": {
            name: float(value) for name, value in zip(class_names, per_class)
        },
    }

    def add_slice(prefix: str, values: list[str]) -> None:
        for value in sorted(set(values)):
            indices = [index for index, current in enumerate(values) if current == value]
            slice_true = [true[index] for index in indices]
            slice_predicted = [predicted[index] for index in indices]
            metrics[f"{prefix}{value}_count"] = len(indices)
            metrics[f"{prefix}{value}_accuracy"] = float(
                accuracy_score(slice_true, slice_predicted)
            )
            metrics[f"{prefix}{value}_macro_f1"] = float(
                f1_score(slice_true, slice_predicted, average="macro", zero_division=0)
            )

    add_slice("", sensors)
    add_slice("scene_", scenes)
    return metrics


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求了CUDA，但当前环境没有可用CUDA设备")
    if requested not in {"cpu", "cuda"}:
        raise ValueError("device 必须是 auto、cpu 或 cuda")
    return torch.device(requested)


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = trainable
    for parameter in model.fc.parameters():
        parameter.requires_grad = True


@torch.inference_mode()
def evaluate_model(model, loader, device, criterion=None):
    model.eval()
    losses: list[float] = []
    true: list[int] = []
    predicted: list[int] = []
    sensors: list[str] = []
    scenes: list[str] = []
    crop_paths: list[str] = []
    source_paths: list[str] = []
    confidences: list[float] = []
    for images, labels, batch_sensors, batch_scenes, batch_crops, batch_sources in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        if criterion is not None:
            losses.append(float(criterion(logits, labels).item()))
        probabilities = logits.softmax(dim=1)
        confidence, guesses = probabilities.max(dim=1)
        true.extend(labels.cpu().tolist())
        predicted.extend(guesses.cpu().tolist())
        confidences.extend(confidence.cpu().tolist())
        sensors.extend(list(batch_sensors))
        scenes.extend(list(batch_scenes))
        crop_paths.extend(list(batch_crops))
        source_paths.extend(list(batch_sources))

    metrics = compute_classification_metrics(
        true, predicted, sensors, scenes, CLASS_NAMES
    )
    metrics["loss"] = float(np.mean(losses)) if losses else None
    metrics["sample_count"] = len(true)
    return (
        metrics,
        true,
        predicted,
        sensors,
        scenes,
        crop_paths,
        source_paths,
        confidences,
    )


def train_target_classifier(
    manifest_path: Path,
    output_dir: Path,
    config: TrainingConfig | None = None,
) -> dict:
    """Train, select on validation Macro-F1, and evaluate a ResNet18 crop classifier."""

    config = config or TrainingConfig()
    if config.epochs < 1 or config.batch_size < 1:
        raise ValueError("epochs 和 batch_size 必须大于等于1")
    if not 0 <= config.freeze_backbone_epochs <= config.epochs:
        raise ValueError("freeze_backbone_epochs 必须位于0到epochs之间")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"运行目录不是空目录，请换用新目录: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(config.seed)
    train_transform, evaluation_transform = build_transforms(
        config.image_size, config.augmentation
    )
    train_dataset = TargetCropDataset(manifest_path, "train", train_transform)
    val_dataset = TargetCropDataset(manifest_path, "val", evaluation_transform)
    test_dataset = TargetCropDataset(manifest_path, "test", evaluation_transform)
    device = resolve_device(config.device)
    loader_kwargs = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    model = build_resnet18(len(CLASS_NAMES), pretrained=config.pretrained).to(device)
    set_backbone_trainable(model, config.freeze_backbone_epochs == 0)
    class_counts = Counter(int(row["class_id"]) for row in train_dataset.rows)
    missing_classes = [
        CLASS_NAMES[index]
        for index in range(len(CLASS_NAMES))
        if class_counts[index] == 0
    ]
    if missing_classes:
        raise ValueError(f"训练集缺少目标类别: {', '.join(missing_classes)}")
    class_weights = torch.tensor(
        [
            len(train_dataset) / (len(CLASS_NAMES) * class_counts[index])
            for index in range(len(CLASS_NAMES))
        ],
        dtype=torch.float32,
        device=device,
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    history: list[dict] = []
    # 与整图主线同样的问题：验证集接近饱和时，只按 Macro-F1 严格大于会锁定在早期轮次。
    # 主指标同分时改用验证集 loss 兜底，详见 docs/诊断报告-场景捷径与模型选择缺陷.md。
    best_val_macro_f1 = -1.0
    best_selection_key = (-1.0, float("-inf"))
    best_epoch = 0
    for epoch in range(1, config.epochs + 1):
        if config.freeze_backbone_epochs and epoch == config.freeze_backbone_epochs + 1:
            set_backbone_trainable(model, True)
        model.train()
        started = time.perf_counter()
        running_loss = 0.0
        seen = 0
        for images, labels, *_ in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.item()) * images.size(0)
            seen += images.size(0)
        scheduler.step()
        val_result = evaluate_model(model, val_loader, device, criterion)
        val_metrics = val_result[0]
        history_row = {
            "epoch": epoch,
            "train_loss": running_loss / max(1, seen),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "seconds": time.perf_counter() - started,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_macro_recall": val_metrics["macro_recall"],
        }
        history.append(history_row)
        print(json.dumps(history_row, ensure_ascii=False))
        selection_key = (val_metrics["macro_f1"], -(val_metrics["loss"] or 0.0))
        if selection_key > best_selection_key:
            best_selection_key = selection_key
            best_val_macro_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            torch.save(
                {
                    "model_name": "resnet18",
                    "state_dict": model.state_dict(),
                    "class_names": CLASS_NAMES,
                    "image_size": config.image_size,
                    "best_val_macro_f1": best_val_macro_f1,
                    "best_val_loss": val_metrics["loss"],
                    "selected_epoch": epoch,
                    "config": asdict(config),
                },
                output_dir / "best.pt",
            )

    checkpoint = torch.load(output_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    val_result = evaluate_model(model, val_loader, device, criterion)
    test_result = evaluate_model(model, test_loader, device, criterion)
    val_metrics = val_result[0]
    (
        test_metrics,
        true,
        predicted,
        sensors,
        scenes,
        crop_paths,
        source_paths,
        confidences,
    ) = test_result
    report = classification_report(
        true,
        predicted,
        labels=range(len(CLASS_NAMES)),
        target_names=CLASS_NAMES,
        zero_division=0,
        output_dict=True,
    )
    matrix = confusion_matrix(true, predicted, labels=range(len(CLASS_NAMES)))
    result = {
        "model": "resnet18",
        "device": str(device),
        "manifest": str(manifest_path.resolve()),
        "config": asdict(config),
        "best_val_macro_f1": best_val_macro_f1,
        "selected_epoch": best_epoch,
        "selection_rule": "验证集Macro-F1为主，同分时取验证集loss更低的轮次",
        "validation": val_metrics,
        "test": test_metrics,
        "classification_report": report,
        "scope_warning": (
            "本基线使用真实标注框裁剪目标，只评估已知位置下的目标分类能力，"
            "不能替代完整目标检测mAP。"
        ),
        "shortcut_warning": (
            "本数据集中类别与场景强绑定（small_aircraft只在air、warship只在sea），"
            "且soldier与tank的框尺寸几乎不重叠。仅用「场景+框尺寸」的浅决策树、"
            "不看任何像素即可达到97.67% Accuracy，因此本实验的Accuracy必须减去该平凡上限"
            "后才代表真实视觉识别增益。用 scene_recognition.target_classifier_module.diagnose_shortcut 复现。"
        ),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (output_dir / "history.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    with (output_dir / "confusion_matrix.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["actual/predicted", *CLASS_NAMES])
        for name, row in zip(CLASS_NAMES, matrix.tolist()):
            writer.writerow([name, *row])
    with (output_dir / "test_predictions.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        fieldnames = [
            "crop_path",
            "source_image_path",
            "sensor",
            "scene",
            "actual",
            "predicted",
            "confidence",
            "correct",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for crop, source, sensor, scene, actual, guess, confidence in zip(
            crop_paths,
            source_paths,
            sensors,
            scenes,
            true,
            predicted,
            confidences,
        ):
            writer.writerow(
                {
                    "crop_path": crop,
                    "source_image_path": source,
                    "sensor": sensor,
                    "scene": scene,
                    "actual": CLASS_NAMES[actual],
                    "predicted": CLASS_NAMES[guess],
                    "confidence": round(confidence, 6),
                    "correct": int(actual == guess),
                }
            )
    return result


def build_resnet18(class_count: int, pretrained: bool = True) -> nn.Module:
    """Build a ResNet18 classifier with an explicit target-class head."""

    if class_count < 2:
        raise ValueError("分类类别数必须至少为2")
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    try:
        model = models.resnet18(weights=weights)
    except Exception as exc:  # noqa: BLE001
        if not pretrained:
            raise
        raise RuntimeError(
            "无法加载ResNet18预训练权重；请先联网缓存权重，或明确使用 --no-pretrained"
        ) from exc
    model.fc = nn.Linear(model.fc.in_features, class_count)
    return model
