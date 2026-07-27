from __future__ import annotations

import csv
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset

from detector_module.boxes import parse_yolo_boxes
from target_classifier_module import CLASS_NAMES
from target_classifier_module.training import (
    build_resnet18,
    build_transforms,
    resolve_device,
    seed_everything,
)


SCENE_NAMES = ("air", "sea", "urban", "forest")


def parse_image_context(image_path: Path) -> tuple[str, str]:
    """Read sensor and scene from the repository's image naming convention."""

    parts = image_path.stem.split("_")
    if not parts or parts[0] not in {"ir", "sar"}:
        raise ValueError(f"无法从文件名识别模态: {image_path}")
    scenes = [part for part in parts if part in SCENE_NAMES]
    if len(scenes) != 1:
        raise ValueError(f"无法从文件名唯一识别场景: {image_path}")
    return parts[0], scenes[0]


def resolve_label_path(image_path: Path) -> Path:
    """定位一张图对应的 YOLO 标签文件。

    支持两种目录布局：
      1. 标签与图像同级（旧的 datasets_r1_base_train 就是这样），直接换后缀即可；
      2. ultralytics 约定的 images/ 与 labels/ 分离（E:\\yolo_augmented 就是这样），
         需把路径中最后一个 images 段替换成 labels。
    找不到时返回同级路径，交由 parse_yolo_boxes 抛出带原始路径的 FileNotFoundError。
    """

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


class WholeImageDataset(Dataset):
    """One full image and a four-dimensional target-presence vector per row."""

    def __init__(self, manifest_path: Path, transform) -> None:
        self.manifest_path = manifest_path
        self.transform = transform
        self.rows: list[dict] = []
        seen_paths: set[str] = set()
        for line_number, raw in enumerate(
            manifest_path.read_text(encoding="utf-8-sig").splitlines(), start=1
        ):
            text = raw.strip()
            if not text:
                continue
            image_path = Path(text)
            if not image_path.is_absolute():
                image_path = (manifest_path.parent / image_path).resolve()
            if not image_path.is_file():
                raise FileNotFoundError(f"整图不存在: {image_path}")
            normalized_path = str(image_path.resolve())
            if normalized_path in seen_paths:
                raise ValueError(f"整图清单包含重复图片: {image_path}")
            seen_paths.add(normalized_path)
            label_path = resolve_label_path(image_path)
            boxes = parse_yolo_boxes(label_path, len(CLASS_NAMES))
            target = np.zeros(len(CLASS_NAMES), dtype=np.float32)
            for box in boxes:
                target[box.class_id] = 1.0
            sensor, scene = parse_image_context(image_path)
            self.rows.append(
                {
                    "image_path": normalized_path,
                    "label_path": str(label_path),
                    "sensor": sensor,
                    "scene": scene,
                    "target": target,
                }
            )
        if not self.rows:
            raise ValueError(f"整图清单为空: {manifest_path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        with Image.open(row["image_path"]) as opened:
            image = opened.convert("RGB")
        return (
            self.transform(image),
            torch.from_numpy(row["target"].copy()),
            row["sensor"],
            row["scene"],
            row["image_path"],
        )


def _safe_average_precision(true: np.ndarray, probabilities: np.ndarray) -> float:
    if len(np.unique(true)) < 2:
        return float("nan")
    return float(average_precision_score(true, probabilities))


def _slice_metrics(true: np.ndarray, predicted: np.ndarray) -> dict:
    return {
        "sample_count": int(len(true)),
        "exact_match_accuracy": float(accuracy_score(true, predicted)),
        "micro_f1": float(f1_score(true, predicted, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(true, predicted, average="macro", zero_division=0)),
    }


def compute_multilabel_metrics(
    true: np.ndarray,
    probabilities: np.ndarray,
    thresholds: list[float],
    sensors: list[str],
    scenes: list[str],
    class_names: list[str],
) -> dict:
    """Compute presence metrics without pretending they are detection mAP."""

    true = np.asarray(true, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float32)
    if true.ndim != 2 or probabilities.shape != true.shape:
        raise ValueError("真实标签和概率必须是相同形状的二维数组")
    if probabilities.shape[1] != len(class_names) or len(thresholds) != len(class_names):
        raise ValueError("类别数、阈值和概率列数不一致")
    if len(true) == 0 or len(sensors) != len(true) or len(scenes) != len(true):
        raise ValueError("样本、模态和场景数量必须一致且非空")
    predicted = (probabilities >= np.asarray(thresholds)).astype(np.int64)
    per_class = {}
    ap_values = []
    for index, name in enumerate(class_names):
        ap = _safe_average_precision(true[:, index], probabilities[:, index])
        ap_values.append(ap)
        per_class[name] = {
            "precision": float(
                precision_score(true[:, index], predicted[:, index], zero_division=0)
            ),
            "recall": float(
                recall_score(true[:, index], predicted[:, index], zero_division=0)
            ),
            "f1": float(f1_score(true[:, index], predicted[:, index], zero_division=0)),
            "average_precision": ap,
            "positive_count": int(true[:, index].sum()),
        }
    metrics = {
        "sample_count": int(len(true)),
        "thresholds": {name: float(value) for name, value in zip(class_names, thresholds)},
        "exact_match_accuracy": float(np.all(true == predicted, axis=1).mean()),
        "hamming_accuracy": float(1.0 - np.abs(true - predicted).mean()),
        "micro_f1": float(f1_score(true, predicted, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(true, predicted, average="macro", zero_division=0)),
        "macro_precision": float(
            precision_score(true, predicted, average="macro", zero_division=0)
        ),
        "macro_recall": float(recall_score(true, predicted, average="macro", zero_division=0)),
        "mean_label_average_precision": float(np.nanmean(ap_values)),
        "per_class": per_class,
    }
    for prefix, values in (("sensor_", sensors), ("scene_", scenes)):
        for value in sorted(set(values)):
            indices = [index for index, current in enumerate(values) if current == value]
            metrics[f"{prefix}{value}"] = _slice_metrics(true[indices], predicted[indices])
    return metrics


def optimize_thresholds(
    true: np.ndarray, probabilities: np.ndarray, class_count: int
) -> list[float]:
    """Choose validation-only per-class thresholds by F1."""

    thresholds = []
    candidates = [round(float(value), 2) for value in np.linspace(0.10, 0.90, 17)]
    for index in range(class_count):
        best_threshold = 0.5
        best_score = -1.0
        for threshold in candidates:
            score = f1_score(
                true[:, index], probabilities[:, index] >= threshold, zero_division=0
            )
            if score > best_score or (
                score == best_score
                and abs(threshold - 0.5) < abs(best_threshold - 0.5)
            ):
                best_score = float(score)
                best_threshold = float(threshold)
        thresholds.append(best_threshold)
    return thresholds


@torch.inference_mode()
def evaluate_whole_image_model(model, loader, device, thresholds: list[float], criterion=None):
    model.eval()
    true: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    sensors: list[str] = []
    scenes: list[str] = []
    image_paths: list[str] = []
    losses: list[float] = []
    for images, targets, batch_sensors, batch_scenes, batch_paths in loader:
        device_targets = targets.to(device, non_blocking=True)
        logits = model(images.to(device, non_blocking=True))
        if criterion is not None:
            losses.append(float(criterion(logits, device_targets).item()) * images.size(0))
        true.append(targets.numpy())
        probabilities.append(logits.sigmoid().cpu().numpy())
        sensors.extend(list(batch_sensors))
        scenes.extend(list(batch_scenes))
        image_paths.extend(list(batch_paths))
    true_array = np.concatenate(true, axis=0).astype(np.int64)
    probability_array = np.concatenate(probabilities, axis=0)
    metrics = compute_multilabel_metrics(
        true_array,
        probability_array,
        thresholds,
        sensors,
        scenes,
        CLASS_NAMES,
    )
    metrics["loss"] = sum(losses) / len(true_array) if losses else None
    return metrics, true_array, probability_array, sensors, scenes, image_paths


@dataclass(frozen=True)
class WholeImageTrainingConfig:
    epochs: int = 12
    batch_size: int = 16
    image_size: int = 224
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    seed: int = 42
    pretrained: bool = True
    num_workers: int = 0
    augmentation: str = "none"
    device: str = "auto"


def train_whole_image_classifier(
    manifest_dir: Path,
    output_dir: Path,
    config: WholeImageTrainingConfig | None = None,
) -> dict:
    config = config or WholeImageTrainingConfig()
    if config.epochs < 1 or config.batch_size < 1:
        raise ValueError("epochs 和 batch_size 必须大于等于1")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"运行目录不是空目录，请换用新目录: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(config.seed)
    train_transform, evaluation_transform = build_transforms(
        config.image_size, config.augmentation
    )
    datasets = {
        split: WholeImageDataset(
            manifest_dir / f"{split}.txt",
            train_transform if split == "train" else evaluation_transform,
        )
        for split in ("train", "val", "test")
    }
    split_paths = {
        split: {row["image_path"] for row in dataset.rows}
        for split, dataset in datasets.items()
    }
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = split_paths[left] & split_paths[right]
        if overlap:
            example = sorted(overlap)[0]
            raise ValueError(f"{left}与{right}存在原图泄漏，例如: {example}")
    device = resolve_device(config.device)
    loader_kwargs = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": device.type == "cuda",
    }
    loaders = {
        split: DataLoader(dataset, shuffle=split == "train", **loader_kwargs)
        for split, dataset in datasets.items()
    }
    model = build_resnet18(len(CLASS_NAMES), pretrained=config.pretrained).to(device)
    train_targets = np.stack([row["target"] for row in datasets["train"].rows]).astype(np.float32)
    positive = train_targets.sum(axis=0)
    if np.any(positive == 0):
        raise ValueError("训练集缺少至少一个目标类别")
    pos_weight = torch.tensor(
        (len(train_targets) - positive) / positive, dtype=torch.float32, device=device
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history = []
    # 验证集在本数据上第1轮即可饱和到 Macro-F1 = 1.0。若只按 Macro-F1 严格大于来选择，
    # best.pt 会被永久锁定在第1轮的欠训练权重。因此主指标同分时改用验证集 loss 兜底，
    # 详见 docs/诊断报告-场景捷径与模型选择缺陷.md 第1节。
    best_selection_key = (-1.0, float("-inf"))
    best_epoch = 0
    for epoch in range(1, config.epochs + 1):
        model.train()
        started = time.perf_counter()
        running_loss = 0.0
        seen = 0
        for images, targets, *_ in loaders["train"]:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                loss = criterion(model(images), targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.item()) * images.size(0)
            seen += images.size(0)
        scheduler.step()
        val_probe = evaluate_whole_image_model(
            model, loaders["val"], device, [0.5] * len(CLASS_NAMES), criterion
        )
        val_thresholds = optimize_thresholds(
            val_probe[1], val_probe[2], len(CLASS_NAMES)
        )
        val_metrics = compute_multilabel_metrics(
            val_probe[1],
            val_probe[2],
            val_thresholds,
            val_probe[3],
            val_probe[4],
            CLASS_NAMES,
        )
        row = {
            "epoch": epoch,
            "train_loss": running_loss / max(1, seen),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "seconds": time.perf_counter() - started,
            "val_loss": val_probe[0]["loss"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_micro_f1": val_metrics["micro_f1"],
            "val_exact_match_accuracy": val_metrics["exact_match_accuracy"],
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))
        selection_key = (row["val_macro_f1"], -(row["val_loss"] or 0.0))
        if selection_key > best_selection_key:
            best_selection_key = selection_key
            best_epoch = epoch
            torch.save(
                {
                    "model_name": "resnet18",
                    "task_type": "whole_image_multilabel",
                    "state_dict": model.state_dict(),
                    "class_names": CLASS_NAMES,
                    "image_size": config.image_size,
                    "thresholds": val_thresholds,
                    "best_val_macro_f1": row["val_macro_f1"],
                    "best_val_loss": row["val_loss"],
                    "selected_epoch": epoch,
                    "config": asdict(config),
                },
                output_dir / "best.pt",
            )
    checkpoint = torch.load(output_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    thresholds = [float(value) for value in checkpoint["thresholds"]]
    val_result = evaluate_whole_image_model(model, loaders["val"], device, thresholds)
    test_result = evaluate_whole_image_model(model, loaders["test"], device, thresholds)
    result = {
        "model": "resnet18_whole_image_multilabel",
        "device": str(device),
        "manifest_dir": str(manifest_dir.resolve()),
        "config": asdict(config),
        "best_val_macro_f1": best_selection_key[0],
        "selected_epoch": best_epoch,
        "selection_rule": "验证集Macro-F1为主，同分时取验证集loss更低的轮次",
        "validation": val_result[0],
        "test": test_result[0],
        "scope_warning": (
            "本实验从整图判断四类目标是否出现，不输出目标框、数量或每个目标的位置；"
            "因此不能替代完整目标检测的Precision/Recall/mAP。"
        ),
        "shortcut_warning": (
            "本数据集中四维存在标签是场景的确定性函数（air/sea/urban/forest 各自只对应"
            "唯一存在向量），且像素消融显示抹掉全部目标后Exact Match几乎不降。"
            "因此该指标衡量的是场景识别而非目标识别，"
            "详见 docs/诊断报告-场景捷径与模型选择缺陷.md。"
        ),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (output_dir / "history.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    metrics, true, probabilities, sensors, scenes, image_paths = test_result
    predicted = (probabilities >= np.asarray(thresholds)).astype(np.int64)
    with (output_dir / "test_predictions.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "image_path",
                "sensor",
                "scene",
                "actual",
                "predicted",
                "probabilities",
                "exact_match",
            ],
        )
        writer.writeheader()
        for path, sensor, scene, actual, guess, probs in zip(
            image_paths, sensors, scenes, true, predicted, probabilities
        ):
            writer.writerow(
                {
                    "image_path": path,
                    "sensor": sensor,
                    "scene": scene,
                    "actual": ",".join(
                        CLASS_NAMES[index]
                        for index, value in enumerate(actual)
                        if value
                    ),
                    "predicted": ",".join(
                        CLASS_NAMES[index]
                        for index, value in enumerate(guess)
                        if value
                    ),
                    "probabilities": json.dumps(
                        dict(
                            zip(
                                CLASS_NAMES,
                                [round(float(probability), 6) for probability in probs],
                            )
                        ),
                        ensure_ascii=False,
                    ),
                    "exact_match": int(np.array_equal(actual, guess)),
                }
            )
    return result


def predict_whole_image(
    image_path: Path, checkpoint_path: Path, device_name: str = "auto"
) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    device = resolve_device(device_name)
    model = build_resnet18(len(CLASS_NAMES), pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    _, evaluation_transform = build_transforms(int(checkpoint.get("image_size", 224)), "none")
    with Image.open(image_path) as opened:
        tensor = evaluation_transform(opened.convert("RGB")).unsqueeze(0).to(device)
    with torch.inference_mode():
        probabilities = model(tensor).sigmoid()[0].cpu().numpy()
    thresholds = checkpoint.get("thresholds", [0.5] * len(CLASS_NAMES))
    predicted = probabilities >= np.asarray(thresholds)
    return {
        "image": str(image_path.resolve()),
        "predicted_targets": [
            name for name, present in zip(CLASS_NAMES, predicted) if present
        ],
        "probabilities": {
            name: round(float(value), 6)
            for name, value in zip(CLASS_NAMES, probabilities)
        },
        "thresholds": {
            name: float(value) for name, value in zip(CLASS_NAMES, thresholds)
        },
        "warning": "此输出只表示整图目标类别是否出现，不包含边界框和目标数量。",
    }
