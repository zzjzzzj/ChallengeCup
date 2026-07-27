"""Turn YOLO detections into whole-image presence vectors for a fair ResNet18 comparison.

三条基线原本口径不同：ResNet18 整图输出四维"目标是否存在"，YOLO 输出框和 mAP，
两者不能直接并排。本模块把 YOLO 的框按类别聚合成同样的四维存在向量
（该类取全图最高置信度作为该类分数），再复用 target_classifier_module.whole_image
里 **同一份** optimize_thresholds / compute_multilabel_metrics，
于是 ResNet18 与 YOLOv8 就落在同一批图、同一套 Exact Match / Macro-F1 口径上。

阈值只在 val 上搜索，test 不参与任何阈值或模型选择。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np

from detector_module.boxes import parse_yolo_boxes
from target_classifier_module import CLASS_NAMES
from target_classifier_module.whole_image import (
    compute_multilabel_metrics,
    optimize_thresholds,
    parse_image_context,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST_DIR = (
    PROJECT_ROOT / "detector_module" / "artifacts" / "detection_dataset" / "manifests"
)
# 必须用极低的 conf 取回全部候选框，否则高分框以外的信息在扫阈值之前就被砍掉了。
LOW_CONFIDENCE_FLOOR = 0.001
SHORTCUT_WARNING = (
    "本数据集中四维存在标签是场景的确定性函数（air/sea/urban/forest 各自只对应唯一存在向量，"
    "750张零例外），像素消融显示抹掉全部目标像素后 Exact Match 仅从99.10%降到98.20%。"
    "因此本文件的整图存在指标衡量的主要是场景识别而非目标识别，"
    "详见 docs/诊断报告-场景捷径与模型选择缺陷.md。"
)
SCOPE_WARNING = (
    "本指标把检测框聚合成整图存在向量，只用于与 ResNet18 整图多标签基线做同口径对比；"
    "它丢弃了框位置、框数量和定位精度，不能替代检测 mAP，也不能反过来说明检测能力。"
)
COMPARABILITY_NOTE = (
    "YOLO 每类分数 = 该图该类所有候选框的最高置信度（无框记 0.0），"
    "与 ResNet18 的每类 sigmoid 概率同位；两侧共用 whole_image.optimize_thresholds "
    "在 val 上选阈值、共用 whole_image.compute_multilabel_metrics 在 test 上算指标。"
)
SIGNIFICANCE_NOTE = (
    "测试集只有111张整图，1张图约等于0.90个百分点，种子噪声实测标准差0.61个百分点；"
    "小于约1.2个百分点的差异不能声称显著。"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="把YOLO检测框聚合成整图四维存在向量，与ResNet18整图基线同口径比较"
    )
    parser.add_argument("--model", type=Path, required=True, help="YOLO 权重 .pt")
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        required=True,
        help="含 train/val/test.txt 的目录，必须与 ResNet18 整图实验完全相同",
    )
    parser.add_argument("--output", type=Path, required=True, help="输出目录")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument(
        "--conf",
        type=float,
        default=LOW_CONFIDENCE_FLOOR,
        help="推理置信度下限，默认极低以便保留全部候选框供扫阈值",
    )
    return parser.parse_args(argv)


def resolve_ultralytics_device(requested: str) -> str:
    """Map the shared 'auto' spelling onto an Ultralytics device string."""

    if requested != "auto":
        return requested
    try:
        import torch
    except ImportError:  # pragma: no cover - torch 是硬依赖，缺失时直接回落 cpu
        return "cpu"
    return "0" if torch.cuda.is_available() else "cpu"


def read_presence_manifest(manifest_path: Path, class_count: int = len(CLASS_NAMES)) -> list[dict]:
    """Read one manifest exactly the way WholeImageDataset does, minus the pixels.

    Returns one row per image with the ground-truth presence vector, sensor and scene,
    so the YOLO side sees literally the same images and labels as the ResNet18 side.
    """

    rows: list[dict] = []
    seen_paths: set[str] = set()
    for raw in manifest_path.read_text(encoding="utf-8-sig").splitlines():
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
        label_path = image_path.with_suffix(".txt")
        boxes = parse_yolo_boxes(label_path, class_count)
        sensor, scene = parse_image_context(image_path)
        rows.append(
            {
                "image_path": normalized_path,
                "label_path": str(label_path),
                "sensor": sensor,
                "scene": scene,
                "target": presence_vector_from_boxes(boxes, class_count),
            }
        )
    if not rows:
        raise ValueError(f"整图清单为空: {manifest_path}")
    return rows


def presence_vector_from_boxes(boxes, class_count: int = len(CLASS_NAMES)) -> np.ndarray:
    """A class is present when at least one ground-truth box carries that class id."""

    vector = np.zeros(class_count, dtype=np.int64)
    for box in boxes:
        class_id = int(box.class_id)
        if not 0 <= class_id < class_count:
            raise ValueError(f"类别编号越界: {class_id}")
        vector[class_id] = 1
    return vector


def max_confidence_scores(
    detections, class_count: int = len(CLASS_NAMES)
) -> np.ndarray:
    """Collapse (class_id, confidence) pairs into one score per class.

    该类没有任何候选框时记 0.0，这样分数向量与 ResNet18 的 sigmoid 概率向量同位同量纲。
    """

    scores = np.zeros(class_count, dtype=np.float32)
    for class_id, confidence in detections:
        index = int(class_id)
        if not 0 <= index < class_count:
            raise ValueError(f"类别编号越界: {index}")
        value = float(confidence)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"置信度必须位于[0,1]: {value}")
        if value > scores[index]:
            scores[index] = value
    return scores


def extract_detections(result) -> list[tuple[int, float]]:
    """Pull (class_id, confidence) pairs out of one Ultralytics Result."""

    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []
    class_ids = np.asarray(_to_numpy(boxes.cls)).reshape(-1)
    confidences = np.asarray(_to_numpy(boxes.conf)).reshape(-1)
    if len(class_ids) != len(confidences):
        raise ValueError("检测框的类别数量与置信度数量不一致")
    return [
        (int(class_id), float(confidence))
        for class_id, confidence in zip(class_ids, confidences)
    ]


def _to_numpy(values):
    if hasattr(values, "detach"):
        return values.detach().cpu().numpy()
    return np.asarray(values)


def apply_thresholds(scores: np.ndarray, thresholds: list[float]) -> np.ndarray:
    """Binarise per-class scores with the same '>=' rule compute_multilabel_metrics uses."""

    scores = np.asarray(scores, dtype=np.float32)
    if scores.ndim != 2:
        raise ValueError("分数必须是二维数组")
    if scores.shape[1] != len(thresholds):
        raise ValueError("阈值数量与类别数不一致")
    return (scores >= np.asarray(thresholds, dtype=np.float32)).astype(np.int64)


def _path_key(path) -> str:
    """Normalise a path so Ultralytics' echoed path can be matched back to a manifest row."""

    return os.path.normcase(os.path.abspath(str(path)))


def predict_presence_scores(
    model,
    image_paths: list[str],
    class_count: int = len(CLASS_NAMES),
    imgsz: int = 640,
    device: str = "cpu",
    batch: int = 8,
    conf: float = LOW_CONFIDENCE_FLOOR,
    work_dir: Path | None = None,
    source_name: str = "predict_sources",
) -> np.ndarray:
    """Run YOLO over the manifest images and return an (N, class_count) score matrix.

    Ultralytics 会把 list 形式的 source 转成 PIL 图（丢掉路径），而读 .txt 清单时又会
    对文件列表排序。所以这里写一份路径清单交给它，再按 result.path 回填，
    绝不依赖返回顺序与输入顺序一致。
    """

    if batch < 1:
        raise ValueError("batch 必须大于等于1")
    scores = np.zeros((len(image_paths), class_count), dtype=np.float32)
    index_by_path = {_path_key(path): index for index, path in enumerate(image_paths)}
    if len(index_by_path) != len(image_paths):
        raise ValueError("图片清单包含重复路径")

    holder = work_dir or Path(tempfile.mkdtemp(prefix="presence_source_"))
    holder.mkdir(parents=True, exist_ok=True)
    source_path = holder / f"{source_name}.txt"
    source_path.write_text(
        "\n".join(str(Path(path)) for path in image_paths) + "\n", encoding="utf-8"
    )

    filled = np.zeros(len(image_paths), dtype=bool)
    results = model.predict(
        source=str(source_path),
        imgsz=imgsz,
        device=device,
        conf=conf,
        max_det=300,
        batch=batch,
        verbose=False,
        save=False,
        save_txt=False,
        stream=True,
        project=str(holder),
        name="ultralytics_predict",
        exist_ok=True,
    )
    for result in results:
        key = _path_key(getattr(result, "path", ""))
        index = index_by_path.get(key)
        if index is None:
            raise RuntimeError(f"推理结果的路径不在清单中: {getattr(result, 'path', '')}")
        if filled[index]:
            raise RuntimeError(f"同一张图返回了多次推理结果: {result.path}")
        scores[index] = max_confidence_scores(extract_detections(result), class_count)
        filled[index] = True
    if not filled.all():
        missing = [image_paths[index] for index in np.flatnonzero(~filled)][:3]
        raise RuntimeError(
            f"有{int((~filled).sum())}张图没有拿到推理结果，例如: {missing}"
        )
    return scores


def collect_split(
    model,
    manifest_path: Path,
    class_names: list[str],
    imgsz: int,
    device: str,
    batch: int,
    conf: float,
    work_dir: Path | None = None,
    source_name: str = "predict_sources",
) -> dict:
    rows = read_presence_manifest(manifest_path, len(class_names))
    true = np.stack([row["target"] for row in rows]).astype(np.int64)
    scores = predict_presence_scores(
        model,
        [row["image_path"] for row in rows],
        class_count=len(class_names),
        imgsz=imgsz,
        device=device,
        batch=batch,
        conf=conf,
        work_dir=work_dir,
        source_name=source_name,
    )
    return {
        "manifest": str(manifest_path.resolve()),
        "rows": rows,
        "true": true,
        "scores": scores,
        "sensors": [row["sensor"] for row in rows],
        "scenes": [row["scene"] for row in rows],
    }


def write_predictions_csv(
    output_path: Path, splits: dict[str, dict], thresholds: list[float], class_names: list[str]
) -> None:
    fieldnames = (
        ["split", "image_path", "sensor", "scene"]
        + [f"true_{name}" for name in class_names]
        + [f"score_{name}" for name in class_names]
        + [f"pred_{name}" for name in class_names]
        + ["exact_match"]
    )
    with output_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for split_name, payload in splits.items():
            predicted = apply_thresholds(payload["scores"], thresholds)
            for index, row in enumerate(payload["rows"]):
                record = {
                    "split": split_name,
                    "image_path": row["image_path"],
                    "sensor": row["sensor"],
                    "scene": row["scene"],
                    "exact_match": int(
                        np.array_equal(payload["true"][index], predicted[index])
                    ),
                }
                for class_index, name in enumerate(class_names):
                    record[f"true_{name}"] = int(payload["true"][index][class_index])
                    record[f"score_{name}"] = round(
                        float(payload["scores"][index][class_index]), 6
                    )
                    record[f"pred_{name}"] = int(predicted[index][class_index])
                writer.writerow(record)


def evaluate_presence(
    model_path: Path,
    manifest_dir: Path,
    output_dir: Path,
    imgsz: int = 640,
    device: str = "auto",
    batch: int = 8,
    conf: float = LOW_CONFIDENCE_FLOOR,
) -> dict:
    from ultralytics import YOLO

    if not model_path.is_file():
        raise FileNotFoundError(f"检测权重不存在: {model_path}")
    manifest_paths = {
        split: manifest_dir / f"{split}.txt" for split in ("val", "test")
    }
    for split, path in manifest_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{split} 清单不存在: {path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    resolved_device = resolve_ultralytics_device(device)
    model = YOLO(str(model_path.resolve()))
    class_names = list(CLASS_NAMES)

    splits = {}
    for split, path in manifest_paths.items():
        splits[split] = collect_split(
            model,
            path,
            class_names,
            imgsz=imgsz,
            device=resolved_device,
            batch=batch,
            conf=conf,
            work_dir=output_dir / "predict_sources",
            source_name=split,
        )
        print(f"{split}: {len(splits[split]['rows'])} 张整图完成推理")

    # 阈值只允许看 val。test 在此之前没有以任何形式参与选择。
    thresholds = optimize_thresholds(
        splits["val"]["true"], splits["val"]["scores"], len(class_names)
    )
    validation_metrics = compute_multilabel_metrics(
        splits["val"]["true"],
        splits["val"]["scores"],
        thresholds,
        splits["val"]["sensors"],
        splits["val"]["scenes"],
        class_names,
    )
    test_metrics = compute_multilabel_metrics(
        splits["test"]["true"],
        splits["test"]["scores"],
        thresholds,
        splits["test"]["sensors"],
        splits["test"]["scenes"],
        class_names,
    )

    report = {
        "model": str(model_path.resolve()),
        "model_family": "yolov8_presence_aggregation",
        "task_type": "whole_image_multilabel",
        "device": resolved_device,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "class_names": class_names,
        "manifest_dir": str(manifest_dir.resolve()),
        "source": {
            "manifest_dir": str(manifest_dir.resolve()),
            "val_manifest": splits["val"]["manifest"],
            "test_manifest": splits["test"]["manifest"],
            "val_image_count": len(splits["val"]["rows"]),
            "test_image_count": len(splits["test"]["rows"]),
            "label_suffix": ".txt",
            "score_definition": "每类分数 = 该图该类候选框的最高置信度，无框记0.0",
            "inference": {"imgsz": imgsz, "conf": conf, "batch": batch, "max_det": 300},
        },
        "thresholds": {
            name: float(value) for name, value in zip(class_names, thresholds)
        },
        "threshold_rule": (
            "复用 target_classifier_module.whole_image.optimize_thresholds，"
            "仅在验证集上按每类F1网格搜索；测试集不参与阈值选择。"
        ),
        "validation": validation_metrics,
        "test": test_metrics,
        "comparability_note": COMPARABILITY_NOTE,
        "scope_warning": SCOPE_WARNING,
        "shortcut_warning": SHORTCUT_WARNING,
        "significance_note": SIGNIFICANCE_NOTE,
    }
    (output_dir / "presence_metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_predictions_csv(
        output_dir / "presence_predictions.csv", splits, thresholds, class_names
    )
    return report


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    report = evaluate_presence(
        model_path=args.model,
        manifest_dir=args.manifest_dir,
        output_dir=args.output,
        imgsz=args.imgsz,
        device=args.device,
        batch=args.batch,
        conf=args.conf,
    )
    print(
        json.dumps(
            {
                "thresholds": report["thresholds"],
                "validation": {
                    "sample_count": report["validation"]["sample_count"],
                    "exact_match_accuracy": report["validation"]["exact_match_accuracy"],
                    "macro_f1": report["validation"]["macro_f1"],
                },
                "test": {
                    "sample_count": report["test"]["sample_count"],
                    "exact_match_accuracy": report["test"]["exact_match_accuracy"],
                    "micro_f1": report["test"]["micro_f1"],
                    "macro_f1": report["test"]["macro_f1"],
                    "hamming_accuracy": report["test"]["hamming_accuracy"],
                    "mean_label_average_precision": report["test"][
                        "mean_label_average_precision"
                    ],
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(SHORTCUT_WARNING)


if __name__ == "__main__":
    main()
