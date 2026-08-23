"""Evaluate an incremental detector with New-mAP, old-class mAP and KRR."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import yaml

from scene_recognition.detector_module import BASE_CLASS_NAMES, INCREMENTAL_CLASS_NAMES
from scene_recognition.detector_module.evaluate_yolo_same_protocol import evaluate_checkpoint
from scene_recognition.detector_module.resnet18_detector import YoloManifestDataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "scene_recognition"
    / "detector_module"
    / "runs"
    / "continual_r2_evaluation.json"
)


def parse_class_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def read_data_config(data_path: Path) -> tuple[dict, list[str]]:
    config = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    names = config["names"]
    class_names = (
        [str(names[index] if index in names else names[str(index)]) for index in range(len(names))]
        if isinstance(names, dict)
        else [str(name) for name in names]
    )
    return config, class_names


def aggregate_classes(metrics: dict, class_names: list[str], metric_name: str) -> dict:
    values: list[float] = []
    missing_support: list[str] = []
    per_class: dict[str, dict] = {}
    for name in class_names:
        row = metrics["per_class"].get(name)
        if row is None or int(row.get("support", 0)) == 0:
            missing_support.append(name)
            continue
        value = float(row[metric_name])
        if not math.isnan(value):
            values.append(value)
            per_class[name] = {
                "support": int(row["support"]),
                metric_name: value,
            }
    return {
        "value": sum(values) / len(values) if values else None,
        "classes": class_names,
        "evaluated_classes": list(per_class),
        "missing_support": missing_support,
        "per_class": per_class,
    }


def build_continual_metrics(
    before_metrics: dict,
    after_metrics: dict,
    old_classes: list[str],
    new_classes: list[str],
) -> dict:
    old_before_50 = aggregate_classes(before_metrics, old_classes, "map50")
    old_after_50 = aggregate_classes(after_metrics, old_classes, "map50")
    new_after_50 = aggregate_classes(after_metrics, new_classes, "map50")
    all_after_50 = aggregate_classes(after_metrics, [*old_classes, *new_classes], "map50")
    old_before_5095 = aggregate_classes(before_metrics, old_classes, "map50_95")
    old_after_5095 = aggregate_classes(after_metrics, old_classes, "map50_95")
    new_after_5095 = aggregate_classes(after_metrics, new_classes, "map50_95")
    all_after_5095 = aggregate_classes(
        after_metrics,
        [*old_classes, *new_classes],
        "map50_95",
    )

    def retention(before: dict, after: dict) -> float | None:
        before_value = before["value"]
        after_value = after["value"]
        if before_value is None or after_value is None or before_value <= 0:
            return None
        return after_value / before_value

    return {
        "map50": {
            "old_map_before": old_before_50,
            "old_map_after": old_after_50,
            "new_map": new_after_50,
            "all_map": all_after_50,
            "krr": retention(old_before_50, old_after_50),
        },
        "map50_95": {
            "old_map_before": old_before_5095,
            "old_map_after": old_after_5095,
            "new_map": new_after_5095,
            "all_map": all_after_5095,
            "krr": retention(old_before_5095, old_after_5095),
        },
        "evaluation_ready": not (
            old_before_50["missing_support"]
            or old_after_50["missing_support"]
            or new_after_50["missing_support"]
        ),
        "score_targets": {"new_map50": 0.60, "krr_map50": 0.95},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评估六类持续检测模型的 New-mAP 和 KRR")
    parser.add_argument("--data", type=Path, required=True, help="含固定 test 清单的六类 YAML")
    parser.add_argument("--before", type=Path, required=True, help="增量前四类 checkpoint")
    parser.add_argument("--after", type=Path, required=True, help="增量后六类 checkpoint")
    parser.add_argument("--old-classes", default=",".join(BASE_CLASS_NAMES))
    parser.add_argument("--new-classes", default=",".join(INCREMENTAL_CLASS_NAMES))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--image-size", type=int, default=640)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (args.data, args.before, args.after):
        if not path.is_file():
            raise FileNotFoundError(path)
    config, class_names = read_data_config(args.data)
    old_classes = parse_class_list(args.old_classes)
    new_classes = parse_class_list(args.new_classes)
    unknown = sorted(set([*old_classes, *new_classes]) - set(class_names))
    if unknown:
        raise ValueError(f"评测类别不在 YAML 中: {', '.join(unknown)}")
    test_value = config.get("test")
    if not test_value:
        raise ValueError("持续学习评测必须提供固定 test 清单")
    test_dataset = YoloManifestDataset(Path(test_value), len(class_names))
    before_metrics = evaluate_checkpoint(
        args.before,
        test_dataset,
        class_names,
        args.device,
        args.image_size,
    )
    after_metrics = evaluate_checkpoint(
        args.after,
        test_dataset,
        class_names,
        args.device,
        args.image_size,
    )
    report = {
        "protocol": "r2-class-increment-v1",
        "data": str(args.data.resolve()),
        "test_images": len(test_dataset),
        "class_order": class_names,
        "old_classes": old_classes,
        "new_classes": new_classes,
        "before_checkpoint": str(args.before.resolve()),
        "after_checkpoint": str(args.after.resolve()),
        "continual_metrics": build_continual_metrics(
            before_metrics,
            after_metrics,
            old_classes,
            new_classes,
        ),
        "raw": {"before": before_metrics, "after": after_metrics},
        "privacy": {"network_used": False, "local_checkpoints_only": True},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["continual_metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
