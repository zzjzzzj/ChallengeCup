from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Callable

import torch
import yaml
from ultralytics import YOLO

from detector_module.dataset import DetectionSample, read_scene_index
from detector_module.metrics import detection_metrics_to_dict


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = PROJECT_ROOT / "detector_module" / "artifacts" / "detection_dataset" / "dataset.yaml"
DEFAULT_INDEX = PROJECT_ROOT / "scene_module" / "artifacts" / "scene_index.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="按总体、传感器和场景评估目标检测模型")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="0" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--overall-only",
        action="store_true",
        help="只评估完整测试集，适用于不允许创建多进程管道的受限环境",
    )
    return parser.parse_args()


def read_names(config: dict) -> list[str]:
    names = config["names"]
    if isinstance(names, dict):
        return [str(names[index] if index in names else names[str(index)]) for index in range(len(names))]
    return [str(name) for name in names]


def write_slice_config(
    base_config: dict,
    samples: list[DetectionSample],
    output_dir: Path,
    slice_name: str,
) -> Path:
    manifests_dir = output_dir / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = (manifests_dir / f"{slice_name}.txt").resolve()
    manifest_path.write_text(
        "\n".join(sample.image_path.as_posix() for sample in samples) + "\n",
        encoding="utf-8",
    )
    config = dict(base_config)
    config["test"] = manifest_path.as_posix()
    config_path = (manifests_dir / f"{slice_name}.yaml").resolve()
    config_path.write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    return config_path


def build_evaluation_slices(
    test_samples: list[DetectionSample], overall_only: bool = False
) -> list[tuple[str, str, Callable[[DetectionSample], bool]]]:
    slices: list[tuple[str, str, Callable[[DetectionSample], bool]]] = [
        ("overall", "all", lambda _sample: True)
    ]
    if overall_only:
        return slices
    for sensor in sorted({sample.sensor for sample in test_samples}):
        slices.append(
            ("sensor", sensor, lambda sample, value=sensor: sample.sensor == value)
        )
    for scene in sorted({sample.scene for sample in test_samples}):
        slices.append(("scene", scene, lambda sample, value=scene: sample.scene == value))
    return slices


def main() -> None:
    args = parse_args()
    for path in (args.model, args.data, args.index):
        if not path.is_file():
            raise FileNotFoundError(path)

    output_dir = args.output or args.model.resolve().parents[1] / "slice_evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    base_config = yaml.safe_load(args.data.read_text(encoding="utf-8"))
    class_names = read_names(base_config)
    test_samples = [sample for sample in read_scene_index(args.index) if sample.split == "test"]
    model = YOLO(str(args.model.resolve()))

    slices = build_evaluation_slices(test_samples, args.overall_only)

    results = []
    for group, value, predicate in slices:
        subset = [sample for sample in test_samples if predicate(sample)]
        slice_name = f"{group}_{value}"
        slice_config = write_slice_config(base_config, subset, output_dir, slice_name)
        metrics = model.val(
            data=str(slice_config),
            split="test",
            imgsz=args.image_size,
            batch=args.batch_size,
            device=args.device,
            workers=args.workers,
            project=str(output_dir.resolve()),
            name=slice_name,
            plots=group == "overall",
            verbose=False,
        )
        converted = detection_metrics_to_dict(metrics, class_names)
        results.append(
            {
                "group": group,
                "value": value,
                "image_count": len(subset),
                **converted,
            }
        )
        print(
            f"{group}/{value}: images={len(subset)} "
            f"mAP50={converted['map50']:.4f} mAP50-95={converted['map50_95']:.4f}"
        )

    report = {
        "model": str(args.model.resolve()),
        "data": str(args.data.resolve()),
        "test_image_count": len(test_samples),
        "slices": results,
        "limitations": [
            "air 场景只有 IR 数据，不能据此评价 SAR-air 泛化能力。",
            "场景切片中的目标类别并不均衡，切片指标需结合每类支持数解释。",
            "overall_only=true 时本报告只包含完整测试集指标，不包含模态/场景切片。",
        ],
        "overall_only": args.overall_only,
    }
    (output_dir / "slice_metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (output_dir / "slice_metrics.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "group",
                "value",
                "image_count",
                "precision",
                "recall",
                "map50",
                "map50_95",
                "map75",
            ],
        )
        writer.writeheader()
        for row in results:
            writer.writerow({name: row[name] for name in writer.fieldnames})


if __name__ == "__main__":
    main()
