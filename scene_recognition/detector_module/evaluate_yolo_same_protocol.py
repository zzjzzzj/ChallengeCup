"""Evaluate existing YOLO checkpoints with the ResNet detector's mAP implementation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from scene_recognition.detector_module.resnet18_detector import YoloManifestDataset, detection_metrics


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = (
    PROJECT_ROOT
    / "detector_module"
    / "artifacts"
    / "comparison_dataset"
    / "data_noaug.yaml"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "docs" / "comparison" / "yolo_clean8_same_evaluator.json"


def read_class_names(data_path: Path) -> list[str]:
    names = yaml.safe_load(data_path.read_text(encoding="utf-8"))["names"]
    if isinstance(names, dict):
        return [str(names[index] if index in names else names[str(index)]) for index in range(len(names))]
    return [str(name) for name in names]


def prediction_from_result(result) -> dict[str, torch.Tensor]:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return {
            "boxes": torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.zeros((0,), dtype=torch.int64),
            "scores": torch.zeros((0,), dtype=torch.float32),
        }
    return {
        "boxes": boxes.xyxy.detach().cpu().float(),
        # Keep class 0 reserved for background, matching torchvision detection.
        "labels": boxes.cls.detach().cpu().to(torch.int64) + 1,
        "scores": boxes.conf.detach().cpu().float(),
    }


def evaluate_checkpoint(
    checkpoint: Path,
    dataset: YoloManifestDataset,
    class_names: list[str],
    device: str,
    image_size: int,
) -> dict:
    from ultralytics import YOLO

    model = YOLO(str(checkpoint))
    results = model.predict(
        source=[str(path) for path in dataset.image_paths],
        imgsz=image_size,
        conf=0.001,
        iou=0.7,
        max_det=200,
        device=device,
        verbose=False,
        stream=False,
    )
    predictions = [prediction_from_result(result) for result in results]
    targets = [dataset[index][1] for index in range(len(dataset))]
    return detection_metrics(predictions, targets, class_names)


def main() -> None:
    parser = argparse.ArgumentParser(description="以与 ResNet18 检测器相同的 mAP 函数评估 YOLO")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--checkpoint", type=Path, help="只评估一个额外 checkpoint")
    parser.add_argument("--run-name", default="candidate", help="单 checkpoint 在输出 JSON 中的名称")
    args = parser.parse_args()

    config = yaml.safe_load(args.data.read_text(encoding="utf-8"))
    class_names = read_class_names(args.data)
    test_dataset = YoloManifestDataset(Path(config["test"]), len(class_names))
    if args.checkpoint is not None:
        runs = {args.run_name: args.checkpoint}
    else:
        runs = {
            tag: PROJECT_ROOT / "scene_recognition" / "detector_module" / "runs" / f"cmp8_yolov8n_{tag}" / "weights" / "best.pt"
            for tag in ("noaug_pretrained", "noaug_scratch", "aug_pretrained", "aug_scratch")
        }
    report = {
        "evaluator": "scene_recognition.detector_module.resnet18_detector.detection_metrics (COCO-style 101-point AP)",
        "test_images": len(test_dataset),
        "data": str(args.data.resolve()),
        "runs": {},
    }
    for tag, checkpoint in runs.items():
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        metrics = evaluate_checkpoint(checkpoint, test_dataset, class_names, args.device, args.image_size)
        report["runs"][tag] = {
            "checkpoint": str(checkpoint.resolve()),
            "metrics": metrics,
        }
        print(f"{tag}: mAP50={metrics['map50']:.4f} mAP50-95={metrics['map50_95']:.4f}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
