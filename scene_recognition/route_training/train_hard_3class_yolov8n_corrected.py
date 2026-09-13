"""Train the corrected hard-scene three-class YOLOv8n detector.

This is the recipe that produced the hard branch used by the current route:
960 square training, shuffled batches, Mosaic enabled, and no rectangular
training. Rectangular 1120x896 is an inference/export comparison only.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from ultralytics import YOLO


def env_path(name: str, default: str) -> Path:
    return Path(os.environ.get(name, default)).expanduser()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=env_path("HARD_DETECTOR_INIT", "yolov8n.pt"),
        help="COCO-pretrained YOLOv8n checkpoint.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=env_path("HARD_DETECTOR_DATA", "hard_3class_dataset/hard_3class.yaml"),
        help="Three-class hard-scene detection YAML.",
    )
    parser.add_argument("--project", type=Path, default=Path("scene_recognition/runs/route_training"))
    parser.add_argument("--name", default="hard_3class_yolov8n_corrected_960")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if not args.model.is_file():
        raise FileNotFoundError(f"hard detector checkpoint not found: {args.model}")
    if not args.data.is_file():
        raise FileNotFoundError(f"hard detector data YAML not found: {args.data}")

    model = YOLO(str(args.model), task="detect")
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        patience=20,
        imgsz=960,
        batch=args.batch_size,
        workers=args.workers,
        device=args.device,
        seed=42,
        deterministic=True,
        optimizer="auto",
        rect=False,
        mosaic=1.0,
        close_mosaic=10,
        cos_lr=True,
        amp=True,
        cache=False,
        save=True,
        save_period=10,
        val=True,
        project=str(args.project),
        name=args.name,
        exist_ok=False,
    )


if __name__ == "__main__":
    main()
