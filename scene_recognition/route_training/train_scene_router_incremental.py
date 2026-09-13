"""Train the four-class scene router used by the routed detector.

This fine-tunes the previous YOLOv8n-cls checkpoint on the combined r1+r2
scene dataset. The route-safe checkpoint selection is evaluated separately on
the detector validation set because scene Top-1 accuracy and route accuracy
are different metrics.
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
        default=env_path(
            "SCENE_ROUTER_INIT",
            "runs/classify/detector_module/runs/scene_cls_yolov8n_v2/weights/best.pt",
        ),
        help="Existing four-scene YOLOv8n-cls checkpoint.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=env_path("SCENE_ROUTER_DATA", "D:/scene_cls_r1_r2_incremental"),
        help="Ultralytics classification dataset with train/ and val/ folders.",
    )
    parser.add_argument("--project", type=Path, default=Path("scene_recognition/runs/route_training"))
    parser.add_argument("--name", default="scene_router_yolov8n_r1r2_incremental_224")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if not args.model.is_file():
        raise FileNotFoundError(f"scene router checkpoint not found: {args.model}")
    if not args.data.is_dir():
        raise FileNotFoundError(f"scene router dataset not found: {args.data}")

    model = YOLO(str(args.model), task="classify")
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        patience=10,
        imgsz=224,
        batch=args.batch_size,
        workers=args.workers,
        device=args.device,
        seed=42,
        deterministic=True,
        optimizer="auto",
        cos_lr=True,
        amp=True,
        cache=False,
        save=True,
        save_period=5,
        val=True,
        project=str(args.project),
        name=args.name,
        exist_ok=False,
    )


if __name__ == "__main__":
    main()
