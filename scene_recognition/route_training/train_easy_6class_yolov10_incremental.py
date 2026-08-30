"""Train the six-class incremental YOLOv10 detector used by the easy route.

The deployed easy ONNX is exported at 640x640 from this detector checkpoint.
The checkpoint itself is trained at 960x960, matching the established
six-class incremental recipe.
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
            "EASY_DETECTOR_INIT",
            "runs/detect/detector_module/runs/r2_yolov10n_960_6class/init6.pt",
        ),
        help="Four-class YOLOv10 checkpoint with a six-class head prepared.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=env_path("EASY_DETECTOR_DATA", "detector_module/configs/data_user_full.yaml"),
        help="Six-class Ultralytics detection YAML.",
    )
    parser.add_argument("--project", type=Path, default=Path("scene_recognition/runs/route_training"))
    parser.add_argument("--name", default="easy_6class_yolov10n_incremental_960")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", action="store_true", help="Resume from the run's last.pt checkpoint.")
    args = parser.parse_args()

    if not args.model.is_file():
        raise FileNotFoundError(f"easy detector checkpoint not found: {args.model}")
    if not args.data.is_file():
        raise FileNotFoundError(f"easy detector data YAML not found: {args.data}")

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
        cos_lr=True,
        name=args.name,
        project=str(args.project),
        exist_ok=True,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
