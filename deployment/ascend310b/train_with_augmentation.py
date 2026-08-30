#!/usr/bin/env python3
"""Board-side helper: selected offline augmentation, then YOLO training.

This wrapper is intentionally thin. It creates/reuses the augmented YOLO
dataset, then calls the existing project training entry point:

    python train.py yolo --data <augmented>/data.yaml ...
"""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEPLOYMENT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "data" / "datasets_r1_base_train"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "datasets_r1_base_train_augmented"
DEFAULT_RUNS = PROJECT_ROOT / "scene_recognition" / "detector_module" / "runs"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run selected YOLO offline augmentation and optionally train on the generated data.yaml."
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    source.add_argument("--images", type=Path, help="Explicit source image directory.")
    parser.add_argument("--labels", type=Path, help="Explicit source label directory, required with --images.")
    parser.add_argument("--split", default="train", help="Split name under --dataset-root. Default: train.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Augmented YOLO dataset output root.")
    parser.add_argument("--classes", type=Path, help="Class-name file. Defaults to <dataset-root>/classes.txt.")
    parser.add_argument("--val-images", type=Path, help="Independent validation image directory/list/file.")
    parser.add_argument("--val-root", type=Path, help="Independent validation YOLO root.")
    parser.add_argument("--val-split", default="val", help="Validation split under --val-root. Default: val.")
    parser.add_argument(
        "--default-modality",
        choices=["ir", "sar"],
        help="Fallback when image names do not start with ir_ or sar_.",
    )
    parser.add_argument(
        "--include-original",
        dest="include_original",
        action="store_true",
        default=True,
        help="Copy original images/labels into the augmented training set. Enabled by default.",
    )
    parser.add_argument(
        "--no-include-original",
        dest="include_original",
        action="store_false",
        help="Use only generated augmented variants.",
    )
    parser.add_argument("--force-augment", action="store_true", help="Rebuild the augmented dataset.")
    parser.add_argument("--augment-only", action="store_true", help="Only generate/reuse the augmented dataset.")

    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--image-size", type=int, default=960)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--project", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--name", default="ascend310b_augmented_yolov8n_960")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--no-builtin-aug", action="store_true")
    parser.add_argument("--freeze", type=int, default=None, help="Freeze the first N YOLO layers for CPU fine-tuning.")
    parser.add_argument("--no-amp", action="store_true", help="Disable AMP. Recommended on CPU-only training.")
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation to reduce board-side overhead.")
    return parser.parse_args(argv)


def build_augmentation_command(args: argparse.Namespace, data_yaml: Path) -> List[str]:
    command = [
        sys.executable,
        str(DEPLOYMENT_DIR / "augment_selected_yolo.py"),
        "--output",
        str(args.output),
        "--data-yaml",
        str(data_yaml),
    ]
    if args.images is not None:
        if args.labels is None:
            raise SystemExit("--labels is required when --images is used.")
        command.extend(["--images", str(args.images), "--labels", str(args.labels)])
    else:
        command.extend(["--dataset-root", str(args.dataset_root), "--split", args.split])
    if args.classes is not None:
        command.extend(["--classes", str(args.classes)])
    if args.val_images is not None:
        command.extend(["--val-images", str(args.val_images)])
    if args.val_root is not None:
        command.extend(["--val-root", str(args.val_root), "--val-split", args.val_split])
    if args.default_modality is not None:
        command.extend(["--default-modality", args.default_modality])
    if args.include_original:
        command.append("--include-original")
    if args.force_augment:
        command.append("--force")
    return command


def build_training_command(args: argparse.Namespace, data_yaml: Path) -> List[str]:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "train.py"),
        "yolo",
        "--data",
        str(data_yaml),
        "--model",
        args.model,
        "--epochs",
        str(args.epochs),
        "--patience",
        str(args.patience),
        "--image-size",
        str(args.image_size),
        "--batch-size",
        str(args.batch_size),
        "--workers",
        str(args.workers),
        "--device",
        args.device,
        "--seed",
        str(args.seed),
        "--project",
        str(args.project),
        "--name",
        args.name,
    ]
    if args.resume:
        command.append("--resume")
    if args.exist_ok:
        command.append("--exist-ok")
    if args.no_pretrained:
        command.append("--no-pretrained")
    if args.no_builtin_aug:
        command.append("--no-builtin-aug")
    if args.freeze is not None:
        command.extend(["--freeze", str(args.freeze)])
    if args.no_amp:
        command.append("--no-amp")
    if args.no_plots:
        command.append("--no-plots")
    return command


def check_training_dependencies() -> None:
    missing = [
        module_name
        for module_name in ("torch", "ultralytics")
        if importlib.util.find_spec(module_name) is None
    ]
    if not missing:
        return
    raise SystemExit(
        "\n".join(
            [
                "[ERROR] YOLO training requires PyTorch because the project training entry calls Ultralytics YOLO.",
                "[ERROR] Missing modules in current Python (%s): %s"
                % (sys.executable, ", ".join(missing)),
                "[HINT] If you only want to generate augmented images, rerun with --augment-only.",
                "[HINT] If you want to train, run this script with the conda Python that has torch installed.",
                "[HINT] Example: $CONDA_PREFIX/bin/python deployment/ascend310b/train_with_augmentation.py ...",
            ]
        )
    )


def run(command: Sequence[str]) -> None:
    print("[INFO] " + " ".join(str(part) for part in command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if (args.images is None) != (args.labels is None):
        raise SystemExit("--images and --labels must be passed together.")
    if args.val_images is not None and args.val_root is not None:
        raise SystemExit("Use either --val-images or --val-root, not both.")

    output_root = args.output.resolve()
    data_yaml = output_root / "data.yaml"

    if data_yaml.is_file() and not args.force_augment:
        print("[INFO] Reuse augmented dataset: %s" % data_yaml, flush=True)
    else:
        run(build_augmentation_command(args, data_yaml))

    if args.augment_only:
        print("[INFO] Augmentation is ready: %s" % data_yaml, flush=True)
        return 0

    check_training_dependencies()
    run(build_training_command(args, data_yaml))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
