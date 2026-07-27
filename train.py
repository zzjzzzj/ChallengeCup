"""Single documented entry point for data preparation, training and evaluation."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
COMMANDS: dict[str, tuple[str, ...]] = {
    "scene-prepare": ("-m", "scene_module.analyze_and_prepare"),
    "scene-extract": ("-m", "scene_module.feature_engineering", "extract"),
    "scene-evaluate": ("-m", "scene_module.feature_engineering", "evaluate"),
    "crop-prepare": ("-m", "target_classifier_module.prepare_crops"),
    "crop-classifier": ("-m", "target_classifier_module.train_classifier"),
    "whole-classifier": ("-m", "target_classifier_module.train_whole_image"),
    "prepare-detection": ("-m", "detector_module.prepare_detection_dataset"),
    "prepare-comparison": ("-m", "detector_module.prepare_comparison_dataset"),
    "yolo": ("-m", "detector_module.train_detector_ablation"),
    "resnet-detector": ("-m", "detector_module.resnet18_detector"),
    "detection-matrix": (str(ROOT / "run_detection_experiments.py"),),
    "yolo-evaluate": ("-m", "detector_module.evaluate_yolo_same_protocol"),
}


def usage() -> str:
    rows = [
        "Usage: python train.py <command> [arguments]",
        "",
        "Commands:",
        "  scene-prepare       audit data and create the scene split",
        "  scene-extract       extract handcrafted scene features",
        "  scene-evaluate      train/evaluate scene classifiers",
        "  crop-prepare        crop targets from ground-truth boxes",
        "  crop-classifier     train the ResNet18 crop classifier",
        "  whole-classifier    train the historical whole-image classifier",
        "  prepare-detection   build the original 525/114/111 detection split",
        "  prepare-comparison  build the final original/augmented 79/76 split",
        "  yolo                train one YOLOv8n detector",
        "  resnet-detector     train one ResNet18-FPN detector",
        "  detection-matrix    run the final eight detection experiments",
        "  yolo-evaluate       re-evaluate YOLO with the shared AP implementation",
        "",
        "Run `python train.py <command> --help` for command-specific options.",
    ]
    return "\n".join(rows)


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        print(usage())
        return
    command_name = sys.argv[1]
    command = COMMANDS.get(command_name)
    if command is None:
        raise SystemExit(f"Unknown command: {command_name}\n\n{usage()}")
    completed = subprocess.run(
        (sys.executable, *command, *sys.argv[2:]),
        cwd=ROOT,
    )
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
