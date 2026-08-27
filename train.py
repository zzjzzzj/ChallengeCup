"""Single documented entry point for data preparation, training and evaluation."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
COMMANDS: dict[str, tuple[str, ...]] = {
    "image-processing": ("-m", "image_processing.cli"),
    "scene-recognition": ("-m", "scene_recognition.cli"),
    "scene-prepare": ("-m", "image_processing.analyze_and_prepare"),
    "scene-extract": ("-m", "image_processing.feature_engineering", "extract"),
    "scene-evaluate": ("-m", "image_processing.feature_engineering", "evaluate"),
    "crop-prepare": ("-m", "image_processing.prepare_crops"),
    "crop-classifier": ("-m", "scene_recognition.target_classifier_module.train_classifier"),
    "whole-classifier": ("-m", "scene_recognition.target_classifier_module.train_whole_image"),
    "prepare-detection": ("-m", "image_processing.prepare_detection_dataset"),
    "prepare-comparison": ("-m", "image_processing.prepare_comparison_dataset"),
    "prepare-continual": ("-m", "scene_recognition.detector_module.prepare_continual_dataset"),
    "prepare-class-il": ("-m", "scene_recognition.detector_module.prepare_class_incremental_dataset"),
    "yolo": ("-m", "scene_recognition.detector_module.train_detector_ablation"),
    "continual-yolo": ("-m", "scene_recognition.detector_module.train_continual_yolo"),
    "class-il-yolo": ("-m", "scene_recognition.detector_module.train_class_incremental_yolo"),
    "continual-evaluate": ("-m", "scene_recognition.detector_module.evaluate_continual"),
    "resnet-detector": ("-m", "scene_recognition.detector_module.resnet18_detector"),
    "detection-matrix": (
        str(ROOT / "scene_recognition" / "experiments" / "run_detection_experiments.py"),
    ),
    "yolo-evaluate": ("-m", "scene_recognition.detector_module.evaluate_yolo_same_protocol"),
}


def usage() -> str:
    rows = [
        "Usage: python train.py <command> [arguments]",
        "",
        "Commands:",
        "  image-processing    image audit, feature extraction and dataset preparation",
        "  scene-recognition   scene training, inference and visualization",
        "  scene-prepare       audit data and create the scene split",
        "  scene-extract       extract handcrafted scene features",
        "  scene-evaluate      train/evaluate scene classifiers",
        "  crop-prepare        crop targets from ground-truth boxes",
        "  crop-classifier     train the ResNet18 crop classifier",
        "  whole-classifier    train the historical whole-image classifier",
        "  prepare-detection   build the original 525/114/111 detection split",
        "  prepare-comparison  build the final original/augmented 79/76 split",
        "  prepare-continual   build local r2 incremental/replay manifests",
        "  prepare-class-il    build six singleton Class-IL stages with 200/500 buffers",
        "  yolo                train one YOLOv8n detector",
        "  continual-yolo      fine-tune a local checkpoint on an incremental round",
        "  class-il-yolo       run six-stage ER or DER Class-IL training",
        "  continual-evaluate  report New-mAP, old-class mAP and KRR",
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
