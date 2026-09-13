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
    "split-yolo": ("-m", "scene_recognition.detector_module.split_yolo_dataset"),
    "prepare-continual": ("-m", "scene_recognition.detector_module.prepare_continual_dataset"),
    "prepare-class-il": ("-m", "scene_recognition.detector_module.prepare_class_incremental_dataset"),
    "yolo": ("-m", "scene_recognition.detector_module.train_detector_ablation"),
    "continual-yolo": ("-m", "scene_recognition.detector_module.train_continual_yolo"),
    "class-il-yolo": ("-m", "scene_recognition.detector_module.train_class_incremental_yolo"),
    "augment-yolo": ("-m", "scene_recognition.detector_module.augment_yolo_dataset"),
    "prepare-batch-il": ("-m", "scene_recognition.detector_module.prepare_batch_incremental_dataset"),
    "batch-il-yolo": ("-m", "scene_recognition.detector_module.train_batch_incremental_yolo"),
    "four-to-six-yolo": ("-m", "scene_recognition.detector_module.run_four_to_six_pipeline"),
    "continual-evaluate": ("-m", "scene_recognition.detector_module.evaluate_continual"),
    "resnet-detector": ("-m", "scene_recognition.detector_module.resnet18_detector"),
    "detection-matrix": (
        str(ROOT / "scene_recognition" / "experiments" / "run_detection_experiments.py"),
    ),
    "yolo-evaluate": ("-m", "scene_recognition.detector_module.evaluate_yolo_same_protocol"),
    "ascend310b-package": (
        str(ROOT / "deployment" / "ascend310b" / "build_ascend310b_package.py"),
    ),
    "ascend310b-augment": (
        str(ROOT / "deployment" / "ascend310b" / "augment_selected_yolo.py"),
    ),
    "ascend310b-train-aug": (
        str(ROOT / "deployment" / "ascend310b" / "train_with_augmentation.py"),
    ),
    "ascend310b-pipeline": (
        str(ROOT / "deployment" / "ascend310b" / "run_end_to_end.py"),
    ),
    "ascend310b-probe-train": (
        str(ROOT / "deployment" / "ascend310b" / "probe_training_env.py"),
    ),
    "ascend310b-cascade": (
        str(ROOT / "deployment" / "ascend310b" / "infer_cascade_npu.py"),
    ),
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
        "  split-yolo          keep train fixed and create leakage-safe val/test splits",
        "  prepare-continual   build local r2 incremental/replay manifests",
        "  prepare-class-il    build six singleton Class-IL stages with 200/500 buffers",
        "  yolo                train one YOLOv8n detector",
        "  continual-yolo      fine-tune a local checkpoint on an incremental round",
        "  class-il-yolo       run six-stage ER or DER Class-IL training",
        "  augment-yolo       build deterministic offline train-only YOLO augmentation",
        "  prepare-batch-il   prepare arbitrary-batch four-to-six Class-IL views",
        "  batch-il-yolo      run arbitrary-batch ER or DER Class-IL training",
        "  four-to-six-yolo   orchestrate base augmentation/training and batch IL",
        "  continual-evaluate  report New-mAP, old-class mAP and KRR",
        "  resnet-detector     train one ResNet18-FPN detector",
        "  detection-matrix    run the final eight detection experiments",
        "  yolo-evaluate       re-evaluate YOLO with the shared AP implementation",
        "  ascend310b-package  build a portable Ascend 310B inference package",
        "  ascend310b-augment  build selected offline YOLO augmentation data",
        "  ascend310b-train-aug augment first, then train YOLO on the generated data",
        "  ascend310b-pipeline  augment data, run an existing ONNX/OM model and save outputs",
        "  ascend310b-probe-train inspect board-side torch/ultralytics/NPU training environment",
        "  ascend310b-cascade  run six-class plus single-class expert NPU inference",
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
