"""Unified command line interface for image and dataset preprocessing."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COMMANDS: dict[str, tuple[str, ...]] = {
    "audit": ("-m", "image_processing.analyze_and_prepare"),
    "features": ("-m", "image_processing.feature_engineering", "extract"),
    "crops": ("-m", "scene_recognition.target_classifier_module.prepare_crops"),
    "detection": ("-m", "scene_recognition.detector_module.prepare_detection_dataset"),
    "comparison": ("-m", "scene_recognition.detector_module.prepare_comparison_dataset"),
}


def usage() -> str:
    return "\n".join(
        [
            "Usage: python -m image_processing.cli <command> [arguments]",
            "",
            "Commands:",
            "  audit       audit source images/labels and create stable splits",
            "  features    extract grayscale, texture and frequency features",
            "  crops       crop targets using ground-truth YOLO boxes",
            "  detection   build the original detection manifests",
            "  comparison  build the final original/augmented comparison split",
            "",
            "Run `python -m image_processing.cli <command> --help` for details.",
        ]
    )


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
