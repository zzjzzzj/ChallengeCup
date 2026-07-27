"""Unified command line interface for scene recognition."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COMMANDS: dict[str, tuple[str, ...]] = {
    "train-features": ("-m", "scene_module.feature_engineering", "evaluate"),
    "train-cnn": ("-m", "scene_module.train_scene_classifier"),
    "infer": ("-m", "scene_module.feature_infer"),
    "dashboard": ("-m", "scene_module.feature_web_app"),
}


def usage() -> str:
    return "\n".join(
        [
            "Usage: python -m scene_recognition.cli <command> [arguments]",
            "",
            "Commands:",
            "  train-features  train/evaluate the handcrafted-feature classifier",
            "  train-cnn       train the image-based scene classifier",
            "  infer           run single-image scene inference",
            "  dashboard       start the local feature-analysis dashboard",
            "",
            "Run `python -m scene_recognition.cli <command> --help` for details.",
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
