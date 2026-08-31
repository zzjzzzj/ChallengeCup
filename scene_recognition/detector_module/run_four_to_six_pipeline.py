"""Orchestrate offline augmentation, four-class training, preparation and IL."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence

import yaml

from scene_recognition.detector_module import ALL_CLASS_NAMES, BASE_CLASS_NAMES
from scene_recognition.detector_module.prepare_batch_incremental_dataset import (
    BUFFER_SIZE_CHOICES,
    _normalise_plan,
)


def _read_names(path: Path) -> list[str]:
    config = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    if not isinstance(config, dict):
        raise ValueError(f"数据 YAML 必须是映射: {path}")
    values = config.get("names")
    if isinstance(values, dict):
        names = [str(values[index] if index in values else values[str(index)]) for index in range(len(values))]
    elif isinstance(values, list):
        names = [str(value) for value in values]
    else:
        raise ValueError(f"数据 YAML 缺少 names: {path}")
    if int(config.get("nc", len(names))) != len(names):
        raise ValueError(f"nc 与 names 数量不一致: {path}")
    return names


def _validate_inputs(base_data: Path, increment_data: Path, generic_model: Path) -> None:
    for path in (base_data, increment_data, generic_model):
        if not path.is_file():
            raise FileNotFoundError(path)
    base_names, increment_names = _read_names(base_data), _read_names(increment_data)
    if base_names != BASE_CLASS_NAMES:
        raise ValueError(f"base-data 必须严格为 {BASE_CLASS_NAMES}，实际为 {base_names}")
    if increment_names != ALL_CLASS_NAMES:
        raise ValueError(f"increment-data 必须严格为 {ALL_CLASS_NAMES}，实际为 {increment_names}")


def build_pipeline_plan(
    base_data: Path,
    increment_data: Path,
    generic_model: Path,
    workspace: Path,
    *,
    num_batches: int | None = None,
    batch_plan: Path | None = None,
    buffer_sizes: Sequence[int] = (200,),
    max_current_images_per_class: int | None = None,
    base_epochs: int = 60,
    increment_epochs: int = 30,
    method: str = "der",
    device: str = "cpu",
    seed: int = 42,
    default_modality: str | None = None,
    sparse_moe: bool = False,
    sparse_options: dict[str, object] | None = None,
) -> dict:
    """Validate and build the complete local command plan without executing it."""

    base_data, increment_data, generic_model, workspace = (Path(value).resolve() for value in (base_data, increment_data, generic_model, workspace))
    _validate_inputs(base_data, increment_data, generic_model)
    if method not in {"er", "der"}:
        raise ValueError("method 必须为 er 或 der")
    if not buffer_sizes or any(int(value) not in BUFFER_SIZE_CHOICES for value in buffer_sizes):
        raise ValueError(f"buffer-size 只允许 {BUFFER_SIZE_CHOICES}")
    if base_epochs <= 0 or increment_epochs <= 0:
        raise ValueError("epochs 必须为正整数")
    plan = _normalise_plan(batch_plan, num_batches, seed)
    root = Path(__file__).resolve().parents[2]
    base_aug = workspace / "base_augmented"
    increment_aug = workspace / "increment_augmented"
    prepared = workspace / "prepared_batch_il"
    runs = workspace / "runs"
    commands: list[list[str]] = []
    python = sys.executable
    augment_args = ["--default-modality", default_modality] if default_modality else []
    commands.append([python, "-m", "scene_recognition.detector_module.augment_yolo_dataset", "--data", str(base_data), "--output", str(base_aug), *augment_args])
    commands.append([python, "-m", "scene_recognition.detector_module.train_detector", "--data", str(base_aug / "data.yaml"), "--model", str(generic_model), "--epochs", str(base_epochs), "--project", str(runs), "--name", "base_four", "--device", device, "--seed", str(seed), "--no-builtin-aug"])
    commands.append([python, "-m", "scene_recognition.detector_module.augment_yolo_dataset", "--data", str(increment_data), "--output", str(increment_aug), *augment_args])
    prepare_command = [python, "-m", "scene_recognition.detector_module.prepare_batch_incremental_dataset", "--base-data", str(base_aug / "data.yaml"), "--increment-data", str(increment_aug / "data.yaml"), "--output", str(prepared), "--seed", str(seed)]
    if batch_plan is not None:
        prepare_command += ["--batch-plan", str(Path(batch_plan).resolve())]
    else:
        prepare_command += ["--num-batches", str(num_batches)]
    for value in dict.fromkeys(int(item) for item in buffer_sizes):
        prepare_command += ["--buffer-size", str(value)]
    if max_current_images_per_class is not None:
        prepare_command += ["--max-current-images-per-class", str(max_current_images_per_class)]
    commands.append(prepare_command)
    train_commands: dict[str, list[str]] = {}
    for value in dict.fromkeys(int(item) for item in buffer_sizes):
        output = runs / f"batch_il_{method}_{value}"
        command = [python, "-m", "scene_recognition.detector_module.train_batch_incremental_yolo", "--prepared", str(prepared), "--initial-checkpoint", str(runs / "base_four" / "weights" / "best.pt"), "--method", method, "--buffer-size", str(value), "--output", str(output), "--epochs", str(increment_epochs), "--seed", str(seed), "--device", device, "--no-builtin-aug"]
        if sparse_moe:
            command.append("--sparse-moe")
            for flag, option_name in (
                ("--expert-count", "expert_count"),
                ("--top-k", "top_k"),
                ("--expert-bottleneck", "expert_bottleneck"),
                ("--router-hidden", "router_hidden"),
                ("--aux-hidden", "aux_hidden"),
                ("--modality-loss-weight", "modality_loss_weight"),
                ("--scene-loss-weight", "scene_loss_weight"),
                ("--balance-loss-weight", "balance_loss_weight"),
                ("--router-z-loss-weight", "router_z_loss_weight"),
                ("--anchor-loss-weight", "anchor_loss_weight"),
                ("--anchor-rho", "anchor_rho"),
                ("--router-temperature-start", "router_temperature_start"),
                ("--router-temperature-end", "router_temperature_end"),
                ("--router-temperature-warmup-epochs", "router_temperature_warmup_epochs"),
            ):
                if sparse_options and option_name in sparse_options:
                    command += [flag, str(sparse_options[option_name])]
            if sparse_options and sparse_options.get("router_temperature") is not None:
                command += ["--router-temperature", str(sparse_options["router_temperature"])]
        train_commands[str(value)] = command
        commands.append(command)
    return {
        "scenario": "four_to_six_batch_incremental_pipeline",
        "offline": True,
        "base_data": str(base_data),
        "increment_data": str(increment_data),
        "generic_model": str(generic_model),
        "workspace": str(workspace),
        "base_augmentation": str(base_aug),
        "increment_augmentation": str(increment_aug),
        "prepared": str(prepared),
        "num_batches": len(plan),
        "plan": {"batches": plan, "source": str(Path(batch_plan).resolve()) if batch_plan else "deterministic_default"},
        "method": method,
        "seed": seed,
        "sparse_moe": sparse_moe,
        "buffer_sizes": list(dict.fromkeys(int(item) for item in buffer_sizes)),
        "base_epochs": base_epochs,
        "increment_epochs": increment_epochs,
        "commands": commands,
        "training_commands": train_commands,
        "audit": {"base_taxonomy": BASE_CLASS_NAMES, "increment_taxonomy": ALL_CLASS_NAMES, "built_in_augmentation": "disabled", "validation": "val", "test": "after_final_batch_only"},
    }


def run_pipeline(args: argparse.Namespace) -> dict:
    generic_arg = getattr(args, "generic_model", None) or getattr(args, "initial_model", None)
    if generic_arg is None:
        raise ValueError("必须提供本地通用初始模型")
    base_data, increment_data, generic_model, workspace = (Path(value).resolve() for value in (args.base_data, args.increment_data, generic_arg, args.workspace))
    buffer_sizes = tuple(getattr(args, "buffer_sizes", None) or (200,))
    sparse_options = {
        "expert_count": getattr(args, "expert_count", 5),
        "top_k": getattr(args, "top_k", 2),
        "expert_bottleneck": getattr(args, "expert_bottleneck", 0.25),
        "router_hidden": getattr(args, "router_hidden", 128),
        "aux_hidden": getattr(args, "aux_hidden", 128),
        "modality_loss_weight": getattr(args, "modality_loss_weight", 0.10),
        "scene_loss_weight": getattr(args, "scene_loss_weight", 0.10),
        "balance_loss_weight": getattr(args, "balance_loss_weight", 0.01),
        "router_z_loss_weight": getattr(args, "router_z_loss_weight", 0.001),
        "anchor_loss_weight": getattr(args, "anchor_loss_weight", 0.001),
        "anchor_rho": getattr(args, "anchor_rho", 0.95),
        "router_temperature": getattr(args, "router_temperature", None),
        "router_temperature_start": getattr(args, "router_temperature_start", 2.0),
        "router_temperature_end": getattr(args, "router_temperature_end", 1.0),
        "router_temperature_warmup_epochs": getattr(args, "router_temperature_warmup_epochs", 3),
    }
    plan = build_pipeline_plan(base_data, increment_data, generic_model, workspace, num_batches=getattr(args, "num_batches", None), batch_plan=getattr(args, "batch_plan", None), buffer_sizes=buffer_sizes, max_current_images_per_class=getattr(args, "max_current_images_per_class", None), base_epochs=getattr(args, "base_epochs", 60), increment_epochs=getattr(args, "increment_epochs", 30), method=getattr(args, "method", "der"), device=getattr(args, "device", "cpu"), seed=getattr(args, "seed", 42), default_modality=getattr(args, "default_modality", None), sparse_moe=getattr(args, "sparse_moe", False), sparse_options=sparse_options)
    if getattr(args, "dry_run", False) or getattr(args, "plan_only", False):
        plan["status"] = "dry_run_ok"
        return plan
    if workspace.exists() and any(workspace.iterdir()):
        raise FileExistsError(f"workspace 非空，请选择新目录: {workspace}")
    workspace.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["YOLO_OFFLINE"] = "true"
    for command in plan["commands"]:
        subprocess.run(command, cwd=Path(__file__).resolve().parents[2], env=environment, check=True)
    plan["status"] = "complete"
    plan["base_checkpoint"] = str((workspace / "runs" / "base_four" / "weights" / "best.pt").resolve())
    (workspace / "pipeline_plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return plan


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="四类离线增广/训练 → 六类增量离线增广/任意批次 ER/DER")
    parser.add_argument("--base-data", type=Path, required=True, help="原始四类 data.yaml")
    parser.add_argument("--increment-data", type=Path, required=True, help="原始六类 data.yaml")
    parser.add_argument("--generic-model", "--initial-model", dest="generic_model", type=Path, required=True, help="本地通用初始模型，不联网下载")
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--num-batches", type=int)
    parser.add_argument("--batch-plan", type=Path)
    parser.add_argument("--max-current-images-per-class", type=int)
    parser.add_argument("--buffer-size", type=int, action="append", dest="buffer_sizes", choices=BUFFER_SIZE_CHOICES, default=None)
    parser.add_argument("--method", choices=("er", "der"), default="der")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base-epochs", type=int, default=60)
    parser.add_argument("--increment-epochs", type=int, default=30)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--default-modality", choices=("ir", "sar"))
    parser.add_argument("--sparse-moe", action="store_true")
    parser.add_argument("--expert-count", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--expert-bottleneck", type=float, default=0.25)
    parser.add_argument("--router-hidden", type=int, default=128)
    parser.add_argument("--aux-hidden", type=int, default=128)
    parser.add_argument("--modality-loss-weight", type=float, default=0.10)
    parser.add_argument("--scene-loss-weight", type=float, default=0.10)
    parser.add_argument("--balance-loss-weight", type=float, default=0.01)
    parser.add_argument("--router-z-loss-weight", type=float, default=0.001)
    parser.add_argument("--anchor-loss-weight", type=float, default=0.001)
    parser.add_argument("--anchor-rho", type=float, default=0.95)
    parser.add_argument("--router-temperature", type=float)
    parser.add_argument("--router-temperature-start", type=float, default=2.0)
    parser.add_argument("--router-temperature-end", type=float, default=1.0)
    parser.add_argument("--router-temperature-warmup-epochs", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.buffer_sizes = tuple(args.buffer_sizes or (200,))
    summary = run_pipeline(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


run_four_to_six_pipeline = run_pipeline
build_pipeline_commands = build_pipeline_plan


if __name__ == "__main__":
    raise SystemExit(main())
