from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from Agent.continual.manager import ContinualLearningManager
from Agent.continual.protocols import load_protocol
from Agent.models.yolo_p2 import write_yolov8n_p2_yaml


DEFAULT_PROTOCOL = Path("Agent/configs/tasks/default_incremental_protocol.json")
DEFAULT_WORKSPACE = Path("Agent/runs/continual_workspace")
DEFAULT_CLASS_COUNT = 4


def _manager(args: argparse.Namespace) -> ContinualLearningManager:
    protocol = load_protocol(args.protocol)
    return ContinualLearningManager(
        workspace=args.workspace,
        protocol=protocol,
        replay_capacity=args.replay_capacity,
    )


def _print(payload: dict) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def plan(args: argparse.Namespace) -> None:
    manager = _manager(args)
    training_plan = manager.build_training_plan(
        index_csv=args.index,
        task_id=args.task_id,
        replay_limit=args.replay_limit,
    )
    if args.output:
        manager.save_training_plan(training_plan, args.output)
    payload = training_plan.to_dict() if args.verbose else training_plan.summary()
    if args.output:
        payload["plan_json"] = str(args.output)
    _print(payload)


def prepare_task(args: argparse.Namespace) -> None:
    manager = _manager(args)
    training_plan = manager.build_training_plan(
        index_csv=args.index,
        task_id=args.task_id,
        replay_limit=args.replay_limit,
    )
    task_dir = args.output_dir or args.workspace / "tasks" / args.task_id
    plan_path = task_dir / "training_plan.json"
    manager.save_training_plan(training_plan, plan_path)
    asset_summary = manager.export_training_assets(
        training_plan,
        task_dir / "dataset",
        include_replay=not args.no_replay,
        copy_images=not args.manifest_only,
    )
    model_yaml = write_yolov8n_p2_yaml(task_dir / "model" / "yolov8n_p2.yaml", class_count=DEFAULT_CLASS_COUNT)
    _print(
        {
            "task": training_plan.summary(),
            "plan_json": str(plan_path),
            "dataset": asset_summary,
            "model_yaml": str(model_yaml),
            "next_step": (
                "Train with Ultralytics using the generated data.yaml and yolov8n_p2.yaml, "
                "then save best.pt/teacher.pt under the task version directory."
            ),
        }
    )


def train_task(args: argparse.Namespace) -> None:
    task_dir = args.task_dir or args.workspace / "tasks" / args.task_id
    data_yaml = args.data or task_dir / "dataset" / "data.yaml"
    model_yaml = args.model or task_dir / "model" / "yolov8n_p2.yaml"
    project = args.project or args.workspace / "versions"
    run_name = args.name or args.task_id
    command = [
        sys.executable,
        "-m",
        "scene_recognition.detector_module.train_detector_ablation",
        "--data",
        str(data_yaml),
        "--model",
        str(model_yaml),
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
        str(args.device),
        "--project",
        str(project),
        "--name",
        run_name,
        "--eval-split",
        args.eval_split,
    ]
    if args.exist_ok:
        command.append("--exist-ok")
    payload = {
        "task_id": args.task_id,
        "data_yaml": str(data_yaml),
        "model_yaml": str(model_yaml),
        "project": str(project),
        "run_name": run_name,
        "command": command,
        "dry_run": not args.run,
    }
    if not args.run:
        _print(payload)
        return
    completed = subprocess.run(command, cwd=Path.cwd())
    payload["returncode"] = completed.returncode
    _print(payload)
    raise SystemExit(completed.returncode)


def update_replay(args: argparse.Namespace) -> None:
    manager = _manager(args)
    records = manager.load_scene_index(args.index)
    stage = manager.protocol.get_stage(args.task_id)
    current_records = manager.filter_records(records, stage)
    summary = manager.update_replay_from_records(
        current_records,
        task_id=args.task_id,
        copy_files=args.copy_files,
    )
    if args.manifest:
        manager.replay.export_manifest(args.manifest)
        summary["manifest"] = str(args.manifest)
    _print(summary)


def summary(args: argparse.Namespace) -> None:
    manager = _manager(args)
    _print(manager.replay.summary())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Continual-learning helper for Agent")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
        p.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
        p.add_argument("--replay-capacity", type=int, default=200)

    p_plan = sub.add_parser("plan", help="build a training plan for one task")
    add_common(p_plan)
    p_plan.add_argument("--index", type=Path, required=True)
    p_plan.add_argument("--task-id", required=True)
    p_plan.add_argument("--replay-limit", type=int, default=64)
    p_plan.add_argument("--output", type=Path)
    p_plan.add_argument("--verbose", action="store_true", help="print full sample lists to terminal")
    p_plan.set_defaults(func=plan)

    p_prepare = sub.add_parser("prepare-task", help="export training plan, task dataset, and YOLOv8n-P2 YAML")
    add_common(p_prepare)
    p_prepare.add_argument("--index", type=Path, required=True)
    p_prepare.add_argument("--task-id", required=True)
    p_prepare.add_argument("--replay-limit", type=int, default=64)
    p_prepare.add_argument("--output-dir", type=Path)
    p_prepare.add_argument("--no-replay", action="store_true", help="export current task samples only")
    p_prepare.add_argument(
        "--manifest-only",
        action="store_true",
        help="do not copy images or write filtered labels; for preview only",
    )
    p_prepare.set_defaults(func=prepare_task)

    p_train = sub.add_parser("train-task", help="print or run the YOLO training command for a prepared task")
    add_common(p_train)
    p_train.add_argument("--task-id", required=True)
    p_train.add_argument("--task-dir", type=Path, help="prepared task directory; defaults to workspace/tasks/<task-id>")
    p_train.add_argument("--data", type=Path, help="override generated data.yaml")
    p_train.add_argument("--model", type=Path, help="override generated model YAML")
    p_train.add_argument("--project", type=Path, help="override training output project")
    p_train.add_argument("--name", help="override run name")
    p_train.add_argument("--epochs", type=int, default=80)
    p_train.add_argument("--patience", type=int, default=20)
    p_train.add_argument("--image-size", type=int, default=768)
    p_train.add_argument("--batch-size", type=int, default=8)
    p_train.add_argument("--workers", type=int, default=2)
    p_train.add_argument("--device", default="0")
    p_train.add_argument("--eval-split", choices=["val", "test"], default="val")
    p_train.add_argument("--exist-ok", action="store_true")
    p_train.add_argument("--run", action="store_true", help="actually start training; default only prints command")
    p_train.set_defaults(func=train_task)

    p_replay = sub.add_parser("update-replay", help="score and store current task samples into replay buffer")
    add_common(p_replay)
    p_replay.add_argument("--index", type=Path, required=True)
    p_replay.add_argument("--task-id", required=True)
    p_replay.add_argument("--copy-files", action="store_true", help="materialize image/label copies inside replay workspace")
    p_replay.add_argument("--manifest", type=Path, help="optional replay CSV manifest output")
    p_replay.set_defaults(func=update_replay)

    p_summary = sub.add_parser("summary", help="show replay buffer summary")
    add_common(p_summary)
    p_summary.set_defaults(func=summary)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
