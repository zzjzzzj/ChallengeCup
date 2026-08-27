from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

from .agent import IntelligentRecognitionAgent
from .config import AgentConfig, PROJECT_ROOT
from .losses import combine_training_losses
from .memory import EpisodeMemory


# Bridge to train.py COMMANDS so preparation/training is reachable from Agent CLI.
TRAIN_BRIDGES: dict[str, str] = {
    "prepare-scene": "scene-prepare",
    "extract-scene-features": "scene-extract",
    "evaluate-scene": "scene-evaluate",
    "prepare-crops": "crop-prepare",
    "prepare-detection": "prepare-detection",
    "prepare-comparison": "prepare-comparison",
    "prepare-continual": "prepare-continual",
    "prepare-class-il": "prepare-class-il",
    "train-detector": "yolo",
    "train-continual": "continual-yolo",
    "train-class-il": "class-il-yolo",
    "evaluate-continual": "continual-evaluate",
    "train-resnet-detector": "resnet-detector",
    "train-target": "crop-classifier",
    "train-whole-target": "whole-classifier",
    "image-processing": "image-processing",
    "scene-recognition": "scene-recognition",
}


def _write_or_print(payload: dict, output: Path | None) -> None:
    """统一处理命令输出：既打印到终端，也可写入JSON文件。"""

    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    print(text)


def _config_from_args(args: argparse.Namespace) -> AgentConfig:
    """把命令行参数转换成AgentConfig。"""

    return AgentConfig.from_values(
        scene_model=args.scene_model,
        scene_metadata=args.scene_metadata,
        scene_cnn_checkpoint=getattr(args, "scene_cnn_checkpoint", None),
        calibration=getattr(args, "calibration", None),
        detector_model=args.detector_model,
        target_checkpoint=args.target_checkpoint,
        memory_path=args.memory,
        scene_threshold=args.scene_threshold,
        detector_confidence=args.detector_confidence,
        image_size=args.image_size,
        device=args.device,
        allow_label_fallback=not args.no_label_fallback,
        remember_runs=not args.no_memory,
    )


def infer(args: argparse.Namespace) -> None:
    """单图推理入口，对应演示时最常用的一条命令。"""

    config = _config_from_args(args)
    modalities = {
        key: value
        for key, value in {
            "visible": args.visible,
            "ir": args.ir,
            "sar": args.sar,
        }.items()
        if value
    }
    agent = IntelligentRecognitionAgent(config)
    report = agent.run(
        args.image,
        sensor_hint=args.sensor,
        modality_images=modalities,
        remember=not args.no_memory,
    )
    _write_or_print(report.to_dict(), args.output)


def batch(args: argparse.Namespace) -> None:
    """批量推理入口。

    manifest可以直接使用image_processing生成的scene_index.csv。
    """

    config = _config_from_args(args)
    agent = IntelligentRecognitionAgent(config)
    rows = list(csv.DictReader(args.manifest.open("r", encoding="utf-8-sig", newline="")))
    if not rows:
        raise ValueError(f"manifest is empty: {args.manifest}")
    if args.split:
        rows = [row for row in rows if row.get("split") == args.split]
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError("no rows matched the requested batch filters")
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    for index, row in enumerate(rows, start=1):
        image = row.get("image") or row.get("image_path")
        if not image:
            raise ValueError("manifest must contain image or image_path column")
        report = agent.run(image, sensor_hint=row.get("sensor") or None, remember=not args.no_memory)
        report_dict = report.to_dict()
        report_path = output_dir / f"{Path(image).stem}_{index:04d}.json"
        report_path.write_text(json.dumps(report_dict, ensure_ascii=False, indent=2), encoding="utf-8")
        summary_rows.append(
            {
                "image": report_dict["image"],
                "report": str(report_path),
                "modality": report_dict["modality"]["label"],
                "scene": report_dict["final_scene"]["label"],
                "target_count": len(report_dict["detections"]),
                "consistency": report_dict["consistency"]["status"],
                "loss_total": report_dict["losses"]["total"],
            }
        )
    summary_path = output_dir / "batch_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    _write_or_print({"count": len(summary_rows), "summary": str(summary_path)}, None)


def feedback(args: argparse.Namespace) -> None:
    """人工反馈入口。"""

    memory = EpisodeMemory(args.memory)
    targets = [item.strip() for item in (args.targets or "").split(",") if item.strip()]
    memory.append_feedback(
        image=str(Path(args.image).expanduser().resolve()),
        corrected_scene=args.scene,
        corrected_modality=args.modality,
        corrected_targets=targets,
        note=args.note,
    )
    _write_or_print({"status": "ok", "memory": memory.summary()}, None)


def loss(args: argparse.Namespace) -> None:
    """损失公式计算入口。"""

    payload = combine_training_losses(
        l_box=args.l_box,
        l_cls=args.l_cls,
        l_dfl=args.l_dfl,
        l_detail=args.l_detail,
        l_scene=args.l_scene,
        l_proto=args.l_proto,
        l_moti=args.l_moti,
        lambda_detail=args.lambda_detail,
        lambda_scene=args.lambda_scene,
        lambda_proto=args.lambda_proto,
        lambda_moti=args.lambda_moti,
    )
    _write_or_print(payload, args.output)


def run_train_bridge(args: argparse.Namespace) -> None:
    """Forward remaining args to ``python train.py <mapped-command> ...``."""

    train_command = TRAIN_BRIDGES[args.command]
    train_script = PROJECT_ROOT / "train.py"
    forwarded = list(getattr(args, "forwarded", []) or [])
    # argparse.REMAINDER keeps the conventional separator. It is only for the
    # Agent parser and must not be forwarded to the child parser, otherwise the
    # child treats every following option as a positional value.
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]
    completed = subprocess.run(
        [sys.executable, str(train_script), train_command, *forwarded],
        cwd=PROJECT_ROOT,
    )
    raise SystemExit(completed.returncode)


def add_common_runtime_args(parser: argparse.ArgumentParser) -> None:
    """给infer/batch添加共享运行参数。"""

    parser.add_argument("--scene-model", type=Path, help="feature SVM/joblib scene model")
    parser.add_argument("--scene-metadata", type=Path, help="scene model metadata JSON")
    parser.add_argument("--scene-cnn-checkpoint", type=Path, help="optional CNN scene checkpoint")
    parser.add_argument("--calibration", type=Path, help="optional quality calibration JSON")
    parser.add_argument("--detector-model", type=Path, help="YOLO .pt detector model")
    parser.add_argument("--target-checkpoint", type=Path, help="ResNet18 crop classifier checkpoint")
    parser.add_argument("--memory", type=Path, help="agent JSONL memory path")
    parser.add_argument("--scene-threshold", type=float, default=None)
    parser.add_argument("--detector-confidence", type=float, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--device", default=None, help="auto/cpu/cuda/0")
    parser.add_argument("--no-label-fallback", action="store_true")
    parser.add_argument("--no-memory", action="store_true")


def _add_bridge_parser(sub: argparse._SubParsersAction, name: str, help_text: str) -> None:
    bridge = sub.add_parser(
        name,
        help=help_text,
        description=f"Bridge to `python train.py {TRAIN_BRIDGES[name]} ...`",
    )
    bridge.add_argument(
        "forwarded",
        nargs=argparse.REMAINDER,
        help="arguments forwarded to train.py (include a leading --)",
    )
    bridge.set_defaults(func=run_train_bridge)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ChallengeCup intelligent recognition agent")
    sub = parser.add_subparsers(dest="command", required=True)

    infer_parser = sub.add_parser("infer", help="run the full flow on one image")
    infer_parser.add_argument("--image", type=Path, required=True)
    infer_parser.add_argument("--sensor", choices=["visible", "ir", "sar"], help="optional modality hint")
    infer_parser.add_argument("--visible", type=Path, help="aligned visible image")
    infer_parser.add_argument("--ir", type=Path, help="aligned IR image")
    infer_parser.add_argument("--sar", type=Path, help="aligned SAR image")
    infer_parser.add_argument("--output", type=Path)
    add_common_runtime_args(infer_parser)
    infer_parser.set_defaults(func=infer)

    batch_parser = sub.add_parser("batch", help="run reports for a CSV manifest")
    batch_parser.add_argument("--manifest", type=Path, required=True)
    batch_parser.add_argument("--output-dir", type=Path, required=True)
    batch_parser.add_argument("--split", choices=["train", "val", "test"], help="optional split filter")
    batch_parser.add_argument("--limit", type=int, help="optional maximum number of rows to run")
    add_common_runtime_args(batch_parser)
    batch_parser.set_defaults(func=batch)

    feedback_parser = sub.add_parser("feedback", help="append human correction to memory")
    feedback_parser.add_argument("--image", type=Path, required=True)
    feedback_parser.add_argument("--scene", choices=["air", "sea", "urban", "forest"])
    feedback_parser.add_argument("--modality", choices=["visible", "ir", "sar"])
    feedback_parser.add_argument("--targets", help="comma-separated target labels")
    feedback_parser.add_argument("--note")
    feedback_parser.add_argument("--memory", type=Path, default=AgentConfig().memory_path)
    feedback_parser.set_defaults(func=feedback)

    loss_parser = sub.add_parser("loss", help="combine training losses with the project formula")
    loss_parser.add_argument("--l-box", type=float, required=True)
    loss_parser.add_argument("--l-cls", type=float, required=True)
    loss_parser.add_argument("--l-dfl", type=float, default=0.0)
    loss_parser.add_argument("--l-detail", type=float, default=0.0)
    loss_parser.add_argument("--l-scene", type=float, default=0.0)
    loss_parser.add_argument("--l-proto", type=float, default=0.0)
    loss_parser.add_argument("--l-moti", type=float, default=0.0)
    loss_parser.add_argument("--lambda-detail", type=float, default=0.2)
    loss_parser.add_argument("--lambda-scene", type=float, default=0.6)
    loss_parser.add_argument("--lambda-proto", type=float, default=0.4)
    loss_parser.add_argument("--lambda-moti", type=float, default=0.3)
    loss_parser.add_argument("--output", type=Path)
    loss_parser.set_defaults(func=loss)

    _add_bridge_parser(sub, "prepare-scene", "bridge to train.py scene-prepare")
    _add_bridge_parser(sub, "extract-scene-features", "bridge to train.py scene-extract")
    _add_bridge_parser(sub, "evaluate-scene", "bridge to train.py scene-evaluate")
    _add_bridge_parser(sub, "prepare-crops", "bridge to train.py crop-prepare")
    _add_bridge_parser(sub, "prepare-detection", "bridge to train.py prepare-detection")
    _add_bridge_parser(sub, "prepare-comparison", "bridge to train.py prepare-comparison")
    _add_bridge_parser(sub, "prepare-continual", "prepare a local r2 continual-learning protocol")
    _add_bridge_parser(sub, "prepare-class-il", "prepare six singleton Class-IL stages")
    _add_bridge_parser(sub, "train-detector", "bridge to train.py yolo")
    _add_bridge_parser(sub, "train-continual", "fine-tune a local checkpoint on an incremental round")
    _add_bridge_parser(sub, "train-class-il", "run six-stage ER or DER Class-IL training")
    _add_bridge_parser(sub, "evaluate-continual", "report New-mAP, old-class mAP and KRR")
    _add_bridge_parser(sub, "train-resnet-detector", "bridge to train.py resnet-detector")
    _add_bridge_parser(sub, "train-target", "bridge to train.py crop-classifier")
    _add_bridge_parser(sub, "train-whole-target", "bridge to train.py whole-classifier")
    _add_bridge_parser(sub, "image-processing", "bridge to train.py image-processing")
    _add_bridge_parser(sub, "scene-recognition", "bridge to train.py scene-recognition")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
