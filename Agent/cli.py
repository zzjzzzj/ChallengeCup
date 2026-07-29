from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .agent import IntelligentRecognitionAgent
from .config import AgentConfig
from .losses import combine_training_losses
from .memory import EpisodeMemory


def _write_or_print(payload: dict, output: Path | None) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    print(text)


def _config_from_args(args: argparse.Namespace) -> AgentConfig:
    return AgentConfig.from_values(
        scene_model=args.scene_model,
        scene_metadata=args.scene_metadata,
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
    config = _config_from_args(args)
    agent = IntelligentRecognitionAgent(config)
    rows = list(csv.DictReader(args.manifest.open("r", encoding="utf-8-sig", newline="")))
    if not rows:
        raise ValueError(f"manifest is empty: {args.manifest}")
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


def add_common_runtime_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--scene-model", type=Path, help="feature SVM/joblib scene model")
    parser.add_argument("--scene-metadata", type=Path, help="scene model metadata JSON")
    parser.add_argument("--detector-model", type=Path, help="YOLO .pt detector model")
    parser.add_argument("--target-checkpoint", type=Path, help="ResNet18 crop classifier checkpoint")
    parser.add_argument("--memory", type=Path, help="agent JSONL memory path")
    parser.add_argument("--scene-threshold", type=float, default=None)
    parser.add_argument("--detector-confidence", type=float, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--device", default=None, help="auto/cpu/cuda/0")
    parser.add_argument("--no-label-fallback", action="store_true")
    parser.add_argument("--no-memory", action="store_true")


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

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
