"""Train the arbitrary-batch four-to-six class incremental protocol.

The dry-run path is intentionally dependency-light: it validates only local
manifests, YAML taxonomies, replay provenance and the batch plan, and does not
import Ultralytics or touch the network.  Real training updates one six-class
student per batch and always selects checkpoints on val; test is evaluated only
after the final batch.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import yaml

from scene_recognition.detector_module import ALL_CLASS_NAMES, BASE_CLASS_NAMES
from scene_recognition.detector_module.boxes import parse_yolo_boxes, resolve_label_path
from scene_recognition.detector_module.metrics import detection_metrics_to_dict
from scene_recognition.detector_module.prepare_batch_incremental_dataset import BUFFER_SIZE_CHOICES
from scene_recognition.detector_module.train_detector import (
    BUILTIN_AUGMENTATION,
    DISABLED_AUGMENTATION,
    default_device,
    import_training_dependencies,
    maybe_import_torch_npu,
)


def _names(config: dict) -> list[str]:
    values = config.get("names")
    if isinstance(values, dict):
        return [str(values[index] if index in values else values[str(index)]) for index in range(len(values))]
    if isinstance(values, list):
        return [str(value) for value in values]
    raise ValueError("训练 YAML 缺少合法 names")


def _normalise_names(value: object) -> list[str]:
    if isinstance(value, dict):
        return [str(value[index] if index in value else value[str(index)]) for index in range(len(value))]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return []


def _load_summary(prepared: Path) -> dict:
    path = prepared / "batch_incremental_dataset_summary.json" if prepared.is_dir() else prepared
    if not path.is_file():
        raise FileNotFoundError(path)
    summary = json.loads(path.read_text(encoding="utf-8-sig"))
    if summary.get("scenario") != "batch_class_incremental":
        raise ValueError("当前训练器只接受 scenario=batch_class_incremental")
    if summary.get("base_classes") != BASE_CLASS_NAMES or summary.get("task_order") != ALL_CLASS_NAMES:
        raise ValueError("prepared 协议的 taxonomy 不是严格四类前缀 + 六类固定顺序")
    batches = summary.get("batches")
    if not isinstance(batches, list) or not batches:
        raise ValueError("prepared 协议没有 batches")
    return summary


def _read_manifest(path: Path) -> list[Path]:
    if not path.is_file():
        raise FileNotFoundError(path)
    paths = [Path(line.strip()).resolve() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"清单包含不存在的图像: {missing[0]}")
    return paths


def _yaml_names(path: Path) -> tuple[dict, list[str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    config = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    if not isinstance(config, dict):
        raise ValueError(f"训练 YAML 必须是映射: {path}")
    names = _names(config)
    if names != ALL_CLASS_NAMES or int(config.get("nc", len(names))) != len(ALL_CLASS_NAMES):
        raise ValueError(f"批次 YAML 必须固定 nc=6/names={ALL_CLASS_NAMES}: {path}")
    return config, names


def _within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _validate_labels(manifest: Path, allowed: set[int]) -> None:
    """Validate labels without importing Ultralytics (used in dry-run)."""

    image_paths = _read_manifest(manifest)
    for image in image_paths:
        label = resolve_label_path(image)
        for box in parse_yolo_boxes(label, len(ALL_CLASS_NAMES)):
            if box.class_id not in allowed:
                raise ValueError(f"未来类别标签泄漏到当前视图 {label}: {box.class_id}")


def _sparse_config_dict(args: argparse.Namespace) -> dict[str, Any]:
    """Return Sparse-MoE settings without importing Ultralytics."""

    fixed_temperature = getattr(args, "router_temperature", None)
    return {
        "expert_count": int(getattr(args, "expert_count", 5)),
        "top_k": int(getattr(args, "top_k", 2)),
        "expert_bottleneck": float(getattr(args, "expert_bottleneck", 0.25)),
        "router_hidden": int(getattr(args, "router_hidden", 128)),
        "aux_hidden": int(getattr(args, "aux_hidden", 128)),
        "modality_loss_weight": float(getattr(args, "modality_loss_weight", 0.10)),
        "scene_loss_weight": float(getattr(args, "scene_loss_weight", 0.10)),
        "balance_loss_weight": float(getattr(args, "balance_loss_weight", 0.01)),
        "router_z_loss_weight": float(getattr(args, "router_z_loss_weight", 0.001)),
        "anchor_loss_weight": float(getattr(args, "anchor_loss_weight", 0.001)),
        "anchor_rho": float(getattr(args, "anchor_rho", 0.95)),
        "router_temperature_start": float(fixed_temperature if fixed_temperature is not None else getattr(args, "router_temperature_start", 2.0)),
        "router_temperature_end": float(fixed_temperature if fixed_temperature is not None else getattr(args, "router_temperature_end", 1.0)),
        "router_temperature_warmup_epochs": int(getattr(args, "router_temperature_warmup_epochs", 3)),
    }


def validate_prepared_protocol(summary: dict, buffer_size: int) -> list[dict]:
    """Validate every prepared batch and return the selected batch plan."""

    if buffer_size not in BUFFER_SIZE_CHOICES:
        raise ValueError(f"buffer-size 只允许 {BUFFER_SIZE_CHOICES}")
    base_root = Path(summary["base_data"]).resolve().parent
    selected: list[dict] = []
    previous_seen: set[str] = set(BASE_CLASS_NAMES)
    for index, batch in enumerate(summary["batches"], start=1):
        buffers = batch.get("buffers", {})
        entry = buffers.get(str(buffer_size))
        if not isinstance(entry, dict):
            raise ValueError(f"批次 {batch.get('id')} 缺少 buffer={buffer_size}")
        data_yaml = Path(entry["data_yaml"])
        config, _ = _yaml_names(data_yaml)
        train_manifest = Path(entry["training"]["manifest"])
        replay_manifest = Path(entry["training"]["replay_manifest"])
        _validate_labels(train_manifest, {ALL_CLASS_NAMES.index(name) for name in batch["seen"]})
        _validate_labels(Path(entry["validation"]["manifest"]), {ALL_CLASS_NAMES.index(name) for name in batch["seen"]})
        if entry.get("test"):
            _validate_labels(Path(entry["test"]["manifest"]), {ALL_CLASS_NAMES.index(name) for name in batch["seen"]})
        replay_paths = _read_manifest(replay_manifest)
        expected_replay = int(entry["replay_before"]["images"])
        if len(replay_paths) != expected_replay or expected_replay > buffer_size:
            raise ValueError(f"批次 {batch.get('id')} replay 数量与容量不一致")
        if index == 1:
            if not replay_paths:
                raise ValueError("第一批必须存在来自 base train 的 replay")
            base_train_root = Path(summary["base_data"]).resolve().parent
            # The materialized replay image names live under prepared/, while
            # their source paths are auditable in entries.  Ensure every entry
            # points back into the base dataset for the first batch.
            entries = entry["replay_before"].get("entries", [])
            if len(entries) != expected_replay:
                raise ValueError("首批 replay_before entries 数量不一致")
            for replay_entry in entries:
                if not _within(Path(replay_entry["image_path"]), base_train_root):
                    raise ValueError("第一批 replay_before 含非 base 数据")
        requested = list(batch.get("requested", []))
        if not requested or any(name not in ALL_CLASS_NAMES for name in requested):
            raise ValueError(f"批次 {batch.get('id')} requested 非法")
        seen = set(batch.get("seen", []))
        if not set(BASE_CLASS_NAMES).issubset(seen):
            raise ValueError("每个批次 seen 必须包含四类 base")
        if not previous_seen.issubset(seen):
            raise ValueError("seen 类别不能回退")
        previous_seen = seen
        selected.append({
            "index": index,
            "id": str(batch["id"]),
            "requested": requested,
            "present": list(batch.get("present", [])),
            "missing": list(batch.get("missing", [])),
            "seen": list(batch.get("seen", [])),
            "newly_seen": list(batch.get("newly_seen", [])),
            "data_yaml": str(data_yaml.resolve()),
            "replay_manifest": str(replay_manifest.resolve()),
            "replay_images": len(replay_paths),
            "config": config,
            "test_manifest": str(Path(entry["test"]["manifest"]).resolve()) if entry.get("test") else None,
            "context_index": entry["training"].get("context_index"),
        })
    return selected


def _batch_metric_payload(
    results: list[dict], class_order: Sequence[str], metric_name: str
) -> dict:
    rows: list[dict[str, float | None]] = []
    arrivals: dict[str, int] = {}
    for index, result in enumerate(results, start=1):
        validation = result.get("validation") or {}
        per_class = validation.get("per_class", {})
        seen = set(result.get("seen", result.get("present", [])))
        row: dict[str, float | None] = {}
        for name in class_order:
            value = per_class.get(name, {}).get(metric_name)
            row[name] = float(value) if name in seen and value is not None else None
        rows.append(row)
        for name in seen:
            arrivals.setdefault(name, index)
    first_arrival = {name: arrivals[name] for name in class_order if name in arrivals}
    final_row = rows[-1] if rows else {}
    forgetting: dict[str, float] = {}
    bwt: dict[str, float] = {}
    for name, arrival in first_arrival.items():
        start = rows[arrival - 1].get(name)
        final = final_row.get(name)
        history = [row[name] for row in rows[arrival - 1 :] if row.get(name) is not None]
        if start is not None and final is not None:
            bwt[name] = final - start
            forgetting[name] = max(history) - final if history else 0.0
    stage_means = [
        sum(value for value in row.values() if value is not None)
        / len([value for value in row.values() if value is not None])
        for row in rows
        if any(value is not None for value in row.values())
    ]
    final_values = [value for value in final_row.values() if value is not None]
    return {
        "metric": metric_name,
        "class_order": list(class_order),
        "rows": rows,
        "first_arrival": first_arrival,
        "final_average": sum(final_values) / len(final_values) if final_values else None,
        "average_seen_accuracy": sum(stage_means) / len(stage_means) if stage_means else None,
        "forgetting": forgetting,
        "average_forgetting": sum(forgetting.values()) / len(forgetting) if forgetting else None,
        "backward_transfer": bwt,
        "average_backward_transfer": sum(bwt.values()) / len(bwt) if bwt else None,
    }


def build_batch_metrics(results: list[dict], class_order: Sequence[str] = ALL_CLASS_NAMES) -> dict:
    """Build both mAP@0.5 and mAP@0.5:0.95 batch matrices and drift metrics."""

    map50 = _batch_metric_payload(results, class_order, "map50")
    map50_95 = _batch_metric_payload(results, class_order, "map50_95")
    # Keep the map50 fields at the top level for simple consumers while
    # exposing a complete, parallel payload for the stricter AP metric.
    return {
        "class_order": list(class_order),
        "first_arrival": map50["first_arrival"],
        "map50": map50,
        "map50_95": map50_95,
        "rows": map50["rows"],
        "final_average": map50["final_average"],
        "average_seen_accuracy": map50["average_seen_accuracy"],
        "forgetting": map50["forgetting"],
        "average_forgetting": map50["average_forgetting"],
        "backward_transfer": map50["backward_transfer"],
        "average_backward_transfer": map50["average_backward_transfer"],
    }


def _make_der_trainer(context: Any):
    """Create DER trainer lazily, preserving the dry-run no-Ultralytics rule."""

    from scene_recognition.detector_module.dark_experience_replay import DarkReplayModel
    from ultralytics.models.yolo.detect import DetectionTrainer

    class ReplayAwareBatchDERTrainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            student = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
            return DarkReplayModel(
                context.teacher_checkpoint,
                student,
                context.replay_paths,
                der_weight=context.der_weight,
                cls_weight=context.cls_weight,
                box_weight=context.box_weight,
                min_confidence=context.min_confidence,
            )

        def set_model_attributes(self):
            super().set_model_attributes()
            model = self.model
            if isinstance(model, DarkReplayModel):
                model.student_model.nc = self.data["nc"]
                model.student_model.names = self.data["names"]
                model.student_model.args = self.args

        def get_validator(self):
            validator = super().get_validator()
            self.loss_names = "box_loss", "cls_loss", "dfl_loss", "der_loss"
            return validator

        def build_optimizer(self, model, name="auto", lr=0.001, momentum=0.9, decay=1e-5, iterations=1e5):
            trainable = model.student_model if isinstance(model, DarkReplayModel) else model
            return super().build_optimizer(trainable, name=name, lr=lr, momentum=momentum, decay=decay, iterations=iterations)

    return ReplayAwareBatchDERTrainer


def _training_kwargs(args: argparse.Namespace, data_yaml: Path, stage_name: str, output: Path, seed: int) -> dict:
    augmentation = DISABLED_AUGMENTATION if args.no_builtin_aug else BUILTIN_AUGMENTATION
    kwargs = {
        "data": str(data_yaml.resolve()), "epochs": args.epochs, "patience": args.patience,
        "imgsz": args.image_size, "batch": args.batch_size, "workers": args.workers,
        "device": args.device, "seed": seed, "deterministic": True, "optimizer": "AdamW",
        "lr0": args.learning_rate, "pretrained": True, "resume": False, "cache": False,
        "amp": not args.no_amp, "project": str(output.resolve()), "name": stage_name,
        "exist_ok": False, "plots": not args.no_plots, "verbose": True,
        "close_mosaic": 0 if args.no_builtin_aug else 10, "freeze": args.freeze,
    }
    kwargs.update(augmentation)
    return kwargs


def run_batch_incremental(args: argparse.Namespace) -> dict:
    os.environ.setdefault("YOLO_OFFLINE", "true")
    prepared = Path(args.prepared).resolve()
    initial_arg = getattr(args, "initial_checkpoint", None) or getattr(args, "initial_model", None)
    if initial_arg is None:
        raise ValueError("必须提供 --initial-checkpoint（四类 checkpoint）")
    initial = Path(initial_arg).resolve()
    if not initial.is_file():
        raise FileNotFoundError(f"四类 initial checkpoint 不存在: {initial}")
    if args.method not in {"er", "der"}:
        raise ValueError("method 必须为 er 或 der")
    summary = _load_summary(prepared)
    plan = validate_prepared_protocol(summary, args.buffer_size)
    stop_arg = getattr(args, "stop_after_batch", None)
    stop_after = len(plan) if stop_arg is None else int(stop_arg)
    if stop_after <= 0 or stop_after > len(plan):
        raise ValueError(f"stop-after-batch 必须位于 1..{len(plan)}")
    selected = plan[:stop_after]
    sparse_enabled = bool(getattr(args, "sparse_moe", False))
    if bool(getattr(args, "dry_run", False)):
        result = {
            "status": "dry_run_ok", "scenario": "batch_class_incremental", "method": args.method.upper(),
            "buffer_size": args.buffer_size, "initial_checkpoint": str(initial),
            "batches": selected, "stop_after_batch": stop_after,
            "der_first_batch_enabled": args.method == "der", "student_head": ALL_CLASS_NAMES,
            "test_used_for_checkpoint_selection": False, "offline": True,
        }
        if sparse_enabled:
            result.update({"sparse_moe": True, "sparse_moe_config": _sparse_config_dict(args)})
        return result

    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"输出目录非空，请选择新实验目录: {args.output}")
    if initial.suffix.casefold() != ".pt":
        raise ValueError("真实批次训练的 initial checkpoint 必须是本地 .pt 文件")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch, ultralytics, YOLO = import_training_dependencies()
    maybe_import_torch_npu(args.device)
    initial_probe = YOLO(str(initial))
    initial_names = _normalise_names(getattr(initial_probe, "names", None))
    if initial_names != BASE_CLASS_NAMES:
        raise ValueError(f"真实训练要求 initial checkpoint names 严格等于 {BASE_CLASS_NAMES}，实际为 {initial_names}")
    previous_checkpoint = initial
    stage_results: list[dict] = []
    started = time.perf_counter()
    for batch in selected:
        index, batch_id = int(batch["index"]), str(batch["id"])
        stage_started = time.perf_counter()
        model = YOLO(str(previous_checkpoint))
        kwargs = _training_kwargs(args, Path(batch["data_yaml"]), batch_id, args.output, args.seed + index - 1)
        replay_paths = _read_manifest(Path(batch["replay_manifest"]))
        der_context = None
        if args.method == "der":
            from dataclasses import dataclass
            @dataclass(frozen=True)
            class Context:
                teacher_checkpoint: Path
                replay_paths: tuple[Path, ...]
                der_weight: float
                cls_weight: float
                box_weight: float
                min_confidence: float
            der_context = Context(previous_checkpoint, tuple(replay_paths), args.der_weight, args.der_cls_weight, args.der_box_weight, args.der_min_confidence)
        if sparse_enabled:
            from scene_recognition.detector_module.train_class_incremental_yolo import build_sparse_moe_config
            from scene_recognition.detector_module.sparse_moe_model import get_sparse_moe_adapter
            from scene_recognition.detector_module.sparse_moe_checkpoint import update_sparse_moe_anchors, write_sparse_moe_artifacts, sparse_moe_metadata
            from scene_recognition.detector_module.sparse_moe_trainer import make_sparse_moe_trainer
            sparse_config = build_sparse_moe_config(args)
            model.train(trainer=make_sparse_moe_trainer(sparse_config, context_index=Path(batch["context_index"]) if batch.get("context_index") else None, der_context=der_context), **kwargs)
        elif der_context is not None:
            model.train(trainer=_make_der_trainer(der_context), **kwargs)
        else:
            model.train(**kwargs)
        run_dir = Path(model.trainer.save_dir)
        best = run_dir / "weights" / "best.pt"
        if not best.is_file():
            raise FileNotFoundError(f"批次 {batch_id} 没有生成 best.pt: {best}")
        trained = YOLO(str(best))
        validation_obj = trained.val(data=str(Path(batch["data_yaml"]).resolve()), split="val", imgsz=args.image_size, batch=args.batch_size, workers=args.workers, device=args.device, project=str(run_dir), name="batch_val", exist_ok=True, plots=not args.no_plots, verbose=False)
        validation = detection_metrics_to_dict(validation_obj, ALL_CLASS_NAMES)
        seen_values = [validation["per_class"][name]["map50"] for name in batch["seen"] if validation["per_class"].get(name, {}).get("map50") is not None]
        seen_mean = sum(seen_values) / len(seen_values) if seen_values else None
        test = None
        if index == len(plan) and batch.get("test_manifest"):
            test_obj = trained.val(data=str(Path(batch["data_yaml"]).resolve()), split="test", imgsz=args.image_size, batch=args.batch_size, workers=args.workers, device=args.device, project=str(run_dir), name="batch_test", exist_ok=True, plots=not args.no_plots, verbose=False)
            test = detection_metrics_to_dict(test_obj, ALL_CLASS_NAMES)
        stage = {
            "index": index, "id": batch_id, "requested": batch["requested"], "present": batch["present"], "missing": batch["missing"], "seen": batch["seen"], "newly_seen": batch["newly_seen"],
            "method": args.method.upper(), "buffer_capacity": args.buffer_size, "replay_images": len(replay_paths), "input_checkpoint": str(previous_checkpoint), "best_model": str(best.resolve()), "data_yaml": str(Path(batch["data_yaml"]).resolve()), "validation": validation, "test": test, "elapsed_seconds": round(time.perf_counter() - stage_started, 3),
            "seen_map50": seen_mean,
            "dark_targets": {"enabled": args.method == "der", "scope": "replay_samples_only", "source": "frozen_previous_batch_checkpoint"} if args.method == "der" else {"enabled": False},
        }
        if sparse_enabled:
            update_sparse_moe_anchors(trained.model)
            trained.save(best)
            stage["sparse_moe"] = sparse_moe_metadata(trained.model)
            stage["sparse_moe_artifacts"] = write_sparse_moe_artifacts(trained.model, run_dir, context_summary={})
        (run_dir / "batch_incremental_stage_summary.json").write_text(json.dumps(stage, ensure_ascii=False, indent=2), encoding="utf-8")
        stage_results.append(stage)
        previous_checkpoint = best.resolve()
    complete = stop_after == len(plan)
    result = {
        "status": "complete" if complete else "partial", "scenario": "batch_class_incremental", "method": args.method.upper(), "buffer_size": args.buffer_size, "initial_checkpoint": str(initial), "final_model": str(previous_checkpoint), "batches": stage_results, "metrics": build_batch_metrics(stage_results), "elapsed_seconds": round(time.perf_counter() - started, 3), "environment": {"finished_at": datetime.now().isoformat(), "python": sys.version, "platform": platform.platform(), "torch": torch.__version__, "ultralytics": ultralytics.__version__, "device": args.device}, "privacy": {"offline_environment_requested": True, "local_checkpoint_required": True, "dataset_upload": False}, "test_used_for_checkpoint_selection": False,
    }
    if sparse_enabled:
        from scene_recognition.detector_module.train_class_incremental_yolo import build_sparse_moe_config
        result.update({"sparse_moe": True, "sparse_moe_config": build_sparse_moe_config(args).to_dict()})
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "batch_incremental_training_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="四类 checkpoint → 任意批次六类 Class-IL：ER / DER")
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--initial-checkpoint", "--initial-model", dest="initial_checkpoint", type=Path, required=True)
    parser.add_argument("--method", choices=("er", "der"), required=True)
    parser.add_argument("--buffer-size", type=int, choices=BUFFER_SIZE_CHOICES, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--freeze", type=int, default=None)
    parser.add_argument("--der-weight", type=float, default=1.0)
    parser.add_argument("--der-cls-weight", type=float, default=1.0)
    parser.add_argument("--der-box-weight", type=float, default=0.25)
    parser.add_argument("--der-min-confidence", type=float, default=0.0)
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
    parser.add_argument("--router-temperature", type=float, default=None)
    parser.add_argument("--router-temperature-start", type=float, default=2.0)
    parser.add_argument("--router-temperature-end", type=float, default=1.0)
    parser.add_argument("--router-temperature-warmup-epochs", type=int, default=3)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--no-builtin-aug", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-after-batch", type=int)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run_batch_incremental(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
