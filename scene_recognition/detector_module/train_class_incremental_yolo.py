"""Run six sequential YOLO Class-IL stages with ER or DER replay."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from scene_recognition.detector_module import ALL_CLASS_NAMES
from scene_recognition.detector_module.context_metadata import (
    context_index_summary,
    read_context_rows,
    resolve_context_metadata,
)
from scene_recognition.detector_module.metrics import detection_metrics_to_dict
from scene_recognition.detector_module.prepare_class_incremental_dataset import BUFFER_SIZE_CHOICES
from scene_recognition.detector_module.sparse_moe_checkpoint import (
    restore_sparse_moe_usage,
    sparse_moe_metadata,
    sparse_moe_usage_state,
    update_sparse_moe_anchors,
    write_sparse_moe_artifacts,
)
from scene_recognition.detector_module.sparse_moe_model import SparseMoEConfig
from scene_recognition.detector_module.sparse_moe_trainer import make_sparse_moe_trainer
from scene_recognition.detector_module.train_detector import (
    BUILTIN_AUGMENTATION,
    DISABLED_AUGMENTATION,
    default_device,
    import_training_dependencies,
    maybe_import_torch_npu,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PREPARED = (
    PROJECT_ROOT
    / "scene_recognition"
    / "detector_module"
    / "artifacts"
    / "class_incremental"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "scene_recognition"
    / "detector_module"
    / "runs"
    / "class_incremental"
)


@dataclass(frozen=True)
class DERContext:
    teacher_checkpoint: Path
    replay_paths: tuple[Path, ...]
    der_weight: float
    cls_weight: float
    box_weight: float
    min_confidence: float


def _read_nonempty_paths(manifest: Path) -> tuple[Path, ...]:
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    paths = tuple(
        Path(line.strip()).resolve()
        for line in manifest.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    )
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"清单包含不存在的图像: {missing[0]}")
    return paths


def _read_yaml_names(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    names = config.get("names") if isinstance(config, dict) else None
    if isinstance(names, dict):
        return [str(names[index] if index in names else names[str(index)]) for index in range(len(names))]
    if isinstance(names, list):
        return [str(name) for name in names]
    raise ValueError(f"训练 YAML 缺少合法 names: {path}")


def _normalize_model_names(names: object) -> list[str]:
    if isinstance(names, dict):
        return [str(names[index] if index in names else names[str(index)]) for index in range(len(names))]
    if isinstance(names, (list, tuple)):
        return [str(name) for name in names]
    return []


def load_prepared_protocol(path: Path) -> dict:
    summary_path = path / "class_incremental_dataset_summary.json" if path.is_dir() else path
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    protocol = json.loads(summary_path.read_text(encoding="utf-8"))
    if protocol.get("scenario") != "class_incremental":
        raise ValueError("当前训练器只接受 scenario=class_incremental")
    if protocol.get("task_order") != ALL_CLASS_NAMES:
        raise ValueError(
            f"六阶段类别顺序必须为 {ALL_CLASS_NAMES}，实际为 {protocol.get('task_order')}"
        )
    stages = protocol.get("stages")
    if not isinstance(stages, list) or len(stages) != len(ALL_CLASS_NAMES):
        raise ValueError("Class-IL 协议必须恰好包含六个阶段")
    for expected_stage, stage in enumerate(stages, start=1):
        if int(stage.get("stage", -1)) != expected_stage:
            raise ValueError("Class-IL 阶段编号必须从 1 连续递增到 6")
        if stage.get("new_classes") != [ALL_CLASS_NAMES[expected_stage - 1]]:
            raise ValueError(f"第 {expected_stage} 阶段必须只引入一个指定类别")
    return protocol


def validate_experiment(
    protocol: dict,
    initial_model: Path,
    method: str,
    buffer_size: int,
) -> list[dict]:
    if not initial_model.is_file():
        raise FileNotFoundError(
            f"初始模型不存在: {initial_model}；本训练器不会联网下载模型"
        )
    if initial_model.suffix.lower() not in {".pt", ".yaml", ".yml"}:
        raise ValueError("初始模型必须是本地 .pt 或 YOLO 架构 .yaml")
    if method not in {"er", "der"}:
        raise ValueError("method 必须为 er 或 der")
    if buffer_size not in BUFFER_SIZE_CHOICES:
        raise ValueError(f"buffer size 只允许 {BUFFER_SIZE_CHOICES}")

    evaluation_split = str(protocol.get("evaluation", {}).get("split", "val"))
    if evaluation_split not in {"val", "test"}:
        raise ValueError(f"协议 evaluation split 非法: {evaluation_split}")
    plan: list[dict] = []
    for stage in protocol["stages"]:
        buffer = stage.get("buffers", {}).get(str(buffer_size))
        if buffer is None:
            raise ValueError(f"第 {stage['stage']} 阶段缺少 buffer={buffer_size} 数据")
        data_yaml = Path(buffer["data_yaml"])
        expected_names = stage["all_learned_classes"]
        if _read_yaml_names(data_yaml) != expected_names:
            raise ValueError(f"第 {stage['stage']} 阶段 YAML 类别与协议不一致")
        data_config = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
        if evaluation_split == "test" and not data_config.get("test"):
            raise ValueError(f"第 {stage['stage']} 阶段缺少独立 test 清单")
        replay_manifest = Path(buffer["training"]["replay_manifest"])
        replay_paths = _read_nonempty_paths(replay_manifest)
        context_index_value = buffer["training"].get("context_index")
        context_index = Path(context_index_value) if context_index_value else None
        if context_index is not None and not context_index.is_file():
            raise FileNotFoundError(f"Class-IL context_index 不存在: {context_index}")
        expected_replay = int(buffer["training"]["replay_images"])
        if len(replay_paths) != expected_replay:
            raise ValueError(f"第 {stage['stage']} 阶段 replay 清单计数不一致")
        if stage["stage"] == 1 and replay_paths:
            raise ValueError("首阶段不应包含旧类回放")
        if stage["stage"] > 1 and not replay_paths:
            raise ValueError(f"第 {stage['stage']} 阶段必须包含旧类回放")
        if len(replay_paths) > buffer_size:
            raise ValueError("replay 清单超过缓冲池容量")
        plan_entry = {
            "stage": stage["stage"],
            "new_class": stage["new_classes"][0],
            "learned_classes": expected_names,
            "data_yaml": str(data_yaml.resolve()),
            "replay_manifest": str(replay_manifest.resolve()),
            "replay_images": len(replay_paths),
            "evaluation_split": evaluation_split,
        }
        if context_index is not None:
            plan_entry["context_index"] = str(context_index.resolve())
        plan.append(plan_entry)
    return plan


def _make_der_trainer(context: DERContext):
    """Create an Ultralytics trainer bound to one stage's DER context."""

    from scene_recognition.detector_module.dark_experience_replay import DarkReplayModel
    from ultralytics.models.yolo.detect import DetectionTrainer

    class ReplayAwareDERTrainer(DetectionTrainer):
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
            # Frozen teacher tensors need no optimizer slots.  Passing only the
            # student also keeps optimizer logs and state size comparable to ER.
            trainable_model = model.student_model if isinstance(model, DarkReplayModel) else model
            return super().build_optimizer(
                trainable_model,
                name=name,
                lr=lr,
                momentum=momentum,
                decay=decay,
                iterations=iterations,
            )

    return ReplayAwareDERTrainer


def _metric_matrix(
    stage_results: list[dict],
    class_order: list[str],
    metric: str,
    evaluation_split: str,
) -> dict:
    rows: list[dict[str, float | None]] = []
    result_key = "validation" if evaluation_split == "val" else "test"
    for result in stage_results:
        learned = set(result["learned_classes"])
        per_class = result[result_key]["per_class"]
        rows.append(
            {
                name: (
                    float(per_class[name][metric])
                    if name in learned and per_class.get(name, {}).get(metric) is not None
                    else None
                )
                for name in class_order
            }
        )
    final_row = rows[-1]
    old_classes = class_order[:-1]
    forgetting_values: list[float] = []
    backward_transfer: list[float] = []
    for class_index, name in enumerate(old_classes):
        history = [row[name] for row in rows[class_index:] if row[name] is not None]
        final_value = final_row[name]
        learned_value = rows[class_index][name]
        if history and final_value is not None:
            forgetting_values.append(max(history) - final_value)
        if learned_value is not None and final_value is not None:
            backward_transfer.append(final_value - learned_value)
    stage_means = [
        sum(value for value in row.values() if value is not None)
        / len([value for value in row.values() if value is not None])
        for row in rows
    ]
    final_values = [value for value in final_row.values() if value is not None]
    return {
        "rows": rows,
        "final_average": sum(final_values) / len(final_values) if final_values else None,
        "average_incremental_accuracy": sum(stage_means) / len(stage_means),
        "average_forgetting": (
            sum(forgetting_values) / len(forgetting_values) if forgetting_values else None
        ),
        "backward_transfer": (
            sum(backward_transfer) / len(backward_transfer) if backward_transfer else None
        ),
    }


def build_class_incremental_metrics(
    stage_results: list[dict],
    class_order: list[str],
    evaluation_split: str = "val",
) -> dict:
    if len(stage_results) != len(class_order):
        raise ValueError("性能矩阵必须包含每个类别对应的一个阶段")
    if evaluation_split not in {"val", "test"}:
        raise ValueError("evaluation_split 必须为 val 或 test")
    official = evaluation_split == "test"
    return {
        "class_order": class_order,
        "map50": _metric_matrix(stage_results, class_order, "map50", evaluation_split),
        "map50_95": _metric_matrix(
            stage_results, class_order, "map50_95", evaluation_split
        ),
        "evaluation_split": evaluation_split,
        "official": official,
        "independent_holdout": official,
        "competition_official": False,
        "reason": (
            "本地独立 test 只用于阶段后评估，不参与早停或 checkpoint 选择；"
            "它不是主办方隐藏测试集。"
            if official
            else "数据集没有独立 test；该矩阵仅用于本地验证与方法对比。"
        ),
    }


def build_sparse_moe_config(args: argparse.Namespace) -> SparseMoEConfig:
    """Build the explicit Sparse-MoE configuration recorded in every run."""

    temperature = getattr(args, "router_temperature", None)
    temperature_start = (
        float(temperature)
        if temperature is not None
        else float(getattr(args, "router_temperature_start", 2.0))
    )
    temperature_end = (
        float(temperature)
        if temperature is not None
        else float(getattr(args, "router_temperature_end", 1.0))
    )
    return SparseMoEConfig(
        expert_count=int(getattr(args, "expert_count", 5)),
        top_k=int(getattr(args, "top_k", 2)),
        expert_bottleneck=float(getattr(args, "expert_bottleneck", 0.25)),
        router_hidden=int(getattr(args, "router_hidden", 128)),
        aux_hidden=int(getattr(args, "aux_hidden", 128)),
        modality_loss_weight=float(getattr(args, "modality_loss_weight", 0.10)),
        scene_loss_weight=float(getattr(args, "scene_loss_weight", 0.10)),
        balance_loss_weight=float(getattr(args, "balance_loss_weight", 0.01)),
        router_z_loss_weight=float(getattr(args, "router_z_loss_weight", 0.001)),
        anchor_loss_weight=float(getattr(args, "anchor_loss_weight", 0.001)),
        anchor_rho=float(getattr(args, "anchor_rho", 0.95)),
        router_temperature_start=temperature_start,
        router_temperature_end=temperature_end,
        router_temperature_warmup_epochs=int(
            getattr(args, "router_temperature_warmup_epochs", 3)
        ),
    )


def _context_summary_for_stage(stage_plan: dict[str, Any]) -> dict[str, Any]:
    """Summarize explicit metadata, with a filename fallback for old prep runs."""

    context_index = stage_plan.get("context_index")
    if context_index:
        return context_index_summary(read_context_rows(Path(context_index)))
    data_yaml = Path(stage_plan["data_yaml"])
    data_config = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    train_value = data_config.get("train") if isinstance(data_config, dict) else None
    if not isinstance(train_value, str):
        return {}
    train_manifest = Path(train_value)
    if not train_manifest.is_absolute():
        train_manifest = data_yaml.parent / train_manifest
    if not train_manifest.is_file():
        return {}
    rows = []
    for image_path in _read_nonempty_paths(train_manifest):
        context = resolve_context_metadata(image_path)
        rows.append(
            {
                "sensor": context["sensor"],
                "scene": context["scene"],
                "metadata_source": context["metadata_source"],
            }
        )
    return context_index_summary(rows)


def _training_arguments(args: argparse.Namespace, data_yaml: Path, stage_name: str, seed: int) -> dict:
    augmentation = DISABLED_AUGMENTATION if args.no_builtin_aug else BUILTIN_AUGMENTATION
    kwargs = {
        "data": str(data_yaml.resolve()),
        "epochs": args.epochs,
        "patience": args.patience,
        "imgsz": args.image_size,
        "batch": args.batch_size,
        "workers": args.workers,
        "device": args.device,
        "seed": seed,
        "deterministic": True,
        "optimizer": "AdamW",
        "lr0": args.learning_rate,
        "pretrained": True,
        "resume": False,
        "cache": False,
        "amp": not args.no_amp,
        "project": str(args.output.resolve()),
        "name": stage_name,
        "exist_ok": False,
        "plots": not args.no_plots,
        "verbose": True,
        "close_mosaic": 0 if args.no_builtin_aug else 10,
        "freeze": args.freeze,
    }
    kwargs.update(augmentation)
    return kwargs


def run_class_incremental(args: argparse.Namespace) -> dict:
    os.environ.setdefault("YOLO_OFFLINE", "true")
    sparse_moe_enabled = bool(getattr(args, "sparse_moe", False))
    sparse_moe_config = build_sparse_moe_config(args)
    protocol = load_prepared_protocol(args.prepared)
    plan = validate_experiment(protocol, args.initial_model, args.method, args.buffer_size)
    stop_after_stage = min(args.stop_after_stage, len(plan))
    selected_plan = plan[:stop_after_stage]
    selected_stages = protocol["stages"][:stop_after_stage]
    evaluation_split = str(protocol.get("evaluation", {}).get("split", "val"))
    if evaluation_split not in {"val", "test"}:
        raise ValueError(f"协议 evaluation split 非法: {evaluation_split}")
    if args.dry_run:
        result = {
            "status": "dry_run_ok",
            "scenario": "class_incremental",
            "method": args.method.upper(),
            "buffer_size": args.buffer_size,
            "initial_model": str(args.initial_model.resolve()),
            "evaluation_split": evaluation_split,
            "stages": selected_plan,
        }
        if sparse_moe_enabled:
            result.update(
                {
                    "sparse_moe": True,
                    "sparse_moe_config": sparse_moe_config.to_dict(),
                }
            )
        return result
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"输出目录非空，请使用新的实验目录: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    torch, ultralytics, YOLO = import_training_dependencies()
    maybe_import_torch_npu(args.device)

    initial_probe = YOLO(str(args.initial_model.resolve()))
    initial_names = _normalize_model_names(getattr(initial_probe, "names", None))
    leaked_target_classes = sorted(set(initial_names) & set(protocol["task_order"]))
    if leaked_target_classes:
        raise ValueError(
            "六阶段 Class-IL 的初始模型不能已经学习项目目标类，否则首阶段存在类别泄漏；"
            f"检测到 {leaked_target_classes}。请使用本地通用预训练模型或本地架构 YAML。"
        )
    del initial_probe

    previous_checkpoint = args.initial_model.resolve()
    stage_results: list[dict] = []
    experiment_started = time.perf_counter()
    for stage, stage_plan in zip(selected_stages, selected_plan):
        stage_started = time.perf_counter()
        stage_number = int(stage["stage"])
        stage_name = f"stage_{stage_number:02d}_{stage_plan['new_class']}"
        data_yaml = Path(stage_plan["data_yaml"])
        replay_paths = _read_nonempty_paths(Path(stage_plan["replay_manifest"]))
        model = YOLO(str(previous_checkpoint))
        kwargs = _training_arguments(args, data_yaml, stage_name, args.seed + stage_number - 1)
        der_context = None
        if args.method == "der" and stage_number > 1:
            context = DERContext(
                teacher_checkpoint=previous_checkpoint,
                replay_paths=replay_paths,
                der_weight=args.der_weight,
                cls_weight=args.der_cls_weight,
                box_weight=args.der_box_weight,
                min_confidence=args.der_min_confidence,
            )
            der_context = context
        if sparse_moe_enabled:
            context_index = (
                Path(stage_plan["context_index"])
                if stage_plan.get("context_index")
                else None
            )
            model.train(
                trainer=make_sparse_moe_trainer(
                    sparse_moe_config,
                    context_index=context_index,
                    der_context=der_context,
                ),
                **kwargs,
            )
        elif der_context is not None:
            model.train(trainer=_make_der_trainer(der_context), **kwargs)
        else:
            model.train(**kwargs)
        run_dir = Path(model.trainer.save_dir)
        best_model = run_dir / "weights" / "best.pt"
        if not best_model.is_file():
            raise FileNotFoundError(f"第 {stage_number} 阶段没有生成 best.pt: {best_model}")
        sparse_training_usage: dict[str, Any] | None = None
        if sparse_moe_enabled:
            training_model = getattr(model.trainer, "model", getattr(model, "model", None))
            sparse_training_usage = sparse_moe_usage_state(training_model)
            if sparse_training_usage is None:
                raise RuntimeError(f"第 {stage_number} 阶段训练完成但未找到 Sparse-MoE usage tracker")
        trained = YOLO(str(best_model))
        if sparse_moe_enabled and not restore_sparse_moe_usage(trained.model, sparse_training_usage):
            raise RuntimeError(f"第 {stage_number} 阶段 best checkpoint 未恢复 Sparse-MoE usage tracker")
        validation_result = trained.val(
            data=str(data_yaml.resolve()),
            split="val",
            imgsz=args.image_size,
            batch=args.batch_size,
            workers=args.workers,
            device=args.device,
            project=str(run_dir),
            name="class_il_val",
            exist_ok=True,
            plots=not args.no_plots,
            verbose=False,
        )
        validation = detection_metrics_to_dict(validation_result, stage["all_learned_classes"])
        test_evaluation = None
        if evaluation_split == "test":
            test_result = trained.val(
                data=str(data_yaml.resolve()),
                split="test",
                imgsz=args.image_size,
                batch=args.batch_size,
                workers=args.workers,
                device=args.device,
                project=str(run_dir),
                name="class_il_test",
                exist_ok=True,
                plots=not args.no_plots,
                verbose=False,
            )
            test_evaluation = detection_metrics_to_dict(
                test_result, stage["all_learned_classes"]
            )
        sparse_artifacts: dict[str, str] = {}
        sparse_metadata: dict[str, Any] = {"enabled": False}
        if sparse_moe_enabled:
            update_sparse_moe_anchors(trained.model)
            # The stage-best checkpoint is the anchor source for the next
            # stage, so persist the updated EMA anchor bank in that checkpoint.
            trained.save(best_model)
            sparse_artifacts = write_sparse_moe_artifacts(
                trained.model,
                run_dir,
                context_summary=_context_summary_for_stage(stage_plan),
            )
            sparse_metadata = sparse_moe_metadata(trained.model)
        stage_summary = {
            "stage": stage_number,
            "task_id": stage["task_id"],
            "new_class": stage_plan["new_class"],
            "learned_classes": stage["all_learned_classes"],
            "method": args.method.upper(),
            "buffer_capacity": args.buffer_size,
            "replay_images": len(replay_paths),
            "input_checkpoint": str(previous_checkpoint),
            "best_model": str(best_model.resolve()),
            "data_yaml": str(data_yaml.resolve()),
            "context_index": stage_plan.get("context_index"),
            "validation": validation,
            "test": test_evaluation,
            "elapsed_seconds": round(time.perf_counter() - stage_started, 3),
            "dark_targets": (
                {
                    "enabled": args.method == "der" and stage_number > 1,
                    "scope": "replay_samples_only",
                    "source": "frozen_previous_stage_checkpoint",
                    "cache": "online_recompute",
                    "weight": args.der_weight,
                    "class_weight": args.der_cls_weight,
                    "box_weight": args.der_box_weight,
                    "min_confidence": args.der_min_confidence,
                }
                if args.method == "der"
                else {"enabled": False}
            ),
            "training_controls": {
                "freeze": args.freeze,
                "amp": not args.no_amp,
                "plots": not args.no_plots,
                "builtin_augmentation_disabled": args.no_builtin_aug,
            },
        }
        if sparse_moe_enabled:
            stage_summary.update(
                {
                    "sparse_moe": sparse_metadata,
                    "sparse_moe_artifacts": sparse_artifacts,
                }
            )
        (run_dir / "class_incremental_stage_summary.json").write_text(
            json.dumps(stage_summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        stage_results.append(stage_summary)
        previous_checkpoint = best_model.resolve()

        partial = {
            "status": "running" if stage_number < stop_after_stage else (
                "complete" if stop_after_stage == len(plan) else "partial"
            ),
            "scenario": "class_incremental",
            "method": args.method.upper(),
            "buffer_size": args.buffer_size,
            "task_order": protocol["task_order"],
            "stages": stage_results,
        }
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "class_incremental_training_summary.json").write_text(
            json.dumps(partial, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    completed_protocol = stop_after_stage == len(plan)
    summary: dict[str, Any] = {
        "status": "complete" if completed_protocol else "partial",
        "protocol_version": protocol["protocol_version"],
        "scenario": "class_incremental",
        "head_policy": "single_expanding_head",
        "method": args.method.upper(),
        "buffer_size": args.buffer_size,
        "task_order": protocol["task_order"],
        "initial_model": str(args.initial_model.resolve()),
        "final_model": str(previous_checkpoint),
        "stages": stage_results,
        "continual_metrics": (
            build_class_incremental_metrics(
                stage_results,
                protocol["task_order"],
                evaluation_split=evaluation_split,
            )
            if completed_protocol
            else {
                "available": False,
                "reason": "只运行了部分阶段；完成六阶段后才计算最终遗忘与 BWT。",
                "completed_stages": stop_after_stage,
            }
        ),
        "elapsed_seconds": round(time.perf_counter() - experiment_started, 3),
        "environment": {
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "ultralytics": ultralytics.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": args.device,
        },
        "privacy": {
            "offline_environment_requested": True,
            "local_checkpoint_required": True,
            "dataset_upload": False,
        },
        "method_scope": {
            "er": "supervised replay from a fixed class-balanced image buffer",
            "der": (
                "ER plus confidence-weighted previous-stage raw class and box response "
                "matching on replay samples only"
            ),
            "task_il_ready": (
                "task_id/scenario are separated in the protocol; a future Task-IL scheduler can reuse "
                "the replay and DER components"
            ),
        },
    }
    if sparse_moe_enabled:
        summary.update(
            {
                "sparse_moe": True,
                "sparse_moe_config": sparse_moe_config.to_dict(),
            }
        )
    (args.output / "class_incremental_training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="六类别逐类 Class-IL：ER / DER")
    parser.add_argument("--prepared", type=Path, default=DEFAULT_PREPARED)
    parser.add_argument("--initial-model", type=Path, required=True, help="本地初始 .pt 或架构 .yaml")
    parser.add_argument("--method", choices=("er", "der"), required=True)
    parser.add_argument("--buffer-size", type=int, choices=BUFFER_SIZE_CHOICES, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--freeze", type=int, default=None, help="Freeze the first N YOLO layers; useful for CPU fine-tuning.")
    parser.add_argument("--der-weight", type=float, default=1.0)
    parser.add_argument("--der-cls-weight", type=float, default=1.0)
    parser.add_argument("--der-box-weight", type=float, default=0.25)
    parser.add_argument("--der-min-confidence", type=float, default=0.0)
    parser.add_argument(
        "--sparse-moe",
        action="store_true",
        help="启用第三阶段 Sparse-MoE v1；未指定时保持原 ER/DER 行为",
    )
    parser.add_argument("--expert-count", type=int, default=5, help="Sparse-MoE 专家数，默认 5")
    parser.add_argument("--top-k", type=int, default=2, help="每张图片激活的专家数，默认 Top-2")
    parser.add_argument(
        "--expert-bottleneck",
        type=float,
        default=0.25,
        help="专家 1x1 瓶颈比例，默认 0.25",
    )
    parser.add_argument("--router-hidden", type=int, default=128, help="路由 MLP 隐层宽度")
    parser.add_argument("--aux-hidden", type=int, default=128, help="模态/场景辅助头隐层宽度")
    parser.add_argument("--modality-loss-weight", type=float, default=0.10)
    parser.add_argument("--scene-loss-weight", type=float, default=0.10)
    parser.add_argument("--balance-loss-weight", type=float, default=0.01)
    parser.add_argument("--router-z-loss-weight", type=float, default=0.001)
    parser.add_argument("--anchor-loss-weight", type=float, default=0.001)
    parser.add_argument("--anchor-rho", type=float, default=0.95)
    parser.add_argument(
        "--router-temperature",
        type=float,
        default=None,
        help="固定路由温度；省略时按 start/end 退火",
    )
    parser.add_argument("--router-temperature-start", type=float, default=2.0)
    parser.add_argument("--router-temperature-end", type=float, default=1.0)
    parser.add_argument("--router-temperature-warmup-epochs", type=int, default=3)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--no-builtin-aug", action="store_true", help="Disable Ultralytics online augmentation for faster CPU fine-tuning.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--stop-after-stage",
        type=int,
        choices=range(1, 7),
        default=6,
        help="仅用于分段执行或冒烟验证；正式对比保持默认 6",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_class_incremental(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
