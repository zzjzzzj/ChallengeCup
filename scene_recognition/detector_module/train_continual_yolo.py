"""Local-only YOLO continual fine-tuning and replay baselines.

This is the executable baseline required before adding research-grade response
distillation. It never accepts a remote model name: the base checkpoint and all
dataset manifests must already exist locally.
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

import torch
import yaml

from scene_recognition.detector_module import BASE_CLASS_NAMES
from scene_recognition.detector_module.metrics import detection_metrics_to_dict


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "scene_recognition"
    / "detector_module"
    / "runs"
    / "continual_r2_replay"
)


def read_class_names(data_path: Path) -> tuple[dict, list[str]]:
    config = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    names = config["names"]
    class_names = (
        [str(names[index] if index in names else names[str(index)]) for index in range(len(names))]
        if isinstance(names, dict)
        else [str(name) for name in names]
    )
    return config, class_names


def normalize_model_names(names: object) -> list[str]:
    if isinstance(names, dict):
        return [str(names[index] if index in names else names[str(index)]) for index in range(len(names))]
    if isinstance(names, (list, tuple)):
        return [str(name) for name in names]
    return []


def validate_strategy_data(data_path: Path, strategy: str) -> None:
    summary_path = data_path.parent / "continual_dataset_summary.json"
    if not summary_path.is_file():
        return
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    replay_images = int(
        summary.get("statistics", {}).get("replay_train", {}).get("images", 0)
    )
    if strategy == "replay" and replay_images == 0:
        raise ValueError(
            "选择了 replay，但数据摘要显示旧类回放图像为 0；"
            "请提供 --base-index 并重新准备数据"
        )
    expected_yaml = "data_replay.yaml" if strategy == "replay" else "data_increment_only.yaml"
    if data_path.name != expected_yaml:
        raise ValueError(
            f"strategy={strategy} 应使用 {expected_yaml}，实际为 {data_path.name}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从本地四类 checkpoint 进行 r2 六类持续微调")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--strategy", choices=("increment_only", "replay"), default="replay")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    return parser.parse_args()


def main() -> None:
    # Prevent implicit HUB/download behavior. A missing dependency or checkpoint
    # must fail locally rather than causing a network request.
    os.environ.setdefault("YOLO_OFFLINE", "true")
    args = parse_args()
    if not args.data.is_file():
        raise FileNotFoundError(args.data)
    if not args.base_model.is_file():
        raise FileNotFoundError(
            f"基础 checkpoint 不存在: {args.base_model}；本命令不会联网下载模型"
        )
    validate_strategy_data(args.data, args.strategy)
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"输出目录非空，请使用新的实验目录: {args.output}")
    data_config, class_names = read_class_names(args.data)
    if class_names[: len(BASE_CLASS_NAMES)] != BASE_CLASS_NAMES:
        raise ValueError(
            f"六类 YAML 必须以前四类 {BASE_CLASS_NAMES} 开头，实际为 {class_names}"
        )
    if len(class_names) <= len(BASE_CLASS_NAMES):
        raise ValueError("训练 YAML 没有新增类别")
    if not data_config.get("val"):
        raise ValueError("持续微调必须提供独立 val 清单用于选择 checkpoint")

    try:
        import ultralytics
        from ultralytics import YOLO
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "缺少 ultralytics；请在本地环境安装 requirements.txt 后重试"
        ) from exc

    model = YOLO(str(args.base_model.resolve()))
    base_model_names = normalize_model_names(getattr(model, "names", None))
    if base_model_names and base_model_names != BASE_CLASS_NAMES:
        raise ValueError(
            "基础 checkpoint 类别顺序与协议不一致；"
            f"期望 {BASE_CLASS_NAMES}，实际 {base_model_names}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    result = model.train(
        data=str(args.data.resolve()),
        epochs=args.epochs,
        patience=args.patience,
        imgsz=args.image_size,
        batch=args.batch_size,
        workers=args.workers,
        device=args.device,
        seed=args.seed,
        deterministic=True,
        optimizer="AdamW",
        lr0=args.learning_rate,
        pretrained=True,
        resume=False,
        cache=False,
        amp=True,
        project=str(args.output.parent.resolve()),
        name=args.output.name,
        exist_ok=False,
        plots=True,
        verbose=True,
        close_mosaic=10,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.15,
        degrees=5.0,
        translate=0.08,
        scale=0.20,
        fliplr=0.5,
        flipud=0.0,
        mosaic=0.6,
        mixup=0.0,
    )
    run_dir = Path(result.save_dir)
    best_model = run_dir / "weights" / "best.pt"
    if not best_model.is_file():
        raise FileNotFoundError(f"训练结束但未生成 best.pt: {best_model}")
    trained = YOLO(str(best_model))
    validation = trained.val(
        data=str(args.data.resolve()),
        split="val",
        imgsz=args.image_size,
        batch=args.batch_size,
        workers=args.workers,
        device=args.device,
        project=str(run_dir),
        name="continual_val",
        plots=True,
        verbose=False,
    )
    summary = {
        "protocol": "r2-class-increment-v1",
        "strategy": args.strategy,
        "base_model": str(args.base_model.resolve()),
        "data": str(args.data.resolve()),
        "class_order": class_names,
        "base_classes": BASE_CLASS_NAMES,
        "new_classes": class_names[len(BASE_CLASS_NAMES) :],
        "best_model": str(best_model.resolve()),
        "validation": detection_metrics_to_dict(validation, class_names),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "environment": {
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "ultralytics": ultralytics.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": args.device,
        },
        "method_scope": {
            "implemented": [
                "local checkpoint fine-tuning",
                "optional replay through data_replay.yaml",
                "six-class head adaptation",
                "fixed-validation checkpoint selection",
            ],
            "not_implemented": [
                "elastic response distillation",
                "P2/P3 feature distillation",
                "expert anchor consolidation",
            ],
            "warning": (
                "This is the naive/replay continual baseline. Do not describe it as "
                "knowledge-distillation training."
            ),
        },
        "privacy": {
            "offline_environment_requested": True,
            "local_checkpoint_required": True,
            "dataset_upload": False,
        },
    }
    (run_dir / "continual_training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
