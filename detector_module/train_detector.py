from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime
from pathlib import Path

import torch
import ultralytics
import yaml
from ultralytics import YOLO

from detector_module.metrics import detection_metrics_to_dict


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = PROJECT_ROOT / "detector_module" / "artifacts" / "detection_dataset" / "dataset.yaml"
DEFAULT_RUNS = PROJECT_ROOT / "detector_module" / "runs"
DEFAULT_MODEL = "yolov8n.pt"
WEIGHT_SUFFIXES = (".pt", ".pth", ".ckpt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练 YOLO 目标检测基线并在独立测试集评估")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument(
        "--model",
        default=None,
        help=f"模型串，缺省 {DEFAULT_MODEL}；配合 --no-pretrained 时缺省自动换成同名 .yaml 架构。",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--project", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--name", default="yolov8n_baseline")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument(
        "--no-builtin-aug",
        action="store_true",
        help=(
            "关闭YOLO自带的全部在线增广（mosaic/fliplr/degrees/scale/hsv等）。"
            "做离线增广数据集的对照实验时必须开启，否则团队增广与YOLO增广混在一起无法归因。"
        ),
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help=(
            "随机初始化骨架，不加载任何COCO预训练权重（命名与 "
            "target_classifier_module/train_classifier.py 对齐）。"
            "缺省 --model 会自动从 yolov8n.pt 换成 yolov8n.yaml。"
        ),
    )
    return parser.parse_args()


# 仓库既有基线的在线增广配方。改动它会使已公布的 mAP 不可复现，因此保持原值。
BUILTIN_AUGMENTATION = {
    "hsv_h": 0.0,
    "hsv_s": 0.0,
    "hsv_v": 0.15,
    "degrees": 5.0,
    "translate": 0.08,
    "scale": 0.20,
    "fliplr": 0.5,
    "flipud": 0.0,
    "mosaic": 0.6,
    "mixup": 0.0,
}

# --no-builtin-aug 时使用：把每一项在线增广都归零，使唯一变量是训练集本身。
DISABLED_AUGMENTATION = {
    "hsv_h": 0.0,
    "hsv_s": 0.0,
    "hsv_v": 0.0,
    "degrees": 0.0,
    "translate": 0.0,
    "scale": 0.0,
    "shear": 0.0,
    "perspective": 0.0,
    "fliplr": 0.0,
    "flipud": 0.0,
    "bgr": 0.0,
    "mosaic": 0.0,
    "mixup": 0.0,
    "copy_paste": 0.0,
    "erasing": 0.0,
}


def resolve_model_spec(model: str | None, no_pretrained: bool) -> tuple[str, bool]:
    """决定真正交给 YOLO() 的模型串，以及 train(pretrained=...) 的取值。

    实测（ultralytics 8.4.100，见 detector_module/tests/test_train_detector_scratch.py 的注释）：
      * ``YOLO('yolov8n.pt')`` 在**构造时**就把 COCO 权重灌进了 nn.Module，
        ``model.ckpt`` 为真；此时 ``.pt`` 与 ``.yaml`` 的第一层卷积统计量完全不同
        （std 0.152257 vs 0.113982，absmax 0.510742 vs 0.192243）。
      * 本版本 ``Model.train()`` 里有 ``weights = None if pretrained is False else self.model``，
        所以 ``.pt`` + ``pretrained=False`` 确实会被重建成随机初始化；
        但这是**依赖版本的实现细节**，且 ``--resume`` 时该分支整条被跳过。
    因此这里不赌 ultralytics 的实现：从零训练一律强制走 ``.yaml`` 架构，
    显式传 ``.pt`` 又要求从零时直接报错，让"假从零"在命令行阶段就暴露。
    """

    if not no_pretrained:
        return (model if model is not None else DEFAULT_MODEL), True
    if model is None:
        return Path(DEFAULT_MODEL).with_suffix(".yaml").as_posix(), False
    if model.lower().endswith(WEIGHT_SUFFIXES):
        raise ValueError(
            f"--no-pretrained 不能与权重文件 --model {model} 同时使用。\n"
            "YOLO('x.pt') 在构造时就已经把权重加载进网络，靠 train(pretrained=False) "
            "来「清零」只是本版 ultralytics 的实现细节（--resume 时甚至完全失效），"
            "一旦换版本就会静默退化成「假从零」实验。\n"
            f"请显式改用架构文件，例如 --model {Path(model).with_suffix('.yaml').name}，"
            "或直接省略 --model 由脚本自动切换。"
        )
    return model, False


def has_split(data_yaml: Path, split: str) -> bool:
    """Report whether the dataset config actually declares a given split."""

    config = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    return bool(config.get(split))


def read_class_names(data_yaml: Path) -> list[str]:
    config = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    names = config["names"]
    if isinstance(names, dict):
        return [str(names[index] if index in names else names[str(index)]) for index in range(len(names))]
    return [str(name) for name in names]


def main() -> None:
    args = parse_args()
    if not args.data.is_file():
        raise FileNotFoundError(
            f"检测数据配置不存在: {args.data}\n"
            "请先运行 python -m detector_module.prepare_detection_dataset"
        )

    model_spec, pretrained = resolve_model_spec(args.model, args.no_pretrained)
    if args.no_pretrained and args.resume:
        raise ValueError(
            "--no-pretrained 与 --resume 互斥：resume 会跳过随机初始化分支，"
            "断点续训的权重来源由 last.pt 决定，不再是「从零」。"
        )

    args.project.mkdir(parents=True, exist_ok=True)
    class_names = read_class_names(args.data)
    environment = {
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "seed": args.seed,
    }

    augmentation = DISABLED_AUGMENTATION if args.no_builtin_aug else BUILTIN_AUGMENTATION
    environment["builtin_augmentation_disabled"] = args.no_builtin_aug

    model = YOLO(model_spec)
    train_result = model.train(
        data=str(args.data.resolve()),
        epochs=args.epochs,
        patience=args.patience,
        imgsz=args.image_size,
        batch=args.batch_size,
        device=args.device,
        workers=args.workers,
        project=str(args.project.resolve()),
        name=args.name,
        exist_ok=args.exist_ok,
        pretrained=pretrained,
        optimizer="auto",
        seed=args.seed,
        deterministic=True,
        amp=True,
        cache=False,
        plots=True,
        verbose=True,
        close_mosaic=0 if args.no_builtin_aug else 10,
        cos_lr=True,
        resume=args.resume,
        **augmentation,
    )

    run_dir = Path(train_result.save_dir)
    best_model_path = run_dir / "weights" / "best.pt"
    if not best_model_path.is_file():
        raise FileNotFoundError(f"训练结束但未找到最佳权重: {best_model_path}")

    best_model = YOLO(str(best_model_path))
    # 增广数据集的配置只有 train/val，没有 test；此时在 val 上评估并如实标注口径，
    # 而不是让 split="test" 直接崩掉或静默回退到 val。
    evaluation_split = "test" if has_split(args.data, "test") else "val"
    evaluation_metrics = best_model.val(
        data=str(args.data.resolve()),
        split=evaluation_split,
        imgsz=args.image_size,
        batch=args.batch_size,
        device=args.device,
        workers=args.workers,
        project=str(run_dir),
        name=evaluation_split,
        plots=True,
        verbose=True,
    )
    summary = {
        "environment": environment,
        "configuration": {
            "data": str(args.data.resolve()),
            "model": model_spec,
            "model_arg": args.model,
            "pretrained": pretrained,
            "weight_init": "coco_pretrained" if pretrained else "random_scratch",
            "epochs": args.epochs,
            "patience": args.patience,
            "image_size": args.image_size,
            "batch_size": args.batch_size,
            "device": args.device,
            "builtin_augmentation_disabled": args.no_builtin_aug,
            "builtin_augmentation": augmentation,
        },
        "run_dir": str(run_dir.resolve()),
        "best_model": str(best_model_path.resolve()),
        "evaluation_split": evaluation_split,
        evaluation_split: detection_metrics_to_dict(evaluation_metrics, class_names),
        "score_targets": {"map_threshold": 0.80, "fps_threshold": 30.0},
        "metric_note": (
            "同时报告 mAP@0.5 与 mAP@0.5:0.95，待主办方确认正式口径。"
            f"本次指标在 {evaluation_split} 划分上计算。"
        ),
    }
    summary_path = run_dir / "baseline_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
