"""YOLOv8 消融实验入口：预训练 vs 从零训练、增广集 vs 原始集。

与 train_detector.py 的差异只有三点，其余超参逐条保持一致，保证结果可比：
  1. `--no-pretrained` 可真正关闭预训练权重（原脚本第82行把 pretrained=True 写死）；
  2. `--eval-split` 可选 val/test（增广数据集没有 test 划分，原脚本固定评 test 会直接报错）；
  3. 结果统一写到 ablation_summary.json，便于跨run汇总成对比表。

注意：ultralytics 自带在线增广（mosaic/翻转/缩放等）在两组里都保持开启且完全相同，
因此「增广集 vs 原始集」量出来的是**离线增广在在线增广之上的增量**，不是增广的全部价值。
"""

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
DEFAULT_RUNS = PROJECT_ROOT / "detector_module" / "runs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 预训练/增广消融实验")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--project", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--name", required=True)
    parser.add_argument("--eval-split", default="val", choices=["val", "test"])
    parser.add_argument("--exist-ok", action="store_true")
    return parser.parse_args()


def read_class_names(data_yaml: Path) -> list[str]:
    config = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    names = config["names"]
    if isinstance(names, dict):
        return [
            str(names[index] if index in names else names[str(index)])
            for index in range(len(names))
        ]
    return [str(name) for name in names]


def main() -> None:
    args = parse_args()
    if not args.data.is_file():
        raise FileNotFoundError(f"数据配置不存在: {args.data}")

    pretrained = not args.no_pretrained
    # 从零训练必须同时满足：用 .yaml 建结构 + pretrained=False，二者缺一都会悄悄加载权重
    model_spec = args.model if pretrained else "yolov8n.yaml"

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
        close_mosaic=10,
        cos_lr=True,
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

    run_dir = Path(train_result.save_dir)
    best_model_path = run_dir / "weights" / "best.pt"
    if not best_model_path.is_file():
        raise FileNotFoundError(f"训练结束但未找到最佳权重: {best_model_path}")

    best_model = YOLO(str(best_model_path))
    eval_metrics = best_model.val(
        data=str(args.data.resolve()),
        split=args.eval_split,
        imgsz=args.image_size,
        batch=args.batch_size,
        device=args.device,
        workers=args.workers,
        project=str(run_dir),
        name=f"eval_{args.eval_split}",
        plots=True,
        verbose=True,
    )

    summary = {
        "environment": environment,
        "configuration": {
            "data": str(args.data.resolve()),
            "model_spec": model_spec,
            "pretrained": pretrained,
            "epochs": args.epochs,
            "patience": args.patience,
            "image_size": args.image_size,
            "batch_size": args.batch_size,
            "device": args.device,
            "eval_split": args.eval_split,
        },
        "run_dir": str(run_dir.resolve()),
        "best_model": str(best_model_path.resolve()),
        "eval": detection_metrics_to_dict(eval_metrics, class_names),
    }
    summary_path = run_dir / "ablation_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
