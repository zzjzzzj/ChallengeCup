from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO

from scene_recognition.detector_module.metrics import detection_metrics_to_dict


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = PROJECT_ROOT / "scene_recognition" / "detector_module" / "artifacts" / "detection_dataset" / "dataset.yaml"


def read_names(data_yaml: Path) -> list[str]:
    names = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))["names"]
    if isinstance(names, dict):
        return [str(names[index] if index in names else names[str(index)]) for index in range(len(names))]
    return [str(value) for value in names]


def read_results(results_csv: Path) -> list[dict]:
    with results_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"训练结果为空: {results_csv}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="按验证集 mAP@0.5 选择可用检测权重")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    run_dir = args.run.resolve()
    results_csv = run_dir / "results.csv"
    rows = read_results(results_csv)
    best_map5095_row = max(rows, key=lambda row: float(row["metrics/mAP50-95(B)"]))
    last_row = rows[-1]
    candidates = [
        {
            "name": "best.pt",
            "path": run_dir / "weights" / "best.pt",
            "validation_epoch": int(float(best_map5095_row["epoch"])),
            "validation_map50": float(best_map5095_row["metrics/mAP50(B)"]),
            "validation_map50_95": float(best_map5095_row["metrics/mAP50-95(B)"]),
            "reason_saved": "Ultralytics 默认按 mAP@0.5:0.95 保存",
        },
        {
            "name": "last.pt",
            "path": run_dir / "weights" / "last.pt",
            "validation_epoch": int(float(last_row["epoch"])),
            "validation_map50": float(last_row["metrics/mAP50(B)"]),
            "validation_map50_95": float(last_row["metrics/mAP50-95(B)"]),
            "reason_saved": "早停或最大轮次结束时的最后权重",
        },
    ]
    missing = [str(candidate["path"]) for candidate in candidates if not candidate["path"].is_file()]
    if missing:
        raise FileNotFoundError("缺少候选权重: " + ", ".join(missing))

    selected = max(candidates, key=lambda candidate: candidate["validation_map50"])
    output_model = run_dir / "weights" / "submission_map50.pt"
    shutil.copy2(selected["path"], output_model)

    model = YOLO(str(output_model))
    metrics = model.val(
        data=str(args.data.resolve()),
        split="test",
        imgsz=640,
        batch=args.batch_size,
        device=args.device,
        workers=2,
        project=str(run_dir),
        name="test_submission_map50",
        plots=True,
        verbose=False,
    )
    report = {
        "selection_metric": "validation mAP@0.5",
        "selection_policy": "仅在训练后实际保留的 best.pt 与 last.pt 中比较验证集指标；不使用测试集选择模型。",
        "candidates": [
            {**candidate, "path": str(candidate["path"].resolve())} for candidate in candidates
        ],
        "selected_source": selected["name"],
        "selected_model": str(output_model.resolve()),
        "test": detection_metrics_to_dict(metrics, read_names(args.data)),
    }
    (run_dir / "checkpoint_selection.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
