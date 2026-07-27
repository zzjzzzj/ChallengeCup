"""对比实验矩阵驱动器。

产出两组对比数据：

1. **ResNet18 vs YOLOv8**：在同数据、同指标的对比面上比较，而不是把
   Accuracy 与 mAP 并排摆。
2. **预训练 vs 从零**：每个模型都有 ImageNet/COCO 预训练与随机初始化两个臂，
   除初始化外超参完全一致，多种子给出均值±标准差。

每个实验单元幂等：若目标目录已有 ``metrics.json`` 则跳过，可安全重跑。
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "comparison" / "runs"
SEEDS = (42, 43, 44)


@dataclass(frozen=True)
class Experiment:
    """一个实验单元：一条可复现的训练/评价命令。"""

    name: str
    track: str
    model: str
    init: str
    epochs: int
    seed: int
    argv: list[str]
    metrics_rel: str = "metrics.json"
    extra: dict = field(default_factory=dict)


def _resnet_crop(init: str, epochs: int, seed: int, output_root: Path) -> Experiment:
    name = f"crop__resnet18__{init}__e{epochs}__s{seed}"
    out = output_root / name
    argv = [
        "-m", "target_classifier_module.train_classifier",
        "--epochs", str(epochs),
        "--batch-size", "32",
        "--image-size", "224",
        "--seed", str(seed),
        "--augmentation", "none",
        # Windows 上 spawn 出的 DataLoader worker 会与本仓库的数据集实现死锁
        # （实测 num_workers=4 时进程 20 分钟只消耗 7 秒 CPU）。仓库既有基线
        # 也一直用 0，保持一致既能跑通也便于与历史数字对齐。
        "--num-workers", "0",
        "--output", str(out),
    ]
    if init == "scratch":
        argv.append("--no-pretrained")
    return Experiment(name, "crop", "resnet18", init, epochs, seed, argv)


def _yolo_crop(init: str, epochs: int, seed: int, output_root: Path) -> Experiment:
    name = f"crop__yolov8ncls__{init}__e{epochs}__s{seed}"
    out = output_root / name
    argv = [
        "-m", "target_classifier_module.train_yolo_cls",
        "--epochs", str(epochs),
        "--batch-size", "32",
        "--image-size", "224",
        "--seed", str(seed),
        "--output", str(out),
    ]
    if init == "scratch":
        argv.append("--no-pretrained")
    return Experiment(name, "crop", "yolov8n-cls", init, epochs, seed, argv)


def _resnet_whole(init: str, epochs: int, seed: int, output_root: Path) -> Experiment:
    name = f"whole__resnet18__{init}__e{epochs}__s{seed}"
    out = output_root / name
    argv = [
        "-m", "target_classifier_module.train_whole_image",
        "--manifest-dir", "detector_module/artifacts/detection_dataset/manifests",
        "--epochs", str(epochs),
        "--batch-size", "32",
        "--image-size", "224",
        "--seed", str(seed),
        "--augmentation", "none",
        "--num-workers", "0",
        "--output", str(out),
    ]
    if init == "scratch":
        argv.append("--no-pretrained")
    return Experiment(name, "whole", "resnet18", init, epochs, seed, argv)


def _yolo_detect(init: str, epochs: int, seed: int, output_root: Path) -> Experiment:
    name = f"detect__yolov8n__{init}__e{epochs}__s{seed}"
    argv = [
        "-m", "detector_module.train_detector",
        "--epochs", str(epochs),
        "--patience", "20",
        "--batch-size", "16",
        "--image-size", "640",
        "--seed", str(seed),
        "--project", str(output_root),
        "--name", name,
        "--exist-ok",
    ]
    if init == "scratch":
        argv.append("--no-pretrained")
    return Experiment(
        name, "detect", "yolov8n", init, epochs, seed, argv,
        metrics_rel="baseline_summary.json",
    )


def build_matrix(
    output_root: Path, tracks: set[str], seeds: tuple[int, ...] = SEEDS
) -> list[Experiment]:
    """构造完整实验矩阵。

    两种 epoch 预算的用意：低预算与现有 baseline 严格同配置；高预算给随机初始化
    臂充分收敛的机会，用于把"预训练确实有用"与"从零只是没训够"区分开。
    """
    matrix: list[Experiment] = []
    for seed in seeds:
        for init in ("pretrained", "scratch"):
            if "crop" in tracks:
                for epochs in (12, 60):
                    matrix.append(_resnet_crop(init, epochs, seed, output_root))
                    matrix.append(_yolo_crop(init, epochs, seed, output_root))
            if "whole" in tracks:
                for epochs in (12, 40):
                    matrix.append(_resnet_whole(init, epochs, seed, output_root))
            if "detect" in tracks:
                for epochs in (100, 300):
                    matrix.append(_yolo_detect(init, epochs, seed, output_root))
    return matrix


def run_one(exp: Experiment, output_root: Path, dry_run: bool) -> dict:
    run_dir = output_root / exp.name
    metrics_path = run_dir / exp.metrics_rel
    record = {
        "name": exp.name,
        "track": exp.track,
        "model": exp.model,
        "init": exp.init,
        "epochs": exp.epochs,
        "seed": exp.seed,
        "run_dir": str(run_dir),
        "metrics_path": str(metrics_path),
        "command": [sys.executable, *exp.argv],
    }
    if metrics_path.is_file():
        record["status"] = "skipped"
        print(f"[SKIP] {exp.name}", flush=True)
        return record
    if dry_run:
        record["status"] = "dry-run"
        print(f"[DRY ] {exp.name}: {' '.join(exp.argv)}", flush=True)
        return record

    # 训练脚本要求目标目录为空，重跑前先清理残留的失败产物。
    if run_dir.exists() and exp.track != "detect":
        shutil.rmtree(run_dir, ignore_errors=True)

    print(f"[RUN ] {exp.name}", flush=True)
    started = time.perf_counter()
    completed = subprocess.run(
        [sys.executable, *exp.argv],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    elapsed = time.perf_counter() - started
    record["seconds"] = round(elapsed, 1)
    record["returncode"] = completed.returncode
    if completed.returncode != 0 or not metrics_path.is_file():
        record["status"] = "failed"
        record["stderr_tail"] = (completed.stderr or "")[-3000:]
        record["stdout_tail"] = (completed.stdout or "")[-1500:]
        print(f"[FAIL] {exp.name} rc={completed.returncode} ({elapsed:.1f}s)", flush=True)
        print((completed.stderr or "")[-1500:], flush=True)
    else:
        record["status"] = "ok"
        print(f"[DONE] {exp.name} ({elapsed:.1f}s)", flush=True)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="运行 ResNet18/YOLOv8 与 预训练/从零 对比实验矩阵")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--tracks",
        default="crop,whole,detect",
        help="逗号分隔，可选 crop / whole / detect",
    )
    parser.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    parser.add_argument(
        "--models",
        default="",
        help="逗号分隔的模型名过滤，留空表示全部；例如 resnet18 或 yolov8n-cls",
    )
    parser.add_argument(
        "--epochs-filter",
        default="",
        help="逗号分隔的 epoch 预算过滤，留空表示全部；例如 12",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--journal", type=Path, default=None)
    args = parser.parse_args()

    seeds = tuple(int(s) for s in args.seeds.split(",") if s.strip())
    tracks = {t.strip() for t in args.tracks.split(",") if t.strip()}
    unknown = tracks - {"crop", "whole", "detect"}
    if unknown:
        raise SystemExit(f"未知 track: {sorted(unknown)}")

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    journal_path = args.journal or (output_root / "matrix_journal.json")

    matrix = build_matrix(output_root, tracks, seeds)
    if args.models:
        wanted = {m.strip() for m in args.models.split(",") if m.strip()}
        matrix = [e for e in matrix if e.model in wanted]
    if args.epochs_filter:
        budgets = {int(e.strip()) for e in args.epochs_filter.split(",") if e.strip()}
        matrix = [e for e in matrix if e.epochs in budgets]
    print(f"实验单元数: {len(matrix)}  tracks={sorted(tracks)}  seeds={seeds}", flush=True)

    records = []
    for exp in matrix:
        records.append(run_one(exp, output_root, args.dry_run))
        journal_path.write_text(
            json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    ok = sum(1 for r in records if r["status"] == "ok")
    skipped = sum(1 for r in records if r["status"] == "skipped")
    failed = sum(1 for r in records if r["status"] == "failed")
    print(f"\n完成 {ok} / 跳过 {skipped} / 失败 {failed}，日志: {journal_path}", flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
