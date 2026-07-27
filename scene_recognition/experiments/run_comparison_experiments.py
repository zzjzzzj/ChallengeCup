"""ResNet18 vs YOLOv8、预训练 vs 从零 的 8 组对比实验调度器。

实验设计
--------
两个自变量各两档，两种模型各跑一遍，共 2x2x2 = 8 组：
  - 模型：ResNet18（真值框裁剪后做四类分类） / YOLOv8n（端到端检测）
  - 数据：增广集 4400 张训练图 / 未增广 595 张训练图
  - 权重：ImageNet(或COCO)预训练 / 从零随机初始化

关键：两种数据设置**共用同一批 155 张验证图**（全部未增广），
且未增广组就是增广集中不含 __aug- 的那 595 张原图，
因此增广组与未增广组之间唯一变量就是「有没有离线增广」。
不要拿仓库里旧的 detection_dataset（525/114/111 划分）作对照——
它与这里的 155 张验证集只有 6 张重合，指标不可比。

可重复执行：已完成的组会被跳过，中断后重跑即可续上。
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CROPS_AUG = ROOT / "scene_recognition/target_classifier_module/artifacts/target_crops_augmented"
MANIFEST_SRC = CROPS_AUG / "manifest.csv"
MANIFEST_AUG = CROPS_AUG / "manifest_aug.csv"
MANIFEST_NOAUG = CROPS_AUG / "manifest_noaug.csv"
RESNET_RUNS = ROOT / "scene_recognition/target_classifier_module/runs"
YOLO_RUNS = ROOT / "scene_recognition/detector_module/runs"
REPORT_DIR = ROOT / "docs/comparison"

RESNET_EPOCHS = 40
YOLO_EPOCHS = 150


def build_manifests() -> dict:
    """派生两份训练清单：增广组 与 未增广对照组。

    两点处理：
      1. 未增广组 = 剔除文件名含 __aug- 的裁剪行，验证集原样保留（本来就没增广）；
      2. training.py 强制要求 test 划分，而增广数据集只给了 train/val。
         这里让 test 复用同一批 val（不额外占磁盘，只多写一份清单行）。
         代价：模型选择与最终汇报在同一批图上，绝对值偏乐观；
         但 8 组实验协议完全一致，横向对比依然成立。
    """

    if not MANIFEST_SRC.is_file():
        raise FileNotFoundError(f"缺少裁剪清单: {MANIFEST_SRC}")

    with MANIFEST_SRC.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    fields = list(rows[0].keys())

    val_rows = [r for r in rows if r["split"] == "val"]
    test_rows = [dict(r, split="test") for r in val_rows]

    def write(path: Path, train_rows: list[dict]) -> None:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(train_rows + val_rows + test_rows)

    aug_train = [r for r in rows if r["split"] == "train"]
    noaug_train = [r for r in aug_train if "__aug-" not in r["source_image_name"]]
    write(MANIFEST_AUG, aug_train)
    write(MANIFEST_NOAUG, noaug_train)

    stats = {
        "aug_train_crops": len(aug_train),
        "noaug_train_crops": len(noaug_train),
        "shared_val_crops": len(val_rows),
        "test_equals_val": True,
    }
    if not noaug_train or len(noaug_train) >= len(aug_train):
        raise ValueError(f"未增广训练集切分异常: {stats}")
    return stats


def experiments() -> list[dict]:
    """先跑快的，让结果早点可看；最慢的增广版 YOLO 放最后。"""

    specs = []
    for data_tag, manifest in (("noaug", MANIFEST_NOAUG), ("aug", MANIFEST_AUG)):
        for weight_tag, extra in (("pretrained", []), ("scratch", ["--no-pretrained"])):
            name = f"cmp_resnet18_{data_tag}_{weight_tag}"
            out = RESNET_RUNS / name
            specs.append(
                {
                    "name": name,
                    "model": "resnet18",
                    "data": data_tag,
                    "weights": weight_tag,
                    "done_marker": out / "metrics.json",
                    "cmd": [
                        # -u 关掉 stdout 缓冲。子进程的 stdout 被重定向到日志文件后默认是
                        # 8KB 块缓冲，training.py 每轮 print 的那行 JSON 会一直攒在缓冲区里，
                        # 导致训练几小时日志始终 0 字节、无法中途查看进度。
                        sys.executable, "-u", "-m", "scene_recognition.target_classifier_module.train_classifier",
                        "--manifest", str(manifest),
                        "--output", str(out),
                        "--epochs", str(RESNET_EPOCHS),
                        "--batch-size", "32",
                        # 必须是 0。training.py 建了 train/val/test 三个 DataLoader 且没设
                        # persistent_workers，Windows 用 spawn 起 worker，每轮都要重建一遍，
                        # 单个 worker 光 import torch 就约 9 秒。实测 num_workers=4 时
                        # 2388 张裁剪跑一轮要 300 秒、GPU 利用率 0%；改成 0 之后 17663 张
                        # 只要 79 秒，等效快约 28 倍。仓库既有基线也一直用 0，口径一致。
                        "--num-workers", "0",
                        "--augmentation", "none",
                        "--seed", "42",
                        *extra,
                    ],
                }
            )

    for data_tag, yaml_name in (("noaug", "data_noaug.yaml"), ("aug", "data_augmented.yaml")):
        for weight_tag, extra in (("pretrained", []), ("scratch", ["--no-pretrained"])):
            name = f"cmp_yolov8n_{data_tag}_{weight_tag}"
            specs.append(
                {
                    "name": name,
                    "model": "yolov8n",
                    "data": data_tag,
                    "weights": weight_tag,
                    "done_marker": YOLO_RUNS / name / "ablation_summary.json",
                    "cmd": [
                        sys.executable, "-m", "scene_recognition.detector_module.train_detector_ablation",
                        "--data", str(ROOT / "scene_recognition/detector_module/configs" / yaml_name),
                        "--name", name,
                        "--epochs", str(YOLO_EPOCHS),
                        "--batch-size", "16",
                        "--workers", "4",
                        "--eval-split", "val",
                        "--seed", "42",
                        "--exist-ok",
                        *extra,
                    ],
                }
            )
    return specs


def run_one(spec: dict, log_dir: Path) -> dict:
    log_path = log_dir / f"{spec['name']}.log"
    started = time.time()
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        process = subprocess.run(
            spec["cmd"], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, text=True
        )
    elapsed = time.time() - started
    return {
        "name": spec["name"],
        "returncode": process.returncode,
        "minutes": round(elapsed / 60, 1),
        "log": str(log_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="运行 ResNet18/YOLOv8 对比实验")
    parser.add_argument("--only", nargs="*", help="只跑指定实验名")
    parser.add_argument("--force", action="store_true", help="忽略已完成标记，全部重跑")
    args = parser.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    log_dir = REPORT_DIR / "logs"
    log_dir.mkdir(exist_ok=True)

    stats = build_manifests()
    print(json.dumps({"manifest_stats": stats}, ensure_ascii=False), flush=True)

    specs = experiments()
    if args.only:
        specs = [s for s in specs if s["name"] in set(args.only)]

    results = []
    for index, spec in enumerate(specs, start=1):
        head = f"[{index}/{len(specs)}] {spec['name']}"
        if spec["done_marker"].is_file() and not args.force:
            print(f"{head}  已完成，跳过", flush=True)
            results.append({"name": spec["name"], "returncode": 0, "skipped": True})
            continue
        print(f"{head}  开始训练 ...", flush=True)
        outcome = run_one(spec, log_dir)
        flag = "OK" if outcome["returncode"] == 0 else f"失败(码 {outcome['returncode']})"
        print(f"{head}  {flag}，耗时 {outcome['minutes']} 分钟", flush=True)
        results.append(outcome)

    (REPORT_DIR / "run_log.json").write_text(
        json.dumps({"manifest_stats": stats, "runs": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    failed = [r for r in results if r.get("returncode")]
    print(f"\n全部结束：{len(results) - len(failed)} 成功 / {len(failed)} 失败", flush=True)
    for item in failed:
        print(f"  失败: {item['name']} -> 日志 {item.get('log')}", flush=True)


if __name__ == "__main__":
    main()
