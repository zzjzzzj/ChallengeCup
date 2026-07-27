"""Run the four full-image ResNet18 detector groups used in the clean comparison."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "detector_module" / "runs"
LOG_DIR = ROOT / "docs" / "comparison" / "logs"


def experiment_specs(noaug_epochs: int, aug_epochs: int, batch_size: int) -> list[dict]:
    specs: list[dict] = []
    for data_tag, yaml_name, epochs in (
        ("noaug", "data_clean_noaug.yaml", noaug_epochs),
        ("aug", "data_clean_aug.yaml", aug_epochs),
    ):
        for weight_tag, extra in (("pretrained", []), ("scratch", ["--no-pretrained"])):
            name = f"cmp8_resnet18det_{data_tag}_{weight_tag}"
            specs.append(
                {
                    "name": name,
                    "done_marker": RUNS / name / "metrics.json",
                    "command": [
                        sys.executable,
                        "-u",
                        "-m",
                        "detector_module.resnet18_detector",
                        "--data",
                        str(ROOT / "detector_module" / "configs" / yaml_name),
                        "--output",
                        str(RUNS / name),
                        "--epochs",
                        str(epochs),
                        "--patience",
                        "10",
                        "--batch-size",
                        str(batch_size),
                        "--workers",
                        "0",
                        "--seed",
                        "42",
                        *extra,
                    ],
                }
            )
    return specs


def main() -> None:
    parser = argparse.ArgumentParser(description="运行 ResNet18 端到端检测的四组干净对比实验")
    parser.add_argument("--only", nargs="*", help="只执行指定实验名")
    parser.add_argument("--noaug-epochs", type=int, default=40)
    parser.add_argument(
        "--aug-epochs",
        type=int,
        default=6,
        help="保持与原始集约相同优化步数；4400 / 595 约为 7.4。",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--force", action="store_true", help="忽略已有 metrics.json 后重跑")
    args = parser.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    specs = experiment_specs(args.noaug_epochs, args.aug_epochs, args.batch_size)
    if args.only:
        requested = set(args.only)
        specs = [spec for spec in specs if spec["name"] in requested]
    if not specs:
        raise ValueError("没有匹配的实验名")

    results = []
    for index, spec in enumerate(specs, start=1):
        if spec["done_marker"].is_file() and not args.force:
            print(f"[{index}/{len(specs)}] {spec['name']} 已完成，跳过", flush=True)
            results.append({"name": spec["name"], "returncode": 0, "skipped": True})
            continue
        log_path = LOG_DIR / f"{spec['name']}.log"
        print(f"[{index}/{len(specs)}] {spec['name']} 开始训练", flush=True)
        started_at = time.perf_counter()
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(spec["command"], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        minutes = round((time.perf_counter() - started_at) / 60, 2)
        result = {
            "name": spec["name"],
            "returncode": completed.returncode,
            "minutes": minutes,
            "log": str(log_path),
        }
        results.append(result)
        print(f"[{index}/{len(specs)}] {spec['name']} 结束，返回码 {completed.returncode}，耗时 {minutes} 分钟", flush=True)
        if completed.returncode:
            break

    output = ROOT / "docs" / "comparison" / "resnet18_detector_run_log.json"
    output.write_text(json.dumps({"runs": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    failures = [result for result in results if result["returncode"]]
    if failures:
        raise SystemExit(f"存在失败实验，详情见 {failures[0]['log']}")


if __name__ == "__main__":
    main()
