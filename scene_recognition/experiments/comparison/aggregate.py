"""汇总对比实验矩阵，输出多种子均值±标准差的对照表。

设计原则（对应 docs/诊断报告-场景捷径与模型选择缺陷.md）：

- 一律多种子报告 ``均值 ± 标准差``，不报单次结果。
- 显著性判定使用实测噪声下限：测试集 111 张图 / 473 个裁剪，
  组间差异小于约 1.2 个百分点一律标注为"落在噪声内"。
- 整图存在性指标带场景捷径警告，不得当作目标识别证据。
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RUNS = PROJECT_ROOT / "comparison" / "runs"

# 组间差异显著性阈值（百分点）。来源：3 种子实测 test Macro-F1 标准差 0.61pp，
# 取 2σ ≈ 1.2pp；同时 1 张测试图 = 0.90pp，1 个裁剪 = 0.21pp。
NOISE_FLOOR_PP = 1.2

RUN_NAME = re.compile(
    r"^(?P<track>crop|whole|detect|presence)__"
    r"(?P<model>[^_]+(?:-[^_]+)*)__"
    r"(?P<init>pretrained|scratch)__"
    r"e(?P<epochs>\d+)__"
    r"s(?P<seed>\d+)$"
)

# 每条 track 的主指标：(json 路径, 展示名)
TRACK_METRICS = {
    "crop": [("test.accuracy", "Accuracy"), ("test.macro_f1", "Macro-F1")],
    "whole": [
        ("test.exact_match_accuracy", "Exact Match"),
        ("test.macro_f1", "Macro-F1"),
    ],
    "presence": [
        ("test.exact_match_accuracy", "Exact Match"),
        ("test.macro_f1", "Macro-F1"),
    ],
    "detect": [
        ("test.map50", "mAP@0.5"),
        ("test.map50_95", "mAP@0.5:0.95"),
        ("test.precision", "Precision"),
        ("test.recall", "Recall"),
    ],
}

TRACK_TITLE = {
    "crop": "对比面 A：真实框裁剪四分类（同数据 2957 个裁剪，同指标）",
    "whole": "对比面 B：整图四类存在性（同 111 张测试图，同指标）",
    "presence": "对比面 B：整图四类存在性（同 111 张测试图，同指标）",
    "detect": "对比面 C：完整目标检测（仅 YOLOv8 可做，ResNet18 无定位能力）",
}


def dig(data: dict, dotted: str):
    node = data
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node if isinstance(node, (int, float)) else None


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return mean, math.sqrt(var)


def load_runs(runs_root: Path) -> list[dict]:
    """扫描 runs 目录，读取所有可解析的实验单元。"""
    records = []
    for run_dir in sorted(p for p in runs_root.iterdir() if p.is_dir()):
        match = RUN_NAME.match(run_dir.name)
        if not match:
            continue
        info = match.groupdict()
        # 分类/整图走 metrics.json，检测走 baseline_summary.json，
        # 存在性对照走 presence_metrics.json。
        for filename in ("metrics.json", "baseline_summary.json", "presence_metrics.json"):
            path = run_dir / filename
            if path.is_file():
                break
        else:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        records.append(
            {
                "run": run_dir.name,
                "track": info["track"],
                "model": info["model"],
                "init": info["init"],
                "epochs": int(info["epochs"]),
                "seed": int(info["seed"]),
                "source": str(path),
                "payload": payload,
            }
        )
    return records


def group(records: list[dict]) -> dict:
    """按 (track, model, init, epochs) 聚合多个种子。"""
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for rec in records:
        buckets[(rec["track"], rec["model"], rec["init"], rec["epochs"])].append(rec)
    summary = {}
    for key, group_records in buckets.items():
        track = key[0]
        entry = {
            "seeds": sorted(r["seed"] for r in group_records),
            "n": len(group_records),
            "metrics": {},
        }
        for dotted, label in TRACK_METRICS.get(track, []):
            values = [dig(r["payload"], dotted) for r in group_records]
            values = [v for v in values if v is not None]
            if not values:
                continue
            mean, std = mean_std(values)
            entry["metrics"][label] = {
                "mean": mean,
                "std": std,
                "n": len(values),
                "values": values,
            }
        summary[key] = entry
    return summary


def fmt(entry: dict, label: str) -> str:
    metric = entry["metrics"].get(label)
    if not metric:
        return "—"
    if metric["n"] == 1:
        return f"{metric['mean']:.2%}"
    return f"{metric['mean']:.2%} ± {metric['std'] * 100:.2f}"


def render(summary: dict) -> str:
    lines: list[str] = []
    lines.append("# ResNet18 vs YOLOv8 / 预训练 vs 从零 对照结果\n")
    lines.append(
        "> 全部数字为多随机种子的 **均值 ± 标准差（百分点）**。\n"
        f"> 测试集仅 111 张图 / 473 个裁剪，实测种子噪声 2σ ≈ {NOISE_FLOOR_PP} 个百分点，\n"
        f"> **组间差异小于 {NOISE_FLOOR_PP}pp 不能声称显著**。\n"
    )

    for track in ("crop", "whole", "presence", "detect"):
        keys = sorted(k for k in summary if k[0] == track)
        if not keys:
            continue
        labels = [label for _, label in TRACK_METRICS[track]]
        lines.append(f"\n## {TRACK_TITLE[track]}\n")
        header = "| 模型 | 初始化 | Epochs | 种子数 | " + " | ".join(labels) + " |"
        sep = "|---|---|---:|---:|" + "|".join(["---:"] * len(labels)) + "|"
        lines.append(header)
        lines.append(sep)
        for key in keys:
            entry = summary[key]
            _, model, init, epochs = key
            init_cn = "预训练" if init == "pretrained" else "从零"
            cells = " | ".join(fmt(entry, label) for label in labels)
            lines.append(f"| {model} | {init_cn} | {epochs} | {entry['n']} | {cells} |")

        # 预训练 vs 从零 的净增益
        lines.append("\n**预训练净增益**（预训练 − 从零，单位百分点）\n")
        lines.append("| 模型 | Epochs | " + " | ".join(labels) + " | 是否超出噪声 |")
        lines.append("|---|---:|" + "|".join(["---:"] * len(labels)) + "|---|")
        models = sorted({k[1] for k in keys})
        budgets = sorted({k[3] for k in keys})
        for model in models:
            for epochs in budgets:
                pre = summary.get((track, model, "pretrained", epochs))
                scr = summary.get((track, model, "scratch", epochs))
                if not pre or not scr:
                    continue
                deltas = []
                cells = []
                for label in labels:
                    a = pre["metrics"].get(label)
                    b = scr["metrics"].get(label)
                    if not a or not b:
                        cells.append("—")
                        continue
                    delta = (a["mean"] - b["mean"]) * 100
                    deltas.append(abs(delta))
                    cells.append(f"{delta:+.2f}")
                verdict = (
                    "✅ 显著" if deltas and max(deltas) >= NOISE_FLOOR_PP else "⚠️ 落在噪声内"
                )
                lines.append(
                    f"| {model} | {epochs} | " + " | ".join(cells) + f" | {verdict} |"
                )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="汇总对比实验矩阵")
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    runs_root = args.runs_root.resolve()
    if not runs_root.is_dir():
        raise SystemExit(f"运行目录不存在: {runs_root}")

    records = load_runs(runs_root)
    if not records:
        raise SystemExit(f"{runs_root} 下没有可解析的实验单元")
    summary = group(records)
    markdown = render(summary)

    output = args.output or (runs_root.parent / "comparison_results.md")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(markdown, encoding="utf-8")

    raw = {
        "|".join(str(part) for part in key): value for key, value in summary.items()
    }
    (runs_root.parent / "comparison_summary.json").write_text(
        json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(markdown)
    print(f"\n已写出: {output}")


if __name__ == "__main__":
    main()
