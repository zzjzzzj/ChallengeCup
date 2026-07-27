"""Compare YOLO detection runs that differ only in their training set.

用于离线增广数据集的对照实验：各组必须使用相同验证集、相同随机种子、
并且都用 --no-builtin-aug 关闭 YOLO 自带在线增广，否则结论无法归因到增广本身。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


CLASS_ORDER = ("soldier", "small_aircraft", "warship", "tank")


def load_run(summary_path: Path) -> dict:
    """Read one baseline_summary.json and pull out the fields needed for comparison."""

    if not summary_path.is_file():
        raise FileNotFoundError(f"找不到运行摘要: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    split = summary.get("evaluation_split")
    if split is None:
        # 旧版摘要只写了 test，没有 evaluation_split 字段。
        split = "test" if "test" in summary else "val"
    if split not in summary:
        raise ValueError(f"运行摘要缺少 {split} 指标: {summary_path}")
    configuration = summary.get("configuration", {})
    return {
        "summary_path": summary_path,
        "evaluation_split": split,
        "metrics": summary[split],
        "epochs": configuration.get("epochs"),
        "data": configuration.get("data"),
        "batch_size": configuration.get("batch_size"),
        "builtin_augmentation_disabled": configuration.get(
            "builtin_augmentation_disabled"
        ),
    }


def _percent(value) -> str:
    if value is None:
        return "—"
    return f"{float(value):.2%}"


def build_comparison(
    runs: dict[str, dict], train_image_counts: dict[str, int] | None = None
) -> str:
    """Render a markdown comparison, making the training budget explicit."""

    if len(runs) < 2:
        raise ValueError("对照实验至少需要两组运行结果")
    train_image_counts = train_image_counts or {}

    splits = {run["evaluation_split"] for run in runs.values()}
    disabled = {run["builtin_augmentation_disabled"] for run in runs.values()}

    lines = ["# 离线增广数据集对照实验", ""]

    warnings = []
    if len(splits) > 1:
        warnings.append(
            f"**各组评估划分不一致（{sorted(splits)}），指标不可比。**"
        )
    if len(disabled) > 1:
        warnings.append(
            "**各组的YOLO内置增广开关不一致，无法把差异归因到离线增广。**"
        )
    if False in disabled:
        warnings.append(
            "至少有一组保留了YOLO内置在线增广（mosaic/fliplr等），"
            "该组的结果是「离线增广 + 在线增广」的合并效果，不能单独归因。"
        )
    if warnings:
        lines.extend(["> ⚠️ " + warning for warning in warnings])
        lines.append("")

    lines.extend(
        [
            "## 训练预算",
            "",
            "| 组 | 训练图片数 | Epoch | 图片呈现次数 | 评估划分 |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for name, run in runs.items():
        images = train_image_counts.get(name)
        epochs = run["epochs"]
        presentations = (
            f"{images * epochs:,}" if images is not None and epochs is not None else "—"
        )
        lines.append(
            f"| {name} | {images if images is not None else '—'} | "
            f"{epochs if epochs is not None else '—'} | {presentations} | "
            f"{run['evaluation_split']} |"
        )

    lines.extend(
        [
            "",
            "「图片呈现次数」= 训练图片数 × Epoch。两组该数值接近时，",
            "差异才不能被「只是训练得更久」解释。",
            "",
            "## 总体指标",
            "",
            "| 组 | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name, run in runs.items():
        metrics = run["metrics"]
        lines.append(
            f"| {name} | {_percent(metrics.get('precision'))} | "
            f"{_percent(metrics.get('recall'))} | {_percent(metrics.get('map50'))} | "
            f"{_percent(metrics.get('map50_95'))} |"
        )

    lines.extend(["", "## 分类别 mAP@0.5", "", "| 组 | " + " | ".join(CLASS_ORDER) + " |"])
    lines.append("|---|" + "---:|" * len(CLASS_ORDER))
    for name, run in runs.items():
        per_class = run["metrics"].get("per_class", {})
        cells = [_percent(per_class.get(cls, {}).get("map50")) for cls in CLASS_ORDER]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")

    lines.extend(["", "## 来源", ""])
    for name, run in runs.items():
        lines.append(f"- {name}: `{run['summary_path'].as_posix()}`")

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="对比多组只在训练集上有差异的YOLO检测实验"
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="名称=运行目录",
        help="可重复。例如 --run 增广组=detector_module/runs/ab_augmented_e60",
    )
    parser.add_argument(
        "--train-images",
        action="append",
        default=[],
        metavar="名称=数量",
        help="可重复。给出该组的训练图片数，用于计算图片呈现次数。",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    runs = {}
    for item in args.run:
        if "=" not in item:
            raise SystemExit(f"--run 需要「名称=运行目录」格式: {item}")
        name, _, directory = item.partition("=")
        runs[name] = load_run(Path(directory) / "baseline_summary.json")

    counts = {}
    for item in args.train_images:
        if "=" not in item:
            raise SystemExit(f"--train-images 需要「名称=数量」格式: {item}")
        name, _, value = item.partition("=")
        counts[name] = int(value)

    markdown = build_comparison(runs, counts)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
