"""汇总 8 组实验结果，产出对比表（Markdown + CSV）。

务必注意指标口径不同，不要把两列数字直接并排当成「谁更强」：
  - ResNet18 报的是**给定真值框裁剪后**的四类分类准确率／宏F1，定位是白送的；
  - YOLOv8 报的是 mAP，必须自己把目标找出来**再**分类对。
  两者天然不可比。要横向比较，请看 yolo_gt_box_classification.py 产出的
  「YOLOv8 在真值框上的分类准确率」，那一列才和 ResNet18 同口径。
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESNET_RUNS = ROOT / "scene_recognition/target_classifier_module/runs"
YOLO_RUNS = ROOT / "scene_recognition/detector_module/runs"
REPORT_DIR = ROOT / "docs/comparison"

DATA_LABEL = {"aug": "增广集(4400张)", "noaug": "原始集(595张)"}
WEIGHT_LABEL = {"pretrained": "预训练", "scratch": "从零训练"}


def pct(value) -> str:
    return "—" if value is None else f"{value * 100:.2f}"


def load_resnet(data_tag: str, weight_tag: str) -> dict | None:
    path = RESNET_RUNS / f"cmp_resnet18_{data_tag}_{weight_tag}" / "metrics.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    evaluation = payload.get("test") or payload.get("validation") or {}
    return {
        "data": data_tag,
        "weights": weight_tag,
        "pretrained": payload.get("config", {}).get("pretrained"),
        "epochs": payload.get("config", {}).get("epochs"),
        "accuracy": evaluation.get("accuracy"),
        "macro_f1": evaluation.get("macro_f1"),
        "macro_recall": evaluation.get("macro_recall"),
        "ir_accuracy": evaluation.get("ir_accuracy"),
        "sar_accuracy": evaluation.get("sar_accuracy"),
        "per_class_recall": evaluation.get("per_class_recall", {}),
    }


def load_yolo(data_tag: str, weight_tag: str) -> dict | None:
    path = YOLO_RUNS / f"cmp_yolov8n_{data_tag}_{weight_tag}" / "ablation_summary.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    evaluation = payload.get("eval", {})
    return {
        "data": data_tag,
        "weights": weight_tag,
        "pretrained": payload.get("configuration", {}).get("pretrained"),
        "epochs": payload.get("configuration", {}).get("epochs"),
        "precision": evaluation.get("precision"),
        "recall": evaluation.get("recall"),
        "map50": evaluation.get("map50"),
        "map50_95": evaluation.get("map50_95"),
        "per_class": evaluation.get("per_class", {}),
    }


def gt_box_accuracy(data_tag: str, weight_tag: str) -> float | None:
    path = YOLO_RUNS / f"cmp_yolov8n_{data_tag}_{weight_tag}" / "gt_box_classification.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8")).get("accuracy")


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    combos = [(d, w) for d in ("noaug", "aug") for w in ("pretrained", "scratch")]
    resnet = {c: load_resnet(*c) for c in combos}
    yolo = {c: load_yolo(*c) for c in combos}

    lines: list[str] = []
    lines.append("# ResNet18 与 YOLOv8 对比实验结果\n")
    lines.append(
        "全部 8 组共用同一批 **155 张未增广验证图**；"
        "「原始集」是增广集中不含 `__aug-` 的那 595 张原图，"
        "因此增广组与原始组之间唯一变量就是有没有离线增广。\n"
    )

    lines.append("\n## 表1 ResNet18：真值框裁剪四类分类\n")
    lines.append("| 训练数据 | 权重 | 准确率(%) | 宏F1(%) | 宏召回(%) | IR准确率(%) | SAR准确率(%) |")
    lines.append("|---|---|---|---|---|---|---|")
    for combo in combos:
        row = resnet[combo]
        if row is None:
            lines.append(f"| {DATA_LABEL[combo[0]]} | {WEIGHT_LABEL[combo[1]]} | 未完成 | | | | |")
            continue
        lines.append(
            f"| {DATA_LABEL[combo[0]]} | {WEIGHT_LABEL[combo[1]]} | {pct(row['accuracy'])} | "
            f"{pct(row['macro_f1'])} | {pct(row['macro_recall'])} | "
            f"{pct(row['ir_accuracy'])} | {pct(row['sar_accuracy'])} |"
        )

    lines.append("\n## 表2 YOLOv8n：端到端检测\n")
    lines.append("| 训练数据 | 权重 | mAP@0.5(%) | mAP@0.5:0.95(%) | 精确率(%) | 召回率(%) |")
    lines.append("|---|---|---|---|---|---|")
    for combo in combos:
        row = yolo[combo]
        if row is None:
            lines.append(f"| {DATA_LABEL[combo[0]]} | {WEIGHT_LABEL[combo[1]]} | 未完成 | | | |")
            continue
        lines.append(
            f"| {DATA_LABEL[combo[0]]} | {WEIGHT_LABEL[combo[1]]} | {pct(row['map50'])} | "
            f"{pct(row['map50_95'])} | {pct(row['precision'])} | {pct(row['recall'])} |"
        )

    lines.append("\n## 表3 预训练带来多少提升（同数据集内纵向对比）\n")
    lines.append("| 模型 | 训练数据 | 预训练 | 从零 | 差值(百分点) |")
    lines.append("|---|---|---|---|---|")
    for data_tag in ("noaug", "aug"):
        pre, scr = resnet[(data_tag, "pretrained")], resnet[(data_tag, "scratch")]
        if pre and scr and pre["macro_f1"] is not None and scr["macro_f1"] is not None:
            delta = (pre["macro_f1"] - scr["macro_f1"]) * 100
            lines.append(
                f"| ResNet18 (宏F1) | {DATA_LABEL[data_tag]} | {pct(pre['macro_f1'])} | "
                f"{pct(scr['macro_f1'])} | {delta:+.2f} |"
            )
    for data_tag in ("noaug", "aug"):
        pre, scr = yolo[(data_tag, "pretrained")], yolo[(data_tag, "scratch")]
        if pre and scr and pre["map50"] is not None and scr["map50"] is not None:
            delta = (pre["map50"] - scr["map50"]) * 100
            lines.append(
                f"| YOLOv8n (mAP@0.5) | {DATA_LABEL[data_tag]} | {pct(pre['map50'])} | "
                f"{pct(scr['map50'])} | {delta:+.2f} |"
            )

    lines.append("\n## 表4 离线增广带来多少提升（同权重设置内纵向对比）\n")
    lines.append("| 模型 | 权重 | 原始集 | 增广集 | 差值(百分点) |")
    lines.append("|---|---|---|---|---|")
    for weight_tag in ("pretrained", "scratch"):
        base, augd = resnet[("noaug", weight_tag)], resnet[("aug", weight_tag)]
        if base and augd and base["macro_f1"] is not None and augd["macro_f1"] is not None:
            delta = (augd["macro_f1"] - base["macro_f1"]) * 100
            lines.append(
                f"| ResNet18 (宏F1) | {WEIGHT_LABEL[weight_tag]} | {pct(base['macro_f1'])} | "
                f"{pct(augd['macro_f1'])} | {delta:+.2f} |"
            )
    for weight_tag in ("pretrained", "scratch"):
        base, augd = yolo[("noaug", weight_tag)], yolo[("aug", weight_tag)]
        if base and augd and base["map50"] is not None and augd["map50"] is not None:
            delta = (augd["map50"] - base["map50"]) * 100
            lines.append(
                f"| YOLOv8n (mAP@0.5) | {WEIGHT_LABEL[weight_tag]} | {pct(base['map50'])} | "
                f"{pct(augd['map50'])} | {delta:+.2f} |"
            )

    lines.append("\n## 表5 同口径对比：两个模型都在真值框上做分类\n")
    lines.append(
        "表1 和表2 的数字**不可直接比较**（一个定位白送、一个要自己找目标）。"
        "下表把 YOLOv8 的预测按最大IoU匹配到真值框后只看分类对错，与 ResNet18 同口径。\n"
    )
    lines.append("| 训练数据 | 权重 | ResNet18 准确率(%) | YOLOv8n 真值框分类准确率(%) |")
    lines.append("|---|---|---|---|")
    for combo in combos:
        row = resnet[combo]
        acc = gt_box_accuracy(*combo)
        left = pct(row["accuracy"]) if row else "未完成"
        lines.append(
            f"| {DATA_LABEL[combo[0]]} | {WEIGHT_LABEL[combo[1]]} | {left} | {pct(acc)} |"
        )

    report = REPORT_DIR / "comparison_report.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")

    csv_path = REPORT_DIR / "comparison_results.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["模型", "训练数据", "权重", "主指标名", "主指标(%)", "次指标名", "次指标(%)"]
        )
        for combo in combos:
            row = resnet[combo]
            if row:
                writer.writerow(
                    ["ResNet18", DATA_LABEL[combo[0]], WEIGHT_LABEL[combo[1]],
                     "准确率", pct(row["accuracy"]), "宏F1", pct(row["macro_f1"])]
                )
        for combo in combos:
            row = yolo[combo]
            if row:
                writer.writerow(
                    ["YOLOv8n", DATA_LABEL[combo[0]], WEIGHT_LABEL[combo[1]],
                     "mAP@0.5", pct(row["map50"]), "mAP@0.5:0.95", pct(row["map50_95"])]
                )

    print("\n".join(lines))
    print(f"\n已写出:\n  {report}\n  {csv_path}")


if __name__ == "__main__":
    main()
