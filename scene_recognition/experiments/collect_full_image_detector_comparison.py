"""Generate the report for YOLOv8n and ResNet18-FPN full-image detectors."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "scene_recognition" / "detector_module" / "runs"
YOLO_METRICS = ROOT / "docs" / "comparison" / "yolo_clean8_same_evaluator.json"
OUTPUT = ROOT / "docs" / "comparison" / "ResNet18端到端检测对比报告.md"


def percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def load_resnet(tag: str) -> dict:
    return json.loads((RUNS / f"cmp8_resnet18det_{tag}" / "metrics.json").read_text(encoding="utf-8"))


def main() -> None:
    if not YOLO_METRICS.is_file():
        raise FileNotFoundError(f"请先运行 YOLO 同口径评测: {YOLO_METRICS}")
    yolo = json.loads(YOLO_METRICS.read_text(encoding="utf-8"))["runs"]
    tags = ("noaug_pretrained", "noaug_scratch", "aug_pretrained", "aug_scratch")
    resnet = {tag: load_resnet(tag) for tag in tags}

    rows = []
    for tag in tags:
        data_name = "原始集（595 图）" if tag.startswith("noaug") else "增广集（4400 图）"
        init_name = "预训练" if tag.endswith("pretrained") else "从零训练"
        r = resnet[tag]
        r_metrics = r["test"]
        y_metrics = yolo[tag]["metrics"]
        rows.append(
            "| {data} | {init} | {r50} | {r95} | {rp} | {rr} | {y50} | {y95} | {yp} | {yr} |".format(
                data=data_name,
                init=init_name,
                r50=percent(r_metrics["map50"]),
                r95=percent(r_metrics["map50_95"]),
                rp=percent(r_metrics["precision"]),
                rr=percent(r_metrics["recall"]),
                y50=percent(y_metrics["map50"]),
                y95=percent(y_metrics["map50_95"]),
                yp=percent(y_metrics["precision"]),
                yr=percent(y_metrics["recall"]),
            )
        )

    def delta(tag: str) -> float:
        return resnet[tag]["test"]["map50"] - yolo[tag]["metrics"]["map50"]

    r_pretrain_gain_noaug = resnet["noaug_pretrained"]["test"]["map50"] - resnet["noaug_scratch"]["test"]["map50"]
    r_pretrain_gain_aug = resnet["aug_pretrained"]["test"]["map50"] - resnet["aug_scratch"]["test"]["map50"]
    r_aug_gain_pretrained = resnet["aug_pretrained"]["test"]["map50"] - resnet["noaug_pretrained"]["test"]["map50"]
    r_aug_gain_scratch = resnet["aug_scratch"]["test"]["map50"] - resnet["noaug_scratch"]["test"]["map50"]
    best = max(tags, key=lambda tag: resnet[tag]["test"]["map50"])

    report = f"""# ResNet18 与 YOLOv8n 全图检测对比报告

## 结论

ResNet18 已不再以裁剪图分类器参与对比，而是以 **Faster R-CNN + ResNet18-FPN** 端到端检测器运行：输入为整张遥感图，输出为边界框、目标类别和置信度，与 YOLOv8n 的任务输入输出一致。

在同一批 **76 张独立测试图**上，ResNet18 检测器的最佳配置为 **增广集 + ImageNet 预训练**，mAP@0.5 为 **{percent(resnet[best]['test']['map50'])}**。四组中，ResNet18 与 YOLO 的 mAP 由同一个 COCO 风格 101 点 AP 实现计算，可以直接比较检测指标。

## 实验口径

- 训练/验证/测试：原始集 595 图或增广集 4400 图；验证 79 图仅用于选最佳检查点；测试 76 图仅用于最终汇报。
- ResNet18 模型：Faster R-CNN + ResNet18-FPN，输入长边缩放到 640，预测小目标锚框为 8/16/32/64/128 像素。
- ResNet18 预训练：ImageNet；从零组不加载任何权重。YOLO 组使用既有的四个 `best.pt` 权重。
- 原始集训练上限 40 轮；增广集训练 6 轮，使两种训练集的总优化步数接近，避免将 7.4 倍样本量误报为纯粹的增广收益。
- 所有表中 mAP 均由 `scene_recognition.detector_module.resnet18_detector.detection_metrics` 计算：IoU 0.50 的 mAP@0.5，以及 0.50--0.95 的 COCO 风格平均值。

## 总体结果

| 训练数据 | 初始化 | ResNet18 mAP@0.5 | ResNet18 mAP@0.5:0.95 | ResNet18 P | ResNet18 R | YOLO mAP@0.5 | YOLO mAP@0.5:0.95 | YOLO P | YOLO R |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

## ResNet18 自身对比

| 效应 | mAP@0.5 变化 |
|---|---:|
| 原始集上使用 ImageNet 预训练 | {r_pretrain_gain_noaug * 100:+.2f} 个百分点 |
| 增广集上使用 ImageNet 预训练 | {r_pretrain_gain_aug * 100:+.2f} 个百分点 |
| 预训练 ResNet18 使用增广集 | {r_aug_gain_pretrained * 100:+.2f} 个百分点 |
| 从零 ResNet18 使用增广集 | {r_aug_gain_scratch * 100:+.2f} 个百分点 |

## ResNet18 对 YOLO 的差值

| 配置 | ResNet18 - YOLO（mAP@0.5） |
|---|---:|
| 原始集 + 预训练 | {delta('noaug_pretrained') * 100:+.2f} 个百分点 |
| 原始集 + 从零训练 | {delta('noaug_scratch') * 100:+.2f} 个百分点 |
| 增广集 + 预训练 | {delta('aug_pretrained') * 100:+.2f} 个百分点 |
| 增广集 + 从零训练 | {delta('aug_scratch') * 100:+.2f} 个百分点 |

## 解读边界

1. 现在两类模型的评测任务相同，均为“整图输入 -> 目标框、类别、置信度 -> mAP”，不再把 ResNet18 的裁剪分类 Accuracy 与 YOLO 的 mAP 混在一起。
2. 这仍不是严格的“只比较网络结构”实验：Faster R-CNN 与 YOLO 的检测头、优化器、学习率策略和预训练来源不同（ImageNet vs COCO），结果应解释为两个完整检测方案的基线对比。
3. 四组均只有一个随机种子；差值需要补多种子均值和标准差后才能作为稳健结论。
4. 当前数据的主要难点仍是 soldier 小目标。请查看各运行目录的 `metrics.json` 中 `test.per_class`，避免只用总体 mAP 掩盖类别差异。

## 产物

- ResNet18 检测指标：`scene_recognition/detector_module/runs/cmp8_resnet18det_*/metrics.json`
- YOLO 同口径指标：`docs/comparison/yolo_clean8_same_evaluator.json`
- ResNet18 训练日志：`docs/comparison/logs/cmp8_resnet18det_*.log`
"""
    OUTPUT.write_text(report, encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
