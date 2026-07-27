from __future__ import annotations

import argparse
import json
from pathlib import Path


def _overall_detection_metrics(report: dict) -> dict:
    if "test" in report:
        return report["test"]
    for row in report.get("slices", []):
        if row.get("group") == "overall" and row.get("value") == "all":
            return row
    raise ValueError("检测报告中没有overall/all或test指标")


def write_baseline_comparison(
    classifier_metrics_path: Path,
    detector_metrics_path: Path,
    output_path: Path,
    whole_image_metrics_path: Path | None = None,
) -> str:
    classifier = json.loads(classifier_metrics_path.read_text(encoding="utf-8"))
    detector = json.loads(detector_metrics_path.read_text(encoding="utf-8"))
    classification = classifier["test"]
    detection = _overall_detection_metrics(detector)
    detection_images = detector.get("test_image_count", "见检测报告")
    classification_result = (
        f"Accuracy {classification['accuracy']:.2%}；"
        f"Macro-F1 {classification['macro_f1']:.2%}"
    )
    detection_result = (
        f"Precision {detection['precision']:.2%}；"
        f"Recall {detection['recall']:.2%}；"
        f"mAP@0.5 {detection['map50']:.2%}；"
        f"mAP@0.5:0.95 {detection['map50_95']:.2%}"
    )
    classification_count = classification.get("sample_count", "见分类报告")
    crop_row = (
        "| ResNet18真实框裁剪分类 | 已知真实目标位置，只判断四类目标 "
        f"| {classification_count}个裁剪目标 | {classification_result} |"
    )
    whole_image_row = ""
    whole_image_explanation = ""
    whole_image_source = ""
    if whole_image_metrics_path is not None:
        whole_image = json.loads(whole_image_metrics_path.read_text(encoding="utf-8"))
        presence = whole_image["test"]
        whole_image_result = (
            f"Exact Match {presence['exact_match_accuracy']:.2%}；"
            f"Macro-F1 {presence['macro_f1']:.2%}"
        )
        whole_image_row = (
            "| ResNet18整图多标签识别 | 输入完整图片，判断四类目标是否存在 "
            f"| {presence.get('sample_count', '见整图报告')}张图片 | {whole_image_result} |\n"
        )
        whole_image_explanation = (
            "- 整图ResNet18是当前直接识别主线，可同时输出多个存在类别，"
            "但不输出位置和数量。\n"
        )
        whole_image_source = f"- 整图指标：`{whole_image_metrics_path.as_posix()}`\n"
    markdown = f"""# 三条识别基线对照

## 结论

三条基线解决的任务不同，**Exact Match、分类Accuracy与检测mAP不能直接比较**。

| 基线 | 输入与前提 | 样本 | 核心结果 |
|---|---|---:|---|
{whole_image_row}{crop_row}
| YOLOv8完整目标检测 | 输入完整图片，同时定位并分类 | {detection_images}张图片 | {detection_result} |

## 正确解释

{whole_image_explanation}- ResNet18真实框裁剪结果是“已知正确框以后，目标能否被分对”的分类上限，不包含寻找目标的难度。
- YOLOv8结果包含定位、分类、漏检和误检，是比赛基础目标检测指标的直接对照。

## 来源

{whole_image_source}- 裁剪分类指标：`{classifier_metrics_path.as_posix()}`
- 检测指标：`{detector_metrics_path.as_posix()}`
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")
    return markdown


def main() -> None:
    parser = argparse.ArgumentParser(description="生成整图、裁剪分类与YOLO检测基线对照报告")
    parser.add_argument("--whole-image", type=Path)
    parser.add_argument("--classifier", type=Path, required=True)
    parser.add_argument("--detector", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        write_baseline_comparison(
            args.classifier,
            args.detector,
            args.output,
            whole_image_metrics_path=args.whole_image,
        )
    )


if __name__ == "__main__":
    main()
