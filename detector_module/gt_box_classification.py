"""把 YOLOv8 拉到和 ResNet18 同一口径：只考核「真值框内的目标分类对不对」。

动机：ResNet18 拿到的是按真值框裁好的图，定位是白送的；YOLOv8 要自己找目标。
直接拿 ResNet18 的准确率和 YOLOv8 的 mAP 并排是错的。
本脚本把 YOLOv8 的预测框按最大 IoU 匹配到真值框，匹配上的只看类别对错，于是：

  matched_accuracy —— 「已经找到目标的前提下分类准不准」，与 ResNet18 准确率同口径
  match_rate       —— 有多少真值框被找到了（ResNet18 恒为 100%，这是它白拿的部分）
  end_to_end_accuracy —— 既要找到又要分对，才是 YOLOv8 的真实难度
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import yaml
from ultralytics import YOLO

from detector_module.boxes import YoloBox, box_iou, parse_yolo_boxes

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def resolve_val_images(data_yaml: Path) -> tuple[list[Path], Path]:
    config = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    base = Path(config.get("path", data_yaml.parent))
    val_entry = config["val"]
    val_path = Path(val_entry)
    if not val_path.is_absolute():
        val_path = base / val_path
    if val_path.is_dir():
        images = sorted(p for p in val_path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    else:
        images = [Path(line.strip()) for line in
                  val_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    names = config["names"]
    class_names = [str(names[i] if i in names else names[str(i)]) for i in range(len(names))]
    return images, class_names


def label_for(image_path: Path) -> Path:
    parts = list(image_path.parts)
    for index in range(len(parts) - 1, -1, -1):
        if parts[index] == "images":
            parts[index] = "labels"
            break
    return Path(*parts).with_suffix(".txt")


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLOv8 在真值框上的分类准确率")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--image-size", type=int, default=640)
    args = parser.parse_args()

    images, class_names = resolve_val_images(args.data)
    model = YOLO(str(args.weights))

    total = matched = correct = 0
    confusion: Counter = Counter()
    per_class_total: Counter = Counter()
    per_class_correct: Counter = Counter()
    per_class_matched: Counter = Counter()

    for image_path in images:
        label_path = label_for(image_path)
        if not label_path.exists():
            continue
        truths = parse_yolo_boxes(label_path, len(class_names))
        if not truths:
            continue

        result = model.predict(
            str(image_path), imgsz=args.image_size, conf=args.conf, verbose=False
        )[0]
        predictions = []
        for box in result.boxes:
            x, y, w, h = box.xywhn[0].tolist()
            predictions.append((int(box.cls.item()), YoloBox(int(box.cls.item()), x, y, w, h)))

        for truth in truths:
            total += 1
            per_class_total[class_names[truth.class_id]] += 1
            best_iou, best_class = 0.0, None
            for predicted_class, predicted_box in predictions:
                iou = box_iou(truth, predicted_box)
                if iou > best_iou:
                    best_iou, best_class = iou, predicted_class
            if best_iou < args.iou_threshold or best_class is None:
                confusion[(class_names[truth.class_id], "漏检")] += 1
                continue
            matched += 1
            per_class_matched[class_names[truth.class_id]] += 1
            confusion[(class_names[truth.class_id], class_names[best_class])] += 1
            if best_class == truth.class_id:
                correct += 1
                per_class_correct[class_names[truth.class_id]] += 1

    summary = {
        "weights": str(args.weights.resolve()),
        "data": str(args.data.resolve()),
        "iou_threshold": args.iou_threshold,
        "conf_threshold": args.conf,
        "gt_box_count": total,
        "matched_count": matched,
        "match_rate": matched / total if total else None,
        # 与 ResNet18 同口径的那一列
        "accuracy": correct / matched if matched else None,
        "end_to_end_accuracy": correct / total if total else None,
        "per_class": {
            name: {
                "gt": per_class_total[name],
                "matched": per_class_matched[name],
                "correct": per_class_correct[name],
                "accuracy": (
                    per_class_correct[name] / per_class_matched[name]
                    if per_class_matched[name]
                    else None
                ),
            }
            for name in class_names
        },
        "confusion": {f"{k[0]}->{k[1]}": v for k, v in sorted(confusion.items())},
        "note": "accuracy 只统计已匹配上的真值框，用于与 ResNet18 裁剪分类同口径比较；"
                "match_rate 是 ResNet18 白拿的定位部分。",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
