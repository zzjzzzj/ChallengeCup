"""从标准YOLO目录（images/{split} + labels/{split}）生成ResNet18目标裁剪数据。

与 prepare_crops.py 的区别：后者依赖 image_processing 的 scene_index.csv（标签与原图同目录），
而增广数据集使用 images/labels 分离的标准YOLO布局，因此单独提供本入口。
裁剪逻辑、padding 与 manifest 字段与 prepare_crops.py 保持一致，保证可比性。
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from PIL import Image

from scene_recognition.detector_module.boxes import parse_yolo_boxes
from scene_recognition.target_classifier_module import CLASS_NAMES
from scene_recognition.target_classifier_module.dataset import normalized_box_to_pixels

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
KNOWN_SENSORS = {"ir", "sar"}
KNOWN_SCENES = {"air", "sea", "urban", "forest"}


def infer_sensor_and_scene(image_name: str) -> tuple[str, str]:
    """从文件名推断传感器与场景，例如 ir_r1_base_air_000001__aug-rot180.png。"""

    stem = Path(image_name).stem
    parts = stem.split("__")[0].split("_")
    sensor = next((p for p in parts if p.lower() in KNOWN_SENSORS), "unknown")
    scene = next((p for p in parts if p.lower() in KNOWN_SCENES), "unknown")
    return sensor.lower(), scene.lower()


def build(
    dataset_root: Path,
    output_dir: Path,
    padding_ratio: float,
    splits: list[str],
) -> dict:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"输出目录不是空目录，请换用新目录: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    for split in splits:
        for class_name in CLASS_NAMES:
            (output_dir / split / class_name).mkdir(parents=True, exist_ok=True)

    crop_counts: Counter = Counter()
    source_counts: Counter = Counter()
    missing_labels: list[str] = []
    manifest_rows: list[dict[str, object]] = []

    for split in splits:
        image_dir = dataset_root / "images" / split
        label_dir = dataset_root / "labels" / split
        if not image_dir.is_dir():
            raise FileNotFoundError(f"缺少图片目录: {image_dir}")
        if not label_dir.is_dir():
            raise FileNotFoundError(f"缺少标签目录: {label_dir}")

        image_paths = sorted(
            p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES
        )
        for image_path in image_paths:
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                missing_labels.append(image_path.name)
                continue

            boxes = parse_yolo_boxes(label_path, len(CLASS_NAMES))
            sensor, scene = infer_sensor_and_scene(image_path.name)
            source_counts[split] += 1

            with Image.open(image_path) as opened:
                image = opened.convert("RGB")
                for box_index, box in enumerate(boxes):
                    bounds = normalized_box_to_pixels(box, image.size, padding_ratio)
                    class_name = CLASS_NAMES[box.class_id]
                    crop_name = f"{image_path.stem}__obj{box_index:03d}.png"
                    crop_path = (output_dir / split / class_name / crop_name).resolve()
                    image.crop(bounds).save(crop_path)
                    left, top, right, bottom = bounds
                    crop_counts[(split, class_name)] += 1
                    manifest_rows.append(
                        {
                            "crop_path": str(crop_path),
                            "source_image_path": str(image_path.resolve()),
                            "source_image_name": image_path.name,
                            "label_path": str(label_path.resolve()),
                            "split": split,
                            "sensor": sensor,
                            "scene": scene,
                            "box_index": box_index,
                            "class_id": box.class_id,
                            "class_name": class_name,
                            "x_center": box.x_center,
                            "y_center": box.y_center,
                            "width": box.width,
                            "height": box.height,
                            "left": left,
                            "top": top,
                            "right": right,
                            "bottom": bottom,
                        }
                    )

    manifest_path = output_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)

    summary = {
        "dataset_root": str(dataset_root),
        "manifest_path": str(manifest_path.resolve()),
        "padding_ratio": padding_ratio,
        "source_images_per_split": dict(source_counts),
        "crops_per_split": {
            split: sum(v for (s, _), v in crop_counts.items() if s == split)
            for split in splits
        },
        "crops_per_split_class": {
            f"{split}/{class_name}": count
            for (split, class_name), count in sorted(crop_counts.items())
        },
        "total_crops": len(manifest_rows),
        "missing_label_count": len(missing_labels),
        "missing_label_examples": missing_labels[:10],
    }
    (output_dir / "crop_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="从标准YOLO目录生成ResNet18四类目标裁剪数据"
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--padding-ratio", type=float, default=0.10)
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    args = parser.parse_args()
    summary = build(args.dataset_root, args.output, args.padding_ratio, args.splits)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
