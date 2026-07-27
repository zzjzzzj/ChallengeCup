from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path

from PIL import Image

from detector_module.boxes import YoloBox, parse_yolo_boxes
from detector_module.dataset import VALID_SPLITS, read_scene_index
from target_classifier_module import CLASS_NAMES


def normalized_box_to_pixels(
    box: YoloBox,
    image_size: tuple[int, int],
    padding_ratio: float = 0.0,
) -> tuple[int, int, int, int]:
    """Convert a normalized YOLO box to clamped PIL crop bounds."""

    image_width, image_height = image_size
    if image_width <= 0 or image_height <= 0:
        raise ValueError("图片宽高必须为正数")
    if padding_ratio < 0:
        raise ValueError("padding_ratio 不能为负数")

    half_width = box.width * (1.0 + 2.0 * padding_ratio) / 2.0
    half_height = box.height * (1.0 + 2.0 * padding_ratio) / 2.0
    left = max(0, math.floor((box.x_center - half_width) * image_width))
    top = max(0, math.floor((box.y_center - half_height) * image_height))
    right = min(image_width, math.ceil((box.x_center + half_width) * image_width))
    bottom = min(image_height, math.ceil((box.y_center + half_height) * image_height))
    if right <= left or bottom <= top:
        raise ValueError("目标框转换后为空")
    return left, top, right, bottom


def build_target_crop_dataset(
    index_csv: Path,
    output_dir: Path,
    padding_ratio: float = 0.10,
    class_names: list[str] | None = None,
) -> dict:
    """Materialize GT object crops while preserving source-image train/val/test splits."""

    class_names = class_names or CLASS_NAMES
    samples = read_scene_index(index_csv)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"输出目录不是空目录，请换用新目录: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    source_splits: dict[Path, str] = {}
    crop_counts: Counter = Counter()
    source_counts: Counter = Counter()
    manifest_rows: list[dict[str, str | int | float]] = []

    for split in VALID_SPLITS:
        for class_name in class_names:
            (output_dir / split / class_name).mkdir(parents=True, exist_ok=True)

    for sample in samples:
        previous_split = source_splits.setdefault(sample.image_path, sample.split)
        if previous_split != sample.split:
            raise ValueError(f"同一原图出现在多个划分中: {sample.image_path}")
        boxes = parse_yolo_boxes(sample.label_path, len(class_names))
        source_counts[sample.split] += 1
        with Image.open(sample.image_path) as opened:
            image = opened.convert("RGB")
        for box_index, box in enumerate(boxes):
            bounds = normalized_box_to_pixels(box, image.size, padding_ratio)
            class_name = class_names[box.class_id]
            crop_name = f"{sample.image_path.stem}__obj{box_index:03d}.png"
            crop_path = (output_dir / sample.split / class_name / crop_name).resolve()
            image.crop(bounds).save(crop_path)
            left, top, right, bottom = bounds
            crop_counts[(sample.split, class_name)] += 1
            manifest_rows.append(
                {
                    "crop_path": str(crop_path),
                    "source_image_path": str(sample.image_path),
                    "source_image_name": sample.image_name,
                    "label_path": str(sample.label_path),
                    "split": sample.split,
                    "sensor": sample.sensor,
                    "scene": sample.scene,
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
    if not manifest_rows:
        raise ValueError("数据集中没有可裁剪的目标框")
    with manifest_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=manifest_rows[0].keys())
        writer.writeheader()
        writer.writerows(manifest_rows)

    summary = {
        "index_csv": str(index_csv.resolve()),
        "output_dir": str(output_dir.resolve()),
        "manifest": str(manifest_path.resolve()),
        "source_images": len(source_splits),
        "crops": len(manifest_rows),
        "class_names": class_names,
        "padding_ratio": padding_ratio,
        "split_policy": "先按原图划分，再裁剪；同一原图的所有目标只能属于同一划分",
        "splits": {
            split: {
                "source_images": int(source_counts[split]),
                "crops": int(sum(crop_counts[(split, name)] for name in class_names)),
                "crops_by_class": {
                    name: int(crop_counts[(split, name)]) for name in class_names
                },
            }
            for split in VALID_SPLITS
        },
    }
    (output_dir / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary
