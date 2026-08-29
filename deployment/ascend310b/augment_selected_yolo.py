#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""对已确定的 YOLO 训练图片执行 R1/R2 筛选后的离线增强。

本工具不划分训练集/验证集，不修改源图片或源标签。输入目录应只包含已经
确定用于训练的图片及其 YOLO 标签；输出目录仅含原图副本（可选）和增强样本。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class Operation:
    """单项增强的名称、说明与适用模态。"""

    key: str
    display_name: str
    modality: str


# 这些操作来自此前消融筛选结果。每张原图仅执行三种独立操作，彼此不叠加。
IR_OPERATIONS = (
    Operation("ir_gamma_bright", "Gamma 提亮", "ir"),
    Operation("invert_255", "255 灰度取反", "ir"),
    Operation("rot180", "180° 旋转", "ir"),
)
SAR_OPERATIONS = (
    Operation("rot180", "180° 旋转", "sar"),
    Operation("sar_rot90_cw", "顺时针 90° 旋转", "sar"),
    Operation("sar_gamma", "Gamma 调整", "sar"),
)


def stable_unit_value(image_name: str, operation_key: str) -> float:
    """为同一图片和操作生成稳定参数，保证重复运行结果一致。"""

    digest = hashlib.sha256(f"selected-augmentation-v1|{image_name}|{operation_key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def gamma_transform(image: Image.Image, gamma: float) -> Image.Image:
    """执行 Gamma 变换，统一输出三通道 RGB 图像。"""

    pixels = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    result = np.clip(255.0 * np.power(pixels, gamma), 0, 255).astype(np.uint8)
    return Image.fromarray(result, mode="RGB")


def selected_operations(modality: str) -> tuple[Operation, ...]:
    """返回对应模态的三种已筛选增强。"""

    if modality == "ir":
        return IR_OPERATIONS
    if modality == "sar":
        return SAR_OPERATIONS
    raise ValueError(f"不支持的模态：{modality}")


def modality_from_name(image_path: Path) -> str:
    """从图片文件名前缀 ir_ / sar_ 判断传感器模态。"""

    modality = image_path.stem.split("_", 1)[0].lower()
    if modality not in {"ir", "sar"}:
        raise ValueError(f"图片名必须以 ir_ 或 sar_ 开头，无法判断模态：{image_path.name}")
    return modality


def apply_operation(image: Image.Image, operation: Operation, image_name: str) -> tuple[Image.Image, str]:
    """执行一项增强，并返回结果及可追溯参数说明。"""

    if operation.key == "ir_gamma_bright":
        gamma = 0.50 + stable_unit_value(image_name, operation.key) * 0.20
        return gamma_transform(image, gamma), f"gamma={gamma:.4f}，范围 [0.50, 0.70]"
    if operation.key == "invert_255":
        table = [255 - value for value in range(256)]
        return image.convert("RGB").point(table * 3), "逐像素 p'=255-p"
    if operation.key == "rot180":
        return image.transpose(Image.Transpose.ROTATE_180), "旋转 180°"
    if operation.key == "sar_rot90_cw":
        return image.transpose(Image.Transpose.ROTATE_270), "顺时针旋转 90°"
    if operation.key == "sar_gamma":
        value = stable_unit_value(image_name, operation.key)
        gamma = 0.45 + value * 0.46 if value < 0.5 else 1.55 + (value - 0.5) * 0.70
        return gamma_transform(image, gamma), f"gamma={gamma:.4f}，范围 [0.45,0.68] 或 [1.55,1.90]"
    raise ValueError(f"未实现的增强操作：{operation.key}")


def parse_yolo_labels(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    """读取并检查 YOLO 检测标签。"""

    records: list[tuple[int, float, float, float, float]] = []
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"{label_path} 第 {line_number} 行不是 5 列 YOLO 标签")
        try:
            class_id = int(fields[0])
            x_center, y_center, width, height = (float(value) for value in fields[1:])
        except ValueError as error:
            raise ValueError(f"{label_path} 第 {line_number} 行包含非数值标签") from error
        if class_id < 0 or not all(0.0 <= value <= 1.0 for value in (x_center, y_center, width, height)):
            raise ValueError(f"{label_path} 第 {line_number} 行类别或坐标越界")
        if width <= 0.0 or height <= 0.0:
            raise ValueError(f"{label_path} 第 {line_number} 行框宽高必须大于 0")
        records.append((class_id, x_center, y_center, width, height))
    return records


def transform_rotation_labels(
    records: Iterable[tuple[int, float, float, float, float]], operation_key: str
) -> list[tuple[int, float, float, float, float]]:
    """对旋转图像同步变换 YOLO 归一化标注。"""

    transformed: list[tuple[int, float, float, float, float]] = []
    for class_id, x_center, y_center, width, height in records:
        if operation_key == "rot180":
            item = (class_id, 1.0 - x_center, 1.0 - y_center, width, height)
        elif operation_key == "sar_rot90_cw":
            item = (class_id, 1.0 - y_center, x_center, height, width)
        else:
            raise ValueError(f"不是几何旋转操作：{operation_key}")
        if not all(0.0 <= value <= 1.0 for value in item[1:]):
            raise ValueError(f"旋转后标签越界：{item}")
        transformed.append(item)
    return transformed


def write_yolo_labels(label_path: Path, records: Iterable[tuple[int, float, float, float, float]]) -> None:
    """以固定精度、UTF-8 写出 YOLO 标签。"""

    rows = [f"{class_id} {x:.6f} {y:.6f} {width:.6f} {height:.6f}" for class_id, x, y, width, height in records]
    label_path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")


def list_images(images_dir: Path) -> list[Path]:
    """列出图片并固定排序，以保证输出顺序可复现。"""

    return sorted(path for path in images_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def validate_source(images_dir: Path, labels_dir: Path) -> list[Path]:
    """验证输入图片、标签配对、可读性与标签坐标。"""

    if not images_dir.is_dir() or not labels_dir.is_dir():
        raise FileNotFoundError("--images 与 --labels 都必须是存在的目录")
    images = list_images(images_dir)
    if not images:
        raise ValueError("输入图片目录为空")
    image_stems = {path.stem for path in images}
    label_stems = {path.stem for path in labels_dir.glob("*.txt")}
    if image_stems != label_stems:
        missing = sorted(image_stems - label_stems)
        orphan = sorted(label_stems - image_stems)
        raise ValueError(f"图片标签不一一对应；缺标签={missing[:3]}，孤立标签={orphan[:3]}")
    for image_path in images:
        parse_yolo_labels(labels_dir / f"{image_path.stem}.txt")
        with Image.open(image_path) as image:
            image.verify()
        modality_from_name(image_path)
    return images


def validate_output(output_root: Path) -> None:
    """验证输出图片标签配对、图片可读取及标签合法。"""

    images_dir, labels_dir = output_root / "images", output_root / "labels"
    images = list_images(images_dir)
    if not images:
        raise ValueError("输出图片为空")
    if {path.stem for path in images} != {path.stem for path in labels_dir.glob("*.txt")}:
        raise ValueError("输出图片和标签未一一对应")
    for image_path in images:
        parse_yolo_labels(labels_dir / f"{image_path.stem}.txt")
        with Image.open(image_path) as image:
            image.load()


def build_augmentation(
    images_dir: Path, labels_dir: Path, output_root: Path, include_original: bool
) -> dict[str, object]:
    """从已确定的训练样本构建增强样本；不涉及任何数据集划分。"""

    source_images = validate_source(images_dir, labels_dir)
    if output_root.exists():
        raise FileExistsError(f"为保护既有文件，输出目录必须不存在：{output_root}")
    output_images = output_root / "images"
    output_labels = output_root / "labels"
    output_images.mkdir(parents=True, exist_ok=False)
    output_labels.mkdir(parents=True, exist_ok=False)

    manifest_rows: list[dict[str, str]] = []
    operation_counter: Counter[str] = Counter()
    modality_counter: Counter[str] = Counter()

    for index, image_path in enumerate(source_images, start=1):
        label_path = labels_dir / f"{image_path.stem}.txt"
        modality = modality_from_name(image_path)
        source_records = parse_yolo_labels(label_path)
        modality_counter[modality] += 1
        if include_original:
            shutil.copy2(image_path, output_images / image_path.name)
            shutil.copy2(label_path, output_labels / label_path.name)
            manifest_rows.append(
                {
                    "source_image": image_path.name,
                    "output_image": image_path.name,
                    "operation_key": "original",
                    "operation_name": "原图副本",
                    "operation_detail": "未增强；原标签副本",
                    "label_transform": "否",
                }
            )

        with Image.open(image_path) as opened:
            source_image = opened.convert("RGB")
            for operation in selected_operations(modality):
                result, detail = apply_operation(source_image, operation, image_path.name)
                target_stem = f"{image_path.stem}__aug-{operation.key}"
                target_image = output_images / f"{target_stem}.png"
                target_label = output_labels / f"{target_stem}.txt"
                result.save(target_image, format="PNG", optimize=False)
                if operation.key in {"rot180", "sar_rot90_cw"}:
                    write_yolo_labels(target_label, transform_rotation_labels(source_records, operation.key))
                    label_transform = "是：同步旋转中心坐标和宽高"
                else:
                    shutil.copy2(label_path, target_label)
                    label_transform = "否：图像几何位置未改变"
                manifest_rows.append(
                    {
                        "source_image": image_path.name,
                        "output_image": target_image.name,
                        "operation_key": operation.key,
                        "operation_name": operation.display_name,
                        "operation_detail": detail,
                        "label_transform": label_transform,
                    }
                )
                operation_counter[f"{modality}/{operation.key}"] += 1
        if index % 50 == 0 or index == len(source_images):
            print(f"已处理 {index}/{len(source_images)} 张训练图片")

    validate_output(output_root)
    with (output_root / "augmentation_manifest.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)
    summary = {
        "source_images": len(source_images),
        "source_by_modality": dict(sorted(modality_counter.items())),
        "include_original": include_original,
        "augmentations_per_source": 3,
        "generated_augmentation_images": len(source_images) * 3,
        "output_images": len(list_images(output_images)),
        "operation_counts": dict(sorted(operation_counter.items())),
        "validation": "已检查所有输出图片可读、图片标签一一对应、YOLO 标签格式和坐标合法。",
    }
    (output_root / "augmentation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="对已确定的 YOLO 训练图片执行 R1/R2 筛选后增强")
    parser.add_argument("--images", type=Path, required=True, help="仅包含待增强训练图片的目录")
    parser.add_argument("--labels", type=Path, required=True, help="与 --images 同名的 YOLO 标签目录")
    parser.add_argument("--output", type=Path, required=True, help="必须不存在的新输出目录")
    parser.add_argument("--include-original", action="store_true", help="同时复制原图和原标签，形成可直接训练的完整训练集")
    arguments = parser.parse_args()

    summary = build_augmentation(
        arguments.images.resolve(), arguments.labels.resolve(), arguments.output.resolve(), arguments.include_original
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"增强完成：{arguments.output.resolve()}")


if __name__ == "__main__":
    main()
