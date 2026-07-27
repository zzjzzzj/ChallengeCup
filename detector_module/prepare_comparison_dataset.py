"""Build portable manifests for the final original/augmented comparison.

The input dataset is expected to use the standard YOLO layout::

    dataset_root/
      images/train
      images/val
      labels/train
      labels/val

The augmented training directory may contain both original images and files
whose names contain ``__aug-``. The original arm excludes those augmented
files. The untouched validation pool is split deterministically, per scene,
into model-selection and final-test manifests.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import yaml


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
CLASS_NAMES = ["soldier", "small_aircraft", "warship", "tank"]
SCENE_PATTERN = re.compile(r"_(air|forest|sea|urban)_", re.IGNORECASE)


def image_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"图片目录不存在: {directory}")
    files = sorted(
        path.resolve()
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not files:
        raise ValueError(f"图片目录为空: {directory}")
    return files


def scene_name(path: Path) -> str:
    match = SCENE_PATTERN.search(path.stem)
    if not match:
        raise ValueError(f"无法从文件名解析场景: {path.name}")
    return match.group(1).lower()


def validate_labels(images: list[Path], label_directory: Path) -> None:
    missing = [image.name for image in images if not (label_directory / f"{image.stem}.txt").is_file()]
    if missing:
        examples = ", ".join(missing[:5])
        raise FileNotFoundError(f"{len(missing)} 张图片缺少 YOLO 标签，例如: {examples}")


def split_holdout(images: list[Path]) -> tuple[list[Path], list[Path]]:
    """Split each scene in sorted order: even rows to val, odd rows to test."""

    grouped: dict[str, list[Path]] = {}
    for image in images:
        grouped.setdefault(scene_name(image), []).append(image)
    validation: list[Path] = []
    test: list[Path] = []
    for scene in sorted(grouped):
        rows = sorted(grouped[scene])
        validation.extend(rows[::2])
        test.extend(rows[1::2])
    if not validation or not test:
        raise ValueError("留出集过小，无法同时生成 val 和 test")
    return sorted(validation), sorted(test)


def write_manifest(path: Path, images: list[Path]) -> None:
    path.write_text("\n".join(image.as_posix() for image in images) + "\n", encoding="utf-8")


def write_data_yaml(path: Path, train: Path, val: Path, test: Path) -> None:
    payload = {
        "train": train.resolve().as_posix(),
        "val": val.resolve().as_posix(),
        "test": test.resolve().as_posix(),
        "nc": len(CLASS_NAMES),
        "names": {index: name for index, name in enumerate(CLASS_NAMES)},
    }
    path.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )


def prepare_comparison_dataset(dataset_root: Path, output: Path) -> dict:
    dataset_root = dataset_root.resolve()
    output = output.resolve()
    train_images = image_files(dataset_root / "images" / "train")
    holdout_images = image_files(dataset_root / "images" / "val")
    validate_labels(train_images, dataset_root / "labels" / "train")
    validate_labels(holdout_images, dataset_root / "labels" / "val")

    original_train = [image for image in train_images if "__aug-" not in image.stem]
    if not original_train or len(original_train) == len(train_images):
        raise ValueError(
            "训练集未同时包含原图和 __aug- 增广图，无法构建原始/增广对照"
        )
    validation, test = split_holdout(holdout_images)

    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "train_noaug": output / "train_noaug.txt",
        "train_aug": output / "train_aug.txt",
        "val": output / "val.txt",
        "test": output / "test.txt",
    }
    write_manifest(paths["train_noaug"], original_train)
    write_manifest(paths["train_aug"], train_images)
    write_manifest(paths["val"], validation)
    write_manifest(paths["test"], test)
    write_data_yaml(
        output / "data_noaug.yaml",
        paths["train_noaug"],
        paths["val"],
        paths["test"],
    )
    write_data_yaml(
        output / "data_aug.yaml",
        paths["train_aug"],
        paths["val"],
        paths["test"],
    )

    stats = {
        "dataset_root": dataset_root.as_posix(),
        "original_train_images": len(original_train),
        "augmented_train_images": len(train_images),
        "validation_images": len(validation),
        "test_images": len(test),
        "validation_by_scene": dict(Counter(scene_name(path) for path in validation)),
        "test_by_scene": dict(Counter(scene_name(path) for path in test)),
        "output": output.as_posix(),
    }
    (output / "dataset_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="生成最终原始/增广对比所需的 train/val/test 清单和 YAML"
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("detector_module/artifacts/comparison_dataset"),
    )
    args = parser.parse_args()
    print(
        json.dumps(
            prepare_comparison_dataset(args.dataset_root, args.output),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
