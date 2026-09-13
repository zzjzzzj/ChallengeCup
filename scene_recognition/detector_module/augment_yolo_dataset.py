"""Offline, deterministic YOLO dataset augmentation for the local protocol.

The Ascend 310B augmentation recipe is deliberately reused instead of being
reimplemented here.  This wrapper adds the normal project ``data.yaml``
contract: train is materialised with originals and three selected variants,
while val/test are copied without augmentation.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from pathlib import Path
from typing import Iterable, Sequence

import yaml
from PIL import Image

from deployment.ascend310b.augment_selected_yolo import (
    apply_operation,
    list_images,
    modality_from_name,
    parse_yolo_labels,
    selected_operations,
    transform_rotation_labels,
    write_yolo_labels,
)

from scene_recognition.detector_module import BASE_CLASS_NAMES, ALL_CLASS_NAMES


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _names(config: dict) -> list[str]:
    values = config.get("names")
    if isinstance(values, dict):
        return [str(values[index] if index in values else values[str(index)]) for index in range(len(values))]
    if isinstance(values, list):
        return [str(value) for value in values]
    raise ValueError("data.yaml 缺少合法 names")


def _dataset_root(config: dict, yaml_path: Path) -> Path:
    raw = config.get("path")
    root = Path(str(raw)) if raw else yaml_path.parent
    if not root.is_absolute():
        root = yaml_path.parent / root
    return root.resolve()


def _resolve_entry(value: object, root: Path, parent: Path) -> list[Path]:
    entries = value if isinstance(value, list) else [value]
    result: list[Path] = []
    for raw in entries:
        if raw is None:
            continue
        path = Path(str(raw))
        if not path.is_absolute():
            candidate = root / path
            path = candidate if candidate.exists() else parent / path
        path = path.resolve()
        if path.is_dir():
            result.extend(sorted(p.resolve() for p in path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES))
        elif path.is_file() and path.suffix.lower() == ".txt":
            for line in path.read_text(encoding="utf-8-sig").splitlines():
                if not line.strip():
                    continue
                item = Path(line.strip())
                if not item.is_absolute():
                    candidate = root / item
                    item = candidate if candidate.exists() else path.parent / item
                if not item.is_file():
                    raise FileNotFoundError(f"清单包含不存在的图像: {item}")
                result.append(item.resolve())
        elif path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            result.append(path)
        else:
            raise FileNotFoundError(f"无法解析数据划分路径: {raw} (resolved={path})")
    unique: list[Path] = []
    seen: set[Path] = set()
    for item in result:
        if item not in seen:
            unique.append(item)
            seen.add(item)
    return unique


def resolve_split_images(config: dict, split: str, root: Path, yaml_path: Path) -> list[Path]:
    """Resolve a YOLO split value that is a directory, txt list, file, or list."""

    images = _resolve_entry(config.get(split), root, yaml_path.parent)
    if config.get(split) and not images:
        raise ValueError(f"{split} 划分没有图像")
    return images


def resolve_label_path(image: Path) -> Path:
    sibling = image.with_suffix(".txt")
    if sibling.is_file():
        return sibling
    parts = list(image.parts)
    for index in range(len(parts) - 1, -1, -1):
        if parts[index].casefold() == "images":
            parts[index] = "labels"
            candidate = Path(*parts).with_suffix(".txt")
            if candidate.is_file():
                return candidate
            break
    raise FileNotFoundError(f"图像缺少 YOLO 标签: {image}")


def _copy_pair(image: Path, target_image: Path, target_label: Path) -> None:
    target_image.parent.mkdir(parents=True, exist_ok=True)
    target_label.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image, target_image)
    shutil.copy2(resolve_label_path(image), target_label)


def _yaml_path(path: Path, root: Path) -> str:
    try:
        relative = os.path.relpath(str(path.resolve()), str(root.resolve()))
        if not relative.startswith(".."):
            return Path(relative).as_posix()
    except ValueError:
        pass
    return path.resolve().as_posix()


def _write_yaml(output: Path, names: list[str], split_paths: dict[str, Path]) -> None:
    config: dict[str, object] = {
        "path": output.resolve().as_posix(),
        "train": _yaml_path(split_paths["train"], output),
        "val": _yaml_path(split_paths["val"], output),
        "nc": len(names),
        "names": {index: name for index, name in enumerate(names)},
    }
    if "test" in split_paths:
        config["test"] = _yaml_path(split_paths["test"], output)
    (output / "data.yaml").write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")


def _ensure_empty_or_absent(output: Path, force: bool = False) -> None:
    if force:
        # This command must never recursively remove an arbitrary user
        # directory.  Rebuilds are intentionally explicit: choose a fresh
        # output path (or remove a known artifact outside this command).
        raise ValueError("--force 已禁用；为避免误删，请选择不存在或为空的输出目录")
    if not output.exists():
        return
    if any(output.iterdir()):
        raise FileExistsError(f"输出目录非空，请选择新的输出目录: {output}")


def augment_yolo_dataset(
    data_yaml: Path,
    output: Path,
    *,
    include_original: bool = True,
    default_modality: str | None = None,
    force: bool = False,
) -> dict:
    """Materialise an offline augmented dataset and return its audit summary."""

    data_yaml = data_yaml.resolve()
    if not data_yaml.is_file():
        raise FileNotFoundError(data_yaml)
    config = yaml.safe_load(data_yaml.read_text(encoding="utf-8-sig"))
    if not isinstance(config, dict):
        raise ValueError("data.yaml 必须是映射")
    names = _names(config)
    if not names or len(names) != len(set(names)):
        raise ValueError("data.yaml names 不能为空且不能重复")
    declared_nc = int(config.get("nc", len(names)))
    if declared_nc != len(names):
        raise ValueError("nc 与 names 数量不一致")
    root = _dataset_root(config, data_yaml)
    splits: dict[str, list[Path]] = {split: resolve_split_images(config, split, root, data_yaml) for split in ("train", "val", "test")}
    if not splits["train"]:
        raise ValueError("train 划分不能为空")
    if not splits["val"]:
        raise ValueError("val 划分不能为空")
    for split, source_images in splits.items():
        names_seen: set[str] = set()
        duplicate_names: set[str] = set()
        for image in source_images:
            key = image.name.casefold()
            if key in names_seen:
                duplicate_names.add(key)
            names_seen.add(key)
        if duplicate_names:
            raise ValueError(f"{split} 划分存在会导致输出冲突的重复文件名: {sorted(duplicate_names)[:5]}")
    output = output.resolve()
    if output in {data_yaml.parent.resolve(), root}:
        raise ValueError("输出目录不能覆盖输入数据目录")
    _ensure_empty_or_absent(output, force)
    output.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    op_counts: dict[str, int] = {}
    output_counts: dict[str, int] = {}
    for split, source_images in splits.items():
        if not source_images:
            continue
        image_dir = output / "images" / split
        label_dir = output / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        for image in source_images:
            source_label = resolve_label_path(image)
            records = parse_yolo_labels(source_label)
            if any(class_id < 0 or class_id >= len(names) for class_id, *_ in records):
                raise ValueError(f"{source_label} 含超出 names 范围的 class id")
            # Reading the image here catches corrupt input before any output is committed.
            with Image.open(image) as opened:
                source_rgb = opened.convert("RGB")
            if split != "train":
                target_image = image_dir / image.name
                target_label = label_dir / f"{image.stem}.txt"
                _copy_pair(image, target_image, target_label)
                rows.append({
                    "split": split,
                    "source_image": str(image),
                    "output_image": str(target_image),
                    "operation_key": "original",
                    "operation": "original",
                    "label_transform": "no",
                    "source_key": image.name.casefold(),
                })
                output_counts[split] = output_counts.get(split, 0) + 1
                continue
            modality = modality_from_name(image, default_modality)
            if include_original:
                target_image = image_dir / image.name
                target_label = label_dir / f"{image.stem}.txt"
                _copy_pair(image, target_image, target_label)
                rows.append({
                    "split": split,
                    "source_image": str(image),
                    "output_image": str(target_image),
                    "operation_key": "original",
                    "operation": "original",
                    "label_transform": "no",
                    "source_key": image.name.casefold(),
                })
                output_counts[split] = output_counts.get(split, 0) + 1
            for operation in selected_operations(modality):
                transformed, detail = apply_operation(source_rgb, operation, image.name)
                target_stem = f"{image.stem}__aug-{operation.key}"
                target_image = image_dir / f"{target_stem}.png"
                target_label = label_dir / f"{target_stem}.txt"
                transformed.save(target_image, format="PNG", optimize=False)
                if operation.key in {"rot180", "sar_rot90_cw"}:
                    write_yolo_labels(target_label, transform_rotation_labels(records, operation.key))
                    transformed_labels = "yes"
                else:
                    shutil.copy2(source_label, target_label)
                    transformed_labels = "no"
                key = operation.key
                op_counts[key] = op_counts.get(key, 0) + 1
                output_counts[split] = output_counts.get(split, 0) + 1
                rows.append({
                    "split": split,
                    "source_image": str(image),
                    "output_image": str(target_image),
                    "operation_key": key,
                    "operation": detail,
                    "label_transform": transformed_labels,
                    "source_key": image.name.casefold(),
                })

    manifest_path = output / "augmentation_manifest.csv"
    with manifest_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "source_image", "output_image", "operation_key", "operation", "label_transform", "source_key"])
        writer.writeheader()
        writer.writerows(rows)
    split_paths = {split: output / "images" / split for split, values in splits.items() if values}
    _write_yaml(output, names, split_paths)
    (output / "classes.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
    summary = {
        "scenario": "offline_yolo_augmentation",
        "source_data_yaml": str(data_yaml),
        "data_yaml": str((output / "data.yaml").resolve()),
        "class_names": names,
        "nc": len(names),
        "taxonomy": {"nc": len(names), "names": list(names)},
        "include_original": include_original,
        "augmented_split": "train",
        "unaugmented_splits": [split for split in ("val", "test") if splits[split]],
        "source_images": {split: len(values) for split, values in splits.items() if values},
        "output_images": output_counts,
        "operation_counts": op_counts,
        "generated_augmentation_images": sum(op_counts.values()),
        "manifest": str(manifest_path.resolve()),
        "augmentation_manifest": str(manifest_path.resolve()),
        "val_test_augmented": False,
        "offline": True,
        "privacy": {"local_only": True, "dataset_upload": False},
    }
    (output / "augmentation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="离线、确定性的标准 YOLO 数据集增广")
    parser.add_argument("--data", "--data-yaml", dest="data_yaml", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--default-modality", choices=("ir", "sar"))
    parser.add_argument("--no-original", dest="include_original", action="store_false")
    parser.set_defaults(include_original=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = augment_yolo_dataset(
        args.data_yaml,
        args.output,
        include_original=args.include_original,
        default_modality=args.default_modality,
        force=False,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


# Friendly aliases for callers migrating from the Ascend augmentation helper.
build_augmentation = augment_yolo_dataset
augment_dataset = augment_yolo_dataset


if __name__ == "__main__":
    raise SystemExit(main())
