#!/usr/bin/env python3
"""Build a selected offline-augmented YOLO dataset.

This script keeps the original explicit form:

    python augment_selected_yolo.py --images images --labels labels --output out --include-original

For the Ascend board project layout, prefer:

    python augment_selected_yolo.py --dataset-root data/datasets_r1_base_train --output outputs/r1_aug --include-original

The generated folder contains paired ``images`` and ``labels`` directories. By
default it also writes ``data.yaml`` so the result can be passed directly to the
project YOLO training command.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
YOLO_RECORD = Tuple[int, float, float, float, float]


@dataclass(frozen=True)
class Operation:
    key: str
    display_name: str
    modality: str


# These are the selected offline augmentations used by the project. Each source
# image receives three independent variants according to its modality.
IR_OPERATIONS = (
    Operation("ir_gamma_bright", "Gamma brighten", "ir"),
    Operation("invert_255", "Invert grayscale", "ir"),
    Operation("rot180", "Rotate 180", "ir"),
)
SAR_OPERATIONS = (
    Operation("rot180", "Rotate 180", "sar"),
    Operation("sar_rot90_cw", "Rotate 90 clockwise", "sar"),
    Operation("sar_gamma", "Gamma adjust", "sar"),
)


def stable_unit_value(image_name: str, operation_key: str) -> float:
    digest = hashlib.sha256(
        ("selected-augmentation-v1|%s|%s" % (image_name, operation_key)).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def gamma_transform(image: Image.Image, gamma: float) -> Image.Image:
    pixels = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    result = np.clip(255.0 * np.power(pixels, gamma), 0, 255).astype(np.uint8)
    return Image.fromarray(result, mode="RGB")


def selected_operations(modality: str) -> Tuple[Operation, ...]:
    if modality == "ir":
        return IR_OPERATIONS
    if modality == "sar":
        return SAR_OPERATIONS
    raise ValueError("Unsupported modality: %s" % modality)


def modality_from_name(image_path: Path, default_modality: Optional[str] = None) -> str:
    prefix = image_path.stem.split("_", 1)[0].lower()
    if prefix in {"ir", "sar"}:
        return prefix
    if default_modality in {"ir", "sar"}:
        return default_modality
    raise ValueError(
        "Image names must start with ir_ or sar_ so the augmentation recipe can "
        "be selected. Pass --default-modality ir or --default-modality sar for "
        "datasets that do not use this project naming convention: %s" % image_path.name
    )


def apply_operation(image: Image.Image, operation: Operation, image_name: str) -> Tuple[Image.Image, str]:
    if operation.key == "ir_gamma_bright":
        gamma = 0.50 + stable_unit_value(image_name, operation.key) * 0.20
        return gamma_transform(image, gamma), "gamma=%.4f range=[0.50,0.70]" % gamma
    if operation.key == "invert_255":
        table = [255 - value for value in range(256)]
        return image.convert("RGB").point(table * 3), "pixel'=255-pixel"
    if operation.key == "rot180":
        return image.transpose(Image.Transpose.ROTATE_180), "rotate 180"
    if operation.key == "sar_rot90_cw":
        return image.transpose(Image.Transpose.ROTATE_270), "rotate 90 clockwise"
    if operation.key == "sar_gamma":
        value = stable_unit_value(image_name, operation.key)
        gamma = 0.45 + value * 0.46 if value < 0.5 else 1.55 + (value - 0.5) * 0.70
        return gamma_transform(image, gamma), "gamma=%.4f range=[0.45,0.68] or [1.55,1.90]" % gamma
    raise ValueError("Unimplemented augmentation operation: %s" % operation.key)


def parse_yolo_labels(label_path: Path) -> List[YOLO_RECORD]:
    records: List[YOLO_RECORD] = []
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError("%s line %d is not a 5-column YOLO label" % (label_path, line_number))
        try:
            class_id = int(fields[0])
            x_center, y_center, width, height = (float(value) for value in fields[1:])
        except ValueError as error:
            raise ValueError("%s line %d contains a non-numeric label" % (label_path, line_number)) from error
        if class_id < 0 or not all(0.0 <= value <= 1.0 for value in (x_center, y_center, width, height)):
            raise ValueError("%s line %d has an out-of-range class or coordinate" % (label_path, line_number))
        if width <= 0.0 or height <= 0.0:
            raise ValueError("%s line %d has non-positive box width/height" % (label_path, line_number))
        records.append((class_id, x_center, y_center, width, height))
    return records


def transform_rotation_labels(records: Iterable[YOLO_RECORD], operation_key: str) -> List[YOLO_RECORD]:
    transformed: List[YOLO_RECORD] = []
    for class_id, x_center, y_center, width, height in records:
        if operation_key == "rot180":
            item = (class_id, 1.0 - x_center, 1.0 - y_center, width, height)
        elif operation_key == "sar_rot90_cw":
            item = (class_id, 1.0 - y_center, x_center, height, width)
        else:
            raise ValueError("Not a geometric rotation operation: %s" % operation_key)
        if not all(0.0 <= value <= 1.0 for value in item[1:]):
            raise ValueError("Transformed YOLO label is out of range: %s" % (item,))
        transformed.append(item)
    return transformed


def write_yolo_labels(label_path: Path, records: Iterable[YOLO_RECORD]) -> None:
    rows = [
        "%d %.6f %.6f %.6f %.6f" % (class_id, x, y, width, height)
        for class_id, x, y, width, height in records
    ]
    label_path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")


def list_images(images_dir: Path) -> List[Path]:
    return sorted(
        path for path in images_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def unique_paths(paths: Iterable[Path]) -> List[Path]:
    result: List[Path] = []
    seen: set = set()
    for path in paths:
        key = str(path.resolve()).casefold()
        if key not in seen:
            result.append(path)
            seen.add(key)
    return result


def label_path_for_image(image_path: Path, images_dir: Path, labels_dir: Path) -> Path:
    candidates: List[Path] = []
    try:
        relative_image = image_path.resolve().relative_to(images_dir.resolve())
        candidates.append((labels_dir / relative_image).with_suffix(".txt"))
    except ValueError:
        pass
    candidates.append(labels_dir / ("%s.txt" % image_path.stem))
    candidates.append(image_path.with_suffix(".txt"))
    for candidate in unique_paths(candidates):
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "Missing YOLO label for %s. Tried: %s"
        % (image_path, ", ".join(str(path) for path in unique_paths(candidates)))
    )


def resolve_source_dirs(
    dataset_root: Optional[Path],
    split: str,
    images_dir: Optional[Path],
    labels_dir: Optional[Path],
) -> Tuple[Path, Path]:
    if dataset_root is not None:
        root = dataset_root.resolve()
        candidates = [
            (root / "images" / split, root / "labels" / split),
            (root / "images", root / "labels"),
            (root, root / "labels"),
            (root, root),
        ]
        for candidate_images, candidate_labels in candidates:
            if candidate_images.is_dir() and candidate_labels.is_dir():
                return candidate_images.resolve(), candidate_labels.resolve()
        searched = ", ".join("%s + %s" % (images, labels) for images, labels in candidates)
        raise FileNotFoundError(
            "Could not find YOLO images/labels under --dataset-root %s. Searched: %s"
            % (root, searched)
        )
    if images_dir is None or labels_dir is None:
        raise ValueError("Pass either --dataset-root or both --images and --labels.")
    return images_dir.resolve(), labels_dir.resolve()


def validate_source(
    images_dir: Path,
    labels_dir: Path,
    default_modality: Optional[str] = None,
) -> List[Path]:
    if not images_dir.is_dir() or not labels_dir.is_dir():
        raise FileNotFoundError("--images and --labels must be existing directories.")
    images = list_images(images_dir)
    if not images:
        raise ValueError("Input image directory is empty: %s" % images_dir)
    output_names = [path.name.casefold() for path in images]
    duplicated_names = sorted(name for name, count in Counter(output_names).items() if count > 1)
    if duplicated_names:
        raise ValueError(
            "Duplicate image file names would collide in the flat output directory: %s"
            % duplicated_names[:5]
        )
    for image_path in images:
        parse_yolo_labels(label_path_for_image(image_path, images_dir, labels_dir))
        with Image.open(image_path) as image:
            image.verify()
        modality_from_name(image_path, default_modality)
    return images


def validate_output(output_root: Path) -> None:
    images_dir, labels_dir = output_root / "images", output_root / "labels"
    images = list_images(images_dir)
    if not images:
        raise ValueError("Output image directory is empty: %s" % images_dir)
    if {path.stem for path in images} != {path.stem for path in labels_dir.glob("*.txt")}:
        raise ValueError("Output images and labels are not paired.")
    for image_path in images:
        parse_yolo_labels(labels_dir / ("%s.txt" % image_path.stem))
        with Image.open(image_path) as image:
            image.load()


def prepare_output_root(output_root: Path, force: bool) -> None:
    if output_root.exists():
        if not force:
            raise FileExistsError(
                "Output directory already exists: %s. Pass --force to rebuild it, "
                "or choose a new --output." % output_root
            )
        if output_root.parent == output_root:
            raise ValueError("Refusing to remove filesystem root: %s" % output_root)
        shutil.rmtree(output_root)


def build_augmentation(
    images_dir: Path,
    labels_dir: Path,
    output_root: Path,
    include_original: bool,
    default_modality: Optional[str] = None,
    force: bool = False,
) -> Dict[str, object]:
    source_images = validate_source(images_dir, labels_dir, default_modality)
    output_root = output_root.resolve()
    if output_root in {images_dir.resolve(), labels_dir.resolve()}:
        raise ValueError("Output directory must not be the same as the source images or labels directory.")
    prepare_output_root(output_root, force)
    output_images = output_root / "images"
    output_labels = output_root / "labels"
    output_images.mkdir(parents=True, exist_ok=False)
    output_labels.mkdir(parents=True, exist_ok=False)

    manifest_rows: List[Dict[str, str]] = []
    operation_counter: Counter[str] = Counter()
    modality_counter: Counter[str] = Counter()

    for index, image_path in enumerate(source_images, start=1):
        label_path = label_path_for_image(image_path, images_dir, labels_dir)
        modality = modality_from_name(image_path, default_modality)
        source_records = parse_yolo_labels(label_path)
        modality_counter[modality] += 1
        if include_original:
            shutil.copy2(image_path, output_images / image_path.name)
            shutil.copy2(label_path, output_labels / ("%s.txt" % image_path.stem))
            manifest_rows.append(
                {
                    "source_image": image_path.name,
                    "output_image": image_path.name,
                    "operation_key": "original",
                    "operation_name": "Original copy",
                    "operation_detail": "not augmented",
                    "label_transform": "no",
                }
            )

        with Image.open(image_path) as opened:
            source_image = opened.convert("RGB")
            for operation in selected_operations(modality):
                result, detail = apply_operation(source_image, operation, image_path.name)
                target_stem = "%s__aug-%s" % (image_path.stem, operation.key)
                target_image = output_images / ("%s.png" % target_stem)
                target_label = output_labels / ("%s.txt" % target_stem)
                result.save(target_image, format="PNG", optimize=False)
                if operation.key in {"rot180", "sar_rot90_cw"}:
                    write_yolo_labels(target_label, transform_rotation_labels(source_records, operation.key))
                    label_transform = "yes"
                else:
                    shutil.copy2(label_path, target_label)
                    label_transform = "no"
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
                operation_counter["%s/%s" % (modality, operation.key)] += 1
        if index % 50 == 0 or index == len(source_images):
            print("[INFO] Augmented %d/%d source images" % (index, len(source_images)), flush=True)

    validate_output(output_root)
    with (output_root / "augmentation_manifest.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)
    summary: Dict[str, object] = {
        "source_images": len(source_images),
        "source_images_dir": str(images_dir.resolve()),
        "source_labels_dir": str(labels_dir.resolve()),
        "source_by_modality": dict(sorted(modality_counter.items())),
        "include_original": include_original,
        "augmentations_per_source": 3,
        "generated_augmentation_images": len(source_images) * 3,
        "output_images": len(list_images(output_images)),
        "output_images_dir": str(output_images.resolve()),
        "output_labels_dir": str(output_labels.resolve()),
        "operation_counts": dict(sorted(operation_counter.items())),
        "validation": "all output images are readable, image/label pairs match, and YOLO labels are valid",
    }
    write_summary(output_root, summary)
    return summary


def find_classes_file(dataset_root: Optional[Path], explicit_classes: Optional[Path]) -> Optional[Path]:
    if explicit_classes is not None:
        return explicit_classes.resolve()
    if dataset_root is not None:
        candidate = dataset_root.resolve() / "classes.txt"
        if candidate.is_file():
            return candidate
    return None


def read_class_names(classes_path: Path) -> List[str]:
    return [
        line.strip()
        for line in classes_path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def infer_class_names(images_dir: Path, labels_dir: Path) -> List[str]:
    max_class_id = -1
    for image_path in list_images(images_dir):
        label_path = label_path_for_image(image_path, images_dir, labels_dir)
        for class_id, _x, _y, _w, _h in parse_yolo_labels(label_path):
            max_class_id = max(max_class_id, class_id)
    if max_class_id < 0:
        raise ValueError("Could not infer classes because all label files are empty.")
    return ["class_%d" % index for index in range(max_class_id + 1)]


def resolve_class_names(
    dataset_root: Optional[Path],
    images_dir: Path,
    labels_dir: Path,
    classes_path: Optional[Path],
) -> Tuple[List[str], str]:
    resolved = find_classes_file(dataset_root, classes_path)
    if resolved is not None:
        names = read_class_names(resolved)
        if not names:
            raise ValueError("Class file is empty: %s" % resolved)
        return names, str(resolved)
    return infer_class_names(images_dir, labels_dir), "inferred_from_labels"


def ensure_class_ids_fit(images_dir: Path, labels_dir: Path, class_names: Sequence[str]) -> None:
    class_count = len(class_names)
    for image_path in list_images(images_dir):
        label_path = label_path_for_image(image_path, images_dir, labels_dir)
        for class_id, _x, _y, _w, _h in parse_yolo_labels(label_path):
            if class_id >= class_count:
                raise ValueError(
                    "%s contains class_id=%d, but only %d class names were provided."
                    % (label_path, class_id, class_count)
                )


def yaml_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def path_for_yaml(path: Path, dataset_root: Path) -> str:
    resolved = path.resolve()
    try:
        relative = Path(os.path.relpath(str(resolved), str(dataset_root.resolve())))
        if not str(relative).startswith(".."):
            return relative.as_posix()
    except ValueError:
        pass
    return resolved.as_posix()


def resolve_validation_images(
    val_images: Optional[Path],
    val_root: Optional[Path],
    val_split: str,
) -> Tuple[Optional[Path], str]:
    if val_images is not None:
        path = val_images.resolve()
        if not path.exists():
            raise FileNotFoundError("Validation image path does not exist: %s" % path)
        return path, "external_val_images"
    if val_root is not None:
        root = val_root.resolve()
        candidates = [root / "images" / val_split, root / "images", root]
        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve(), "external_val_root"
        raise FileNotFoundError("Could not find validation images under --val-root: %s" % root)
    return None, "train_as_val"


def write_data_yaml(
    output_root: Path,
    data_yaml: Path,
    class_names: Sequence[str],
    val_images: Optional[Path],
    validation_policy: str,
) -> None:
    output_root = output_root.resolve()
    data_yaml.parent.mkdir(parents=True, exist_ok=True)
    train_value = path_for_yaml(output_root / "images", output_root)
    val_value = train_value if val_images is None else path_for_yaml(val_images, output_root)
    lines = [
        "path: %s" % yaml_quote(output_root.as_posix()),
        "train: %s" % yaml_quote(train_value),
        "val: %s" % yaml_quote(val_value),
        "nc: %d" % len(class_names),
        "names:",
    ]
    for index, name in enumerate(class_names):
        lines.append("  %d: %s" % (index, yaml_quote(str(name))))
    data_yaml.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary(output_root: Path, summary: Dict[str, object]) -> None:
    (output_root / "augmentation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a selected offline-augmented YOLO dataset.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        help="Project YOLO root. The script searches images/<split> + labels/<split>, then images + labels.",
    )
    parser.add_argument("--split", default="train", help="Split name used under --dataset-root. Default: train.")
    parser.add_argument("--images", type=Path, help="Source training image directory.")
    parser.add_argument("--labels", type=Path, help="Source YOLO label directory paired with --images.")
    parser.add_argument("--output", type=Path, required=True, help="New output dataset directory.")
    parser.add_argument("--include-original", action="store_true", help="Copy original images/labels into output too.")
    parser.add_argument("--force", action="store_true", help="Delete and rebuild --output if it already exists.")
    parser.add_argument(
        "--default-modality",
        choices=["ir", "sar"],
        help="Fallback recipe for image names that do not start with ir_ or sar_.",
    )
    parser.add_argument("--classes", type=Path, help="Class-name file. Defaults to <dataset-root>/classes.txt.")
    parser.add_argument("--no-data-yaml", action="store_true", help="Do not write YOLO data.yaml.")
    parser.add_argument("--data-yaml", type=Path, help="Output YAML path. Default: <output>/data.yaml.")
    parser.add_argument("--val-images", type=Path, help="Independent validation image directory/list/file.")
    parser.add_argument("--val-root", type=Path, help="Independent YOLO validation root. Searches images/<val-split>.")
    parser.add_argument("--val-split", default="val", help="Validation split name under --val-root. Default: val.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.dataset_root is not None and (args.images is not None or args.labels is not None):
        raise SystemExit("Use either --dataset-root or --images/--labels, not both.")
    if (args.images is None) != (args.labels is None):
        raise SystemExit("--images and --labels must be passed together.")
    if args.val_images is not None and args.val_root is not None:
        raise SystemExit("Use either --val-images or --val-root, not both.")

    images_dir, labels_dir = resolve_source_dirs(args.dataset_root, args.split, args.images, args.labels)
    output_root = args.output.resolve()
    summary = build_augmentation(
        images_dir,
        labels_dir,
        output_root,
        args.include_original,
        default_modality=args.default_modality,
        force=args.force,
    )

    if not args.no_data_yaml:
        class_names, class_source = resolve_class_names(args.dataset_root, images_dir, labels_dir, args.classes)
        ensure_class_ids_fit(images_dir, labels_dir, class_names)
        val_images, validation_policy = resolve_validation_images(args.val_images, args.val_root, args.val_split)
        data_yaml = args.data_yaml.resolve() if args.data_yaml is not None else output_root / "data.yaml"
        write_data_yaml(output_root, data_yaml, class_names, val_images, validation_policy)
        (output_root / "classes.txt").write_text("\n".join(class_names) + "\n", encoding="utf-8")
        summary.update(
            {
                "data_yaml": str(data_yaml.resolve()),
                "class_names": list(class_names),
                "class_source": class_source,
                "validation_images": str(val_images.resolve()) if val_images is not None else str((output_root / "images").resolve()),
                "validation_policy": validation_policy,
            }
        )
        write_summary(output_root, summary)
        if validation_policy == "train_as_val":
            print(
                "[WARN] data.yaml uses the augmented training images as val. "
                "This is runnable, but not an independent metric.",
                flush=True,
            )
        print("[INFO] YOLO data YAML: %s" % data_yaml.resolve(), flush=True)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("[INFO] Augmentation complete: %s" % output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
