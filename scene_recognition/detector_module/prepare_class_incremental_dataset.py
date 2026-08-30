"""Build a six-stage, local-only class-incremental YOLO protocol.

Each stage introduces exactly one class.  Training views expose labels for the
current class plus a fixed-capacity replay buffer; validation and optional test
views expose all classes learned so far.  Images are hard-linked when possible
and copied only when the filesystem cannot create a hard link.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import yaml

from scene_recognition.detector_module import ALL_CLASS_NAMES
from scene_recognition.detector_module.boxes import YoloBox, parse_yolo_boxes, resolve_label_path
from scene_recognition.detector_module.context_metadata import (
    CONTEXT_INDEX_FIELDS,
    SCENE_NAMES,
    SENSOR_NAMES,
    build_context_row,
    context_index_summary,
    resolve_context_metadata,
    read_context_rows,
    write_context_index,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "scene_recognition"
    / "detector_module"
    / "artifacts"
    / "class_incremental"
)
BUFFER_SIZE_CHOICES = (200, 500)
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}


@dataclass(frozen=True)
class SourceSample:
    """One source image and its complete six-class annotation."""

    image_path: Path
    split: str
    boxes: tuple[YoloBox, ...]
    source_key: str
    operation: str
    context: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ReplayEntry:
    """One replay slot; labels are frozen to the class that acquired the slot."""

    image_path: Path
    class_id: int
    source_key: str


def parse_class_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def read_data_config(data_yaml: Path) -> tuple[dict, list[str], Path]:
    """Read a YOLO YAML and resolve its dataset root without importing Ultralytics."""

    if not data_yaml.is_file():
        raise FileNotFoundError(data_yaml)
    config = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or "names" not in config:
        raise ValueError(f"数据 YAML 缺少 names: {data_yaml}")
    names = config["names"]
    class_names = (
        [str(names[index] if index in names else names[str(index)]) for index in range(len(names))]
        if isinstance(names, dict)
        else [str(name) for name in names]
    )
    if not class_names or len(class_names) != len(set(class_names)):
        raise ValueError("类别表为空或包含重复类别")
    declared_nc = int(config.get("nc", len(class_names)))
    if declared_nc != len(class_names):
        raise ValueError(f"nc={declared_nc} 与 names 数量 {len(class_names)} 不一致")
    raw_root = config.get("path")
    dataset_root = Path(raw_root) if raw_root else data_yaml.parent
    if not dataset_root.is_absolute():
        dataset_root = data_yaml.parent / dataset_root
    return config, class_names, dataset_root.resolve()


def _resolve_relative_path(value: str, dataset_root: Path, manifest_parent: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    dataset_candidate = (dataset_root / path).resolve()
    if dataset_candidate.exists():
        return dataset_candidate
    return (manifest_parent / path).resolve()


def _images_from_entry(value: str, dataset_root: Path, yaml_parent: Path) -> list[Path]:
    path = _resolve_relative_path(value, dataset_root, yaml_parent)
    if path.is_dir():
        return sorted(
            candidate.resolve()
            for candidate in path.rglob("*")
            if candidate.is_file() and candidate.suffix.lower() in IMAGE_SUFFIXES
        )
    if path.is_file() and path.suffix.lower() == ".txt":
        images: list[Path] = []
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            if line.strip():
                images.append(_resolve_relative_path(line.strip(), dataset_root, path.parent))
        return images
    if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
        return [path]
    raise FileNotFoundError(f"无法解析数据划分路径: {value} (resolved={path})")


def resolve_split_images(
    config: dict,
    split: str,
    dataset_root: Path,
    yaml_parent: Path,
) -> list[Path]:
    """Resolve a YOLO split whose value may be a directory, image list, or list of either."""

    value = config.get(split)
    if not value:
        return []
    entries = value if isinstance(value, list) else [value]
    images: list[Path] = []
    for entry in entries:
        images.extend(_images_from_entry(str(entry), dataset_root, yaml_parent))
    unique = list(dict.fromkeys(path.resolve() for path in images))
    missing = [path for path in unique if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"{split} 清单包含不存在的图像: {missing[0]}")
    if not unique:
        raise ValueError(f"{split} 划分没有图像")
    return unique


def read_provenance_details(manifest_path: Path | None) -> dict[str, dict[str, str]]:
    """Map output filenames to provenance and optional authoritative context."""

    if manifest_path is None or not manifest_path.is_file():
        return {}
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    mapping: dict[str, dict[str, str]] = {}
    for row in rows:
        output_name = Path(row.get("output_image") or "").name
        if not output_name:
            continue
        sensor_value = str(row.get("sensor") or "").strip().casefold()
        scene_value = str(row.get("scene") or "").strip().casefold()
        mapping[output_name.casefold()] = {
            "source_image": str(row.get("source_image") or output_name),
            "operation": str(row.get("operation") or row.get("operation_detail") or "unknown"),
            "sensor": sensor_value,
            "scene": scene_value,
            "metadata_source": (
                "authoritative_manifest"
                if sensor_value in SENSOR_NAMES or scene_value in SCENE_NAMES
                else ""
            ),
        }
    return mapping


def read_provenance(manifest_path: Path | None) -> dict[str, tuple[str, str]]:
    """Backward-compatible map to ``(source frame, augmentation operation)``."""

    return {
        key: (value["source_image"], value["operation"])
        for key, value in read_provenance_details(manifest_path).items()
    }


def _fallback_source_key(image_path: Path) -> str:
    stem = image_path.stem.split("__aug-", maxsplit=1)[0]
    return f"{stem}{image_path.suffix.lower()}".casefold()


def scan_samples(
    images: Iterable[Path],
    split: str,
    class_count: int,
    provenance: dict[str, tuple[str, str] | dict[str, str]],
) -> list[SourceSample]:
    samples: list[SourceSample] = []
    for image_path in images:
        boxes = tuple(parse_yolo_boxes(resolve_label_path(image_path), class_count))
        provenance_value = provenance.get(image_path.name.casefold())
        if isinstance(provenance_value, dict):
            source_image = provenance_value.get("source_image") or image_path.name
            operation = provenance_value.get("operation") or "unknown"
            context = resolve_context_metadata(
                source_image,
                sensor=provenance_value.get("sensor"),
                scene=provenance_value.get("scene"),
                metadata_source=provenance_value.get("metadata_source") or None,
            )
        elif isinstance(provenance_value, tuple):
            source_image, operation = provenance_value
            context = resolve_context_metadata(source_image)
        else:
            source_image, operation = _fallback_source_key(image_path), "unknown"
            context = resolve_context_metadata(source_image)
        # Some manifests identify the source frame without carrying the
        # sensor/scene token, while the materialized filename does. Fill only
        # still-unknown fields from that compatibility fallback; authoritative
        # values remain untouched.
        filename_context = resolve_context_metadata(image_path.name)
        for context_field in ("sensor", "scene"):
            if (
                context[context_field] == "unknown"
                and filename_context[context_field] != "unknown"
            ):
                context[context_field] = filename_context[context_field]
        if context["metadata_source"] == "unknown" and any(
            context[field] != "unknown" for field in ("sensor", "scene")
        ):
            context["metadata_source"] = "filename_fallback"
        samples.append(
            SourceSample(
                image_path=image_path.resolve(),
                split=split,
                boxes=boxes,
                source_key=str(source_image).casefold(),
                operation=operation,
                context={"source_image": str(source_image), **context},
            )
        )
    return samples


def _class_candidates(
    samples: list[SourceSample],
    class_id: int,
    seed: int,
) -> list[SourceSample]:
    """Order candidates by source diversity before consuming augmented variants."""

    groups: defaultdict[str, list[SourceSample]] = defaultdict(list)
    for sample in samples:
        if any(box.class_id == class_id for box in sample.boxes):
            groups[sample.source_key].append(sample)
    keys = sorted(groups)
    random.Random(seed + class_id * 1009).shuffle(keys)
    for key in keys:
        groups[key].sort(
            key=lambda sample: (
                sample.operation.casefold() != "original",
                sample.operation.casefold(),
                sample.image_path.name.casefold(),
            )
        )
    ordered: list[SourceSample] = []
    depth = 0
    while True:
        progressed = False
        for key in keys:
            if depth < len(groups[key]):
                ordered.append(groups[key][depth])
                progressed = True
        if not progressed:
            return ordered
        depth += 1


def select_balanced_buffer(
    train_samples: list[SourceSample],
    learned_class_ids: list[int],
    limit: int,
    seed: int,
) -> list[ReplayEntry]:
    """Select a deterministic class-balanced image buffer with a hard capacity."""

    if limit <= 0:
        raise ValueError("buffer size 必须为正整数")
    if not learned_class_ids:
        return []
    queues = {
        class_id: _class_candidates(train_samples, class_id, seed)
        for class_id in learned_class_ids
    }
    offsets = {class_id: 0 for class_id in learned_class_ids}
    selected: list[ReplayEntry] = []
    used_paths: set[Path] = set()
    while len(selected) < limit:
        progressed = False
        for class_id in learned_class_ids:
            queue = queues[class_id]
            while offsets[class_id] < len(queue):
                sample = queue[offsets[class_id]]
                offsets[class_id] += 1
                if sample.image_path in used_paths:
                    continue
                selected.append(ReplayEntry(sample.image_path, class_id, sample.source_key))
                used_paths.add(sample.image_path)
                progressed = True
                break
            if len(selected) >= limit:
                break
        if not progressed:
            break
    return selected


def _write_label(path: Path, boxes: Iterable[YoloBox]) -> None:
    lines = [
        f"{box.class_id} {box.x_center:.8f} {box.y_center:.8f} {box.width:.8f} {box.height:.8f}"
        for box in boxes
    ]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _materialized_name(image_path: Path, role: str, class_id: int | None) -> str:
    identity = f"{image_path.resolve()}|{role}|{class_id}"
    digest = hashlib.sha1(identity.encode("utf-8")).hexdigest()[:12]
    return f"{digest}__{image_path.name}"


def _link_or_copy(source: Path, target: Path) -> str:
    try:
        os.link(source, target)
        return "hardlink"
    except OSError:
        shutil.copy2(source, target)
        return "copy"


def materialize_training_view(
    train_samples: list[SourceSample],
    current_class_id: int,
    replay_entries: list[ReplayEntry],
    output_dir: Path,
    *,
    stage: int = 0,
) -> dict:
    """Write current-task and replay examples into one standard YOLO view."""

    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    by_path = {sample.image_path: sample for sample in train_samples}
    current_samples = [
        sample
        for sample in train_samples
        if any(box.class_id == current_class_id for box in sample.boxes)
    ]
    manifest: list[str] = []
    replay_manifest: list[str] = []
    link_modes: Counter[str] = Counter()
    object_counts: Counter[int] = Counter()
    context_rows: list[dict[str, str]] = []

    def add_sample(sample: SourceSample, role: str, label_class_id: int) -> Path:
        target_image = images_dir / _materialized_name(sample.image_path, role, label_class_id)
        link_modes[_link_or_copy(sample.image_path, target_image)] += 1
        selected_boxes = [box for box in sample.boxes if box.class_id == label_class_id]
        if not selected_boxes:
            raise ValueError(f"{sample.image_path} 不包含回放槽声明的类别 {label_class_id}")
        _write_label(labels_dir / f"{target_image.stem}.txt", selected_boxes)
        object_counts[label_class_id] += len(selected_boxes)
        manifest.append(target_image.resolve().as_posix())
        context = sample.context
        context_rows.append(
            build_context_row(
                materialized_image_path=target_image,
                source_image=context.get("source_image", sample.source_key),
                sensor=context.get("sensor"),
                scene=context.get("scene"),
                split="train",
                stage=stage,
                sample_role=role,
                augmentation_operation=sample.operation,
                metadata_source=context.get("metadata_source"),
            )
        )
        return target_image

    for sample in current_samples:
        add_sample(sample, "current", current_class_id)
    for entry in replay_entries:
        sample = by_path.get(entry.image_path)
        if sample is None:
            raise ValueError(f"回放样本不在训练源集合中: {entry.image_path}")
        target = add_sample(sample, "replay", entry.class_id)
        replay_manifest.append(target.resolve().as_posix())

    manifest_path = output_dir / "train.txt"
    replay_manifest_path = output_dir / "replay.txt"
    manifest_path.write_text("\n".join(manifest) + "\n", encoding="utf-8")
    replay_manifest_path.write_text(
        "\n".join(replay_manifest) + ("\n" if replay_manifest else ""),
        encoding="utf-8",
    )
    context_index_path = write_context_index(context_rows, output_dir / "context_index.csv")
    return {
        "current_images": len(current_samples),
        "replay_images": len(replay_entries),
        "mixed_images": len(manifest),
        "objects_by_class_id": {str(key): value for key, value in sorted(object_counts.items())},
        "manifest": str(manifest_path.resolve()),
        "replay_manifest": str(replay_manifest_path.resolve()),
        "context_index": str(context_index_path),
        "context_summary": context_index_summary(context_rows),
        "materialization": dict(link_modes),
    }


def materialize_validation_view(
    val_samples: list[SourceSample],
    learned_class_ids: set[int],
    output_dir: Path,
    *,
    split: str = "val",
    stage: int = 0,
) -> dict:
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[str] = []
    link_modes: Counter[str] = Counter()
    object_counts: Counter[int] = Counter()
    context_rows: list[dict[str, str]] = []
    for sample in val_samples:
        boxes = [box for box in sample.boxes if box.class_id in learned_class_ids]
        if not boxes:
            continue
        target_image = images_dir / _materialized_name(sample.image_path, split, None)
        link_modes[_link_or_copy(sample.image_path, target_image)] += 1
        _write_label(labels_dir / f"{target_image.stem}.txt", boxes)
        manifest.append(target_image.resolve().as_posix())
        object_counts.update(box.class_id for box in boxes)
        context = sample.context
        context_rows.append(
            build_context_row(
                materialized_image_path=target_image,
                source_image=context.get("source_image", sample.source_key),
                sensor=context.get("sensor"),
                scene=context.get("scene"),
                split=split,
                stage=stage,
                sample_role=split,
                augmentation_operation=sample.operation,
                metadata_source=context.get("metadata_source"),
            )
        )
    manifest_path = output_dir / f"{split}.txt"
    manifest_path.write_text("\n".join(manifest) + ("\n" if manifest else ""), encoding="utf-8")
    context_index_path = write_context_index(context_rows, output_dir / "context_index.csv")
    return {
        "images": len(manifest),
        "objects_by_class_id": {str(key): value for key, value in sorted(object_counts.items())},
        "manifest": str(manifest_path.resolve()),
        "context_index": str(context_index_path),
        "context_summary": context_index_summary(context_rows),
        "materialization": dict(link_modes),
    }


def _write_stage_yaml(
    path: Path,
    train_manifest: str,
    val_manifest: str,
    learned_names: list[str],
    test_manifest: str | None = None,
    context_index: str | None = None,
) -> str:
    payload = {
        "train": train_manifest,
        "val": val_manifest,
        "nc": len(learned_names),
        "names": {index: name for index, name in enumerate(learned_names)},
    }
    if test_manifest:
        payload["test"] = test_manifest
    if context_index:
        payload["context_index"] = context_index
    path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return str(path.resolve())


def _buffer_payload(entries: list[ReplayEntry], class_names: list[str]) -> dict:
    counts = Counter(entry.class_id for entry in entries)
    return {
        "images": len(entries),
        "classes": {class_names[class_id]: counts[class_id] for class_id in sorted(counts)},
        "entries": [
            {
                "image_path": str(entry.image_path),
                "class_id": entry.class_id,
                "class_name": class_names[entry.class_id],
                "source_key": entry.source_key,
            }
            for entry in entries
        ],
    }


def prepare_class_incremental_dataset(
    data_yaml: Path,
    output_dir: Path,
    *,
    buffer_sizes: tuple[int, ...] = BUFFER_SIZE_CHOICES,
    seed: int = 42,
    class_order: list[str] | None = None,
    provenance_manifest: Path | None = None,
) -> dict:
    """Prepare singleton Class-IL stages and ER/DER replay views."""

    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"输出目录非空，请使用新目录: {output_dir}")
    if not buffer_sizes or any(size <= 0 for size in buffer_sizes):
        raise ValueError("至少提供一个正整数 buffer size")
    if len(buffer_sizes) != len(set(buffer_sizes)):
        raise ValueError("buffer size 不能重复")

    config, dataset_names, dataset_root = read_data_config(data_yaml)
    ordered_names = list(class_order or dataset_names)
    if ordered_names != dataset_names:
        raise ValueError(
            "当前实现要求 Class-IL 顺序与数据集 class id 顺序一致，"
            f"数据集={dataset_names}, 请求={ordered_names}"
        )
    if len(ordered_names) != 6:
        raise ValueError(f"六阶段 Class-IL 必须恰好包含 6 类，实际为 {len(ordered_names)}")
    if dataset_names != ALL_CLASS_NAMES:
        raise ValueError(f"项目六类顺序应为 {ALL_CLASS_NAMES}，实际为 {dataset_names}")

    train_images = resolve_split_images(config, "train", dataset_root, data_yaml.parent)
    val_images = resolve_split_images(config, "val", dataset_root, data_yaml.parent)
    test_images = resolve_split_images(config, "test", dataset_root, data_yaml.parent)
    if not val_images:
        raise ValueError("Class-IL 每阶段必须有固定 val")
    if provenance_manifest is None:
        candidate = data_yaml.parent / "dataset_manifest.csv"
        provenance_manifest = candidate if candidate.is_file() else None
    provenance = read_provenance_details(provenance_manifest)
    train_samples = scan_samples(train_images, "train", len(dataset_names), provenance)
    val_samples = scan_samples(val_images, "val", len(dataset_names), provenance)
    test_samples = scan_samples(test_images, "test", len(dataset_names), provenance)

    output_dir.mkdir(parents=True, exist_ok=True)
    stages: list[dict] = []
    total_materialization: Counter[str] = Counter()
    for class_id, class_name in enumerate(ordered_names):
        stage_number = class_id + 1
        stage_dir = output_dir / f"stage_{stage_number:02d}_{class_name}"
        learned_names = ordered_names[:stage_number]
        val_stats = materialize_validation_view(
            val_samples,
            set(range(stage_number)),
            stage_dir / "val_all",
            stage=stage_number,
        )
        total_materialization.update(val_stats["materialization"])
        test_stats = (
            materialize_validation_view(
                test_samples,
                set(range(stage_number)),
                stage_dir / "test_all",
                split="test",
                stage=stage_number,
            )
            if test_samples
            else None
        )
        if test_stats is not None:
            total_materialization.update(test_stats["materialization"])
        buffers: dict[str, dict] = {}
        for buffer_size in buffer_sizes:
            replay_before = select_balanced_buffer(
                train_samples,
                list(range(class_id)),
                buffer_size,
                seed,
            )
            buffer_after = select_balanced_buffer(
                train_samples,
                list(range(stage_number)),
                buffer_size,
                seed,
            )
            view_dir = stage_dir / f"train_buffer_{buffer_size}"
            train_stats = materialize_training_view(
                train_samples,
                class_id,
                replay_before,
                view_dir,
                stage=stage_number,
            )
            total_materialization.update(train_stats["materialization"])
            yaml_path = _write_stage_yaml(
                stage_dir / f"data_buffer_{buffer_size}.yaml",
                train_stats["manifest"],
                val_stats["manifest"],
                learned_names,
                test_stats["manifest"] if test_stats is not None else None,
                train_stats["context_index"],
            )
            replay_before_path = stage_dir / f"buffer_before_{buffer_size}.json"
            replay_before_payload = _buffer_payload(replay_before, ordered_names)
            replay_before_path.write_text(
                json.dumps(replay_before_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            buffer_path = stage_dir / f"buffer_after_{buffer_size}.json"
            buffer_payload = _buffer_payload(buffer_after, ordered_names)
            buffer_path.write_text(
                json.dumps(buffer_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            buffers[str(buffer_size)] = {
                "capacity": buffer_size,
                "replay_before": {
                    "images": replay_before_payload["images"],
                    "classes": replay_before_payload["classes"],
                    "path": str(replay_before_path.resolve()),
                },
                "buffer_after": {
                    "images": buffer_payload["images"],
                    "classes": buffer_payload["classes"],
                    "path": str(buffer_path.resolve()),
                },
                "training": train_stats,
                "data_yaml": yaml_path,
                "context_index": train_stats["context_index"],
            }
        stage_context_rows: list[dict[str, str]] = []
        stage_context_rows.extend(read_context_rows(Path(val_stats["context_index"])))
        if test_stats is not None:
            stage_context_rows.extend(read_context_rows(Path(test_stats["context_index"])))
        for buffer_payload in buffers.values():
            stage_context_rows.extend(
                read_context_rows(Path(buffer_payload["context_index"]))
            )
        stage_context_path = write_context_index(stage_context_rows, stage_dir / "context_index.csv")
        stages.append(
            {
                "stage": stage_number,
                "task_id": f"class_{class_id}",
                "new_classes": [class_name],
                "old_classes": ordered_names[:class_id],
                "all_learned_classes": learned_names,
                "validation": val_stats,
                "test": test_stats,
                "context_index": str(stage_context_path),
                "context_summary": context_index_summary(stage_context_rows),
                "buffers": buffers,
            }
        )

    report = {
        "protocol_version": "2.0",
        "scenario": "class_incremental",
        "head_policy": "single_expanding_head",
        "task_order": ordered_names,
        "buffer_methods": ["ER", "DER"],
        "buffer_sizes": list(buffer_sizes),
        "seed": seed,
        "source_data_yaml": str(data_yaml.resolve()),
        "provenance_manifest": str(provenance_manifest.resolve()) if provenance_manifest else None,
        "context_index_fields": list(CONTEXT_INDEX_FIELDS),
        "source_statistics": {
            "train_images": len(train_samples),
            "val_images": len(val_samples),
            "test_images": len(test_samples),
        },
        "evaluation": {
            "split": "test" if test_samples else "val",
            "official": bool(test_samples),
            "independent_holdout": bool(test_samples),
            "competition_official": False,
            "scope": "local_experiment_protocol",
            "test_usage": (
                "post_stage_evaluation_only_not_checkpoint_selection"
                if test_samples
                else "unavailable"
            ),
        },
        "stages": stages,
        "privacy": {
            "local_only": True,
            "dataset_upload": False,
            "hardlinked_images": total_materialization["hardlink"],
            "fallback_copied_images": total_materialization["copy"],
        },
        "task_il_extension": {
            "reserved": True,
            "note": "task_id 与 scenario 已独立记录；未来可增加 task_incremental 调度而不改变 ER/DER 缓冲池接口。",
        },
    }
    summary_path = output_dir / "class_incremental_dataset_summary.json"
    summary_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="准备六类别逐类 Class-IL 的 ER/DER 数据协议")
    parser.add_argument("--data", type=Path, required=True, help="原始六类 YOLO data.yaml")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--buffer-size",
        action="append",
        type=int,
        choices=BUFFER_SIZE_CHOICES,
        dest="buffer_sizes",
        help="回放池容量，可重复指定；默认同时生成 200 与 500",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--class-order", help="逗号分隔的六类顺序；默认使用 YAML 顺序")
    parser.add_argument("--provenance-manifest", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = prepare_class_incremental_dataset(
        args.data,
        args.output,
        buffer_sizes=tuple(args.buffer_sizes or BUFFER_SIZE_CHOICES),
        seed=args.seed,
        class_order=parse_class_list(args.class_order) if args.class_order else None,
        provenance_manifest=args.provenance_manifest,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
