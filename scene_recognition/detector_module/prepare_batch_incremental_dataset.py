"""Prepare arbitrary-batch, class-subset incremental YOLO views.

This protocol is intentionally separate from ``prepare_class_incremental_dataset``.
The latter models six singleton stages from a generic model; this module starts
from an already-trained four-class checkpoint and schedules samples from a
six-class increment dataset in arbitrary batches.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import yaml

from scene_recognition.detector_module import ALL_CLASS_NAMES, BASE_CLASS_NAMES
from scene_recognition.detector_module.boxes import YoloBox, parse_yolo_boxes, resolve_label_path
from scene_recognition.detector_module.context_metadata import (
    build_context_row,
    context_index_summary,
    write_context_index,
)


BUFFER_SIZE_CHOICES = (200, 500)
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
_SAFE_BATCH_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


@dataclass(frozen=True)
class Sample:
    image_path: Path
    boxes: tuple[YoloBox, ...]
    source_key: str
    operation: str
    context: dict[str, str]


@dataclass(frozen=True)
class ReplaySlot:
    sample: Sample
    class_id: int


def _class_names(config: dict) -> list[str]:
    values = config.get("names")
    if isinstance(values, dict):
        return [str(values[index] if index in values else values[str(index)]) for index in range(len(values))]
    if isinstance(values, list):
        return [str(value) for value in values]
    raise ValueError("data.yaml 缺少合法 names")


def _read_yaml(path: Path) -> tuple[dict, list[str], Path]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    config = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    if not isinstance(config, dict):
        raise ValueError(f"data.yaml 必须是映射: {path}")
    names = _class_names(config)
    if not names or len(names) != len(set(names)):
        raise ValueError(f"类别表为空或包含重复类别: {path}")
    if int(config.get("nc", len(names))) != len(names):
        raise ValueError(f"nc 与 names 数量不一致: {path}")
    raw_root = config.get("path")
    root = Path(str(raw_root)) if raw_root else path.parent
    if not root.is_absolute():
        root = path.parent / root
    return config, names, root.resolve()


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
                result.append(item.resolve())
        elif path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            result.append(path)
        else:
            raise FileNotFoundError(f"无法解析划分路径: {raw} (resolved={path})")
    unique = list(dict.fromkeys(result))
    missing = [item for item in unique if not item.is_file()]
    if missing:
        raise FileNotFoundError(f"清单包含不存在的图像: {missing[0]}")
    return unique


def _split_images(config: dict, split: str, root: Path, parent: Path, *, required: bool = False) -> list[Path]:
    result = _resolve_entry(config.get(split), root, parent)
    if required and not result:
        raise ValueError(f"{split} 划分不能为空")
    return result


def _read_manifest(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None or not path.is_file():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    mapping: dict[str, dict[str, str]] = {}
    for row in rows:
        output = Path(str(row.get("output_image", ""))).name.casefold()
        if output:
            source = str(row.get("source_image") or output)
            mapping[output] = {
                "source_key": source.casefold(),
                "operation": str(row.get("operation_key") or row.get("operation") or "unknown"),
            }
    return mapping


def _manifest_for(data_yaml: Path, config: dict) -> Path | None:
    candidates: list[Path] = []
    for value in (config.get("augmentation_manifest"), data_yaml.parent / "augmentation_manifest.csv"):
        if value:
            path = Path(str(value))
            if not path.is_absolute():
                path = data_yaml.parent / path
            candidates.append(path.resolve())
    return next((path for path in candidates if path.is_file()), None)


def _source_key(image: Path, manifest: dict[str, dict[str, str]]) -> tuple[str, str]:
    info = manifest.get(image.name.casefold())
    if info:
        return info["source_key"], info["operation"]
    stem = image.stem.split("__aug-", 1)[0]
    return f"{stem}{image.suffix.lower()}".casefold(), "unknown"


def _scan(data_yaml: Path, split: str, expected_count: int, *, required: bool = False) -> list[Sample]:
    config, _names, root = _read_yaml(data_yaml)
    paths = _split_images(config, split, root, data_yaml.parent, required=required)
    manifest = _read_manifest(_manifest_for(data_yaml.resolve(), config))
    samples: list[Sample] = []
    for image in paths:
        label = resolve_label_path(image)
        boxes = tuple(parse_yolo_boxes(label, expected_count))
        source_key, operation = _source_key(image, manifest)
        context = {"source_image": source_key, "sensor": "unknown", "scene": "unknown", "metadata_source": "unknown"}
        samples.append(Sample(image.resolve(), boxes, source_key, operation, context))
    return samples


def _validate_taxonomies(base_data: Path, increment_data: Path) -> tuple[dict, dict]:
    base_config, base_names, _ = _read_yaml(base_data)
    increment_config, increment_names, _ = _read_yaml(increment_data)
    if base_names != BASE_CLASS_NAMES:
        raise ValueError(f"base-data 必须严格使用四类 {BASE_CLASS_NAMES}，实际为 {base_names}")
    if increment_names != ALL_CLASS_NAMES:
        raise ValueError(f"increment-data 必须严格使用六类 {ALL_CLASS_NAMES}，实际为 {increment_names}")
    return base_config, increment_config


def _normalise_plan(plan: object | None, num_batches: int | None, seed: int) -> list[dict[str, object]]:
    if plan is not None:
        if isinstance(plan, (str, Path)):
            path = Path(plan)
            if not path.is_file():
                raise FileNotFoundError(path)
            plan = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(plan, dict) or not isinstance(plan.get("batches"), list):
            raise ValueError("batch-plan 必须是 {'batches': [{'id': ..., 'classes': [...]}]}")
        raw_batches = plan["batches"]
        if not raw_batches:
            raise ValueError("batch-plan 至少包含一个批次")
        if num_batches is not None and int(num_batches) != len(raw_batches):
            raise ValueError("--num-batches 与 batch-plan 的批次数不一致")
        result: list[dict[str, object]] = []
        seen_ids: set[str] = set()
        for index, raw in enumerate(raw_batches, start=1):
            if not isinstance(raw, dict):
                raise ValueError(f"第 {index} 个 batch 必须是对象")
            batch_id = str(raw.get("id", "")).strip()
            classes = raw.get("classes")
            if not batch_id or not _SAFE_BATCH_ID.fullmatch(batch_id):
                raise ValueError(f"第 {index} 个 batch id 非法: {batch_id!r}")
            if batch_id in seen_ids:
                raise ValueError(f"batch id 重复: {batch_id}")
            if not isinstance(classes, list) or not classes:
                raise ValueError(f"批次 {batch_id} 必须包含非空 classes")
            normalized = [str(value).strip() for value in classes]
            if any(not item for item in normalized):
                raise ValueError(f"批次 {batch_id} 包含空类别")
            if len(normalized) != len(set(normalized)):
                raise ValueError(f"批次 {batch_id} 的 classes 不能重复")
            unknown = sorted(set(normalized) - set(ALL_CLASS_NAMES))
            if unknown:
                raise ValueError(f"批次 {batch_id} 包含未知类别: {unknown}")
            result.append({"id": batch_id, "classes": normalized})
            seen_ids.add(batch_id)
        required_new = set(ALL_CLASS_NAMES[len(BASE_CLASS_NAMES):])
        covered = {name for batch in result for name in batch["classes"]}
        if not required_new.issubset(covered):
            raise ValueError(f"batch-plan 必须覆盖两个新增类: {sorted(required_new - covered)}")
        return result
    if num_batches is None or num_batches <= 0:
        raise ValueError("未提供 batch-plan 时 --num-batches 必须为正整数")
    # Without an explicit plan every class is eligible in every batch.  The
    # deterministic source-group assignment below decides which classes are
    # actually present in each batch (and records missing classes naturally).
    return [{"id": f"batch_{index:02d}", "classes": list(ALL_CLASS_NAMES)} for index in range(1, num_batches + 1)]


def _assign_groups(samples: list[Sample], batches: list[dict[str, object]], seed: int) -> dict[str, dict[str, int]]:
    """Assign each class/source group to one occurrence of that class."""

    groups: dict[str, dict[str, list[Sample]]] = defaultdict(lambda: defaultdict(list))
    for sample in samples:
        for class_id in {box.class_id for box in sample.boxes}:
            groups[ALL_CLASS_NAMES[class_id]][sample.source_key].append(sample)
    result: dict[str, dict[str, int]] = {}
    for class_name, by_source in groups.items():
        occurrences = [index for index, batch in enumerate(batches) if class_name in batch["classes"]]
        if not occurrences:
            continue
        keys = sorted(by_source)
        random.Random(seed + ALL_CLASS_NAMES.index(class_name) * 1009).shuffle(keys)
        result[class_name] = {key: occurrences[index % len(occurrences)] for index, key in enumerate(keys)}
    return result


def _select_current(
    samples: list[Sample],
    batch_index: int,
    requested: list[str],
    assignments: dict[str, dict[str, int]],
    consumed: dict[str, set[str]],
    max_per_class: int | None,
) -> dict[int, list[Sample]]:
    selected: dict[int, list[Sample]] = defaultdict(list)
    for class_name in dict.fromkeys(requested):
        class_id = ALL_CLASS_NAMES.index(class_name)
        by_group: dict[str, list[Sample]] = defaultdict(list)
        for sample in samples:
            if any(box.class_id == class_id for box in sample.boxes):
                if assignments.get(class_name, {}).get(sample.source_key) == batch_index and sample.source_key not in consumed[class_name]:
                    by_group[sample.source_key].append(sample)
        groups = list(by_group)
        random.Random(17 + class_id * 7919 + batch_index).shuffle(groups)
        for key in groups:
            by_group[key].sort(key=lambda item: (item.operation.casefold() != "original", item.operation.casefold(), item.image_path.name.casefold()))
        limit = max_per_class if max_per_class is not None else sum(len(value) for value in by_group.values())
        if limit <= 0:
            continue
        # Round-robin variants gives distinct source diversity before consuming
        # an augmented family.  A source family is always assigned to one batch.
        depth = 0
        while len(selected[class_id]) < limit:
            progressed = False
            for key in groups:
                variants = by_group[key]
                if depth < len(variants) and len(selected[class_id]) < limit:
                    selected[class_id].append(variants[depth])
                    progressed = True
            if not progressed:
                break
            depth += 1
        # A group touched by this batch is consumed as a group, preventing its
        # original/augmented siblings from crossing into another batch.
        consumed[class_name].update(groups)
    return selected


def _balanced_replay(samples: list[Sample], class_ids: Iterable[int], capacity: int, seed: int) -> list[ReplaySlot]:
    class_ids = list(dict.fromkeys(class_ids))
    if capacity not in BUFFER_SIZE_CHOICES:
        raise ValueError(f"buffer size 只允许 {BUFFER_SIZE_CHOICES}")
    pools: dict[int, dict[str, list[Sample]]] = {class_id: defaultdict(list) for class_id in class_ids}
    for sample in samples:
        present = {box.class_id for box in sample.boxes}
        for class_id in class_ids:
            if class_id in present:
                pools[class_id][sample.source_key].append(sample)
    keys_by_class: dict[int, list[str]] = {}
    for class_id, groups in pools.items():
        keys = sorted(groups)
        random.Random(seed + class_id * 1009).shuffle(keys)
        for key in keys:
            groups[key].sort(key=lambda item: (item.operation.casefold() != "original", item.operation.casefold(), item.image_path.name.casefold()))
        keys_by_class[class_id] = keys
    result: list[ReplaySlot] = []
    used: set[tuple[Path, int]] = set()
    queues: dict[int, list[Sample]] = {}
    for class_id in class_ids:
        keys = keys_by_class[class_id]
        queues[class_id] = [
            pools[class_id][key][depth]
            for depth in range(max((len(pools[class_id][key]) for key in keys), default=0))
            for key in keys
            if depth < len(pools[class_id][key])
        ]
    offsets = {class_id: 0 for class_id in class_ids}
    while len(result) < capacity:
        progressed = False
        for class_id in class_ids:
            queue = queues[class_id]
            while offsets[class_id] < len(queue):
                sample = queue[offsets[class_id]]
                offsets[class_id] += 1
                marker = (sample.image_path, class_id)
                if marker in used:
                    continue
                result.append(ReplaySlot(sample, class_id))
                used.add(marker)
                progressed = True
                break
            if len(result) >= capacity:
                break
        if not progressed:
            break
    return result


def _write_label(path: Path, boxes: Iterable[YoloBox]) -> None:
    path.write_text("".join(f"{box.class_id} {box.x_center:.8f} {box.y_center:.8f} {box.width:.8f} {box.height:.8f}\n" for box in boxes), encoding="utf-8")


def _copy_or_link(source: Path, target: Path) -> str:
    try:
        os.link(source, target)
        return "hardlink"
    except OSError:
        shutil.copy2(source, target)
        return "copy"


def _materialized_name(sample: Sample, role: str, class_id: int, ordinal: int) -> str:
    identity = f"{sample.image_path.resolve()}|{role}|{class_id}|{ordinal}"
    digest = hashlib.sha1(identity.encode("utf-8")).hexdigest()[:12]
    return f"{digest}__{sample.image_path.stem}{sample.image_path.suffix.lower()}"


def _write_view(
    output: Path,
    current: dict[int, list[Sample]],
    replay: list[ReplaySlot],
    *,
    learned_ids: set[int],
    split_samples: list[Sample] | None = None,
    split: str | None = None,
    stage: int,
) -> dict:
    images = output / "images"
    labels = output / "labels"
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    replay_lines: list[str] = []
    context_rows: list[dict[str, str]] = []
    modes: Counter[str] = Counter()
    object_counts: Counter[int] = Counter()
    ordinal = 0

    def add(sample: Sample, class_id: int, role: str) -> str:
        nonlocal ordinal
        ordinal += 1
        name = _materialized_name(sample, role, class_id, ordinal)
        image_target = images / name
        label_target = labels / f"{Path(name).stem}.txt"
        modes[_copy_or_link(sample.image_path, image_target)] += 1
        selected = [box for box in sample.boxes if box.class_id == class_id and box.class_id in learned_ids]
        if not selected:
            raise ValueError(f"{sample.image_path} 不包含可见类别 {class_id}")
        _write_label(label_target, selected)
        object_counts[class_id] += len(selected)
        lines.append(image_target.resolve().as_posix())
        if role == "replay":
            replay_lines.append(image_target.resolve().as_posix())
        context_rows.append(build_context_row(
            materialized_image_path=image_target,
            source_image=sample.context.get("source_image", sample.source_key),
            sensor=sample.context.get("sensor"),
            scene=sample.context.get("scene"),
            split=split or "train",
            stage=stage,
            sample_role=role,
            augmentation_operation=sample.operation,
            metadata_source=sample.context.get("metadata_source"),
        ))
        return image_target.resolve().as_posix()

    if split_samples is not None and split is not None:
        seen_paths: set[Path] = set()
        for sample in split_samples:
            for class_id in sorted({box.class_id for box in sample.boxes if box.class_id in learned_ids}):
                if sample.image_path in seen_paths:
                    # Validation keeps all visible labels together, unlike the
                    # single-class training copies.
                    continue
                seen_paths.add(sample.image_path)
                ordinal += 1
                name = _materialized_name(sample, split, -1, ordinal)
                target = images / name
                modes[_copy_or_link(sample.image_path, target)] += 1
                _write_label(labels / f"{Path(name).stem}.txt", [box for box in sample.boxes if box.class_id in learned_ids])
                lines.append(target.resolve().as_posix())
                object_counts.update(box.class_id for box in sample.boxes if box.class_id in learned_ids)
                context_rows.append(build_context_row(
                    materialized_image_path=target,
                    source_image=sample.context.get("source_image", sample.source_key),
                    sensor=sample.context.get("sensor"), scene=sample.context.get("scene"),
                    split=split, stage=stage, sample_role=split,
                    augmentation_operation=sample.operation,
                    metadata_source=sample.context.get("metadata_source"),
                ))
                break
    else:
        for class_id in sorted(current):
            for sample in current[class_id]:
                add(sample, class_id, "current")
        for slot in replay:
            add(slot.sample, slot.class_id, "replay")

    manifest = output / (f"{split}.txt" if split else "train.txt")
    manifest.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    if split is None:
        replay_manifest = output / "replay.txt"
        replay_manifest.write_text("\n".join(replay_lines) + ("\n" if replay_lines else ""), encoding="utf-8")
    index_path = write_context_index(context_rows, output / "context_index.csv")
    return {
        "images": len(lines),
        "current_images": sum(len(value) for value in current.values()) if split is None else 0,
        "replay_images": len(replay_lines) if split is None else 0,
        "mixed_images": len(lines) if split is None else None,
        "manifest": str(manifest.resolve()),
        "replay_manifest": str((output / "replay.txt").resolve()) if split is None else None,
        "objects_by_class": {ALL_CLASS_NAMES[key]: value for key, value in sorted(object_counts.items())},
        "context_index": str(index_path.resolve()),
        "context_summary": context_index_summary(context_rows),
        "materialization": dict(modes),
    }


def _write_batch_yaml(path: Path, train_view: dict, val_view: dict, test_view: dict | None) -> None:
    payload: dict[str, object] = {
        "train": train_view["manifest"],
        "val": val_view["manifest"],
        "nc": len(ALL_CLASS_NAMES),
        "names": {index: name for index, name in enumerate(ALL_CLASS_NAMES)},
    }
    if test_view is not None:
        payload["test"] = test_view["manifest"]
    payload["context_index"] = train_view["context_index"]
    path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")


def _buffer_summary(slots: list[ReplaySlot]) -> dict:
    counts = Counter(slot.class_id for slot in slots)
    return {
        "images": len(slots),
        "classes": {ALL_CLASS_NAMES[key]: counts[key] for key in sorted(counts)},
        "entries": [
            {"image_path": str(slot.sample.image_path), "class_id": slot.class_id, "class_name": ALL_CLASS_NAMES[slot.class_id], "source_key": slot.sample.source_key}
            for slot in slots
        ],
    }


def prepare_batch_incremental_dataset(
    base_data: Path,
    increment_data: Path,
    output: Path,
    num_batches: int | None = None,
    *,
    batch_plan: object | None = None,
    buffer_sizes: Sequence[int] = (200,),
    seed: int = 42,
    max_current_images_per_class: int | None = None,
    buffer_size: int | None = None,
) -> dict:
    """Build arbitrary class-subset batches and fixed-capacity replay views."""

    base_data, increment_data, output = Path(base_data).resolve(), Path(increment_data).resolve(), Path(output).resolve()
    _validate_taxonomies(base_data, increment_data)
    if buffer_size is not None:
        buffer_sizes = (buffer_size,)
    if any(int(value) not in BUFFER_SIZE_CHOICES for value in buffer_sizes):
        raise ValueError(f"buffer-size 只允许 {BUFFER_SIZE_CHOICES}")
    buffer_sizes = tuple(dict.fromkeys(int(value) for value in buffer_sizes))
    if not buffer_sizes:
        raise ValueError("至少提供一个 buffer-size")
    if max_current_images_per_class is not None and max_current_images_per_class <= 0:
        raise ValueError("max-current-images-per-class 必须为正整数")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"输出目录非空，请选择新目录: {output}")
    output.mkdir(parents=True, exist_ok=True)
    plan = _normalise_plan(batch_plan, num_batches, seed)
    base_train = _scan(base_data, "train", len(BASE_CLASS_NAMES), required=True)
    increment_train = _scan(increment_data, "train", len(ALL_CLASS_NAMES), required=True)
    increment_val = _scan(increment_data, "val", len(ALL_CLASS_NAMES), required=True)
    increment_test = _scan(increment_data, "test", len(ALL_CLASS_NAMES), required=False)
    present_base = {box.class_id for sample in base_train for box in sample.boxes}
    missing_base = sorted(set(range(len(BASE_CLASS_NAMES))) - present_base)
    if missing_base:
        raise ValueError(f"base-data train 缺少基础类别: {[BASE_CLASS_NAMES[index] for index in missing_base]}")
    assignments = _assign_groups(increment_train, plan, seed)
    consumed: dict[str, set[str]] = defaultdict(set)
    previous_seen: set[int] = set(range(len(BASE_CLASS_NAMES)))
    consumed_samples: list[Sample] = []
    previous_after: dict[int, list[ReplaySlot]] = {size: [] for size in buffer_sizes}
    batches_summary: list[dict] = []
    for batch_index, batch in enumerate(plan):
        batch_id = str(batch["id"])
        requested = list(dict.fromkeys(str(item) for item in batch["classes"]))
        current = _select_current(increment_train, batch_index, requested, assignments, consumed, max_current_images_per_class)
        present = [ALL_CLASS_NAMES[class_id] for class_id in sorted(current) if current[class_id]]
        missing = [name for name in requested if name not in present]
        newly_seen = [name for name in present if ALL_CLASS_NAMES.index(name) not in previous_seen]
        seen_ids = previous_seen | {ALL_CLASS_NAMES.index(name) for name in present}
        seen_names = [name for name in ALL_CLASS_NAMES if ALL_CLASS_NAMES.index(name) in seen_ids]
        for class_id in sorted(current):
            consumed_samples.extend(current[class_id])
        batch_root = output / batch_id
        batch_root.mkdir(parents=True, exist_ok=False)
        buffers: dict[str, dict] = {}
        for buffer_size in buffer_sizes:
            view_root = batch_root / f"buffer_{buffer_size}"
            view_root.mkdir(parents=True, exist_ok=False)
            # The first batch is the four-class checkpoint's replay anchor.
            # Every later batch starts from the exact previous buffer_after;
            # that makes replay evolution auditable instead of reselecting a
            # different pre-update pool during training.
            replay = (
                _balanced_replay(base_train, list(range(len(BASE_CLASS_NAMES))), buffer_size, seed + buffer_size)
                if batch_index == 0
                else list(previous_after[buffer_size])
            )
            # First replay must be sourced from base data only.
            if batch_index == 0 and any(slot.sample not in base_train for slot in replay):
                raise AssertionError("首批 replay 只能来自四类 base train")
            train_view = _write_view(view_root / "train", current, replay, learned_ids=seen_ids, stage=batch_index + 1)
            val_view = _write_view(view_root / "val", {}, [], learned_ids=seen_ids, split_samples=increment_val, split="val", stage=batch_index + 1)
            test_view = None
            if increment_test:
                test_view = _write_view(view_root / "test", {}, [], learned_ids=seen_ids, split_samples=increment_test, split="test", stage=batch_index + 1)
            data_yaml = view_root / "data.yaml"
            _write_batch_yaml(data_yaml, train_view, val_view, test_view)
            after_pool = base_train + consumed_samples
            after_slots = _balanced_replay(after_pool, sorted(seen_ids), buffer_size, seed + batch_index * 17 + buffer_size + 1)
            previous_after[buffer_size] = after_slots
            buffers[str(buffer_size)] = {
                "data_yaml": str(data_yaml.resolve()),
                "training": train_view,
                "validation": val_view,
                "test": test_view,
                "replay_before": _buffer_summary(replay),
                "buffer_after": _buffer_summary(after_slots),
                "buffer_capacity": buffer_size,
            }
        context_paths = [Path(value["training"]["context_index"]) for value in buffers.values()]
        before_by_buffer = {size: value["replay_before"] for size, value in buffers.items()}
        after_by_buffer = {size: value["buffer_after"] for size, value in buffers.items()}
        mixed_by_buffer = {size: value["training"]["mixed_images"] for size, value in buffers.items()}
        batches_summary.append({
            "index": batch_index + 1,
            "id": batch_id,
            "requested": requested,
            "present": present,
            "missing": missing,
            "seen": seen_names,
            "newly_seen": newly_seen,
            "K": max_current_images_per_class,
            "current": {"images": sum(len(value) for value in current.values()), "by_class": {ALL_CLASS_NAMES[key]: len(value) for key, value in sorted(current.items())}},
            "replay": {"buffer_sizes": list(buffer_sizes), "before": before_by_buffer, "after": after_by_buffer},
            "mixed": mixed_by_buffer,
            "buffer_before": before_by_buffer,
            "buffer_after": after_by_buffer,
            "buffers": buffers,
            "context": {"indices": [str(path) for path in context_paths]},
            "evaluation": {"model_selection": "val", "final_report": "test" if increment_test else "val", "test_after_all_batches": bool(increment_test)},
        })
        previous_seen = seen_ids
    first_arrival_batch: dict[str, int] = {}
    for batch in batches_summary:
        for class_name in batch["present"]:
            if class_name in ALL_CLASS_NAMES[len(BASE_CLASS_NAMES):]:
                first_arrival_batch.setdefault(class_name, int(batch["index"]))
    summary = {
        "protocol_version": "batch-class-incremental-v1",
        "scenario": "batch_class_incremental",
        "head_policy": "fixed_six_class_head",
        "base_classes": list(BASE_CLASS_NAMES),
        "incremental_classes": list(ALL_CLASS_NAMES[len(BASE_CLASS_NAMES):]),
        "task_order": list(ALL_CLASS_NAMES),
        "base_data": str(base_data),
        "increment_data": str(increment_data),
        "output": str(output),
        "plan": {"batches": [{"id": batch["id"], "classes": batch["classes"]} for batch in plan], "seed": seed, "deterministic": True},
        "num_batches": len(plan),
        "buffer_sizes": list(buffer_sizes),
        "max_current_images_per_class": max_current_images_per_class,
        "batches": batches_summary,
        "first_arrival_batch": first_arrival_batch,
        "evaluation": {"model_selection_split": "val", "reporting_split": "test" if increment_test else "val", "test_is_never_used_for_checkpoint_selection": True},
        "privacy": {"offline": True, "dataset_upload": False, "absolute_paths_are_local_artifact_metadata": True},
    }
    summary_path = output / "batch_incremental_dataset_summary.json"
    summary["summary"] = str(summary_path.resolve())
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def parse_batch_plan(plan: object, num_batches: int | None = None, seed: int = 42) -> list[dict[str, object]]:
    """Public plan validator used by audits and callers outside the CLI."""

    return _normalise_plan(plan, num_batches, seed)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="准备四类 checkpoint → 任意批次六类 Class-IL 数据视图")
    parser.add_argument("--base-data", type=Path, required=True, help="四类离线增广 data.yaml")
    parser.add_argument("--increment-data", type=Path, required=True, help="六类离线增广 data.yaml")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-batches", type=int)
    parser.add_argument("--batch-plan", type=Path, help="JSON 文件：{\"batches\":[{\"id\":...,\"classes\":[...]}]}")
    parser.add_argument("--buffer-size", type=int, action="append", dest="buffer_sizes", default=None, choices=BUFFER_SIZE_CHOICES)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-current-images-per-class", type=int)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = prepare_batch_incremental_dataset(
        args.base_data,
        args.increment_data,
        args.output,
        args.num_batches,
        batch_plan=args.batch_plan,
        buffer_sizes=tuple(args.buffer_sizes or (200,)),
        seed=args.seed,
        max_current_images_per_class=args.max_current_images_per_class,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
