"""Create a leakage-safe train/validation/test split for a YOLO dataset.

The existing training split is kept unchanged.  The existing validation pool
is treated as an untouched holdout and divided into validation and test sets.
All variants derived from the same source image are assigned together.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml

from scene_recognition.detector_module.boxes import YoloBox, resolve_label_path
from scene_recognition.detector_module.prepare_class_incremental_dataset import (
    SourceSample,
    read_data_config,
    read_provenance,
    resolve_split_images,
    scan_samples,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = PROJECT_ROOT / "scene_recognition" / "detector_module" / "artifacts" / "tvt"


@dataclass(frozen=True)
class SourceGroup:
    """One indivisible source frame and all of its image variants."""

    key: str
    samples: tuple[SourceSample, ...]
    class_ids: tuple[int, ...]
    box_counts: tuple[int, ...]
    provenances: tuple[str, ...]


def _read_manifest_rows(path: Path | None) -> list[dict[str, str]]:
    if path is None or not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _provenance_by_output(rows: Iterable[dict[str, str]]) -> dict[str, str]:
    return {
        Path(row.get("output_image") or "").name.casefold(): str(
            row.get("provenance") or "unknown"
        )
        for row in rows
        if row.get("output_image")
    }


def _build_groups(
    samples: Iterable[SourceSample],
    class_count: int,
    provenance_by_output: dict[str, str],
) -> list[SourceGroup]:
    grouped: defaultdict[str, list[SourceSample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.source_key].append(sample)
    result: list[SourceGroup] = []
    for key in sorted(grouped):
        rows = tuple(sorted(grouped[key], key=lambda sample: sample.image_path.name.casefold()))
        boxes = [box for sample in rows for box in sample.boxes]
        counts = Counter(box.class_id for box in boxes)
        result.append(
            SourceGroup(
                key=key,
                samples=rows,
                class_ids=tuple(sorted(counts)),
                box_counts=tuple(counts[class_id] for class_id in range(class_count)),
                provenances=tuple(
                    sorted(
                        {
                            provenance_by_output.get(
                                sample.image_path.name.casefold(), "unknown"
                            )
                            for sample in rows
                        }
                    )
                ),
            )
        )
    return result


def _feature_vector(group: SourceGroup) -> dict[str, float]:
    features: dict[str, float] = {}
    for class_id in group.class_ids:
        features[f"presence:{class_id}"] = 1.0
    for class_id, count in enumerate(group.box_counts):
        if count:
            features[f"boxes:{class_id}"] = float(count)
    for provenance in group.provenances:
        features[f"provenance:{provenance}"] = 1.0
    signature = ",".join(str(class_id) for class_id in group.class_ids)
    features[f"signature:{signature}"] = 1.0
    return features


def _feature_weight(name: str) -> float:
    if name.startswith("presence:"):
        return 4.0
    if name.startswith("boxes:"):
        return 1.0
    if name.startswith("provenance:"):
        return 1.0
    return 0.5


def _split_score(
    selected: set[int],
    vectors: list[dict[str, float]],
    totals: Counter[str],
    target_ratio: float,
) -> float:
    actual: Counter[str] = Counter()
    for index in selected:
        actual.update(vectors[index])
    score = 0.0
    for name, total in totals.items():
        target = total * target_ratio
        denominator = max(1.0, min(target, total - target))
        error = (actual[name] - target) / denominator
        score += _feature_weight(name) * error * error
        if name.startswith("presence:") and total >= 2:
            if actual[name] <= 0 or actual[name] >= total:
                score += 1_000.0
    return score


def select_test_groups(
    groups: list[SourceGroup],
    *,
    test_fraction: float,
    seed: int,
    search_trials: int,
) -> set[str]:
    """Select a deterministic multilabel-stratified subset of source groups."""

    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction 必须位于 (0, 1)")
    if len(groups) < 2:
        raise ValueError("留出原图组少于 2，无法拆分 val/test")
    if search_trials <= 0:
        raise ValueError("search_trials 必须为正整数")
    target_size = max(1, min(len(groups) - 1, math.floor(len(groups) * test_fraction)))
    target_ratio = target_size / len(groups)
    vectors = [_feature_vector(group) for group in groups]
    totals: Counter[str] = Counter()
    for vector in vectors:
        totals.update(vector)

    rng = random.Random(seed)
    best: set[int] | None = None
    best_score = math.inf
    population = list(range(len(groups)))
    for _ in range(search_trials):
        candidate = set(rng.sample(population, target_size))
        score = _split_score(candidate, vectors, totals, target_ratio)
        if score < best_score:
            best = candidate
            best_score = score
    if best is None:
        raise RuntimeError("未能生成 val/test 划分")

    # Deterministic pair swaps refine the best sampled assignment while
    # preserving the exact number of source groups in the test set.
    for _ in range(20):
        improved = False
        current_score = _split_score(best, vectors, totals, target_ratio)
        best_swap: tuple[int, int] | None = None
        best_swap_score = current_score
        validation_indices = sorted(set(population) - best)
        for test_index in sorted(best):
            for validation_index in validation_indices:
                candidate = (best - {test_index}) | {validation_index}
                score = _split_score(candidate, vectors, totals, target_ratio)
                if score + 1e-12 < best_swap_score:
                    best_swap = (test_index, validation_index)
                    best_swap_score = score
        if best_swap is not None:
            best.remove(best_swap[0])
            best.add(best_swap[1])
            improved = True
        if not improved:
            break
    return {groups[index].key for index in best}


def _link_or_copy(source: Path, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
        return "hardlink"
    except OSError:
        shutil.copy2(source, target)
        return "copy"


def _materialize_samples(
    samples: Iterable[SourceSample],
    split: str,
    output_dir: Path,
) -> Counter[str]:
    modes: Counter[str] = Counter()
    seen_names: set[str] = set()
    for sample in samples:
        folded_name = sample.image_path.name.casefold()
        if folded_name in seen_names:
            raise ValueError(f"{split} 内存在重复图像文件名: {sample.image_path.name}")
        seen_names.add(folded_name)
        source_label = resolve_label_path(sample.image_path)
        if not source_label.is_file():
            raise FileNotFoundError(source_label)
        target_image = output_dir / "images" / split / sample.image_path.name
        target_label = output_dir / "labels" / split / f"{sample.image_path.stem}.txt"
        modes[f"image_{_link_or_copy(sample.image_path, target_image)}"] += 1
        modes[f"label_{_link_or_copy(source_label, target_label)}"] += 1
    return modes


def _split_statistics(
    samples: list[SourceSample],
    class_names: list[str],
) -> dict:
    presence: Counter[int] = Counter()
    boxes: Counter[int] = Counter()
    for sample in samples:
        present = {box.class_id for box in sample.boxes}
        presence.update(present)
        boxes.update(box.class_id for box in sample.boxes)
    return {
        "images": len(samples),
        "source_groups": len({sample.source_key for sample in samples}),
        "class_image_presence": {
            name: presence[class_id] for class_id, name in enumerate(class_names)
        },
        "class_boxes": {name: boxes[class_id] for class_id, name in enumerate(class_names)},
    }


def _write_updated_manifest(
    output_path: Path,
    source_rows: list[dict[str, str]],
    split_by_name: dict[str, str],
) -> None:
    if not source_rows:
        return
    fieldnames = list(source_rows[0])
    if "split" not in fieldnames:
        fieldnames.insert(0, "split")
    updated: list[dict[str, str]] = []
    for row in source_rows:
        output_name = Path(row.get("output_image") or "").name.casefold()
        if output_name not in split_by_name:
            raise ValueError(f"数据清单含无法匹配的图像: {row.get('output_image')}")
        updated.append({**row, "split": split_by_name[output_name]})
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(updated)


def split_yolo_dataset(
    data_yaml: Path,
    output_dir: Path,
    *,
    test_fraction: float = 0.5,
    seed: int = 42,
    search_trials: int = 2_000,
    provenance_manifest: Path | None = None,
) -> dict:
    """Keep train fixed and split the existing holdout into val and test."""

    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"输出目录非空，请使用新目录: {output_dir}")
    config, class_names, dataset_root = read_data_config(data_yaml)
    train_images = resolve_split_images(config, "train", dataset_root, data_yaml.parent)
    holdout_images = resolve_split_images(config, "val", dataset_root, data_yaml.parent)
    if resolve_split_images(config, "test", dataset_root, data_yaml.parent):
        raise ValueError("输入数据已经声明 test；本命令只拆分尚无独立 test 的数据集")
    if provenance_manifest is None:
        candidate = data_yaml.parent / "dataset_manifest.csv"
        provenance_manifest = candidate if candidate.is_file() else None
    manifest_rows = _read_manifest_rows(provenance_manifest)
    provenance = read_provenance(provenance_manifest)
    class_count = len(class_names)
    train_samples = scan_samples(train_images, "train", class_count, provenance)
    holdout_samples = scan_samples(holdout_images, "val", class_count, provenance)
    train_keys = {sample.source_key for sample in train_samples}
    holdout_keys = {sample.source_key for sample in holdout_samples}
    overlap = train_keys & holdout_keys
    if overlap:
        raise ValueError(f"train 与留出集存在原图组泄漏，例如: {sorted(overlap)[0]}")

    groups = _build_groups(
        holdout_samples,
        class_count,
        _provenance_by_output(manifest_rows),
    )
    test_keys = select_test_groups(
        groups,
        test_fraction=test_fraction,
        seed=seed,
        search_trials=search_trials,
    )
    validation_samples = [sample for sample in holdout_samples if sample.source_key not in test_keys]
    test_samples = [sample for sample in holdout_samples if sample.source_key in test_keys]
    split_samples = {
        "train": train_samples,
        "val": validation_samples,
        "test": test_samples,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    materialization: Counter[str] = Counter()
    split_by_name: dict[str, str] = {}
    for split, samples in split_samples.items():
        materialization.update(_materialize_samples(samples, split, output_dir))
        for sample in samples:
            key = sample.image_path.name.casefold()
            previous = split_by_name.get(key)
            if previous is not None and previous != split:
                raise ValueError(f"同名图像跨集合冲突: {sample.image_path.name}")
            split_by_name[key] = split

    output_yaml = output_dir / "data.yaml"
    output_yaml.write_text(
        yaml.safe_dump(
            {
                "path": output_dir.as_posix(),
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
                "nc": class_count,
                "names": {index: name for index, name in enumerate(class_names)},
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    output_manifest = output_dir / "dataset_manifest.csv"
    _write_updated_manifest(output_manifest, manifest_rows, split_by_name)

    key_sets = {
        split: {sample.source_key for sample in samples}
        for split, samples in split_samples.items()
    }
    report = {
        "policy": "fixed_train_stratified_holdout_split_by_source_group",
        "seed": seed,
        "test_fraction_within_original_holdout": test_fraction,
        "source_data_yaml": str(data_yaml.resolve()),
        "output_data_yaml": str(output_yaml.resolve()),
        "provenance_manifest": (
            str(provenance_manifest.resolve()) if provenance_manifest else None
        ),
        "splits": {
            split: _split_statistics(samples, class_names)
            for split, samples in split_samples.items()
        },
        "leakage_audit": {
            "grouping_key": "dataset_manifest.source_image with filename fallback",
            "train_val_source_overlap": len(key_sets["train"] & key_sets["val"]),
            "train_test_source_overlap": len(key_sets["train"] & key_sets["test"]),
            "val_test_source_overlap": len(key_sets["val"] & key_sets["test"]),
            "augmentation_variants_kept_together": True,
        },
        "materialization": dict(materialization),
        "privacy": {
            "local_only": True,
            "dataset_upload": False,
            "git_tracking_allowed": False,
        },
    }
    (output_dir / "split_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="保留现有 train，并按原图组分层拆分 val/test"
    )
    parser.add_argument("--data", type=Path, required=True, help="只有 train/val 的 YOLO YAML")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.5,
        help="test 占原留出集的比例；默认 0.5，对总数据约为 10%%",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--search-trials", type=int, default=2_000)
    parser.add_argument("--provenance-manifest", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = split_yolo_dataset(
        args.data,
        args.output,
        test_fraction=args.test_fraction,
        seed=args.seed,
        search_trials=args.search_trials,
        provenance_manifest=args.provenance_manifest,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
