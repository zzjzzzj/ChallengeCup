"""Prepare local-only manifests for the official r2 continual-learning round.

The r2 directory is a training injection, not a replacement for the fixed r1
evaluation set. This module therefore keeps every r2 image in training unless
explicit smoke-test holdout ratios are provided. It writes absolute-path
manifests only under ignored artifact directories and never copies or uploads
source images.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from scene_recognition.detector_module import BASE_CLASS_NAMES
from scene_recognition.detector_module.boxes import parse_yolo_boxes, resolve_label_path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "scene_recognition"
    / "detector_module"
    / "artifacts"
    / "continual_r2"
)
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
R2_NAME_RE = re.compile(
    r"^(?P<sensor>ir|sar)_r(?P<round>\d+)_inc_"
    r"(?P<scene>air|sea|urban|forest)_(?P<index>\d{6})$"
)


def parse_class_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def read_class_names(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"类别文件不存在: {path}")
    names = [
        line.strip()
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    if not names or len(names) != len(set(names)):
        raise ValueError(f"类别文件为空或包含重复项: {path}")
    return names


def validate_taxonomy(class_names: list[str], base_classes: list[str]) -> list[str]:
    if class_names[: len(base_classes)] != base_classes:
        raise ValueError(
            "增量类别表必须保留基础类别的编号和顺序；期望前缀为 "
            f"{base_classes}，实际为 {class_names[:len(base_classes)]}"
        )
    new_classes = class_names[len(base_classes) :]
    if not new_classes:
        raise ValueError("类别表没有新增类别，无法构建增量协议")
    return new_classes


def validate_image_and_label(image_path: Path, class_count: int) -> None:
    if not image_path.is_file():
        raise FileNotFoundError(f"图像不存在: {image_path}")
    parse_yolo_boxes(resolve_label_path(image_path), class_count)


def read_base_index(index_path: Path | None, class_count: int) -> list[dict[str, str]]:
    if index_path is None:
        return []
    if not index_path.is_file():
        raise FileNotFoundError(f"基础数据索引不存在: {index_path}")
    with index_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        if row.get("split") not in {"train", "val", "test"}:
            raise ValueError(f"基础索引包含非法 split: {row.get('split')}")
        image_path = Path(row.get("image_path") or row.get("image") or "")
        validate_image_and_label(image_path, class_count)
        row["image_path"] = str(image_path.resolve())
        row.setdefault("sensor", "unknown")
        row.setdefault("scene", "unknown")
        row["source"] = "r1_base"
    return rows


def discover_increment_rows(
    dataset_root: Path,
    class_count: int,
    split: str = "train",
) -> list[dict[str, str]]:
    if not dataset_root.is_dir():
        raise NotADirectoryError(f"增量数据目录不存在: {dataset_root}")
    images = sorted(
        path
        for path in dataset_root.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        raise ValueError(f"增量数据目录中没有图像: {dataset_root}")
    rows: list[dict[str, str]] = []
    for image_path in images:
        match = R2_NAME_RE.match(image_path.stem)
        if not match:
            raise ValueError(f"无法解析 r2 文件名: {image_path.name}")
        validate_image_and_label(image_path, class_count)
        rows.append(
            {
                "image_path": str(image_path.resolve()),
                "sensor": match.group("sensor"),
                "scene": match.group("scene"),
                "data_round": match.group("round"),
                "sequence_index": match.group("index"),
                "split": split,
                "source": "r2_increment",
            }
        )
    return rows


def split_increment_rows(
    rows: list[dict[str, str]],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> list[dict[str, str]]:
    if val_ratio < 0 or test_ratio < 0 or val_ratio + test_ratio >= 1:
        raise ValueError("增量 val/test 比例必须非负且总和小于 1")
    if val_ratio == 0 and test_ratio == 0:
        return rows
    grouped: defaultdict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["sensor"], row["scene"])].append(row)
    for group_index, key in enumerate(sorted(grouped)):
        group = grouped[key]
        random.Random(seed + group_index).shuffle(group)
        val_count = max(1, round(len(group) * val_ratio)) if val_ratio else 0
        test_count = max(1, round(len(group) * test_ratio)) if test_ratio else 0
        if val_count + test_count >= len(group):
            raise ValueError(f"{key} 子组太小，无法按指定比例留出 val/test")
        for index, row in enumerate(group):
            if index < test_count:
                row["split"] = "test"
            elif index < test_count + val_count:
                row["split"] = "val"
            else:
                row["split"] = "train"
    return rows


def select_replay(rows: list[dict[str, str]], limit: int, seed: int) -> list[dict[str, str]]:
    candidates = [row for row in rows if row["split"] == "train"]
    if limit <= 0 or not candidates:
        return []
    grouped: defaultdict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in candidates:
        grouped[(row.get("sensor", "unknown"), row.get("scene", "unknown"))].append(row)
    for group_index, key in enumerate(sorted(grouped)):
        random.Random(seed + group_index).shuffle(grouped[key])
    selected: list[dict[str, str]] = []
    keys = sorted(grouped)
    while len(selected) < min(limit, len(candidates)):
        progressed = False
        for key in keys:
            if grouped[key] and len(selected) < limit:
                selected.append(grouped[key].pop())
                progressed = True
        if not progressed:
            break
    return selected


def unique_paths(rows: list[dict[str, str]]) -> list[Path]:
    seen: set[Path] = set()
    paths: list[Path] = []
    for row in rows:
        path = Path(row["image_path"]).resolve()
        if path not in seen:
            seen.add(path)
            paths.append(path)
    return paths


def write_manifest(path: Path, images: list[Path]) -> str:
    path.write_text(
        "\n".join(image.as_posix() for image in images) + ("\n" if images else ""),
        encoding="utf-8",
    )
    return str(path.resolve())


def summarize(images: list[Path], class_names: list[str]) -> dict:
    object_counts: Counter[str] = Counter()
    images_by_class: Counter[str] = Counter()
    for image in images:
        present: set[str] = set()
        for box in parse_yolo_boxes(resolve_label_path(image), len(class_names)):
            name = class_names[box.class_id]
            object_counts[name] += 1
            present.add(name)
        images_by_class.update(present)
    return {
        "images": len(images),
        "objects": int(sum(object_counts.values())),
        "objects_by_class": {name: int(object_counts[name]) for name in class_names},
        "images_by_class": {name: int(images_by_class[name]) for name in class_names},
    }


def write_data_yaml(
    path: Path,
    train_manifest: str,
    class_names: list[str],
    val_manifest: str | None,
    test_manifest: str | None,
) -> str:
    payload: dict = {
        "train": train_manifest,
        "nc": len(class_names),
        "names": {index: name for index, name in enumerate(class_names)},
    }
    if val_manifest:
        payload["val"] = val_manifest
    if test_manifest:
        payload["test"] = test_manifest
    path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return str(path.resolve())


def prepare_continual_dataset(
    increment_root: Path,
    output_dir: Path,
    *,
    base_index: Path | None = None,
    increment_val_root: Path | None = None,
    increment_test_root: Path | None = None,
    replay_limit: int = 200,
    increment_val_ratio: float = 0.0,
    increment_test_ratio: float = 0.0,
    seed: int = 42,
    base_classes: list[str] | None = None,
) -> dict:
    base_classes = list(base_classes or BASE_CLASS_NAMES)
    class_names = read_class_names(increment_root / "classes.txt")
    new_classes = validate_taxonomy(class_names, base_classes)
    base_rows = read_base_index(base_index, len(class_names))
    if (increment_val_root or increment_test_root) and (
        increment_val_ratio or increment_test_ratio
    ):
        raise ValueError("独立增量 val/test 目录与 smoke holdout 比例不能同时使用")
    increment_rows = discover_increment_rows(increment_root, len(class_names), "train")
    if increment_val_root:
        val_classes_path = increment_val_root / "classes.txt"
        if val_classes_path.is_file() and read_class_names(val_classes_path) != class_names:
            raise ValueError("增量 val 类别表与 train 不一致")
        increment_rows.extend(
            discover_increment_rows(increment_val_root, len(class_names), "val")
        )
    if increment_test_root:
        test_classes_path = increment_test_root / "classes.txt"
        if test_classes_path.is_file() and read_class_names(test_classes_path) != class_names:
            raise ValueError("增量 test 类别表与 train 不一致")
        increment_rows.extend(
            discover_increment_rows(increment_test_root, len(class_names), "test")
        )
    increment_rows = split_increment_rows(
        increment_rows,
        increment_val_ratio,
        increment_test_ratio,
        seed,
    )
    base_paths = {Path(row["image_path"]).resolve() for row in base_rows}
    increment_paths = {Path(row["image_path"]).resolve() for row in increment_rows}
    overlap = base_paths & increment_paths
    if overlap:
        raise ValueError(f"基础集和增量集包含重复路径: {next(iter(overlap))}")

    replay_rows = select_replay(base_rows, replay_limit, seed)
    increment_train = unique_paths([row for row in increment_rows if row["split"] == "train"])
    replay_train = unique_paths(replay_rows)
    mixed_train = unique_paths(
        [*([row for row in increment_rows if row["split"] == "train"]), *replay_rows]
    )
    val_images = unique_paths(
        [row for row in [*base_rows, *increment_rows] if row["split"] == "val"]
    )
    test_images = unique_paths(
        [row for row in [*base_rows, *increment_rows] if row["split"] == "test"]
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    manifests = {
        "train_increment": write_manifest(output_dir / "train_increment.txt", increment_train),
        "train_replay": write_manifest(output_dir / "train_replay.txt", replay_train),
        "train_mixed": write_manifest(output_dir / "train_mixed.txt", mixed_train),
        "val": write_manifest(output_dir / "val.txt", val_images) if val_images else None,
        "test": write_manifest(output_dir / "test.txt", test_images) if test_images else None,
    }
    yamls = {
        "increment_only": write_data_yaml(
            output_dir / "data_increment_only.yaml",
            manifests["train_increment"],
            class_names,
            manifests["val"],
            manifests["test"],
        ),
        "replay": write_data_yaml(
            output_dir / "data_replay.yaml",
            manifests["train_mixed"],
            class_names,
            manifests["val"],
            manifests["test"],
        ),
    }
    test_stats = summarize(test_images, class_names)
    missing_old_test = [name for name in base_classes if test_stats["objects_by_class"][name] == 0]
    missing_new_test = [name for name in new_classes if test_stats["objects_by_class"][name] == 0]
    report = {
        "protocol_version": "r2-class-increment-v1",
        "class_order": class_names,
        "base_classes": base_classes,
        "new_classes": new_classes,
        "increment_root": str(increment_root.resolve()),
        "increment_val_root": str(increment_val_root.resolve()) if increment_val_root else None,
        "increment_test_root": str(increment_test_root.resolve()) if increment_test_root else None,
        "base_index": str(base_index.resolve()) if base_index else None,
        "seed": seed,
        "replay_limit": replay_limit,
        "smoke_holdout": {
            "increment_val_ratio": increment_val_ratio,
            "increment_test_ratio": increment_test_ratio,
            "official": increment_val_ratio == 0 and increment_test_ratio == 0,
            "warning": (
                "Non-zero ratios split the training injection for local smoke tests only; "
                "do not report those scores as official independent-test metrics."
            ),
        },
        "manifests": manifests,
        "yamls": yamls,
        "statistics": {
            "increment_train": summarize(increment_train, class_names),
            "replay_train": summarize(replay_train, class_names),
            "mixed_train": summarize(mixed_train, class_names),
            "validation": summarize(val_images, class_names),
            "test": test_stats,
        },
        "evaluation_ready": not missing_old_test and not missing_new_test,
        "missing_old_test_classes": missing_old_test,
        "missing_new_test_classes": missing_new_test,
        "privacy": {
            "source_images_copied": False,
            "absolute_paths_are_local_only": True,
            "network_used": False,
            "output_must_remain_git_ignored": True,
        },
        "metrics": {
            "new_map": "mean AP of new classes after update",
            "old_map_before": "mean AP of base classes before update",
            "old_map_after": "mean AP of base classes after update",
            "krr": "old_map_after / old_map_before",
            "all_map": "mean AP of all learned classes after update",
        },
    }
    summary_path = output_dir / "continual_dataset_summary.json"
    summary_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="准备本地 r2 类增量训练与回放清单")
    parser.add_argument("--increment-dataset", type=Path, required=True)
    parser.add_argument("--base-index", type=Path)
    parser.add_argument("--increment-val-dataset", type=Path)
    parser.add_argument("--increment-test-dataset", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--base-classes", default=",".join(BASE_CLASS_NAMES))
    parser.add_argument("--replay-limit", type=int, default=200)
    parser.add_argument("--increment-val-ratio", type=float, default=0.0)
    parser.add_argument("--increment-test-ratio", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = prepare_continual_dataset(
        args.increment_dataset,
        args.output,
        base_index=args.base_index,
        increment_val_root=args.increment_val_dataset,
        increment_test_root=args.increment_test_dataset,
        replay_limit=args.replay_limit,
        increment_val_ratio=args.increment_val_ratio,
        increment_test_ratio=args.increment_test_ratio,
        seed=args.seed,
        base_classes=parse_class_list(args.base_classes),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
