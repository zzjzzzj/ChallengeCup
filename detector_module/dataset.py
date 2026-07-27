from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml

from detector_module import CLASS_NAMES


VALID_SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class DetectionSample:
    image_path: Path
    image_name: str
    sensor: str
    scene: str
    split: str
    sequence_index: int

    @property
    def label_path(self) -> Path:
        return self.image_path.with_suffix(".txt")


def read_scene_index(index_csv: Path) -> list[DetectionSample]:
    with index_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"场景索引为空: {index_csv}")

    samples = []
    for row_number, row in enumerate(rows, start=2):
        missing = [
            name
            for name in ("image_path", "image_name", "sensor", "scene", "split", "sequence_index")
            if not row.get(name)
        ]
        if missing:
            raise ValueError(f"索引第 {row_number} 行缺少字段: {', '.join(missing)}")
        if row["split"] not in VALID_SPLITS:
            raise ValueError(f"索引第 {row_number} 行包含未知划分: {row['split']}")
        samples.append(
            DetectionSample(
                image_path=Path(row["image_path"]).resolve(),
                image_name=row["image_name"],
                sensor=row["sensor"],
                scene=row["scene"],
                split=row["split"],
                sequence_index=int(row["sequence_index"]),
            )
        )
    return samples


def parse_yolo_label(label_path: Path, class_count: int) -> list[int]:
    if not label_path.is_file():
        raise FileNotFoundError(f"标签文件不存在: {label_path}")

    class_ids = []
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"{label_path}:{line_number} 应包含5列，实际为 {len(parts)} 列")
        try:
            class_id = int(parts[0])
            coordinates = [float(value) for value in parts[1:]]
        except ValueError as exc:
            raise ValueError(f"{label_path}:{line_number} 包含非数值字段") from exc
        if not 0 <= class_id < class_count:
            raise ValueError(f"{label_path}:{line_number} 类别编号越界: {class_id}")
        x_center, y_center, width, height = coordinates
        if not (0 <= x_center <= 1 and 0 <= y_center <= 1):
            raise ValueError(f"{label_path}:{line_number} 中心坐标不在 [0,1] 范围内")
        if not (0 < width <= 1 and 0 < height <= 1):
            raise ValueError(f"{label_path}:{line_number} 宽高不在 (0,1] 范围内")
        class_ids.append(class_id)
    return class_ids


def validate_samples(
    samples: Iterable[DetectionSample], class_names: list[str] | None = None
) -> dict:
    class_names = class_names or CLASS_NAMES
    samples = list(samples)
    duplicate_paths = [
        path for path, count in Counter(sample.image_path for sample in samples).items() if count > 1
    ]
    if duplicate_paths:
        raise ValueError(f"索引包含重复图片: {duplicate_paths[0]}")

    image_counts: Counter = Counter()
    object_counts: Counter = Counter()
    for sample in samples:
        if not sample.image_path.is_file():
            raise FileNotFoundError(f"图片不存在: {sample.image_path}")
        class_ids = parse_yolo_label(sample.label_path, len(class_names))
        image_counts[(sample.split, sample.sensor, sample.scene)] += 1
        for class_id in class_ids:
            object_counts[(sample.split, class_names[class_id])] += 1

    for split in VALID_SPLITS:
        if not any(sample.split == split for sample in samples):
            raise ValueError(f"数据划分缺少 {split} 样本")
        missing_classes = [name for name in class_names if object_counts[(split, name)] == 0]
        if missing_classes:
            raise ValueError(f"{split} 划分缺少目标类别: {', '.join(missing_classes)}")

    return {
        "image_count": len(samples),
        "object_count": int(sum(object_counts.values())),
        "splits": {
            split: {
                "images": sum(count for (part, _sensor, _scene), count in image_counts.items() if part == split),
                "objects": int(sum(object_counts[(split, name)] for name in class_names)),
                "objects_by_class": {name: int(object_counts[(split, name)]) for name in class_names},
                "images_by_sensor": {
                    sensor: int(
                        sum(
                            count
                            for (part, current_sensor, _scene), count in image_counts.items()
                            if part == split and current_sensor == sensor
                        )
                    )
                    for sensor in sorted({sample.sensor for sample in samples})
                },
                "images_by_scene": {
                    scene: int(
                        sum(
                            count
                            for (part, _sensor, current_scene), count in image_counts.items()
                            if part == split and current_scene == scene
                        )
                    )
                    for scene in sorted({sample.scene for sample in samples})
                },
            }
            for split in VALID_SPLITS
        },
        "class_names": class_names,
        "split_policy": "复用场景模块按传感器×场景×序列隔离的 train/val/test 划分",
    }


def prepare_detection_dataset(
    index_csv: Path,
    output_dir: Path,
    class_names: list[str] | None = None,
) -> dict:
    class_names = class_names or CLASS_NAMES
    samples = read_scene_index(index_csv)
    stats = validate_samples(samples, class_names)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir = output_dir / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)

    manifest_paths = {}
    for split in VALID_SPLITS:
        manifest_path = (manifests_dir / f"{split}.txt").resolve()
        paths = [sample.image_path.as_posix() for sample in samples if sample.split == split]
        manifest_path.write_text("\n".join(paths) + "\n", encoding="utf-8")
        manifest_paths[split] = manifest_path

    dataset_yaml = (output_dir / "dataset.yaml").resolve()
    dataset_config = {
        "train": manifest_paths["train"].as_posix(),
        "val": manifest_paths["val"].as_posix(),
        "test": manifest_paths["test"].as_posix(),
        "nc": len(class_names),
        "names": {index: name for index, name in enumerate(class_names)},
    }
    dataset_yaml.write_text(
        yaml.safe_dump(dataset_config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    stats.update(
        {
            "index_csv": str(index_csv.resolve()),
            "dataset_yaml": str(dataset_yaml),
            "manifests": {name: str(path) for name, path in manifest_paths.items()},
        }
    )
    stats_path = output_dir / "dataset_stats.json"
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return stats
