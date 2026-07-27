from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import yaml

from detector_module import CLASS_NAMES
from detector_module.boxes import YoloBox, parse_yolo_boxes
from detector_module.create_incremental_protocol import build_protocol, parse_class_list
from detector_module.dataset import DetectionSample, read_scene_index


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INDEX = PROJECT_ROOT / "scene_module" / "artifacts" / "scene_index.csv"
DEFAULT_PROTOCOL = PROJECT_ROOT / "detector_module" / "configs" / "incremental_protocol.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "detector_module" / "artifacts" / "incremental_dataset"


@dataclass(frozen=True)
class StageView:
    role: str
    source_split: str
    selector_classes: list[str]
    label_classes: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build materialized YOLO datasets for incremental learning.")
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--base", default=None, help="Optional comma-separated base classes.")
    parser.add_argument(
        "--round",
        action="append",
        dest="rounds",
        default=None,
        help="Optional comma-separated incremental classes. Repeat for multiple rounds.",
    )
    return parser.parse_args()


def load_protocol(protocol_path: Path, base: str | None, rounds: list[str] | None) -> dict:
    if base is not None or rounds is not None:
        if base is None or rounds is None:
            raise ValueError("--base and --round must be provided together when overriding protocol.")
        return build_protocol(parse_class_list(base), [parse_class_list(value) for value in rounds])
    if not protocol_path.is_file():
        raise FileNotFoundError(protocol_path)
    return json.loads(protocol_path.read_text(encoding="utf-8"))


def remap_boxes(boxes: list[YoloBox], label_classes: list[str]) -> list[YoloBox]:
    class_to_stage_id = {name: index for index, name in enumerate(label_classes)}
    remapped = []
    for box in boxes:
        class_name = CLASS_NAMES[box.class_id]
        if class_name in class_to_stage_id:
            remapped.append(
                YoloBox(
                    class_to_stage_id[class_name],
                    box.x_center,
                    box.y_center,
                    box.width,
                    box.height,
                    box.confidence,
                )
            )
    return remapped


def has_any_class(boxes: list[YoloBox], class_names: list[str]) -> bool:
    selected = set(class_names)
    return any(CLASS_NAMES[box.class_id] in selected for box in boxes)


def write_label(path: Path, boxes: list[YoloBox]) -> None:
    lines = [
        f"{box.class_id} {box.x_center:.8f} {box.y_center:.8f} {box.width:.8f} {box.height:.8f}"
        for box in boxes
    ]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def materialize_view(
    samples: list[DetectionSample],
    boxes_by_image: dict[Path, list[YoloBox]],
    view_dir: Path,
    view: StageView,
) -> dict:
    images_dir = view_dir / "images"
    labels_dir = view_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    object_counts = {name: 0 for name in view.label_classes}
    selected_samples = 0

    for sample in samples:
        if sample.split != view.source_split:
            continue
        original_boxes = boxes_by_image[sample.image_path]
        if not has_any_class(original_boxes, view.selector_classes):
            continue
        remapped = remap_boxes(original_boxes, view.label_classes)
        if not remapped:
            continue

        image_name = sample.image_path.name
        target_image = images_dir / image_name
        if not target_image.exists():
            shutil.copy2(sample.image_path, target_image)
        write_label(labels_dir / f"{target_image.stem}.txt", remapped)
        manifest.append(target_image.resolve().as_posix())
        selected_samples += 1
        for box in remapped:
            object_counts[view.label_classes[box.class_id]] += 1

    manifest_path = view_dir / "images.txt"
    manifest_path.write_text("\n".join(manifest) + ("\n" if manifest else ""), encoding="utf-8")
    return {
        "role": view.role,
        "source_split": view.source_split,
        "selector_classes": view.selector_classes,
        "label_classes": view.label_classes,
        "images": selected_samples,
        "objects": int(sum(object_counts.values())),
        "objects_by_class": object_counts,
        "manifest": str(manifest_path.resolve()),
    }


def stage_views(stage: dict) -> list[StageView]:
    new_classes = list(stage["new_classes"])
    old_classes = list(stage["old_classes"])
    learned_classes = list(stage["all_learned_classes"])
    views = [
        StageView("train_new", "train", new_classes, learned_classes),
        StageView("train_replay", "train", learned_classes, learned_classes),
        StageView("val_all", "val", learned_classes, learned_classes),
        StageView("test_all", "test", learned_classes, learned_classes),
        StageView("test_new", "test", new_classes, learned_classes),
    ]
    if old_classes:
        views.append(StageView("test_old", "test", old_classes, learned_classes))
    return views


def write_stage_yaml(stage_dir: Path, role_stats: dict[str, dict], stage: dict) -> dict[str, str]:
    names = {index: name for index, name in enumerate(stage["all_learned_classes"])}
    yaml_paths = {}
    for train_role in ("train_new", "train_replay"):
        config = {
            "train": role_stats[train_role]["manifest"],
            "val": role_stats["val_all"]["manifest"],
            "test": role_stats["test_all"]["manifest"],
            "nc": len(names),
            "names": names,
        }
        config_path = (stage_dir / f"{train_role}.yaml").resolve()
        config_path.write_text(
            yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
        )
        yaml_paths[train_role] = str(config_path)
    return yaml_paths


def build_incremental_dataset(index_csv: Path, protocol: dict, output_dir: Path) -> dict:
    samples = read_scene_index(index_csv)
    boxes_by_image = {
        sample.image_path: parse_yolo_boxes(sample.label_path, len(CLASS_NAMES)) for sample in samples
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    stages = []
    for stage in protocol["stages"]:
        stage_dir = output_dir / f"stage_{stage['stage']}_{stage['name']}"
        role_stats = {}
        for view in stage_views(stage):
            role_dir = stage_dir / view.role
            role_stats[view.role] = materialize_view(samples, boxes_by_image, role_dir, view)
        yaml_paths = write_stage_yaml(stage_dir, role_stats, stage)
        stages.append(
            {
                "stage": stage["stage"],
                "name": stage["name"],
                "new_classes": stage["new_classes"],
                "old_classes": stage["old_classes"],
                "all_learned_classes": stage["all_learned_classes"],
                "roles": role_stats,
                "yamls": yaml_paths,
            }
        )
    report = {
        "index": str(index_csv.resolve()),
        "output_dir": str(output_dir.resolve()),
        "class_order": CLASS_NAMES,
        "stages": stages,
        "notes": [
            "train_new excludes old-class-only images; train_replay includes all learned-class images.",
            "Labels are filtered and class ids are remapped per stage to the learned class list.",
            "Official incremental rules may restrict replay; confirm before reporting replay results.",
        ],
    }
    (output_dir / "incremental_dataset_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    args = parse_args()
    protocol = load_protocol(args.protocol, args.base, args.rounds)
    report = build_incremental_dataset(args.index, protocol, args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
