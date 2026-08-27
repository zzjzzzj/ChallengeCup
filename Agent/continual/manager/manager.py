from __future__ import annotations

import csv
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from Agent.common.schemas import ImageRecord, TaskStage
from Agent.common.utils.jsonio import write_json
from Agent.continual.protocols import ContinualProtocol
from Agent.continual.replay import ReplayBuffer, ReplayItem, score_replay_candidate
from Agent.models.prototypes import PrototypeBank


CLASS_ID_TO_NAME = {
    0: "soldier",
    1: "small_aircraft",
    2: "warship",
    3: "tank",
}
CLASS_NAME_TO_ID = {name: class_id for class_id, name in CLASS_ID_TO_NAME.items()}
DEFAULT_CLASS_NAMES = [CLASS_ID_TO_NAME[index] for index in sorted(CLASS_ID_TO_NAME)]


@dataclass
class TrainingPlan:
    task: TaskStage
    current_samples: list[ImageRecord]
    replay_items: list[ReplayItem]
    teacher_checkpoint: Path | None = None
    prototype_path: Path | None = None
    anchor_path: Path | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task.to_dict(),
            "current_sample_count": len(self.current_samples),
            "replay_count": len(self.replay_items),
            "current_samples": [sample.to_dict() for sample in self.current_samples],
            "replay_items": [item.to_dict() for item in self.replay_items],
            "teacher_checkpoint": str(self.teacher_checkpoint) if self.teacher_checkpoint else None,
            "prototype_path": str(self.prototype_path) if self.prototype_path else None,
            "anchor_path": str(self.anchor_path) if self.anchor_path else None,
            "notes": self.notes,
        }

    def summary(self) -> dict[str, Any]:
        """Small terminal-friendly plan summary."""

        current_by_split: dict[str, int] = {}
        current_by_scene: dict[str, int] = {}
        current_by_modality: dict[str, int] = {}
        for sample in self.current_samples:
            current_by_split[sample.split or "unknown"] = current_by_split.get(sample.split or "unknown", 0) + 1
            current_by_scene[sample.scene or "unknown"] = current_by_scene.get(sample.scene or "unknown", 0) + 1
            current_by_modality[sample.modality or "unknown"] = current_by_modality.get(sample.modality or "unknown", 0) + 1
        replay_by_task: dict[str, int] = {}
        for item in self.replay_items:
            replay_by_task[item.task_id] = replay_by_task.get(item.task_id, 0) + 1
        return {
            "task_id": self.task.task_id,
            "task_name": self.task.name,
            "current_sample_count": len(self.current_samples),
            "current_by_split": current_by_split,
            "current_by_scene": current_by_scene,
            "current_by_modality": current_by_modality,
            "replay_count": len(self.replay_items),
            "replay_by_task": replay_by_task,
            "teacher_checkpoint": str(self.teacher_checkpoint) if self.teacher_checkpoint else None,
            "prototype_path": str(self.prototype_path) if self.prototype_path else None,
            "anchor_path": str(self.anchor_path) if self.anchor_path else None,
            "notes": self.notes,
        }


class ContinualLearningManager:
    """Coordinate protocol stages, image replay, teacher versions, and prototypes."""

    def __init__(
        self,
        workspace: Path,
        protocol: ContinualProtocol,
        replay_capacity: int = 200,
    ) -> None:
        self.workspace = workspace
        self.protocol = protocol
        self.replay = ReplayBuffer(workspace / "replay", capacity=replay_capacity)
        self.prototype_bank = PrototypeBank()
        self.version_dir = workspace / "versions"
        self.prototype_path = workspace / "prototypes" / "prototype_bank.json"
        self.anchor_path = workspace / "anchors" / "expert_anchors.json"
        self.workspace.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def parse_label_classes(label_path: Path | None) -> list[str]:
        if label_path is None or not label_path.is_file():
            return []
        classes = []
        for raw in label_path.read_text(encoding="utf-8-sig").splitlines():
            if not raw.strip():
                continue
            class_id = int(raw.split()[0])
            classes.append(CLASS_ID_TO_NAME.get(class_id, f"class_{class_id}"))
        return sorted(set(classes))

    @staticmethod
    def _label_path_for(image_path: Path) -> Path | None:
        candidate = image_path.with_suffix(".txt")
        return candidate if candidate.is_file() else None

    @staticmethod
    def load_scene_index(index_csv: Path) -> list[ImageRecord]:
        with index_csv.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        records = []
        for row in rows:
            image_path = Path(row["image_path"])
            label_path = ContinualLearningManager._label_path_for(image_path)
            records.append(
                ImageRecord(
                    image_path=image_path,
                    label_path=label_path,
                    modality=row.get("sensor"),
                    scene=row.get("scene"),
                    split=row.get("split"),
                    metadata={key: value for key, value in row.items() if key not in {"image_path", "sensor", "scene", "split"}},
                )
            )
        return records

    def filter_records(self, records: list[ImageRecord], task: TaskStage) -> list[ImageRecord]:
        result = []
        for record in records:
            classes = self.parse_label_classes(record.label_path)
            if task.modalities and record.modality not in task.modalities:
                continue
            if task.scenes and record.scene not in task.scenes:
                continue
            if task.classes and not any(class_name in task.classes for class_name in classes):
                continue
            result.append(
                ImageRecord(
                    image_path=record.image_path,
                    label_path=record.label_path,
                    modality=record.modality,
                    scene=record.scene,
                    split=record.split,
                    task_id=task.task_id,
                    metadata={**record.metadata, "classes": classes},
                )
            )
        return result

    def build_training_plan(
        self,
        index_csv: Path,
        task_id: str,
        replay_limit: int = 64,
    ) -> TrainingPlan:
        records = self.load_scene_index(index_csv)
        task = self.protocol.get_stage(task_id)
        current = self.filter_records(records, task)
        previous = self.protocol.previous_stages(task_id)
        teacher_checkpoint = self.teacher_checkpoint_for(previous[-1].task_id) if previous else None
        replay_items = self.replay.select(replay_limit)
        notes = [
            "Use current task images plus selected replay images.",
            "Keep old task images because replay is allowed.",
            "Apply KD/feature distillation when teacher_checkpoint is available.",
        ]
        return TrainingPlan(
            task=task,
            current_samples=current,
            replay_items=replay_items,
            teacher_checkpoint=teacher_checkpoint,
            prototype_path=self.prototype_path if self.prototype_path.is_file() else None,
            anchor_path=self.anchor_path if self.anchor_path.is_file() else None,
            notes=notes,
        )

    def update_replay_from_records(
        self,
        records: list[ImageRecord],
        task_id: str,
        copy_files: bool = False,
    ) -> dict[str, Any]:
        class_counts: dict[str, int] = {}
        modality_counts: dict[str, int] = {}
        scene_counts: dict[str, int] = {}
        for item in self.replay.load():
            for class_name in item.classes:
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
            if item.modality:
                modality_counts[item.modality] = modality_counts.get(item.modality, 0) + 1
            if item.scene:
                scene_counts[item.scene] = scene_counts.get(item.scene, 0) + 1

        for record in records:
            classes = self.parse_label_classes(record.label_path)
            min_box_area = _min_box_area(record.label_path)
            score, reason = score_replay_candidate(
                classes=classes,
                modality=record.modality,
                scene=record.scene,
                uncertainty=float(record.metadata.get("uncertainty", 0.0) or 0.0),
                min_box_area=min_box_area,
                class_counts=class_counts,
                modality_counts=modality_counts,
                scene_counts=scene_counts,
            )
            item = ReplayItem(
                image_path=record.image_path,
                label_path=record.label_path,
                task_id=task_id,
                modality=record.modality,
                scene=record.scene,
                classes=classes,
                score=score,
                reason=reason,
                metadata=record.metadata,
            )
            self.replay.add(item, copy_files=copy_files)
        return self.replay.summary()

    def save_training_plan(self, plan: TrainingPlan, path: Path) -> None:
        write_json(path, plan.to_dict())

    def export_training_assets(
        self,
        plan: TrainingPlan,
        output_dir: Path,
        *,
        include_replay: bool = True,
        class_names: list[str] | None = None,
        copy_images: bool = True,
    ) -> dict[str, Any]:
        """Export manifests, filtered labels, and YOLO data.yaml for a task.

        The generated dataset is intentionally materialized into task-specific
        folders. This lets class-incremental stages filter labels to currently
        known classes without changing the original dataset.
        """

        class_names = class_names or DEFAULT_CLASS_NAMES
        allowed_class_names = set(plan.task.classes or class_names)
        output_dir.mkdir(parents=True, exist_ok=True)
        images_dir = output_dir / "images"
        labels_dir = output_dir / "labels"
        manifests_dir = output_dir / "manifests"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        manifests_dir.mkdir(parents=True, exist_ok=True)

        rows: list[tuple[str, Path, Path, str, str]] = []
        for sample in plan.current_samples:
            rows.append(("current", sample.image_path, sample.label_path or sample.image_path.with_suffix(".txt"), sample.split or "train", sample.scene or "unknown"))
        if include_replay:
            for item in plan.replay_items:
                rows.append(("replay", item.image_path, item.label_path or item.image_path.with_suffix(".txt"), "train", item.scene or "unknown"))

        manifest_paths: dict[str, Path] = {}
        exported_counts = {"train": 0, "val": 0, "test": 0}
        filtered_label_count = 0
        for split in ("train", "val", "test"):
            split_rows = [row for row in rows if row[3] == split]
            manifest_path = manifests_dir / f"{split}.txt"
            manifest_paths[split] = manifest_path
            manifest_lines = []
            for row_index, (source, image_path, label_path, _split, _scene) in enumerate(split_rows, start=1):
                stem = f"{source}_{image_path.stem}_{row_index:05d}"
                target_image = images_dir / f"{stem}{image_path.suffix.lower()}"
                target_label = labels_dir / f"{stem}.txt"
                if copy_images:
                    shutil.copy2(image_path, target_image)
                    kept = _write_filtered_label(
                        label_path=label_path,
                        output_label=target_label,
                        allowed_class_names=allowed_class_names,
                    )
                else:
                    # Do not modify original labels. This mode is only for a
                    # quick manifest preview, not class-incremental training.
                    target_image = image_path.resolve()
                    kept = _count_allowed_labels(label_path, allowed_class_names)
                filtered_label_count += kept
                manifest_lines.append(target_image.as_posix())
                exported_counts[split] += 1
            manifest_path.write_text("\n".join(manifest_lines) + ("\n" if manifest_lines else ""), encoding="utf-8")

        data_yaml = output_dir / "data.yaml"
        config = {
            "train": manifest_paths["train"].resolve().as_posix(),
            "val": manifest_paths["val"].resolve().as_posix(),
            "test": manifest_paths["test"].resolve().as_posix(),
            "nc": len(class_names),
            "names": {index: name for index, name in enumerate(class_names)},
        }
        data_yaml.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
        summary = {
            "task_id": plan.task.task_id,
            "output_dir": str(output_dir),
            "data_yaml": str(data_yaml),
            "manifests": {split: str(path) for split, path in manifest_paths.items()},
            "image_copy_enabled": copy_images,
            "class_names": class_names,
            "allowed_label_classes": sorted(allowed_class_names),
            "exported_counts": exported_counts,
            "kept_label_count": filtered_label_count,
            "warning": "copy_images=False is manifest-only; original labels are not filtered." if not copy_images else "",
        }
        write_json(output_dir / "asset_summary.json", summary)
        return summary

    def teacher_checkpoint_for(self, task_id: str) -> Path | None:
        candidate = self.version_dir / task_id / "teacher.pt"
        return candidate if candidate.is_file() else None


def _min_box_area(label_path: Path | None) -> float | None:
    if label_path is None or not label_path.is_file():
        return None
    values = []
    for raw in label_path.read_text(encoding="utf-8-sig").splitlines():
        if not raw.strip():
            continue
        parts = raw.split()
        if len(parts) >= 5:
            values.append(float(parts[3]) * float(parts[4]))
    return min(values) if values else None


def _write_filtered_label(label_path: Path, output_label: Path, allowed_class_names: set[str]) -> int:
    if not label_path.is_file():
        output_label.parent.mkdir(parents=True, exist_ok=True)
        output_label.write_text("", encoding="utf-8")
        return 0
    kept_lines = []
    for raw in label_path.read_text(encoding="utf-8-sig").splitlines():
        if not raw.strip():
            continue
        parts = raw.split()
        class_id = int(parts[0])
        class_name = CLASS_ID_TO_NAME.get(class_id)
        if class_name in allowed_class_names:
            kept_lines.append(" ".join(parts[:5]))
    output_label.parent.mkdir(parents=True, exist_ok=True)
    output_label.write_text("\n".join(kept_lines) + ("\n" if kept_lines else ""), encoding="utf-8")
    return len(kept_lines)


def _count_allowed_labels(label_path: Path, allowed_class_names: set[str]) -> int:
    if not label_path.is_file():
        return 0
    count = 0
    for raw in label_path.read_text(encoding="utf-8-sig").splitlines():
        if not raw.strip():
            continue
        class_id = int(raw.split()[0])
        if CLASS_ID_TO_NAME.get(class_id) in allowed_class_names:
            count += 1
    return count
