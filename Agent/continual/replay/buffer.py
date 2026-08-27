from __future__ import annotations

import csv
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from Agent.common.schemas.core import to_jsonable
from Agent.common.utils.jsonio import read_jsonl, write_json


@dataclass
class ReplayItem:
    """One old-task image retained for replay."""

    image_path: Path
    label_path: Path | None
    task_id: str
    modality: str | None = None
    scene: str | None = None
    classes: list[str] = field(default_factory=list)
    score: float = 0.0
    reason: dict[str, float] = field(default_factory=dict)
    added_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(asdict(self))

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ReplayItem":
        return cls(
            image_path=Path(payload["image_path"]),
            label_path=Path(payload["label_path"]) if payload.get("label_path") else None,
            task_id=payload["task_id"],
            modality=payload.get("modality"),
            scene=payload.get("scene"),
            classes=list(payload.get("classes", [])),
            score=float(payload.get("score", 0.0)),
            reason=dict(payload.get("reason", {})),
            added_at=payload.get("added_at", datetime.now().isoformat(timespec="seconds")),
            metadata=dict(payload.get("metadata", {})),
        )


def score_replay_candidate(
    *,
    classes: list[str],
    modality: str | None,
    scene: str | None,
    uncertainty: float = 0.0,
    min_box_area: float | None = None,
    class_counts: dict[str, int] | None = None,
    modality_counts: dict[str, int] | None = None,
    scene_counts: dict[str, int] | None = None,
) -> tuple[float, dict[str, float]]:
    """Compute replay value from rarity, uncertainty, hardness, and coverage."""

    class_counts = class_counts or {}
    modality_counts = modality_counts or {}
    scene_counts = scene_counts or {}
    rare_class = max((1.0 / (1.0 + class_counts.get(name, 0)) for name in classes), default=0.0)
    rare_modality = 1.0 / (1.0 + modality_counts.get(modality or "", 0))
    rare_scene = 1.0 / (1.0 + scene_counts.get(scene or "", 0))
    rare = 0.60 * rare_class + 0.20 * rare_modality + 0.20 * rare_scene

    hard = 0.0
    if "soldier" in classes:
        hard += 0.55
    if "tank" in classes:
        hard += 0.20
    if modality == "sar":
        hard += 0.15
    if scene in {"urban", "forest"}:
        hard += 0.15
    if min_box_area is not None and min_box_area < 0.001:
        hard += 0.25
    hard = min(hard, 1.0)

    coverage = 0.5 * rare_modality + 0.5 * rare_scene
    components = {
        "rare": rare,
        "uncertain": max(0.0, min(1.0, uncertainty)),
        "hard": hard,
        "coverage": coverage,
    }
    score = (
        0.30 * components["rare"]
        + 0.25 * components["uncertain"]
        + 0.30 * components["hard"]
        + 0.15 * components["coverage"]
    )
    return float(score), components


class ReplayBuffer:
    """Image-level replay buffer with optional materialized copies."""

    def __init__(self, root: Path, capacity: int = 200) -> None:
        self.root = root
        self.capacity = capacity
        self.index_path = root / "replay_index.jsonl"
        self.images_dir = root / "images"
        self.labels_dir = root / "labels"
        self.root.mkdir(parents=True, exist_ok=True)

    def load(self) -> list[ReplayItem]:
        return [ReplayItem.from_dict(row) for row in read_jsonl(self.index_path)]

    def save(self, items: list[ReplayItem]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        text = "\n".join(
            __import__("json").dumps(item.to_dict(), ensure_ascii=False) for item in items
        )
        self.index_path.write_text(text + ("\n" if text else ""), encoding="utf-8")

    def add(self, item: ReplayItem, copy_files: bool = False) -> ReplayItem:
        if copy_files:
            item = self._materialize(item)
        items = [old for old in self.load() if old.image_path != item.image_path]
        items.append(item)
        items.sort(key=lambda current: current.score, reverse=True)
        self.save(items[: self.capacity])
        return item

    def select(self, limit: int, task_id: str | None = None) -> list[ReplayItem]:
        items = self.load()
        if task_id is not None:
            items = [item for item in items if item.task_id == task_id]
        return sorted(items, key=lambda item: item.score, reverse=True)[:limit]

    def summary(self) -> dict[str, Any]:
        items = self.load()
        by_task: dict[str, int] = {}
        by_class: dict[str, int] = {}
        for item in items:
            by_task[item.task_id] = by_task.get(item.task_id, 0) + 1
            for class_name in item.classes:
                by_class[class_name] = by_class.get(class_name, 0) + 1
        return {
            "root": str(self.root),
            "capacity": self.capacity,
            "size": len(items),
            "by_task": by_task,
            "by_class": by_class,
        }

    def export_manifest(self, path: Path, limit: int | None = None) -> Path:
        items = self.select(limit or self.capacity)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["image_path", "label_path", "task_id", "modality", "scene", "classes", "score"],
            )
            writer.writeheader()
            for item in items:
                writer.writerow(
                    {
                        "image_path": item.image_path,
                        "label_path": item.label_path or "",
                        "task_id": item.task_id,
                        "modality": item.modality or "",
                        "scene": item.scene or "",
                        "classes": ",".join(item.classes),
                        "score": round(item.score, 6),
                    }
                )
        return path

    def _materialize(self, item: ReplayItem) -> ReplayItem:
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)
        image_dest = self.images_dir / item.image_path.name
        shutil.copy2(item.image_path, image_dest)
        label_dest = None
        if item.label_path and item.label_path.is_file():
            label_dest = self.labels_dir / item.label_path.name
            shutil.copy2(item.label_path, label_dest)
        return ReplayItem(
            image_path=image_dest,
            label_path=label_dest,
            task_id=item.task_id,
            modality=item.modality,
            scene=item.scene,
            classes=item.classes,
            score=item.score,
            reason=item.reason,
            added_at=item.added_at,
            metadata={**item.metadata, "source_image": str(item.image_path), "source_label": str(item.label_path or "")},
        )

    def write_summary(self, path: Path) -> None:
        write_json(path, self.summary())
