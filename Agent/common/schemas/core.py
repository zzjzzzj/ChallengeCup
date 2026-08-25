from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


CLASS_NAMES = ("soldier", "small_aircraft", "warship", "tank")
MODALITY_NAMES = ("ir", "sar")
SCENE_NAMES = ("air", "sea", "urban", "forest")


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:  # noqa: BLE001
            pass
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


@dataclass(frozen=True)
class ProbabilityVector:
    """One probability distribution and its winning label."""

    label: str
    confidence: float
    probabilities: dict[str, float]

    @classmethod
    def from_scores(cls, scores: dict[str, float]) -> "ProbabilityVector":
        total = sum(max(float(value), 0.0) for value in scores.values())
        if total <= 0:
            probability = 1.0 / max(1, len(scores))
            probabilities = {key: probability for key in scores}
        else:
            probabilities = {key: max(float(value), 0.0) / total for key, value in scores.items()}
        label = max(probabilities, key=probabilities.get)
        return cls(label=label, confidence=float(probabilities[label]), probabilities=probabilities)

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(asdict(self))


@dataclass
class DetectionPrediction:
    """Unified detection result used by first-pass and second-pass inference."""

    class_name: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]
    expert_ids: list[int] = field(default_factory=list)
    pass_id: int = 1
    source: str = "model"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox_xyxy
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["area"] = self.area
        data["class"] = data.pop("class_name")
        data["pass"] = data.pop("pass_id")
        return to_jsonable(data)


@dataclass(frozen=True)
class ImageRecord:
    """Dataset item used by replay, task protocols, and batch inference."""

    image_path: Path
    label_path: Path | None = None
    modality: str | None = None
    scene: str | None = None
    split: str | None = None
    task_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(asdict(self))


@dataclass(frozen=True)
class TaskStage:
    """One continual-learning stage.

    The stage can describe modality incremental, scene incremental, class
    incremental, or any mixture of the three. Users can edit JSON protocols
    without changing training code.
    """

    task_id: str
    name: str
    modalities: list[str] = field(default_factory=list)
    scenes: list[str] = field(default_factory=list)
    classes: list[str] = field(default_factory=list)
    train_filter: dict[str, list[str]] = field(default_factory=dict)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(asdict(self))
