from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from Agent.common.schemas import DetectionPrediction


@dataclass(frozen=True)
class CascadeDecision:
    triggered: bool
    reasons: list[str] = field(default_factory=list)
    uncertainty: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "triggered": self.triggered,
            "reasons": self.reasons,
            "uncertainty": self.uncertainty,
        }


def should_trigger_second_evaluation(
    detections: list[DetectionPrediction],
    uncertainty: float,
    *,
    threshold: float = 0.55,
    soldier_conf_threshold: float = 0.45,
    tiny_area_threshold: float = 0.001,
    scene_conflict: bool = False,
) -> CascadeDecision:
    """Decide whether to launch expensive slicing/high-resolution inference."""

    reasons: list[str] = []
    if uncertainty > threshold:
        reasons.append("uncertainty_above_threshold")
    if not detections:
        reasons.append("no_detection")
    soldier_boxes = [box for box in detections if box.class_name == "soldier"]
    if soldier_boxes and all(box.confidence < soldier_conf_threshold for box in soldier_boxes):
        reasons.append("only_low_confidence_soldier")
    if any(box.class_name == "soldier" and box.area < tiny_area_threshold for box in detections):
        reasons.append("tiny_soldier")
    if scene_conflict:
        reasons.append("scene_detection_conflict")
    return CascadeDecision(triggered=bool(reasons), reasons=reasons, uncertainty=float(uncertainty))
