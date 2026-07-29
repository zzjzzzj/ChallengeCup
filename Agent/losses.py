from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .schemas import DetectionBox, ProbabilityResult


@dataclass
class LossWeights:
    modality: float = 0.30
    scene: float = 0.60
    box: float = 1.00
    cls: float = 1.00
    detail: float = 0.20
    proto: float = 0.40


def estimate_runtime_losses(
    modality: ProbabilityResult,
    scene: ProbabilityResult,
    detections: list[DetectionBox],
    consistency: dict[str, Any],
    environment: dict[str, Any],
    weights: LossWeights | None = None,
) -> dict[str, Any]:
    weights = weights or LossWeights()
    if detections:
        mean_det_conf = sum(box.confidence for box in detections) / len(detections)
        mean_cls_conf = sum(box.confidence for box in detections if box.class_name != "unknown") / max(
            1, len([box for box in detections if box.class_name != "unknown"])
        )
    else:
        mean_det_conf = 0.0
        mean_cls_conf = 0.0

    noise_penalty = 0.10 if environment.get("noise_level") == "high" else 0.0
    clarity_penalty = 0.10 if environment.get("clarity_level") == "low" else 0.0
    invalid_rate = consistency.get("final_invalid_count", 0) / max(1, len(detections))
    components = {
        "L_moti": round(1.0 - modality.confidence, 6),
        "L_env": round(1.0 - scene.confidence, 6),
        "L_box": round(1.0 - mean_det_conf if detections else 0.0, 6),
        "L_cls": round(1.0 - mean_cls_conf if detections else 0.0, 6),
        "L_detail": round(noise_penalty + clarity_penalty, 6),
        "L_proto": round(float(invalid_rate), 6),
    }
    total = (
        weights.modality * components["L_moti"]
        + weights.scene * components["L_env"]
        + weights.box * components["L_box"]
        + weights.cls * components["L_cls"]
        + weights.detail * components["L_detail"]
        + weights.proto * components["L_proto"]
    )
    return {
        "type": "runtime_proxy",
        "formula": "L = w_moti*L_moti + w_env*L_env + w_box*L_box + w_cls*L_cls + w_detail*L_detail + w_proto*L_proto",
        "components": components,
        "weights": asdict(weights),
        "total": round(total, 6),
        "note": "Runtime losses are confidence/consistency proxies; use real training losses during model updates.",
    }


def combine_training_losses(
    *,
    l_box: float,
    l_cls: float,
    l_dfl: float = 0.0,
    l_detail: float = 0.0,
    l_scene: float = 0.0,
    l_proto: float = 0.0,
    l_moti: float = 0.0,
    lambda_detail: float = 0.2,
    lambda_scene: float = 0.6,
    lambda_proto: float = 0.4,
    lambda_moti: float = 0.3,
) -> dict[str, float]:
    total = (
        l_box
        + l_cls
        + l_dfl
        + lambda_detail * l_detail
        + lambda_scene * l_scene
        + lambda_proto * l_proto
        + lambda_moti * l_moti
    )
    return {
        "L_box": float(l_box),
        "L_cls": float(l_cls),
        "L_dfl": float(l_dfl),
        "L_detail": float(l_detail),
        "L_scene": float(l_scene),
        "L_proto": float(l_proto),
        "L_moti": float(l_moti),
        "lambda_detail": float(lambda_detail),
        "lambda_scene": float(lambda_scene),
        "lambda_proto": float(lambda_proto),
        "lambda_moti": float(lambda_moti),
        "L_total": round(float(total), 8),
    }
