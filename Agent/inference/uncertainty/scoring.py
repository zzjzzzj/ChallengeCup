from __future__ import annotations

import math

import numpy as np


def _entropy(probabilities: np.ndarray) -> float:
    probabilities = np.asarray(probabilities, dtype=np.float32)
    probabilities = probabilities / max(float(probabilities.sum()), 1e-12)
    valid = probabilities[probabilities > 0]
    if len(valid) == 0:
        return 0.0
    return float(-(valid * np.log(valid)).sum() / math.log(max(len(probabilities), 2)))


def uncertainty_score(
    class_probabilities: np.ndarray,
    *,
    expert_disagreement: float = 0.0,
    scene_disagreement: float = 0.0,
    weights: tuple[float, float, float, float] = (0.35, 0.25, 0.20, 0.20),
) -> dict[str, float]:
    """Compute U(x) from entropy, p_max, expert and scene disagreement."""

    probabilities = np.asarray(class_probabilities, dtype=np.float32).ravel()
    if probabilities.size == 0:
        probabilities = np.asarray([1.0], dtype=np.float32)
    probabilities = probabilities / max(float(probabilities.sum()), 1e-12)
    entropy = _entropy(probabilities)
    one_minus_pmax = 1.0 - float(probabilities.max())
    expert_disagreement = max(0.0, min(1.0, float(expert_disagreement)))
    scene_disagreement = max(0.0, min(1.0, float(scene_disagreement)))
    score = (
        weights[0] * entropy
        + weights[1] * one_minus_pmax
        + weights[2] * expert_disagreement
        + weights[3] * scene_disagreement
    )
    return {
        "uncertainty": float(score),
        "entropy": entropy,
        "one_minus_pmax": one_minus_pmax,
        "expert_disagreement": expert_disagreement,
        "scene_disagreement": scene_disagreement,
    }
