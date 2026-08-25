from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def default_scene_projection(scene_names: list[str], class_names: list[str]) -> np.ndarray:
    """Small, soft scene-to-class projection matrix W_s.

    Values are intentionally weak priors, not hard rules. They nudge logits
    without overwriting detector evidence.
    """

    priors = {
        "air": {"small_aircraft": 1.0},
        "sea": {"warship": 1.0},
        "urban": {"soldier": 0.7, "tank": 0.8},
        "forest": {"soldier": 0.8, "tank": 0.7},
    }
    matrix = np.zeros((len(class_names), len(scene_names)), dtype=np.float32)
    for scene_idx, scene in enumerate(scene_names):
        for class_idx, class_name in enumerate(class_names):
            matrix[class_idx, scene_idx] = priors.get(scene, {}).get(class_name, 0.0)
    return matrix


def default_modality_projection(modality_names: list[str], class_names: list[str]) -> np.ndarray:
    """Small modality-to-class projection matrix W_m."""

    priors = {
        "ir": {"soldier": 0.5, "small_aircraft": 0.4, "tank": 0.4},
        "sar": {"warship": 0.5, "tank": 0.4, "soldier": 0.2},
    }
    matrix = np.zeros((len(class_names), len(modality_names)), dtype=np.float32)
    for modality_idx, modality in enumerate(modality_names):
        for class_idx, class_name in enumerate(class_names):
            matrix[class_idx, modality_idx] = priors.get(modality, {}).get(class_name, 0.0)
    return matrix


@dataclass
class SoftContextFusion:
    """Apply z_final = z_det + alpha W_s p_scene + beta W_m p_modality."""

    class_names: list[str]
    scene_names: list[str]
    modality_names: list[str]
    alpha: float = 0.20
    beta: float = 0.10
    scene_projection: np.ndarray | None = None
    modality_projection: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.scene_projection is None:
            self.scene_projection = default_scene_projection(self.scene_names, self.class_names)
        if self.modality_projection is None:
            self.modality_projection = default_modality_projection(self.modality_names, self.class_names)
        self.scene_projection = np.asarray(self.scene_projection, dtype=np.float32)
        self.modality_projection = np.asarray(self.modality_projection, dtype=np.float32)
        if self.scene_projection.shape != (len(self.class_names), len(self.scene_names)):
            raise ValueError("scene_projection shape must be [num_classes, num_scenes]")
        if self.modality_projection.shape != (len(self.class_names), len(self.modality_names)):
            raise ValueError("modality_projection shape must be [num_classes, num_modalities]")

    def fuse_numpy(
        self,
        det_logits: np.ndarray,
        scene_prob: np.ndarray,
        modality_prob: np.ndarray,
    ) -> np.ndarray:
        det_logits = np.asarray(det_logits, dtype=np.float32)
        scene_prob = np.asarray(scene_prob, dtype=np.float32)
        modality_prob = np.asarray(modality_prob, dtype=np.float32)
        scene_bias = self.scene_projection @ scene_prob
        modality_bias = self.modality_projection @ modality_prob
        return det_logits + self.alpha * scene_bias + self.beta * modality_bias

    def fuse_torch(self, det_logits, scene_prob, modality_prob):
        """Torch implementation for training/inference integration."""

        import torch

        scene_projection = torch.as_tensor(
            self.scene_projection,
            dtype=det_logits.dtype,
            device=det_logits.device,
        )
        modality_projection = torch.as_tensor(
            self.modality_projection,
            dtype=det_logits.dtype,
            device=det_logits.device,
        )
        scene_bias = scene_prob @ scene_projection.T
        modality_bias = modality_prob @ modality_projection.T
        return det_logits + self.alpha * scene_bias + self.beta * modality_bias
