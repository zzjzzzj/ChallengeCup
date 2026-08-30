"""Trainable modality/scene auxiliary heads for Sparse-MoE v1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class AuxiliaryHeadOutputs:
    """Logits and shared context embedding emitted by the auxiliary heads."""

    modality_logits: torch.Tensor
    scene_logits: torch.Tensor
    embedding: torch.Tensor | None = None

    @property
    def modality_probabilities(self) -> torch.Tensor:
        return torch.softmax(self.modality_logits, dim=-1)

    @property
    def scene_probabilities(self) -> torch.Tensor:
        return torch.softmax(self.scene_logits, dim=-1)


class ModalitySceneAuxiliaryHeads(nn.Module):
    """Standard ``nn.Module`` heads over one or more feature maps.

    ``in_channels`` accepts either the historical single integer or the
    dynamic Detect input channel list (for example ``[64, 128, 256]``). Each
    scale is globally pooled and projected before the two classifiers share a
    small context MLP.
    """

    def __init__(
        self,
        in_channels: int | Sequence[int],
        hidden_channels: int = 128,
        num_modalities: int = 2,
        num_scenes: int = 4,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        channels = [int(in_channels)] if isinstance(in_channels, int) else [int(v) for v in in_channels]
        if not channels or any(value <= 0 for value in channels):
            raise ValueError("in_channels must contain positive channel counts")
        if hidden_channels <= 0 or num_modalities <= 0 or num_scenes <= 0:
            raise ValueError("hidden_channels and class counts must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.in_channels = tuple(channels)
        self.hidden_channels = int(hidden_channels)
        self.num_modalities = int(num_modalities)
        self.num_scenes = int(num_scenes)
        projection_width = max(8, hidden_channels // len(channels))
        self.projections = nn.ModuleList(
            [nn.Linear(channels_i, projection_width) for channels_i in channels]
        )
        self.module = nn.Sequential(
            nn.Linear(projection_width * len(channels), hidden_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.modality_head = nn.Linear(hidden_channels, num_modalities)
        self.scene_head = nn.Linear(hidden_channels, num_scenes)

    def forward(self, features: torch.Tensor | Sequence[torch.Tensor]) -> AuxiliaryHeadOutputs:
        feature_list = [features] if isinstance(features, torch.Tensor) else list(features)
        if len(feature_list) != len(self.projections):
            raise ValueError(
                f"expected {len(self.projections)} feature maps, got {len(feature_list)}"
            )
        pooled = []
        for feature, projection, expected_channels in zip(
            feature_list, self.projections, self.in_channels
        ):
            if feature.ndim != 4 or feature.shape[1] != expected_channels:
                raise ValueError(
                    f"expected BCHW feature with {expected_channels} channels, "
                    f"got {tuple(feature.shape)}"
                )
            pooled.append(projection(F.adaptive_avg_pool2d(feature, 1).flatten(1)))
        embedding = self.module(torch.cat(pooled, dim=1))
        return AuxiliaryHeadOutputs(
            modality_logits=self.modality_head(embedding),
            scene_logits=self.scene_head(embedding),
            embedding=embedding,
        )


def _coerce_targets(targets: object, class_names: Mapping[str, int] | None = None) -> torch.Tensor:
    if isinstance(targets, torch.Tensor):
        return targets.long()
    if isinstance(targets, str):
        targets = [targets]
    if isinstance(targets, Sequence):
        values = []
        for target in targets:
            if isinstance(target, str):
                if class_names is None or target not in class_names:
                    values.append(-1)
                else:
                    values.append(int(class_names[target]))
            else:
                values.append(int(target))
        return torch.tensor(values, dtype=torch.long)
    raise TypeError("targets must be a tensor or a sequence")


def masked_cross_entropy(
    logits: torch.Tensor,
    targets: object,
    mask: torch.Tensor | Sequence[bool] | None = None,
    *,
    class_names: Mapping[str, int] | None = None,
) -> torch.Tensor:
    """Cross entropy over known metadata rows only.

    Unknown modality/scene labels are represented by ``-1`` or a false mask.
    An all-unknown batch returns a differentiable zero instead of NaN.
    """

    target_tensor = _coerce_targets(targets, class_names).to(logits.device)
    if target_tensor.ndim != 1 or target_tensor.shape[0] != logits.shape[0]:
        raise ValueError("targets must have one value per logit row")
    valid = target_tensor.ge(0)
    if mask is not None:
        valid = valid & torch.as_tensor(mask, dtype=torch.bool, device=logits.device)
    valid = valid & target_tensor.lt(logits.shape[-1])
    if not bool(valid.any()):
        return logits.sum() * 0.0
    return F.cross_entropy(logits[valid], target_tensor[valid])


def masked_auxiliary_loss(
    outputs: AuxiliaryHeadOutputs,
    modality_targets: object,
    scene_targets: object,
    modality_mask: torch.Tensor | Sequence[bool] | None = None,
    scene_mask: torch.Tensor | Sequence[bool] | None = None,
    *,
    modality_weight: float = 1.0,
    scene_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Compute masked IR/SAR and four-scene losses and their weighted total."""

    modality_loss = masked_cross_entropy(
        outputs.modality_logits,
        modality_targets,
        modality_mask,
        class_names={"ir": 0, "sar": 1},
    )
    scene_loss = masked_cross_entropy(
        outputs.scene_logits,
        scene_targets,
        scene_mask,
        class_names={"air": 0, "sea": 1, "urban": 2, "forest": 3},
    )
    total = modality_loss * float(modality_weight) + scene_loss * float(scene_weight)
    return {"modality": modality_loss, "scene": scene_loss, "total": total}


# Compatibility aliases for callers that use the longer descriptive names.
masked_aux_loss = masked_auxiliary_loss
compute_masked_auxiliary_loss = masked_auxiliary_loss


__all__ = [
    "AuxiliaryHeadOutputs",
    "ModalitySceneAuxiliaryHeads",
    "compute_masked_auxiliary_loss",
    "masked_aux_loss",
    "masked_auxiliary_loss",
    "masked_cross_entropy",
]
