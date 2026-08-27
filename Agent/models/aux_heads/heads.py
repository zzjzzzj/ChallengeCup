from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AuxiliaryHeadOutputs:
    modality_logits: object
    scene_logits: object


class ModalitySceneAuxiliaryHeads:
    """Torch auxiliary heads for IR/SAR and scene supervision.

    The class is intentionally small and can be attached to GAP features from
    YOLO's shared backbone. It imports torch lazily so non-training utilities
    can still run in lightweight environments.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        num_modalities: int = 2,
        num_scenes: int = 4,
        dropout: float = 0.10,
    ) -> None:
        import torch
        from torch import nn

        self.torch = torch
        self.nn = nn
        self.module = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, hidden_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.modality_head = nn.Linear(hidden_channels, num_modalities)
        self.scene_head = nn.Linear(hidden_channels, num_scenes)

    def parameters(self):
        yield from self.module.parameters()
        yield from self.modality_head.parameters()
        yield from self.scene_head.parameters()

    def __call__(self, feature_map) -> AuxiliaryHeadOutputs:
        embedding = self.module(feature_map)
        return AuxiliaryHeadOutputs(
            modality_logits=self.modality_head(embedding),
            scene_logits=self.scene_head(embedding),
        )
