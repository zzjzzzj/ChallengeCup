"""PyTorch sparse expert adapters used by Sparse-MoE v1.

The NumPy router in :mod:`router` is kept for the historical Agent API.  The
classes in this module are the trainable implementation used immediately
before the YOLO detection head.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class SparseExpertAdapter(nn.Module):
    """A bottleneck residual adapter for one scale and one expert.

    The last projection is zero initialized, so ``forward(x)`` is an exact
    identity at construction time.  ``forward_residual`` exposes only the
    learned residual for sparse weighted aggregation.
    """

    def __init__(self, channels: int, bottleneck_ratio: float = 0.25) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        if not 0.0 < bottleneck_ratio <= 1.0:
            raise ValueError("bottleneck_ratio must be in (0, 1]")
        self.channels = int(channels)
        self.bottleneck_ratio = float(bottleneck_ratio)
        self.bottleneck_channels = max(1, int(math.ceil(channels * bottleneck_ratio)))
        self.down = nn.Conv2d(channels, self.bottleneck_channels, kernel_size=1)
        self.depthwise = nn.Conv2d(
            self.bottleneck_channels,
            self.bottleneck_channels,
            kernel_size=3,
            padding=1,
            groups=self.bottleneck_channels,
        )
        self.activation = nn.SiLU()
        self.up = nn.Conv2d(self.bottleneck_channels, channels, kernel_size=1)
        self.forward_calls = 0
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize the bottleneck and make the residual projection zero."""

        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        if self.down.bias is not None:
            nn.init.zeros_(self.down.bias)
        nn.init.kaiming_uniform_(self.depthwise.weight, a=math.sqrt(5))
        if self.depthwise.bias is not None:
            nn.init.zeros_(self.depthwise.bias)
        nn.init.zeros_(self.up.weight)
        if self.up.bias is not None:
            nn.init.zeros_(self.up.bias)

    def forward_residual(self, features: torch.Tensor) -> torch.Tensor:
        """Return the learned residual and record an execution for diagnostics."""

        if features.ndim != 4 or features.shape[1] != self.channels:
            raise ValueError(
                "adapter expects BCHW features with "
                f"{self.channels} channels, got {tuple(features.shape)}"
            )
        self.forward_calls += 1
        hidden = self.down(features)
        hidden = self.depthwise(hidden)
        hidden = self.activation(hidden)
        return self.up(hidden)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Apply the adapter as an identity-preserving residual transform."""

        return features + self.forward_residual(features)


class SparseExpertAdapterBank(nn.Module):
    """One scale's expert pool with genuinely sparse top-k execution."""

    def __init__(
        self,
        channels: int,
        expert_count: int = 5,
        bottleneck_ratio: float = 0.25,
    ) -> None:
        super().__init__()
        if expert_count <= 0:
            raise ValueError("expert_count must be positive")
        self.channels = int(channels)
        self.expert_count = int(expert_count)
        self.adapters = nn.ModuleList(
            [SparseExpertAdapter(channels, bottleneck_ratio) for _ in range(expert_count)]
        )

    def forward(
        self,
        features: torch.Tensor,
        expert_ids: torch.Tensor,
        expert_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Add weighted residuals for the selected experts only.

        ``expert_ids`` and ``expert_weights`` have shape ``[batch, top_k]``.
        Each expert is called on only the images selecting it, which avoids
        the common pseudo-sparse implementation that evaluates all experts.
        """

        if features.ndim != 4 or features.shape[1] != self.channels:
            raise ValueError(
                f"expected BCHW with {self.channels} channels, got {tuple(features.shape)}"
            )
        if expert_ids.ndim != 2 or expert_weights.shape != expert_ids.shape:
            raise ValueError("expert_ids and expert_weights must have matching [batch, top_k] shape")
        if expert_ids.shape[0] != features.shape[0]:
            raise ValueError("routing batch size must match feature batch size")
        if expert_ids.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
            expert_ids = expert_ids.long()
        if bool((expert_ids < 0).any()) or bool((expert_ids >= self.expert_count).any()):
            raise ValueError("expert_ids contain an out-of-range expert")

        output = features.clone()
        for expert_index, adapter in enumerate(self.adapters):
            selected = expert_ids.eq(expert_index)
            image_mask = selected.any(dim=1)
            if not bool(image_mask.any()):
                continue
            weights = expert_weights.masked_fill(~selected, 0.0).sum(dim=1)[image_mask]
            residual = adapter.forward_residual(features[image_mask])
            output[image_mask] = output[image_mask] + residual * weights[:, None, None, None]
        return output

    def execution_counts(self) -> list[int]:
        """Return per-expert forward counts for sparse-execution tests."""

        return [adapter.forward_calls for adapter in self.adapters]


# Short aliases make the component easy to discover without changing the
# neutral expert naming used in the model configuration.
ExpertAdapter = SparseExpertAdapter
ExpertAdapterBank = SparseExpertAdapterBank


__all__ = [
    "ExpertAdapter",
    "ExpertAdapterBank",
    "SparseExpertAdapter",
    "SparseExpertAdapterBank",
]
