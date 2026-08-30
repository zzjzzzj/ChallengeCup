"""Trainable image-level sparse routing and usage diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch
from torch import nn


@dataclass
class TorchExpertRoute:
    """Differentiable routing result for a batch of images."""

    logits: torch.Tensor
    probabilities: torch.Tensor
    expert_ids: torch.Tensor
    expert_weights: torch.Tensor
    entropy: torch.Tensor

    @property
    def normalized_entropy(self) -> torch.Tensor:
        expert_count = int(self.probabilities.shape[-1])
        return self.entropy / max(math.log(expert_count), 1e-12)

    def to_dict(self, index: int | None = None) -> dict[str, Any]:
        """Return JSON-ready diagnostics for one image or the whole batch."""

        def convert(value: torch.Tensor) -> Any:
            value = value.detach().cpu()
            return value.item() if value.ndim == 0 else value.tolist()

        if index is None:
            return {
                "expert_ids": convert(self.expert_ids),
                "expert_weights": convert(self.expert_weights),
                "probabilities": convert(self.probabilities),
                "router_entropy": convert(self.entropy),
                "router_entropy_normalized": convert(self.normalized_entropy),
            }
        return {
            "expert_ids": [int(value) for value in self.expert_ids[index].detach().cpu().tolist()],
            "expert_weights": [float(value) for value in self.expert_weights[index].detach().cpu().tolist()],
            "probabilities": [float(value) for value in self.probabilities[index].detach().cpu().tolist()],
            "router_entropy": float(self.entropy[index].detach().cpu().item()),
            "router_entropy_normalized": float(self.normalized_entropy[index].detach().cpu().item()),
        }


class TorchSparseExpertRouter(nn.Module):
    """Image-level MLP router with a differentiable soft Top-K weighting."""

    def __init__(
        self,
        input_dim: int,
        expert_count: int = 5,
        top_k: int = 2,
        hidden_dim: int = 128,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if expert_count <= 0:
            raise ValueError("expert_count must be positive")
        if top_k <= 0 or top_k > expert_count:
            raise ValueError("top_k must be in [1, expert_count]")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.input_dim = int(input_dim)
        self.expert_count = int(expert_count)
        self.top_k = int(top_k)
        self.hidden_dim = int(hidden_dim)
        self.temperature = float(temperature)
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, expert_count),
        )

    def forward(
        self,
        query: torch.Tensor,
        *,
        temperature: float | torch.Tensor | None = None,
    ) -> TorchExpertRoute:
        if query.ndim != 2 or query.shape[1] != self.input_dim:
            raise ValueError(
                f"router expects [batch, {self.input_dim}] query, got {tuple(query.shape)}"
            )
        route_temperature = self.temperature if temperature is None else temperature
        if isinstance(route_temperature, torch.Tensor):
            if route_temperature.numel() != 1 or bool((route_temperature <= 0).any()):
                raise ValueError("temperature tensor must be a positive scalar")
        elif float(route_temperature) <= 0:
            raise ValueError("temperature must be positive")
        logits = self.network(query)
        probabilities = torch.softmax(logits / route_temperature, dim=-1)
        values, expert_ids = probabilities.topk(self.top_k, dim=-1)
        expert_weights = values / values.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        entropy = -(probabilities.clamp_min(1e-12) * probabilities.clamp_min(1e-12).log()).sum(dim=-1)
        return TorchExpertRoute(
            logits=logits,
            probabilities=probabilities,
            expert_ids=expert_ids,
            expert_weights=expert_weights,
            entropy=entropy,
        )

    @staticmethod
    def load_balance_loss(route: TorchExpertRoute) -> torch.Tensor:
        """Switch-style ``E * sum(mean(p) * fraction(tokens routed))`` loss."""

        probabilities = route.probabilities
        batch_size, expert_count = probabilities.shape
        if batch_size == 0:
            return probabilities.sum() * 0.0
        selected = torch.nn.functional.one_hot(route.expert_ids, num_classes=expert_count).float()
        fractions = selected.reshape(batch_size * route.expert_ids.shape[1], expert_count).mean(dim=0)
        mean_probability = probabilities.mean(dim=0)
        return expert_count * (mean_probability * fractions).sum()

    @staticmethod
    def router_z_loss(route: TorchExpertRoute) -> torch.Tensor:
        """Small Switch-style z-loss on router logits."""

        if route.logits.shape[0] == 0:
            return route.logits.sum() * 0.0
        return torch.logsumexp(route.logits.float(), dim=-1).square().mean()


class TorchExpertUsageTracker:
    """Batch-agnostic expert usage tracker for reports and anchor importance."""

    def __init__(self, expert_count: int, expert_names: list[str] | None = None) -> None:
        if expert_count <= 0:
            raise ValueError("expert_count must be positive")
        self.expert_count = int(expert_count)
        self.expert_names = list(expert_names or [f"expert_{i}" for i in range(expert_count)])
        if len(self.expert_names) != expert_count:
            raise ValueError("expert_names length must equal expert_count")
        self.top_counts = [0 for _ in range(expert_count)]
        self.soft_probability_sum = [0.0 for _ in range(expert_count)]
        self.entropy_sum = 0.0
        self.total_images = 0

    def update(self, route: TorchExpertRoute) -> None:
        ids = route.expert_ids.detach().cpu()
        probabilities = route.probabilities.detach().cpu()
        entropy = route.entropy.detach().cpu()
        if ids.ndim != 2 or ids.shape[1] <= 0:
            raise ValueError("route expert_ids must have shape [batch, top_k]")
        if probabilities.ndim != 2 or probabilities.shape != (ids.shape[0], self.expert_count):
            raise ValueError("route probabilities must have shape [batch, expert_count]")
        if entropy.ndim != 1 or entropy.shape[0] != ids.shape[0]:
            raise ValueError("route entropy must have shape [batch]")
        if bool((ids < 0).any()) or bool((ids >= self.expert_count).any()):
            raise ValueError("route expert_ids contain an out-of-range expert")
        for expert_id in ids.reshape(-1).tolist():
            self.top_counts[int(expert_id)] += 1
        self.soft_probability_sum = [
            total + float(value)
            for total, value in zip(self.soft_probability_sum, probabilities.sum(dim=0).tolist())
        ]
        self.entropy_sum += float(entropy.sum().item())
        self.total_images += int(ids.shape[0])

    def state_dict(self) -> dict[str, Any]:
        """Return the non-parameter state needed to carry usage across stages."""

        return {
            "expert_count": self.expert_count,
            "expert_names": list(self.expert_names),
            "top_counts": list(self.top_counts),
            "soft_probability_sum": list(self.soft_probability_sum),
            "entropy_sum": self.entropy_sum,
            "total_images": self.total_images,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore usage state without changing the tracker object identity."""

        if int(state.get("expert_count", self.expert_count)) != self.expert_count:
            raise ValueError("usage tracker expert_count does not match")
        names = list(state.get("expert_names", self.expert_names))
        if names != self.expert_names:
            raise ValueError("usage tracker expert_names do not match")
        top_counts = [int(value) for value in state.get("top_counts", [])]
        probability_sum = [float(value) for value in state.get("soft_probability_sum", [])]
        if len(top_counts) != self.expert_count or len(probability_sum) != self.expert_count:
            raise ValueError("usage tracker state has the wrong expert count")
        self.top_counts = top_counts
        self.soft_probability_sum = probability_sum
        self.entropy_sum = float(state.get("entropy_sum", 0.0))
        self.total_images = int(state.get("total_images", 0))

    @property
    def total_routes(self) -> int:
        return self.total_images

    def importance(self) -> dict[str, float]:
        denominator = max(self.total_images, 1)
        return {
            name: count / denominator
            for name, count in zip(self.expert_names, self.top_counts)
        }

    def mean_probability(self) -> dict[str, float]:
        denominator = max(self.total_images, 1)
        return {
            name: value / denominator
            for name, value in zip(self.expert_names, self.soft_probability_sum)
        }

    def to_dict(self) -> dict[str, Any]:
        usage = self.importance()
        return {
            "expert_names": list(self.expert_names),
            "total_images": self.total_images,
            "top_k_activations": dict(zip(self.expert_names, self.top_counts)),
            "top_k_activation_frequency": usage,
            "mean_probability": self.mean_probability(),
            "router_entropy": self.entropy_sum / max(self.total_images, 1),
            "max_occupancy": max(usage.values(), default=0.0),
            "load_variation": _coefficient_of_variation(list(usage.values())),
        }


def _coefficient_of_variation(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    if mean <= 1e-12:
        return 0.0
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return variance**0.5 / mean


# Names used by a few downstream experiments.
SparseRouter = TorchSparseExpertRouter
ExpertUsageTrackerTorch = TorchExpertUsageTracker


__all__ = [
    "ExpertUsageTrackerTorch",
    "SparseRouter",
    "TorchExpertRoute",
    "TorchExpertUsageTracker",
    "TorchSparseExpertRouter",
]
