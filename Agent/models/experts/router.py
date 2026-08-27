from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ExpertRoute:
    expert_ids: list[int]
    expert_names: list[str]
    weights: list[float]
    all_scores: list[float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "expert_ids": self.expert_ids,
            "expert_names": self.expert_names,
            "weights": self.weights,
            "all_scores": self.all_scores,
        }


class SparseExpertRouter:
    """Top-K sparse router for latent expert roles."""

    def __init__(
        self,
        expert_names: list[str],
        input_dim: int,
        top_k: int = 2,
        seed: int = 42,
    ) -> None:
        if top_k <= 0 or top_k > len(expert_names):
            raise ValueError("top_k must be in [1, num_experts]")
        self.expert_names = expert_names
        self.input_dim = input_dim
        self.top_k = top_k
        rng = np.random.default_rng(seed)
        self.weight = rng.normal(0.0, 0.02, size=(len(expert_names), input_dim)).astype(np.float32)
        self.bias = np.zeros(len(expert_names), dtype=np.float32)

    @staticmethod
    def build_query(
        gap_feature: np.ndarray,
        modality_prob: np.ndarray,
        scene_prob: np.ndarray,
        runtime_stats: np.ndarray,
    ) -> np.ndarray:
        return np.concatenate(
            [
                np.asarray(gap_feature, dtype=np.float32).ravel(),
                np.asarray(modality_prob, dtype=np.float32).ravel(),
                np.asarray(scene_prob, dtype=np.float32).ravel(),
                np.asarray(runtime_stats, dtype=np.float32).ravel(),
            ],
            axis=0,
        )

    def route(self, query: np.ndarray) -> ExpertRoute:
        query = np.asarray(query, dtype=np.float32).ravel()
        if query.shape[0] != self.input_dim:
            raise ValueError(f"query dim {query.shape[0]} != router input_dim {self.input_dim}")
        logits = self.weight @ query + self.bias
        logits = logits - float(logits.max())
        prob = np.exp(logits)
        prob = prob / max(float(prob.sum()), 1e-12)
        top_ids = np.argsort(prob)[-self.top_k :][::-1].tolist()
        top_weights = prob[top_ids]
        top_weights = top_weights / max(float(top_weights.sum()), 1e-12)
        return ExpertRoute(
            expert_ids=[int(idx) for idx in top_ids],
            expert_names=[self.expert_names[idx] for idx in top_ids],
            weights=[float(value) for value in top_weights],
            all_scores=[float(value) for value in prob],
        )


class ExpertUsageTracker:
    """Track expert activation frequency for anchor consolidation."""

    def __init__(self, expert_names: list[str]) -> None:
        self.expert_names = expert_names
        self.counts = {name: 0 for name in expert_names}
        self.total = 0

    def update(self, route: ExpertRoute) -> None:
        self.total += 1
        for name in route.expert_names:
            self.counts[name] = self.counts.get(name, 0) + 1

    def importance(self) -> dict[str, float]:
        if self.total <= 0:
            return {name: 0.0 for name in self.expert_names}
        return {name: self.counts.get(name, 0) / self.total for name in self.expert_names}

    def to_dict(self) -> dict[str, Any]:
        return {
            "expert_names": self.expert_names,
            "counts": self.counts,
            "total": self.total,
            "importance": self.importance(),
        }
