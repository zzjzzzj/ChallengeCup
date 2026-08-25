from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from Agent.common.utils.jsonio import read_json, write_json


class ExpertAnchorBank:
    """Soft consolidation anchors for expert parameters."""

    def __init__(self, rho: float = 0.95) -> None:
        if not 0.0 <= rho < 1.0:
            raise ValueError("rho must be in [0, 1)")
        self.rho = rho
        self.anchors: dict[str, list[float]] = {}
        self.importance: dict[str, float] = {}

    def update_anchor(self, expert_name: str, flat_parameters: np.ndarray, activation_frequency: float) -> None:
        params = np.asarray(flat_parameters, dtype=np.float32).ravel()
        old = self.anchors.get(expert_name)
        if old is None:
            updated = params
        else:
            updated = self.rho * np.asarray(old, dtype=np.float32) + (1.0 - self.rho) * params
        self.anchors[expert_name] = updated.astype(np.float32).tolist()
        self.importance[expert_name] = float(activation_frequency)

    def penalty_numpy(self, expert_name: str, flat_parameters: np.ndarray) -> float:
        anchor = self.anchors.get(expert_name)
        if anchor is None:
            return 0.0
        params = np.asarray(flat_parameters, dtype=np.float32).ravel()
        ref = np.asarray(anchor, dtype=np.float32)
        weight = self.importance.get(expert_name, 0.0)
        return float(weight * ((params - ref) ** 2).mean())

    def to_dict(self) -> dict[str, Any]:
        return {"rho": self.rho, "anchors": self.anchors, "importance": self.importance}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ExpertAnchorBank":
        bank = cls(rho=float(payload.get("rho", 0.95)))
        bank.anchors = {key: list(value) for key, value in payload.get("anchors", {}).items()}
        bank.importance = {key: float(value) for key, value in payload.get("importance", {}).items()}
        return bank

    def save(self, path: Path) -> None:
        write_json(path, self.to_dict())

    @classmethod
    def load(cls, path: Path) -> "ExpertAnchorBank":
        return cls.from_dict(read_json(path))
