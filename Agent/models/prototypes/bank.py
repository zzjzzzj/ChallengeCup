from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from Agent.common.utils.jsonio import read_json, write_json


@dataclass
class PrototypeEntry:
    vector: list[float]
    count: int = 0


class PrototypeBank:
    """Class-modality prototypes for unpaired IR/SAR data."""

    def __init__(self, momentum: float = 0.90) -> None:
        if not 0.0 <= momentum < 1.0:
            raise ValueError("momentum must be in [0, 1)")
        self.momentum = momentum
        self.entries: dict[str, PrototypeEntry] = {}

    @staticmethod
    def _key(class_name: str, modality: str) -> str:
        return f"{class_name}::{modality}"

    @staticmethod
    def _normalize(vector: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-12:
            return vector.astype(np.float32)
        return (vector / norm).astype(np.float32)

    def update(self, class_name: str, modality: str, features: np.ndarray) -> None:
        """Update P_{class, modality} with one or more target features."""

        features = np.asarray(features, dtype=np.float32)
        if features.ndim == 1:
            features = features[None, :]
        if features.ndim != 2 or features.shape[0] == 0:
            raise ValueError("features must be a non-empty 1D or 2D array")
        current = self._normalize(features.mean(axis=0))
        key = self._key(class_name, modality)
        old = self.entries.get(key)
        if old is None:
            self.entries[key] = PrototypeEntry(vector=current.tolist(), count=int(features.shape[0]))
            return
        old_vector = np.asarray(old.vector, dtype=np.float32)
        updated = self._normalize(self.momentum * old_vector + (1.0 - self.momentum) * current)
        self.entries[key] = PrototypeEntry(vector=updated.tolist(), count=old.count + int(features.shape[0]))

    def modality_prototype(self, class_name: str, modality: str) -> np.ndarray | None:
        entry = self.entries.get(self._key(class_name, modality))
        if entry is None:
            return None
        return np.asarray(entry.vector, dtype=np.float32)

    def shared_prototype(self, class_name: str) -> np.ndarray | None:
        """Build J_class from all available modality prototypes."""

        vectors = []
        weights = []
        prefix = f"{class_name}::"
        for key, entry in self.entries.items():
            if key.startswith(prefix):
                vectors.append(np.asarray(entry.vector, dtype=np.float32))
                weights.append(max(entry.count, 1))
        if not vectors:
            return None
        matrix = np.stack(vectors, axis=0)
        weight = np.asarray(weights, dtype=np.float32)
        combined = (matrix * weight[:, None]).sum(axis=0) / max(float(weight.sum()), 1.0)
        return self._normalize(combined)

    def proto_logits(self, features: np.ndarray, class_names: list[str], temperature: float = 0.1) -> np.ndarray:
        features = np.asarray(features, dtype=np.float32)
        if features.ndim == 1:
            features = features[None, :]
        normalized = np.stack([self._normalize(row) for row in features], axis=0)
        prototypes = []
        for class_name in class_names:
            proto = self.shared_prototype(class_name)
            if proto is None:
                proto = np.zeros(normalized.shape[1], dtype=np.float32)
            prototypes.append(proto)
        proto_matrix = np.stack(prototypes, axis=0)
        return normalized @ proto_matrix.T / max(temperature, 1e-6)

    def to_dict(self) -> dict[str, Any]:
        return {
            "momentum": self.momentum,
            "entries": {key: asdict(value) for key, value in self.entries.items()},
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PrototypeBank":
        bank = cls(momentum=float(payload.get("momentum", 0.90)))
        bank.entries = {
            key: PrototypeEntry(vector=list(value["vector"]), count=int(value.get("count", 0)))
            for key, value in payload.get("entries", {}).items()
        }
        return bank

    def save(self, path: Path) -> None:
        write_json(path, self.to_dict())

    @classmethod
    def load(cls, path: Path) -> "PrototypeBank":
        return cls.from_dict(read_json(path))
