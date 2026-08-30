"""Differentiable expert-anchor consolidation for Sparse-MoE v1."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import torch


def _parameter_list(parameters: object) -> list[torch.Tensor]:
    if isinstance(parameters, torch.Tensor):
        return [parameters]
    if hasattr(parameters, "parameters"):
        return [value for value in parameters.parameters() if isinstance(value, torch.Tensor)]
    if isinstance(parameters, Mapping):
        return [value for value in parameters.values() if isinstance(value, torch.Tensor)]
    if isinstance(parameters, Iterable):
        return [value for value in parameters if isinstance(value, torch.Tensor)]
    raise TypeError("parameters must be a tensor, module, mapping, or iterable of tensors")


def _flatten_parameters(parameters: object) -> torch.Tensor:
    values = _parameter_list(parameters)
    if not values:
        raise ValueError("expert parameters must not be empty")
    return torch.cat([value.reshape(-1) for value in values])


class TorchExpertAnchorBank:
    """EMA anchor bank with a differentiable parameter-drift penalty.

    Anchors are stored detached on CPU and are therefore not trainable.  The
    current expert tensors remain in the computation graph when ``penalty`` is
    called.  T1 naturally has no anchors and returns a differentiable zero.
    """

    def __init__(self, rho: float = 0.95) -> None:
        if not 0.0 <= rho < 1.0:
            raise ValueError("rho must be in [0, 1)")
        self.rho = float(rho)
        self.anchors: dict[str, torch.Tensor] = {}
        self.importance: dict[str, float] = {}

    @property
    def has_anchors(self) -> bool:
        return bool(self.anchors)

    def update_anchor(
        self,
        expert_name: str,
        parameters: object,
        activation_frequency: float = 0.0,
    ) -> None:
        current = _flatten_parameters(parameters).detach().float().cpu()
        if expert_name in self.anchors:
            old = self.anchors[expert_name]
            if old.shape != current.shape:
                raise ValueError(
                    f"anchor shape changed for {expert_name}: {tuple(old.shape)} vs {tuple(current.shape)}"
                )
            current = self.rho * old + (1.0 - self.rho) * current
        self.anchors[expert_name] = current
        self.importance[expert_name] = max(0.0, float(activation_frequency))

    def update_from_experts(
        self,
        experts: Mapping[str, object],
        usage: Mapping[str, float] | None = None,
    ) -> dict[str, float]:
        """Update all expert anchors from a name-to-parameters mapping."""

        usage = usage or {}
        for name, parameters in experts.items():
            self.update_anchor(name, parameters, float(usage.get(name, 0.0)))
        return dict(self.importance)

    def penalty(self, expert_name: str, parameters: object) -> torch.Tensor:
        current = _flatten_parameters(parameters)
        anchor = self.anchors.get(expert_name)
        if anchor is None:
            return current.sum() * 0.0
        reference = anchor.to(device=current.device, dtype=current.dtype)
        if reference.numel() != current.numel():
            raise ValueError(
                f"anchor shape changed for {expert_name}: {reference.numel()} vs {current.numel()}"
            )
        weight = float(self.importance.get(expert_name, 0.0))
        return weight * (current - reference).square().mean()

    def penalty_from_experts(self, experts: Mapping[str, object]) -> torch.Tensor:
        if not experts:
            raise ValueError("experts must not be empty")
        penalties = [self.penalty(name, parameters) for name, parameters in experts.items()]
        return torch.stack(penalties).sum()

    # ``loss`` is a readable alias for model/trainer code.
    loss = penalty_from_experts

    def to_dict(self) -> dict[str, Any]:
        return {
            "rho": self.rho,
            "anchors": {name: value.tolist() for name, value in self.anchors.items()},
            "importance": dict(self.importance),
        }

    def summary(self) -> dict[str, Any]:
        """Return JSON-sized anchor metadata without duplicating parameter vectors."""

        return {
            "rho": self.rho,
            "expert_names": sorted(self.anchors),
            "shapes": {name: list(value.shape) for name, value in self.anchors.items()},
            "importance": dict(self.importance),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TorchExpertAnchorBank":
        bank = cls(rho=float(payload.get("rho", 0.95)))
        bank.anchors = {
            str(name): torch.tensor(values, dtype=torch.float32)
            for name, values in dict(payload.get("anchors", {})).items()
        }
        bank.importance = {
            str(name): float(value)
            for name, value in dict(payload.get("importance", {})).items()
        }
        return bank

    def state_dict(self) -> dict[str, Any]:
        return self.to_dict()

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        restored = self.from_dict(state)
        self.rho = restored.rho
        self.anchors = restored.anchors
        self.importance = restored.importance


TorchExpertAnchor = TorchExpertAnchorBank


__all__ = ["TorchExpertAnchor", "TorchExpertAnchorBank"]
