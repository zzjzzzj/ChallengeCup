"""Checkpoint metadata helpers for Sparse-MoE v1."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn

from scene_recognition.detector_module.sparse_moe_model import (
    SparseMoEConfig,
    get_sparse_moe_adapter,
)


def sparse_moe_metadata(model: nn.Module) -> dict[str, Any]:
    adapter = get_sparse_moe_adapter(model)
    if adapter is None:
        return {"enabled": False}
    return {"enabled": True, **adapter.metadata()}


def update_sparse_moe_anchors(model: nn.Module) -> dict[str, float]:
    adapter = get_sparse_moe_adapter(model)
    if adapter is None:
        return {}
    return adapter.update_anchors()


def write_sparse_moe_artifacts(
    model: nn.Module,
    output_dir: Path,
    *,
    context_summary: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Write human-readable config/usage and a torch anchor artifact."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = sparse_moe_metadata(model)
    paths: dict[str, str] = {}
    config_path = output_dir / "sparse_moe_config.json"
    usage_path = output_dir / "expert_usage.json"
    anchors_path = output_dir / "expert_anchors.pt"
    context_path = output_dir / "context_metadata_summary.json"
    config_payload = {
        "enabled": bool(metadata.get("enabled")),
        "config": metadata.get("config", {}),
        "feature_channels": metadata.get("feature_channels", []),
    }
    config_path.write_text(json.dumps(config_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    usage_path.write_text(
        json.dumps(metadata.get("usage", {}), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    # Keep full vectors in the binary sidecar; compact JSON metadata should
    # only describe their shapes and importance to avoid multi-megabyte stage
    # summaries.
    adapter = get_sparse_moe_adapter(model)
    torch.save(adapter.anchor_bank.to_dict() if adapter is not None else {}, anchors_path)
    context_path.write_text(
        json.dumps(context_summary or {}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    paths.update(
        {
            "config": str(config_path.resolve()),
            "usage": str(usage_path.resolve()),
            "anchors": str(anchors_path.resolve()),
            "context": str(context_path.resolve()),
        }
    )
    return paths


def load_sparse_moe_checkpoint(
    checkpoint: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> nn.Module:
    """Load a project checkpoint and return its sparse model object."""

    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    try:
        payload = torch.load(checkpoint, map_location=map_location, weights_only=False)
    except TypeError:  # torch<2.6 compatibility
        payload = torch.load(checkpoint, map_location=map_location)
    if isinstance(payload, nn.Module):
        model = payload
    elif isinstance(payload, dict):
        model = payload.get("ema") or payload.get("model")
        if not isinstance(model, nn.Module):
            raise ValueError("checkpoint does not contain a serialized torch model")
    else:
        raise ValueError("unsupported checkpoint payload")
    if get_sparse_moe_adapter(model) is None:
        raise ValueError(f"checkpoint is not a Sparse-MoE model: {checkpoint}")
    return model


def save_sparse_moe_checkpoint(
    model: nn.Module,
    path: str | Path,
    *,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Save a lightweight standalone round-trip checkpoint for unit/inference use."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = sparse_moe_metadata(model)
    torch.save(
        {
            "model": copy.deepcopy(model).cpu(),
            "sparse_moe": metadata,
            "extra": extra or {},
        },
        path,
    )
    return path.resolve()


def read_sparse_moe_config(path: str | Path) -> SparseMoEConfig:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    config = payload.get("config", payload)
    return SparseMoEConfig.from_dict(config)


__all__ = [
    "load_sparse_moe_checkpoint",
    "read_sparse_moe_config",
    "save_sparse_moe_checkpoint",
    "sparse_moe_metadata",
    "update_sparse_moe_anchors",
    "write_sparse_moe_artifacts",
]
