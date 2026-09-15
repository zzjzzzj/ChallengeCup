"""Helpers for normalizing Ultralytics loss return values."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import torch


BASE_LOSS_ALIASES: tuple[tuple[str, ...], ...] = (
    ("box_loss", "loss_box", "box", "bbox", "metrics/box_loss", "train/box_loss"),
    ("cls_loss", "loss_cls", "cls", "class", "metrics/cls_loss", "train/cls_loss"),
    ("dfl_loss", "loss_dfl", "dfl", "metrics/dfl_loss", "train/dfl_loss"),
)


def _reference_device_dtype(reference: Any) -> tuple[torch.device, torch.dtype]:
    if isinstance(reference, torch.Tensor):
        dtype = reference.dtype if reference.is_floating_point() else torch.float32
        return reference.device, dtype
    return torch.device("cpu"), torch.float32


def _as_tensor(value: Any, reference: Any) -> torch.Tensor:
    device, dtype = _reference_device_dtype(reference)
    if isinstance(value, torch.Tensor):
        tensor = value
        if not tensor.is_floating_point():
            tensor = tensor.float()
        return tensor.to(device=device, dtype=dtype).reshape(-1)
    if isinstance(value, (int, float)):
        return torch.tensor([float(value)], device=device, dtype=dtype)
    raise TypeError("unsupported loss value type: %s" % type(value).__name__)


def _dict_lookup(data: Mapping[str, Any], aliases: Iterable[str]) -> Any:
    lowered = {str(key).lower(): key for key in data.keys()}
    for alias in aliases:
        key = lowered.get(alias.lower())
        if key is not None:
            return data[key]
    for alias in aliases:
        alias_lower = alias.lower()
        for lower_key, original_key in lowered.items():
            if lower_key.endswith("/" + alias_lower) or lower_key.endswith("_" + alias_lower):
                return data[original_key]
    raise KeyError(tuple(aliases))


def _flatten_loss_items(value: Any, reference: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor) or isinstance(value, (int, float)):
        return _as_tensor(value, reference)
    if isinstance(value, Mapping):
        for key in ("loss_items", "items", "components"):
            if key in value:
                return _flatten_loss_items(value[key], reference)
        ordered = []
        for aliases in BASE_LOSS_ALIASES:
            try:
                ordered.append(_as_tensor(_dict_lookup(value, aliases), reference).sum())
            except (KeyError, TypeError):
                ordered = []
                break
        if ordered:
            return torch.stack(ordered).reshape(-1)
        numeric = []
        for item in value.values():
            try:
                numeric.append(_as_tensor(item, reference).sum())
            except TypeError:
                continue
        if numeric:
            return torch.stack(numeric).reshape(-1)
        raise TypeError("loss item dict has no tensor-like values")
    if isinstance(value, (list, tuple)):
        tensors = [_as_tensor(item, reference) for item in value]
        return torch.cat(tensors).reshape(-1) if tensors else _as_tensor(0.0, reference)
    raise TypeError("unsupported loss items type: %s" % type(value).__name__)


def loss_items_to_vector(value: Any, reference: Any, expected_length: int = 3) -> torch.Tensor:
    """Return a fixed-length vector for trainer logging.

    Newer Ultralytics versions may return dictionaries for loss items while
    older versions return tensors.  The custom trainers need a tensor aligned
    with ``loss_names``.
    """

    vector = _flatten_loss_items(value, reference).detach().reshape(-1)
    if expected_length <= 0:
        return vector
    if vector.numel() == expected_length:
        return vector
    device, dtype = _reference_device_dtype(reference)
    output = torch.zeros(expected_length, device=device, dtype=dtype)
    keep = min(expected_length, int(vector.numel()))
    if keep:
        output[:keep] = vector[:keep].to(device=device, dtype=dtype)
    return output


def loss_value_to_scalar(value: Any, reference: Any) -> torch.Tensor:
    """Return a scalar loss suitable for ``backward()``."""

    if isinstance(value, Mapping):
        for key in ("loss", "total_loss", "total", "L_total"):
            if key in value:
                return _flatten_loss_items(value[key], reference).sum()
    return _flatten_loss_items(value, reference).sum()
