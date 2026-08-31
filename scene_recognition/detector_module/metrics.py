from __future__ import annotations

from typing import Any


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _class_result_rows(box: Any, class_count: int) -> dict[int, int] | None:
    """Map raw class ids to Metric.class_result row indices when available.

    Ultralytics stores metrics only for classes present in the evaluated
    labels.  ``class_result(i)`` indexes that compact present-class array,
    while ``maps`` is expanded back to all classes by filling absent classes
    with the overall mAP.  A missing/invalid ``ap_class_index`` is retained as
    a legacy compatibility case, but is deliberately treated as unknown
    presence rather than guessing from ``maps``.
    """

    marker = object()
    raw = getattr(box, "ap_class_index", marker)
    if raw is marker or raw is None:
        return None
    try:
        if hasattr(raw, "detach"):
            raw = raw.detach().cpu()
        if hasattr(raw, "tolist"):
            raw = raw.tolist()
        if not isinstance(raw, (list, tuple)):
            raw = [raw]
        class_ids = [int(value) for value in raw]
    except (AttributeError, TypeError, ValueError):
        return None
    if any(class_id < 0 or class_id >= class_count for class_id in class_ids):
        return None
    if len(set(class_ids)) != len(class_ids):
        return None
    return {class_id: row_index for row_index, class_id in enumerate(class_ids)}


def detection_metrics_to_dict(metrics: Any, class_names: list[str]) -> dict:
    """Convert an Ultralytics detection metrics object into stable JSON data."""
    box = metrics.box
    maps = list(getattr(box, "maps", []))
    class_rows = _class_result_rows(box, len(class_names))
    per_class = {}
    for class_id, name in enumerate(class_names):
        class_values = []
        row_index = class_id if class_rows is None else class_rows.get(class_id)
        if row_index is not None:
            try:
                class_values = list(box.class_result(row_index))
            except (AttributeError, IndexError, TypeError):
                pass
        per_class[name] = {
            "precision": _safe_float(class_values[0]) if len(class_values) > 0 else None,
            "recall": _safe_float(class_values[1]) if len(class_values) > 1 else None,
            "map50": _safe_float(class_values[2]) if len(class_values) > 2 else None,
            "map50_95": (
                _safe_float(class_values[3])
                if len(class_values) > 3
                else (
                    _safe_float(maps[class_id])
                    if class_rows is not None and class_id in class_rows and class_id < len(maps)
                    else None
                )
            ),
        }

    speed = {
        str(name): round(_safe_float(value), 6)
        for name, value in getattr(metrics, "speed", {}).items()
    }
    return {
        "precision": _safe_float(getattr(box, "mp", 0.0)),
        "recall": _safe_float(getattr(box, "mr", 0.0)),
        "map50": _safe_float(getattr(box, "map50", 0.0)),
        "map50_95": _safe_float(getattr(box, "map", 0.0)),
        "map75": _safe_float(getattr(box, "map75", 0.0)),
        "fitness": _safe_float(getattr(metrics, "fitness", 0.0)),
        "per_class": per_class,
        "speed_ms_per_image": speed,
        "raw_results": {
            str(name): _safe_float(value)
            for name, value in getattr(metrics, "results_dict", {}).items()
        },
    }
