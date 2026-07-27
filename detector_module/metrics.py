from __future__ import annotations

from typing import Any


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def detection_metrics_to_dict(metrics: Any, class_names: list[str]) -> dict:
    """Convert an Ultralytics detection metrics object into stable JSON data."""
    box = metrics.box
    maps = list(getattr(box, "maps", []))
    per_class = {}
    for class_id, name in enumerate(class_names):
        class_values = []
        try:
            class_values = list(box.class_result(class_id))
        except (AttributeError, IndexError, TypeError):
            pass
        per_class[name] = {
            "precision": _safe_float(class_values[0]) if len(class_values) > 0 else None,
            "recall": _safe_float(class_values[1]) if len(class_values) > 1 else None,
            "map50": _safe_float(class_values[2]) if len(class_values) > 2 else None,
            "map50_95": (
                _safe_float(class_values[3])
                if len(class_values) > 3
                else (_safe_float(maps[class_id]) if class_id < len(maps) else None)
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
