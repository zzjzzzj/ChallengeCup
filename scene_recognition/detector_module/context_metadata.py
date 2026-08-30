"""Context metadata indexing for materialized Class-IL images.

The source dataset may only provide provenance rows.  This module preserves
authoritative sensor/scene columns when present and falls back to conservative
filename parsing. Unknown values are explicit so auxiliary losses can mask
them instead of silently learning a guessed target.
"""

from __future__ import annotations

import csv
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping


CONTEXT_INDEX_FIELDS = (
    "materialized_image_path",
    "source_image",
    "sensor",
    "scene",
    "split",
    "stage",
    "sample_role",
    "augmentation_operation",
    "metadata_source",
)
SENSOR_NAMES = ("ir", "sar")
SCENE_NAMES = ("air", "sea", "urban", "forest")
UNKNOWN = "unknown"


def normalize_context_path(path: str | Path) -> str:
    return str(Path(path).expanduser().resolve()).casefold()


def _tokens(value: str | Path) -> list[str]:
    return [token for token in re.split(r"[^a-z0-9]+", str(value).casefold()) if token]


def parse_context_from_filename(value: str | Path) -> dict[str, str]:
    """Parse only exact sensor/scene tokens from a compatible filename."""

    tokens = set(_tokens(value))
    sensor = next((name for name in SENSOR_NAMES if name in tokens), UNKNOWN)
    scene = next((name for name in SCENE_NAMES if name in tokens), UNKNOWN)
    return {"sensor": sensor, "scene": scene}


def resolve_context_metadata(
    source_image: str | Path,
    *,
    sensor: str | None = None,
    scene: str | None = None,
    metadata_source: str | None = None,
) -> dict[str, str]:
    """Resolve authoritative values first, then use filename compatibility fallback."""

    parsed = parse_context_from_filename(source_image)
    normalized_sensor = str(sensor or "").strip().casefold()
    normalized_scene = str(scene or "").strip().casefold()
    sensor_known = normalized_sensor in SENSOR_NAMES
    scene_known = normalized_scene in SCENE_NAMES
    final_sensor = normalized_sensor if sensor_known else parsed["sensor"]
    final_scene = normalized_scene if scene_known else parsed["scene"]
    if metadata_source:
        source = metadata_source
    elif sensor_known or scene_known:
        source = "authoritative_manifest"
    elif final_sensor != UNKNOWN or final_scene != UNKNOWN:
        source = "filename_fallback"
    else:
        source = "unknown"
    return {
        "sensor": final_sensor,
        "scene": final_scene,
        "metadata_source": source,
    }


def build_context_row(
    *,
    materialized_image_path: str | Path,
    source_image: str | Path,
    split: str,
    stage: int,
    sample_role: str,
    augmentation_operation: str,
    sensor: str | None = None,
    scene: str | None = None,
    metadata_source: str | None = None,
) -> dict[str, str]:
    context = resolve_context_metadata(
        source_image,
        sensor=sensor,
        scene=scene,
        metadata_source=metadata_source,
    )
    return {
        "materialized_image_path": str(Path(materialized_image_path).resolve()),
        "source_image": str(source_image),
        "sensor": context["sensor"],
        "scene": context["scene"],
        "split": str(split),
        "stage": str(int(stage)),
        "sample_role": str(sample_role),
        "augmentation_operation": str(augmentation_operation or "unknown"),
        "metadata_source": context["metadata_source"],
    }


def write_context_index(rows: Iterable[Mapping[str, Any]], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CONTEXT_INDEX_FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: str(row.get(field, "")) for field in CONTEXT_INDEX_FIELDS})
    return path.resolve()


def read_context_rows(path: Path) -> list[dict[str, str]]:
    if not Path(path).is_file():
        raise FileNotFoundError(path)
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        missing = [field for field in CONTEXT_INDEX_FIELDS if field not in fieldnames]
        if missing:
            raise ValueError(f"context_index 缺少字段: {', '.join(missing)}")
        rows = list(reader)
    if not rows:
        return []
    return [{field: str(row.get(field, "")) for field in CONTEXT_INDEX_FIELDS} for row in rows]


def read_context_index(path: Path) -> dict[str, dict[str, str]]:
    """Load rows keyed by normalized materialized path for trainer lookup."""

    return {
        normalize_context_path(row["materialized_image_path"]): row
        for row in read_context_rows(path)
    }


def context_index_summary(rows: Iterable[Mapping[str, str]]) -> dict[str, Any]:
    rows = list(rows)
    sensor_counts = Counter(row.get("sensor", UNKNOWN) for row in rows)
    scene_counts = Counter(row.get("scene", UNKNOWN) for row in rows)
    return {
        "images": len(rows),
        "known_sensor_images": sum(value for key, value in sensor_counts.items() if key != UNKNOWN),
        "known_scene_images": sum(value for key, value in scene_counts.items() if key != UNKNOWN),
        "sensor_counts": dict(sorted(sensor_counts.items())),
        "scene_counts": dict(sorted(scene_counts.items())),
        "metadata_sources": dict(sorted(Counter(row.get("metadata_source", UNKNOWN) for row in rows).items())),
        "fields": list(CONTEXT_INDEX_FIELDS),
    }


__all__ = [
    "CONTEXT_INDEX_FIELDS",
    "SCENE_NAMES",
    "SENSOR_NAMES",
    "UNKNOWN",
    "build_context_row",
    "context_index_summary",
    "normalize_context_path",
    "parse_context_from_filename",
    "read_context_index",
    "read_context_rows",
    "resolve_context_metadata",
    "write_context_index",
]
