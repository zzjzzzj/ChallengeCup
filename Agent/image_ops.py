from __future__ import annotations

from pathlib import Path
from typing import Any

from image_processing.feature_engineering import extract_one
from image_processing.scene_runtime import (
    choose_enhancement,
    quality_levels as runtime_quality_levels,
    quality_metrics as runtime_quality_metrics,
)
from scene_recognition.detector_module.boxes import resolve_label_path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def ensure_image(path: str | Path) -> Path:
    image_path = Path(path).expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"Image does not exist: {image_path}")
    if image_path.suffix.lower() not in IMAGE_SUFFIXES:
        raise ValueError(f"Unsupported image suffix: {image_path.suffix}")
    return image_path


def quality_metrics(path: Path) -> dict[str, float | int | str]:
    """Delegate to image_processing.scene_runtime (single source of truth)."""

    return runtime_quality_metrics(path)


def quality_levels(metrics: dict[str, Any]) -> dict[str, str]:
    """Delegate to image_processing.scene_runtime quality level mapping."""

    return runtime_quality_levels(metrics)


def extract_handcrafted_features(path: Path) -> dict[str, float]:
    """Extract repository features via image_processing.feature_engineering."""

    try:
        return {key: float(value) for key, value in extract_one(path).items()}
    except Exception:  # noqa: BLE001
        metrics = quality_metrics(path)
        return {
            "int_mean": float(metrics["mean_gray"]) / 255.0,
            "int_std": float(metrics["contrast_std"]) / 255.0,
            "int_dynamic_range": float(metrics["dynamic_range_p01_p99"]) / 255.0,
            "tex_grad_mean": float(metrics["sharpness_gradient"]) / 255.0,
            "tex_edge_density": float(metrics["edge_density"]),
            "freq_high_low_ratio": float(metrics["high_frequency_noise"])
            / max(float(metrics["contrast_std"]), 1.0),
        }


def build_augmentation_plan(
    modality: str, scene: str, metrics: dict[str, Any], levels: dict[str, str]
) -> list[dict[str, Any]]:
    """Build preprocess / incremental-dataset augmentation suggestions.

    Primary enhancement comes from ``scene_runtime.choose_enhancement``;
    Agent-specific incremental-training tips are appended afterwards.
    """

    plan: list[dict[str, Any]] = []
    enhancement = choose_enhancement(modality, levels)
    enhancement_purposes = {
        "contrast_stretch": "improve weak target-background separation",
        "speckle_denoise": "reduce SAR high-frequency clutter before detection",
        "edge_preserving_denoise": "reduce high-frequency clutter before detection",
        "mild_sharpen": "recover small target boundaries",
        "none": "image quality is acceptable",
    }
    plan.append(
        {
            "name": enhancement if enhancement != "speckle_denoise" else "sar_speckle_denoise",
            "purpose": enhancement_purposes.get(enhancement, enhancement),
            "when": "preprocess",
            "source": "image_processing.scene_runtime.choose_enhancement",
        }
    )
    if modality == "ir":
        plan.append(
            {
                "name": "ir_gamma_invert_ablation",
                "purpose": "few-shot IR robustness experiment; validate before enabling online",
                "when": "incremental_dataset",
            }
        )
    if scene in {"urban", "forest"}:
        plan.append(
            {
                "name": "small_rotation_and_flip",
                "purpose": "increase samples for occlusion and complex background cases",
                "when": "incremental_dataset",
            }
        )
    return plan


def resolve_yolo_label_path(image_path: Path) -> Path:
    """Delegate to detector_module.boxes.resolve_label_path."""

    return resolve_label_path(image_path)


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def normalize_scores(scores: dict[str, float]) -> dict[str, float]:
    cleaned = {key: max(0.0, float(value)) for key, value in scores.items()}
    total = sum(cleaned.values())
    if total <= 0:
        uniform = 1.0 / max(1, len(cleaned))
        return {key: uniform for key in cleaned}
    return {key: value / total for key, value in cleaned.items()}
