from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def ensure_image(path: str | Path) -> Path:
    image_path = Path(path).expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"Image does not exist: {image_path}")
    if image_path.suffix.lower() not in IMAGE_SUFFIXES:
        raise ValueError(f"Unsupported image suffix: {image_path.suffix}")
    return image_path


def quality_metrics(path: Path) -> dict[str, float | int | str]:
    with Image.open(path) as opened:
        opened.load()
        rgb = opened.convert("RGB")
        gray_image = opened.convert("L")
        gray = np.asarray(gray_image, dtype=np.float32)
        rgb_values = np.asarray(rgb, dtype=np.float32)
        blurred = np.asarray(gray_image.filter(ImageFilter.GaussianBlur(radius=1.0)), dtype=np.float32)
        width, height = opened.size
        mode = opened.mode

    gx = np.diff(gray, axis=1)
    gy = np.diff(gray, axis=0)
    residual = gray - blurred
    q01, q05, q50, q95, q99 = np.percentile(gray, [1, 5, 50, 95, 99])
    rg = rgb_values[..., 0] - rgb_values[..., 1]
    yb = 0.5 * (rgb_values[..., 0] + rgb_values[..., 1]) - rgb_values[..., 2]
    colorfulness = math.sqrt(float(rg.std()) ** 2 + float(yb.std()) ** 2) + 0.3 * math.sqrt(
        float(rg.mean()) ** 2 + float(yb.mean()) ** 2
    )
    edge_strength = (float(np.abs(gx).mean()) + float(np.abs(gy).mean())) / 2.0
    return {
        "width": int(width),
        "height": int(height),
        "mode": str(mode),
        "mean_gray": round(float(gray.mean()), 6),
        "contrast_std": round(float(gray.std()), 6),
        "dynamic_range_p01_p99": round(float(q99 - q01), 6),
        "p05_gray": round(float(q05), 6),
        "p50_gray": round(float(q50), 6),
        "p95_gray": round(float(q95), 6),
        "sharpness_gradient": round(edge_strength, 6),
        "high_frequency_noise": round(float(np.mean(np.abs(residual))), 6),
        "edge_density": round(float((np.abs(gx).mean(axis=0).mean() + np.abs(gy).mean(axis=1).mean()) / 255.0), 6),
        "colorfulness": round(float(colorfulness), 6),
        "aspect_ratio": round(float(width / max(1, height)), 6),
    }


def level(value: float, low: float, high: float) -> str:
    if value < low:
        return "low"
    if value < high:
        return "medium"
    return "high"


def quality_levels(metrics: dict[str, Any]) -> dict[str, str]:
    return {
        "contrast_level": level(float(metrics["contrast_std"]), 30.0, 65.0),
        "dynamic_range_level": level(float(metrics["dynamic_range_p01_p99"]), 80.0, 160.0),
        "clarity_level": level(float(metrics["sharpness_gradient"]), 4.5, 12.0),
        "noise_level": level(float(metrics["high_frequency_noise"]), 2.5, 8.0),
        "color_level": level(float(metrics["colorfulness"]), 8.0, 24.0),
    }


def extract_handcrafted_features(path: Path) -> dict[str, float]:
    """Extract repository features when available, otherwise return basic features."""

    try:
        from image_processing.feature_engineering import extract_one

        return {key: float(value) for key, value in extract_one(path).items()}
    except Exception:  # noqa: BLE001
        metrics = quality_metrics(path)
        return {
            "int_mean": float(metrics["mean_gray"]) / 255.0,
            "int_std": float(metrics["contrast_std"]) / 255.0,
            "int_dynamic_range": float(metrics["dynamic_range_p01_p99"]) / 255.0,
            "tex_grad_mean": float(metrics["sharpness_gradient"]) / 255.0,
            "tex_edge_density": float(metrics["edge_density"]),
            "freq_high_low_ratio": float(metrics["high_frequency_noise"]) / max(float(metrics["contrast_std"]), 1.0),
        }


def build_augmentation_plan(
    modality: str, scene: str, metrics: dict[str, Any], levels: dict[str, str]
) -> list[dict[str, Any]]:
    plan: list[dict[str, Any]] = []
    if levels["contrast_level"] == "low" or levels["dynamic_range_level"] == "low":
        plan.append(
            {
                "name": "contrast_stretch",
                "purpose": "improve weak target-background separation",
                "when": "preprocess/incremental_dataset",
            }
        )
    if levels["noise_level"] == "high":
        plan.append(
            {
                "name": "sar_speckle_denoise" if modality == "sar" else "edge_preserving_denoise",
                "purpose": "reduce high-frequency clutter before detection",
                "when": "preprocess",
            }
        )
    if levels["clarity_level"] == "low":
        plan.append(
            {
                "name": "mild_sharpen",
                "purpose": "recover small target boundaries",
                "when": "preprocess",
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
    if not plan:
        plan.append({"name": "none", "purpose": "image quality is acceptable", "when": "preprocess"})
    return plan


def resolve_yolo_label_path(image_path: Path) -> Path:
    sibling = image_path.with_suffix(".txt")
    if sibling.is_file():
        return sibling
    parts = list(image_path.parts)
    for index in range(len(parts) - 1, -1, -1):
        if parts[index].lower() == "images":
            parts[index] = "labels"
            candidate = Path(*parts).with_suffix(".txt")
            if candidate.is_file():
                return candidate
            break
    return sibling


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def normalize_scores(scores: dict[str, float]) -> dict[str, float]:
    cleaned = {key: max(0.0, float(value)) for key, value in scores.items()}
    total = sum(cleaned.values())
    if total <= 0:
        uniform = 1.0 / max(1, len(cleaned))
        return {key: uniform for key in cleaned}
    return {key: value / total for key, value in cleaned.items()}
