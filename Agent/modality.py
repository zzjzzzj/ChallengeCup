from __future__ import annotations

from pathlib import Path

from .image_ops import normalize_scores, quality_metrics
from .schemas import MODALITY_LABELS, ProbabilityResult


FILENAME_ALIASES = {
    "ir": "ir",
    "infrared": "ir",
    "thermal": "ir",
    "sar": "sar",
    "radar": "sar",
    "vis": "visible",
    "rgb": "visible",
    "visible": "visible",
}


def normalize_modality(value: str | None) -> str | None:
    if value is None:
        return None
    key = value.strip().lower().replace("-", "_")
    return FILENAME_ALIASES.get(key, key if key in MODALITY_LABELS else None)


def _from_filename(path: Path) -> str | None:
    stem = path.stem.lower()
    tokens = [token for chunk in stem.split("__") for token in chunk.replace("-", "_").split("_")]
    for token in tokens:
        resolved = FILENAME_ALIASES.get(token)
        if resolved:
            return resolved
    return None


class ModalityRecognizer:
    """Recognize visible/IR/SAR modality from hints, names, and image statistics."""

    def predict(self, image_path: Path, sensor_hint: str | None = None) -> ProbabilityResult:
        hinted = normalize_modality(sensor_hint)
        if hinted:
            probabilities = {name: 0.01 for name in MODALITY_LABELS}
            probabilities[hinted] = 0.98
            return ProbabilityResult(hinted, probabilities[hinted], probabilities, "user_hint")

        named = _from_filename(image_path)
        if named:
            probabilities = {name: 0.03 for name in MODALITY_LABELS}
            probabilities[named] = 0.94
            return ProbabilityResult(named, probabilities[named], probabilities, "filename_rule")

        metrics = quality_metrics(image_path)
        colorfulness = float(metrics["colorfulness"])
        contrast = float(metrics["contrast_std"])
        noise = float(metrics["high_frequency_noise"])
        sharpness = float(metrics["sharpness_gradient"])

        visible_score = 0.25 + min(colorfulness / 35.0, 2.0)
        sar_score = 0.25 + min(noise / 7.0, 1.7) + min(contrast / 95.0, 1.0)
        ir_score = 0.75 + max(0.0, 1.0 - colorfulness / 20.0) + min(sharpness / 18.0, 0.7)
        if colorfulness > 22.0:
            ir_score *= 0.45
            sar_score *= 0.50
        if noise > 9.0 and colorfulness < 10.0:
            sar_score *= 1.35
        probabilities = normalize_scores(
            {
                "visible": visible_score,
                "ir": ir_score,
                "sar": sar_score,
            }
        )
        label = max(probabilities, key=probabilities.get)
        return ProbabilityResult(
            label=label,
            confidence=round(probabilities[label], 6),
            probabilities={key: round(value, 6) for key, value in probabilities.items()},
            source="image_statistic_heuristic",
            details={
                "colorfulness": colorfulness,
                "contrast_std": contrast,
                "high_frequency_noise": noise,
                "sharpness_gradient": sharpness,
            },
        )
