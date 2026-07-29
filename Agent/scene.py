from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .image_ops import extract_handcrafted_features, normalize_scores, quality_levels, quality_metrics
from .schemas import SCENE_LABELS, ProbabilityResult


FILENAME_SCENES = {
    "air": "air",
    "sky": "air",
    "sea": "sea",
    "ocean": "sea",
    "urban": "urban",
    "city": "urban",
    "forest": "forest",
    "woods": "forest",
}


def _scene_from_filename(path: Path) -> str | None:
    tokens = path.stem.lower().replace("-", "_").split("_")
    for token in tokens:
        if token in FILENAME_SCENES:
            return FILENAME_SCENES[token]
    return None


class SceneRecognizer:
    """Scene recognizer with SVM/joblib adapter and deterministic fallback."""

    def __init__(self, model_path: Path | None = None, metadata_path: Path | None = None) -> None:
        self.model_path = model_path
        self.metadata_path = metadata_path
        self._model: Any | None = None
        self._metadata: dict[str, Any] | None = None
        self.warnings: list[str] = []

    def _load(self) -> bool:
        if self._model is not None and self._metadata is not None:
            return True
        if not self.model_path or not self.metadata_path:
            return False
        if not self.model_path.is_file() or not self.metadata_path.is_file():
            return False
        try:
            import joblib

            self._model = joblib.load(self.model_path)
            self._metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            return True
        except Exception as exc:  # noqa: BLE001
            self.warnings.append(f"Scene model load failed; fallback is used: {exc}")
            self._model = None
            self._metadata = None
            return False

    def predict(self, image_path: Path, modality: str) -> ProbabilityResult:
        named_scene = _scene_from_filename(image_path)
        if self._load():
            try:
                return self._predict_with_model(image_path)
            except Exception as exc:  # noqa: BLE001
                self.warnings.append(f"Scene model inference failed; fallback is used: {exc}")
        return self._predict_with_heuristic(image_path, modality, named_scene)

    def _predict_with_model(self, image_path: Path) -> ProbabilityResult:
        import pandas as pd

        assert self._model is not None
        assert self._metadata is not None
        features = extract_handcrafted_features(image_path)
        input_features = list(self._metadata["input_features"])
        missing = [name for name in input_features if name not in features]
        if missing:
            raise ValueError(f"missing features: {', '.join(missing[:5])}")
        frame = pd.DataFrame([{name: features[name] for name in input_features}])
        probabilities_raw = self._model.predict_proba(frame)[0]
        scene_names = list(self._metadata.get("scene_names", SCENE_LABELS))
        probabilities = {
            name: round(float(value), 6)
            for name, value in zip(scene_names, probabilities_raw)
        }
        label = max(probabilities, key=probabilities.get)
        return ProbabilityResult(
            label=label,
            confidence=probabilities[label],
            probabilities=probabilities,
            source="feature_svm_model",
            details={
                "model": str(self.model_path),
                "metadata": str(self.metadata_path),
                "selected_feature_count": len(self._metadata.get("selected_features", [])),
            },
        )

    def _predict_with_heuristic(
        self, image_path: Path, modality: str, named_scene: str | None
    ) -> ProbabilityResult:
        if named_scene:
            probabilities = {name: 0.025 for name in SCENE_LABELS}
            probabilities[named_scene] = 0.925
            return ProbabilityResult(named_scene, 0.925, probabilities, "filename_rule")

        metrics = quality_metrics(image_path)
        contrast = float(metrics["contrast_std"])
        dynamic = float(metrics["dynamic_range_p01_p99"])
        sharpness = float(metrics["sharpness_gradient"])
        noise = float(metrics["high_frequency_noise"])
        color = float(metrics["colorfulness"])

        smoothness = max(0.0, 1.0 - sharpness / 16.0)
        texture = min((sharpness + noise) / 18.0, 2.0)
        scores = {
            "air": 0.35 + smoothness + max(0.0, (float(metrics["mean_gray"]) - 120.0) / 180.0),
            "sea": 0.35 + smoothness + min(dynamic / 180.0, 0.8),
            "urban": 0.35 + texture + min(contrast / 95.0, 0.9),
            "forest": 0.35 + texture + max(0.0, 1.0 - color / 25.0),
        }
        if modality == "sar":
            scores["sea"] *= 1.15
            scores["urban"] *= 1.05
        if modality == "visible":
            scores["urban"] *= 1.10
            scores["forest"] *= 1.10
        probabilities = normalize_scores(scores)
        label = max(probabilities, key=probabilities.get)
        return ProbabilityResult(
            label=label,
            confidence=round(probabilities[label], 6),
            probabilities={key: round(value, 6) for key, value in probabilities.items()},
            source="image_statistic_heuristic",
            details={key: metrics[key] for key in ("contrast_std", "sharpness_gradient", "high_frequency_noise", "colorfulness")},
        )


def build_environment_state(
    image_path: Path, modality_result: ProbabilityResult, scene_result: ProbabilityResult
) -> dict[str, Any]:
    metrics = quality_metrics(image_path)
    levels = quality_levels(metrics)
    return {
        "sensor_type": modality_result.label,
        "scene_label": scene_result.label,
        "scene_confidence": scene_result.confidence,
        **levels,
        "raw_metrics": metrics,
        "state_vector": {
            "modality_confidence": round(modality_result.confidence, 6),
            "scene_confidence": round(scene_result.confidence, 6),
            "contrast_norm": round(min(float(metrics["contrast_std"]) / 100.0, 1.0), 6),
            "clarity_norm": round(min(float(metrics["sharpness_gradient"]) / 18.0, 1.0), 6),
            "noise_norm": round(min(float(metrics["high_frequency_noise"]) / 12.0, 1.0), 6),
            "color_norm": round(min(float(metrics["colorfulness"]) / 35.0, 1.0), 6),
        },
    }
