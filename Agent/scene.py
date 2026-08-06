from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from image_processing.scene_runtime import build_environment_and_policy, predict_scene_cnn
from scene_recognition.feature_infer import predict_scene_from_features

from .image_ops import extract_handcrafted_features, normalize_scores, quality_metrics
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
    """从项目数据集命名中解析场景。"""

    tokens = path.stem.lower().replace("-", "_").split("_")
    for token in tokens:
        if token in FILENAME_SCENES:
            return FILENAME_SCENES[token]
    return None


class SceneRecognizer:
    """场景识别适配器。

    优先级：
    1. feature SVM（image_processing / feature_infer）；
    2. 可选 CNN 场景检查点（scene_runtime.predict_scene_cnn）；
    3. 文件名 / 图像统计启发式回退。
    """

    def __init__(
        self,
        model_path: Path | None = None,
        metadata_path: Path | None = None,
        *,
        cnn_checkpoint: Path | None = None,
        scene_threshold: float = 0.45,
        calibration: dict[str, Any] | Path | None = None,
    ) -> None:
        self.model_path = model_path
        self.metadata_path = metadata_path
        self.cnn_checkpoint = cnn_checkpoint
        self.scene_threshold = scene_threshold
        self.calibration = self._load_calibration(calibration)
        self.warnings: list[str] = []

    @staticmethod
    def _load_calibration(calibration: dict[str, Any] | Path | None) -> dict[str, Any] | None:
        if calibration is None:
            return None
        if isinstance(calibration, dict):
            return calibration
        path = Path(calibration)
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def predict(self, image_path: Path, modality: str) -> ProbabilityResult:
        named_scene = _scene_from_filename(image_path)

        if self.model_path and self.metadata_path:
            try:
                return self._predict_with_feature_svm(image_path)
            except Exception as exc:  # noqa: BLE001
                self.warnings.append(f"Scene SVM inference failed; trying next backend: {exc}")

        if self.cnn_checkpoint and self.cnn_checkpoint.is_file():
            try:
                return self._predict_with_cnn(image_path)
            except Exception as exc:  # noqa: BLE001
                self.warnings.append(f"Scene CNN inference failed; fallback is used: {exc}")

        return self._predict_with_heuristic(image_path, modality, named_scene)

    def _predict_with_feature_svm(self, image_path: Path) -> ProbabilityResult:
        assert self.model_path is not None
        assert self.metadata_path is not None
        features = extract_handcrafted_features(image_path)
        result = predict_scene_from_features(
            image_path,
            self.model_path,
            self.metadata_path,
            extracted=features,
        )
        probabilities = {
            name: float(result["probabilities"].get(name, 0.0)) for name in SCENE_LABELS
        }
        for name, value in result["probabilities"].items():
            probabilities[name] = float(value)
        label = result["scene"]
        return ProbabilityResult(
            label=label,
            confidence=float(result["confidence"]),
            probabilities={key: round(value, 6) for key, value in probabilities.items()},
            source="feature_svm_model",
            details={
                "model": str(self.model_path),
                "metadata": str(self.metadata_path),
                "selected_feature_count": int(result.get("selected_feature_count", 0)),
                "backend": "scene_recognition.feature_infer.predict_scene_from_features",
            },
        )

    def _predict_with_cnn(self, image_path: Path) -> ProbabilityResult:
        assert self.cnn_checkpoint is not None
        result = predict_scene_cnn(
            image_path,
            self.cnn_checkpoint,
            scene_threshold=self.scene_threshold,
        )
        probabilities = {
            name: float(result["probabilities"].get(name, 0.0)) for name in SCENE_LABELS
        }
        for name, value in result["probabilities"].items():
            probabilities[name] = float(value)
        # Prefer raw label for downstream fusion; keep uncertain as detail.
        label = result["raw_label"] if result["label"] == "uncertain" else result["label"]
        return ProbabilityResult(
            label=label,
            confidence=float(result["confidence"]),
            probabilities={key: round(value, 6) for key, value in probabilities.items()},
            source="scene_cnn_model",
            details={
                "checkpoint": result.get("checkpoint"),
                "thresholded_label": result["label"],
                "backend": "image_processing.scene_runtime.predict_scene_cnn",
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
            details={
                key: metrics[key]
                for key in ("contrast_std", "sharpness_gradient", "high_frequency_noise", "colorfulness")
            },
        )


def build_environment_state(
    image_path: Path,
    modality_result: ProbabilityResult,
    scene_result: ProbabilityResult,
    *,
    calibration: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """构造环境状态向量，复用 image_processing.scene_runtime。"""

    bundled = build_environment_and_policy(
        image_path,
        modality_result.label,
        scene_result.label,
        scene_result.confidence,
        calibration=calibration,
    )
    environment = bundled["environment"]
    environment["state_vector"] = {
        "modality_confidence": round(modality_result.confidence, 6),
        **environment.get("state_vector", {}),
    }
    environment["policy_seed"] = bundled["decision"]
    return environment
