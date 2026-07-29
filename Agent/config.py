from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .schemas import TARGET_LABELS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENE_RUN = PROJECT_ROOT / "scene_recognition" / "runs" / "feature_baseline"
DEFAULT_MEMORY = PROJECT_ROOT / "Agent" / "artifacts" / "agent_memory.jsonl"


def _existing(path: Path) -> Path | None:
    return path if path.is_file() else None


def _path(value: str | Path | None) -> Path | None:
    if value is None or str(value).strip() == "":
        return None
    return Path(value).expanduser().resolve()


@dataclass
class AgentConfig:
    """Runtime configuration for the recognition agent.

    All model paths are optional. When a model is unavailable the agent falls
    back to deterministic rules so the full pipeline can still be exercised.
    """

    scene_model: Path | None = field(
        default_factory=lambda: _existing(DEFAULT_SCENE_RUN / "scene_feature_svm.joblib")
    )
    scene_metadata: Path | None = field(
        default_factory=lambda: _existing(DEFAULT_SCENE_RUN / "model_metadata.json")
    )
    detector_model: Path | None = None
    target_checkpoint: Path | None = None
    memory_path: Path = field(default_factory=lambda: DEFAULT_MEMORY)
    class_names: list[str] = field(default_factory=lambda: list(TARGET_LABELS))
    scene_threshold: float = 0.45
    detector_confidence: float = 0.25
    image_size: int = 640
    device: str = "auto"
    allow_label_fallback: bool = True
    use_scene_prior_for_unknown_targets: bool = True
    remember_runs: bool = True

    @classmethod
    def from_values(
        cls,
        *,
        scene_model: str | Path | None = None,
        scene_metadata: str | Path | None = None,
        detector_model: str | Path | None = None,
        target_checkpoint: str | Path | None = None,
        memory_path: str | Path | None = None,
        scene_threshold: float | None = None,
        detector_confidence: float | None = None,
        image_size: int | None = None,
        device: str | None = None,
        allow_label_fallback: bool | None = None,
        remember_runs: bool | None = None,
    ) -> "AgentConfig":
        config = cls()
        if scene_model is not None:
            config.scene_model = _path(scene_model)
        if scene_metadata is not None:
            config.scene_metadata = _path(scene_metadata)
        if detector_model is not None:
            config.detector_model = _path(detector_model)
        if target_checkpoint is not None:
            config.target_checkpoint = _path(target_checkpoint)
        if memory_path is not None:
            config.memory_path = _path(memory_path) or config.memory_path
        if scene_threshold is not None:
            config.scene_threshold = scene_threshold
        if detector_confidence is not None:
            config.detector_confidence = detector_confidence
        if image_size is not None:
            config.image_size = image_size
        if device is not None:
            config.device = device
        if allow_label_fallback is not None:
            config.allow_label_fallback = allow_label_fallback
        if remember_runs is not None:
            config.remember_runs = remember_runs
        return config

    def validate(self) -> list[str]:
        warnings: list[str] = []
        optional_paths = {
            "scene_model": self.scene_model,
            "scene_metadata": self.scene_metadata,
            "detector_model": self.detector_model,
            "target_checkpoint": self.target_checkpoint,
        }
        for name, path in optional_paths.items():
            if path is not None and not path.is_file():
                warnings.append(f"{name} does not exist: {path}")
        if self.scene_model and not self.scene_metadata:
            warnings.append("scene_model is set but scene_metadata is missing; scene fallback may be used.")
        if not 0.0 <= self.scene_threshold <= 1.0:
            warnings.append("scene_threshold should be in [0, 1].")
        if not 0.0 <= self.detector_confidence <= 1.0:
            warnings.append("detector_confidence should be in [0, 1].")
        return warnings
