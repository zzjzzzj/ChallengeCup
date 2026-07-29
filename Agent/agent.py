from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from .config import AgentConfig
from .detection import TargetDetector
from .image_ops import build_augmentation_plan, ensure_image, extract_handcrafted_features, quality_levels, quality_metrics
from .losses import estimate_runtime_losses
from .memory import EpisodeMemory
from .modality import ModalityRecognizer, normalize_modality
from .reasoning import (
    build_consistency_report,
    build_decision,
    describe_scene,
    resolve_final_scene,
    summarize_detection_confidence,
)
from .scene import SceneRecognizer, build_environment_state
from .schemas import AgentReport, PipelineStage, ProbabilityResult
from .target import TargetClassifier


class IntelligentRecognitionAgent:
    """End-to-end agent that follows the designed intelligent recognition flow."""

    def __init__(self, config: AgentConfig | None = None) -> None:
        self.config = config or AgentConfig()
        self.memory = EpisodeMemory(self.config.memory_path)
        self.modality_recognizer = ModalityRecognizer()
        self.scene_recognizer = SceneRecognizer(self.config.scene_model, self.config.scene_metadata)
        self.detector = TargetDetector(
            self.config.detector_model,
            self.config.class_names,
            confidence=self.config.detector_confidence,
            image_size=self.config.image_size,
            device=self.config.device,
            allow_label_fallback=self.config.allow_label_fallback,
        )
        self.target_classifier = TargetClassifier(
            self.config.target_checkpoint,
            self.config.class_names,
            device=self.config.device,
            use_scene_prior=self.config.use_scene_prior_for_unknown_targets,
        )

    def run(
        self,
        image: str | Path,
        *,
        sensor_hint: str | None = None,
        modality_images: dict[str, str | Path] | None = None,
        remember: bool | None = None,
    ) -> AgentReport:
        warnings = self.config.validate()
        stages: list[PipelineStage] = []
        primary_image = ensure_image(image)
        input_modalities = self._resolve_input_modalities(primary_image, sensor_hint, modality_images)

        started = time.perf_counter()
        metrics = quality_metrics(primary_image)
        features = extract_handcrafted_features(primary_image)
        levels = quality_levels(metrics)
        stages.append(
            PipelineStage(
                "image_processing",
                "ok",
                time.perf_counter() - started,
                "quality metrics and handcrafted features extracted",
                {"feature_count": len(features)},
            )
        )

        started = time.perf_counter()
        modality = self.modality_recognizer.predict(primary_image, sensor_hint=sensor_hint)
        stages.append(PipelineStage("modality_recognition", "ok", time.perf_counter() - started, modality.source))

        started = time.perf_counter()
        scene = self.scene_recognizer.predict(primary_image, modality.label)
        warnings.extend(self.scene_recognizer.warnings)
        stages.append(PipelineStage("scene_classification", "ok", time.perf_counter() - started, scene.source))

        environment = build_environment_state(primary_image, modality, scene)
        augmentation_plan = build_augmentation_plan(modality.label, scene.label, metrics, levels)

        started = time.perf_counter()
        detections = self.detector.detect(primary_image)
        warnings.extend(self.detector.warnings)
        stages.append(
            PipelineStage(
                "target_localization",
                "ok",
                time.perf_counter() - started,
                f"{len(detections)} target boxes",
                summarize_detection_confidence(detections),
            )
        )

        started = time.perf_counter()
        detections = self.target_classifier.refine(primary_image, detections, scene.label)
        warnings.extend(self.target_classifier.warnings)
        stages.append(
            PipelineStage(
                "target_classification",
                "ok",
                time.perf_counter() - started,
                f"{len(detections)} target labels",
                summarize_detection_confidence(detections),
            )
        )

        started = time.perf_counter()
        final_scene = resolve_final_scene(scene, modality, detections)
        consistency = build_consistency_report(scene, final_scene, detections)
        decision = build_decision(modality, final_scene, environment, detections)
        decision["description"] = describe_scene(modality, final_scene, detections, consistency)
        stages.append(
            PipelineStage(
                "scene_target_reasoning",
                "ok",
                time.perf_counter() - started,
                consistency["status"],
                {"final_scene": final_scene.label},
            )
        )

        losses = estimate_runtime_losses(modality, final_scene, detections, consistency, environment)
        preprocessing = {
            "augmentation_plan": augmentation_plan,
            "feature_summary": {
                "feature_count": len(features),
                "sample_features": {key: round(float(features[key]), 6) for key in list(features)[:8]},
            },
            "aligned_modalities": input_modalities,
        }
        memory_summary = self.memory.summary(limit=200)

        report = AgentReport(
            image=str(primary_image),
            input_modalities=input_modalities,
            modality=modality,
            scene=scene,
            final_scene=final_scene,
            environment=environment,
            preprocessing=preprocessing,
            detections=detections,
            consistency=consistency,
            decision=decision,
            losses=losses,
            memory=memory_summary,
            warnings=sorted(set(warnings)),
            stages=stages,
        )

        should_remember = self.config.remember_runs if remember is None else remember
        if should_remember:
            self.memory.append_report(report.to_dict())
            report.memory = self.memory.summary(limit=200)
        return report

    def _resolve_input_modalities(
        self,
        primary_image: Path,
        sensor_hint: str | None,
        modality_images: dict[str, str | Path] | None,
    ) -> dict[str, str]:
        resolved: dict[str, str] = {}
        for key, value in (modality_images or {}).items():
            modality = normalize_modality(key)
            if modality and value:
                resolved[modality] = str(ensure_image(value))
        hinted = normalize_modality(sensor_hint)
        if hinted:
            resolved.setdefault(hinted, str(primary_image))
        if not resolved:
            predicted = self.modality_recognizer.predict(primary_image)
            resolved[predicted.label] = str(primary_image)
        return resolved


def run_single_image(
    image: str | Path,
    config: AgentConfig | None = None,
    sensor_hint: str | None = None,
    modality_images: dict[str, str | Path] | None = None,
    remember: bool | None = None,
) -> dict[str, Any]:
    agent = IntelligentRecognitionAgent(config)
    return agent.run(
        image,
        sensor_hint=sensor_hint,
        modality_images=modality_images,
        remember=remember,
    ).to_dict()
