from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

from scene_recognition.target_classifier_module.infer import load_crop_classifier, predict_crop_image

from .schemas import DetectionBox, TARGET_LABELS


SCENE_TARGET_PRIORS = {
    "air": {
        "small_aircraft": 0.82,
        "warship": 0.03,
        "soldier": 0.05,
        "tank": 0.04,
        "patrol_boat": 0.02,
        "armored_vehicle": 0.04,
    },
    "sea": {
        "warship": 0.50,
        "patrol_boat": 0.35,
        "small_aircraft": 0.05,
        "soldier": 0.03,
        "tank": 0.03,
        "armored_vehicle": 0.04,
    },
    "urban": {
        "soldier": 0.25,
        "tank": 0.25,
        "armored_vehicle": 0.36,
        "small_aircraft": 0.05,
        "warship": 0.04,
        "patrol_boat": 0.05,
    },
    "forest": {
        "soldier": 0.28,
        "tank": 0.24,
        "armored_vehicle": 0.36,
        "small_aircraft": 0.04,
        "warship": 0.03,
        "patrol_boat": 0.05,
    },
}


class TargetClassifier:
    """Refine detector classes using the shared crop classifier API or scene priors."""

    def __init__(
        self,
        checkpoint_path: Path | None,
        class_names: list[str] | None = None,
        device: str = "auto",
        use_scene_prior: bool = True,
    ) -> None:
        self.checkpoint_path = checkpoint_path
        self.class_names = class_names or list(TARGET_LABELS)
        self.device_name = device
        self.use_scene_prior = use_scene_prior
        self._runtime: dict[str, Any] | None = None
        self.warnings: list[str] = []

    def refine(
        self, image_path: Path, detections: list[DetectionBox], scene_label: str
    ) -> list[DetectionBox]:
        if not detections:
            return detections
        if self.checkpoint_path and self.checkpoint_path.is_file():
            try:
                return self._classify_crops(image_path, detections)
            except Exception as exc:  # noqa: BLE001
                self.warnings.append(
                    f"Target crop classifier failed; detector classes are kept: {exc}"
                )
        if self.use_scene_prior:
            return [self._fill_unknown_with_scene_prior(box, scene_label) for box in detections]
        return detections

    def _load_runtime(self) -> dict[str, Any]:
        if self._runtime is not None:
            return self._runtime
        assert self.checkpoint_path is not None
        self._runtime = load_crop_classifier(self.checkpoint_path, self.device_name)
        return self._runtime

    def _classify_crops(self, image_path: Path, detections: list[DetectionBox]) -> list[DetectionBox]:
        runtime = self._load_runtime()
        class_names = list(runtime["class_names"])
        refined: list[DetectionBox] = []
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
            width, height = image.size
            for index, box in enumerate(detections):
                # A legacy four-class crop classifier cannot classify the r2
                # classes. Preserve a confident detector label instead of
                # silently mapping it back to one of the old classes.
                if box.class_name != "unknown" and box.class_name not in class_names:
                    refined.append(box)
                    continue
                crop = image.crop(box.xyxy_pixels(width, height, padding_ratio=0.08))
                prediction = predict_crop_image(runtime, crop)
                class_id = int(prediction["predicted_id"])
                previous = {
                    "class_id": box.class_id,
                    "class_name": box.class_name,
                    "confidence": box.confidence,
                    "source": box.source,
                }
                refined.append(
                    DetectionBox(
                        x_center=box.x_center,
                        y_center=box.y_center,
                        width=box.width,
                        height=box.height,
                        class_id=class_id,
                        class_name=class_names[class_id],
                        confidence=round(float(prediction["confidence"]), 6),
                        source="target_crop_classifier",
                        track_id=box.track_id or f"crop-{index}",
                        metadata={
                            **box.metadata,
                            "previous_detection": previous,
                            "class_probabilities": {
                                name: round(float(prob), 6)
                                for name, prob in prediction["probabilities"].items()
                            },
                            "backend": "scene_recognition.target_classifier_module.infer",
                        },
                    )
                )
        return refined

    def _fill_unknown_with_scene_prior(self, box: DetectionBox, scene_label: str) -> DetectionBox:
        if box.class_name != "unknown" and box.class_id is not None:
            return box
        prior = SCENE_TARGET_PRIORS.get(scene_label, {})
        if not prior:
            return box
        class_name = max(prior, key=prior.get)
        class_id = self.class_names.index(class_name) if class_name in self.class_names else None
        return DetectionBox(
            x_center=box.x_center,
            y_center=box.y_center,
            width=box.width,
            height=box.height,
            class_id=class_id,
            class_name=class_name,
            confidence=round(float(prior[class_name]), 6),
            source="scene_prior_target_classifier",
            track_id=box.track_id,
            metadata={**box.metadata, "prior_scene": scene_label},
        )
