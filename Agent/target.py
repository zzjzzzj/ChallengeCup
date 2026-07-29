from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

from .schemas import DetectionBox, TARGET_LABELS


SCENE_TARGET_PRIORS = {
    "air": {"small_aircraft": 0.82, "warship": 0.04, "soldier": 0.07, "tank": 0.07},
    "sea": {"warship": 0.82, "small_aircraft": 0.08, "soldier": 0.05, "tank": 0.05},
    "urban": {"soldier": 0.45, "tank": 0.42, "small_aircraft": 0.08, "warship": 0.05},
    "forest": {"soldier": 0.48, "tank": 0.40, "small_aircraft": 0.06, "warship": 0.06},
}


class TargetClassifier:
    """Refine detector classes using a crop classifier or scene priors."""

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
                self.warnings.append(f"Target crop classifier failed; detector classes are kept: {exc}")
        if self.use_scene_prior:
            return [self._fill_unknown_with_scene_prior(box, scene_label) for box in detections]
        return detections

    def _load_runtime(self) -> dict[str, Any]:
        if self._runtime is not None:
            return self._runtime
        import torch

        from scene_recognition.target_classifier_module.training import (
            build_resnet18,
            build_transforms,
            resolve_device,
        )

        assert self.checkpoint_path is not None
        try:
            checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
        class_names = list(checkpoint.get("class_names", self.class_names))
        image_size = int(checkpoint.get("image_size", 224))
        device = resolve_device(self.device_name)
        model = build_resnet18(len(class_names), pretrained=False)
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device).eval()
        _, evaluation_transform = build_transforms(image_size, augmentation="none")
        self._runtime = {
            "torch": torch,
            "model": model,
            "device": device,
            "transform": evaluation_transform,
            "class_names": class_names,
        }
        return self._runtime

    def _classify_crops(self, image_path: Path, detections: list[DetectionBox]) -> list[DetectionBox]:
        runtime = self._load_runtime()
        torch = runtime["torch"]
        model = runtime["model"]
        transform = runtime["transform"]
        device = runtime["device"]
        class_names = runtime["class_names"]
        refined: list[DetectionBox] = []
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
            width, height = image.size
            for index, box in enumerate(detections):
                crop = image.crop(box.xyxy_pixels(width, height, padding_ratio=0.08))
                tensor = transform(crop).unsqueeze(0).to(device)
                with torch.inference_mode():
                    probabilities = model(tensor).softmax(dim=1)[0].detach().cpu().tolist()
                class_id = int(max(range(len(probabilities)), key=probabilities.__getitem__))
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
                        confidence=round(float(probabilities[class_id]), 6),
                        source="target_crop_classifier",
                        track_id=box.track_id or f"crop-{index}",
                        metadata={
                            **box.metadata,
                            "previous_detection": previous,
                            "class_probabilities": {
                                name: round(float(prob), 6)
                                for name, prob in zip(class_names, probabilities)
                            },
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
