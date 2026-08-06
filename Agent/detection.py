from __future__ import annotations

from pathlib import Path
from typing import Any

from scene_recognition.detector_module.boxes import parse_yolo_boxes, resolve_label_path

from .schemas import DetectionBox, TARGET_LABELS


def _name_from_id(class_id: int, class_names: list[str]) -> str:
    if 0 <= class_id < len(class_names):
        return class_names[class_id]
    return "unknown"


class TargetDetector:
    """目标定位适配器。

    - 有 YOLO 权重时调用 ultralytics；
    - 否则用 detector_module.boxes 解析同名 YOLO 标签。
    """

    def __init__(
        self,
        model_path: Path | None,
        class_names: list[str] | None = None,
        confidence: float = 0.25,
        image_size: int = 640,
        device: str = "auto",
        allow_label_fallback: bool = True,
    ) -> None:
        self.model_path = model_path
        self.class_names = class_names or list(TARGET_LABELS)
        self.confidence = confidence
        self.image_size = image_size
        self.device = device
        self.allow_label_fallback = allow_label_fallback
        self._model: Any | None = None
        self.warnings: list[str] = []

    def detect(self, image_path: Path) -> list[DetectionBox]:
        if self.model_path and self.model_path.is_file():
            try:
                boxes = self._detect_with_yolo(image_path)
                if boxes:
                    return boxes
            except Exception as exc:  # noqa: BLE001
                self.warnings.append(f"YOLO detector failed; fallback is used: {exc}")

        if self.allow_label_fallback:
            boxes = self._detect_from_yolo_label(image_path)
            if boxes:
                return boxes
        return []

    def _load_yolo(self) -> Any:
        if self._model is None:
            from ultralytics import YOLO

            assert self.model_path is not None
            self._model = YOLO(str(self.model_path))
        return self._model

    def _detect_with_yolo(self, image_path: Path) -> list[DetectionBox]:
        model = self._load_yolo()
        device = None if self.device == "auto" else self.device
        results = model.predict(
            source=str(image_path),
            conf=self.confidence,
            imgsz=self.image_size,
            device=device,
            verbose=False,
        )
        if not results:
            return []
        result = results[0]
        names = getattr(result, "names", None) or getattr(model, "names", None) or self.class_names
        boxes = []
        for index, box in enumerate(result.boxes):
            xywhn = box.xywhn[0].detach().cpu().tolist()
            class_id = int(box.cls[0].detach().cpu().item())
            confidence = float(box.conf[0].detach().cpu().item())
            if isinstance(names, dict):
                class_name = str(names.get(class_id, _name_from_id(class_id, self.class_names)))
            else:
                class_name = (
                    str(names[class_id])
                    if 0 <= class_id < len(names)
                    else _name_from_id(class_id, self.class_names)
                )
            boxes.append(
                DetectionBox(
                    x_center=float(xywhn[0]),
                    y_center=float(xywhn[1]),
                    width=float(xywhn[2]),
                    height=float(xywhn[3]),
                    class_id=class_id,
                    class_name=class_name,
                    confidence=round(confidence, 6),
                    source="yolo_model",
                    track_id=f"det-{index}",
                )
            )
        return boxes

    def _detect_from_yolo_label(self, image_path: Path) -> list[DetectionBox]:
        label_path = resolve_label_path(image_path)
        if not label_path.is_file():
            self.warnings.append(
                "No detector model or sidecar YOLO label was found; target boxes are empty."
            )
            return []
        parsed = parse_yolo_boxes(label_path, len(self.class_names), allow_confidence=True)
        boxes: list[DetectionBox] = []
        for index, box in enumerate(parsed, start=1):
            boxes.append(
                DetectionBox(
                    x_center=box.x_center,
                    y_center=box.y_center,
                    width=box.width,
                    height=box.height,
                    class_id=box.class_id,
                    class_name=_name_from_id(box.class_id, self.class_names),
                    confidence=round(float(box.confidence if box.confidence is not None else 1.0), 6),
                    source="sidecar_yolo_label",
                    track_id=f"label-{index}",
                    metadata={
                        "label_path": str(label_path),
                        "backend": "scene_recognition.detector_module.boxes",
                    },
                )
            )
        return boxes
