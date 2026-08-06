from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class YoloBox:
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float
    confidence: float | None = None

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def xyxy(self) -> tuple[float, float, float, float]:
        half_width = self.width / 2
        half_height = self.height / 2
        return (
            self.x_center - half_width,
            self.y_center - half_height,
            self.x_center + half_width,
            self.y_center + half_height,
        )


def resolve_label_path(image_path: Path) -> Path:
    """Resolve both sibling-label and images/labels YOLO directory layouts."""

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


def parse_yolo_boxes(
    label_path: Path,
    class_count: int,
    *,
    allow_confidence: bool = False,
) -> list[YoloBox]:
    if not label_path.is_file():
        raise FileNotFoundError(f"标签文件不存在: {label_path}")
    boxes = []
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split()
        expected = {5, 6} if allow_confidence else {5}
        if len(parts) not in expected:
            expected_text = "5 or 6" if allow_confidence else "5"
            raise ValueError(
                f"{label_path}:{line_number} expected {expected_text} columns, got {len(parts)}"
            )
        try:
            class_id = int(parts[0])
            x_center, y_center, width, height = [float(value) for value in parts[1:5]]
            confidence = float(parts[5]) if len(parts) == 6 else None
        except ValueError as exc:
            raise ValueError(f"{label_path}:{line_number} contains non-numeric values") from exc
        if not 0 <= class_id < class_count:
            raise ValueError(f"{label_path}:{line_number} class id out of range: {class_id}")
        if not (0.0 <= x_center <= 1.0 and 0.0 <= y_center <= 1.0):
            raise ValueError(f"{label_path}:{line_number} center must be within [0,1]")
        if not (0.0 < width <= 1.0 and 0.0 < height <= 1.0):
            raise ValueError(f"{label_path}:{line_number} size must be within (0,1]")
        boxes.append(YoloBox(class_id, x_center, y_center, width, height, confidence))
    return boxes


def box_iou(first: YoloBox, second: YoloBox) -> float:
    first_x1, first_y1, first_x2, first_y2 = first.xyxy
    second_x1, second_y1, second_x2, second_y2 = second.xyxy
    inter_x1 = max(first_x1, second_x1)
    inter_y1 = max(first_y1, second_y1)
    inter_x2 = min(first_x2, second_x2)
    inter_y2 = min(first_y2, second_y2)
    inter_width = max(0.0, inter_x2 - inter_x1)
    inter_height = max(0.0, inter_y2 - inter_y1)
    intersection = inter_width * inter_height
    union = first.area + second.area - intersection
    return intersection / union if union > 0 else 0.0


def size_bucket(box: YoloBox) -> str:
    if box.area < 0.0025:
        return "tiny(<0.25%)"
    if box.area < 0.01:
        return "small(0.25%-1%)"
    if box.area < 0.04:
        return "medium(1%-4%)"
    return "large(>=4%)"
