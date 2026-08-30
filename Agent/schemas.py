from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from scene_recognition.detector_module import ALL_CLASS_NAMES


SCENE_LABELS = ("air", "sea", "urban", "forest")
MODALITY_LABELS = ("visible", "ir", "sar")
TARGET_LABELS = tuple(ALL_CLASS_NAMES)

SCENE_CN = {
    "air": "天空",
    "sea": "海洋",
    "urban": "城市",
    "forest": "森林",
    "uncertain": "不确定",
}

MODALITY_CN = {
    "visible": "可见光",
    "ir": "红外",
    "sar": "SAR",
    "unknown": "未知模态",
}

TARGET_CN = {
    "soldier": "士兵",
    "small_aircraft": "战斗机/小型飞机",
    "warship": "轮船/舰船",
    "tank": "坦克",
    "patrol_boat": "巡逻艇",
    "armored_vehicle": "装甲车辆",
    "unknown": "未知目标",
}


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:  # noqa: BLE001
            pass
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


@dataclass
class ProbabilityResult:
    label: str
    confidence: float
    probabilities: dict[str, float]
    source: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass
class DetectionBox:
    x_center: float
    y_center: float
    width: float
    height: float
    class_id: int | None = None
    class_name: str = "unknown"
    confidence: float = 0.0
    source: str = "unknown"
    track_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    @property
    def xyxy_norm(self) -> tuple[float, float, float, float]:
        half_w = self.width / 2.0
        half_h = self.height / 2.0
        return (
            self.x_center - half_w,
            self.y_center - half_h,
            self.x_center + half_w,
            self.y_center + half_h,
        )

    def xyxy_pixels(
        self, image_width: int, image_height: int, padding_ratio: float = 0.0
    ) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = self.xyxy_norm
        pad_x = self.width * padding_ratio
        pad_y = self.height * padding_ratio
        x1 = max(0.0, x1 - pad_x)
        y1 = max(0.0, y1 - pad_y)
        x2 = min(1.0, x2 + pad_x)
        y2 = min(1.0, y2 + pad_y)
        return (
            int(round(x1 * image_width)),
            int(round(y1 * image_height)),
            int(round(x2 * image_width)),
            int(round(y2 * image_height)),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["area"] = round(self.area, 8)
        data["xyxy_norm"] = [round(v, 8) for v in self.xyxy_norm]
        data["label_cn"] = TARGET_CN.get(self.class_name, self.class_name)
        return _jsonable(data)


@dataclass
class PipelineStage:
    name: str
    status: str
    seconds: float
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass
class AgentReport:
    image: str
    input_modalities: dict[str, str]
    modality: ProbabilityResult
    scene: ProbabilityResult
    final_scene: ProbabilityResult
    environment: dict[str, Any]
    preprocessing: dict[str, Any]
    detections: list[DetectionBox]
    consistency: dict[str, Any]
    decision: dict[str, Any]
    losses: dict[str, Any]
    memory: dict[str, Any]
    warnings: list[str] = field(default_factory=list)
    stages: list[PipelineStage] = field(default_factory=list)
    sparse_moe: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "image": self.image,
            "input_modalities": _jsonable(self.input_modalities),
            "modality": self.modality.to_dict(),
            "scene": self.scene.to_dict(),
            "final_scene": self.final_scene.to_dict(),
            "environment": _jsonable(self.environment),
            "preprocessing": _jsonable(self.preprocessing),
            "detections": [box.to_dict() for box in self.detections],
            "consistency": _jsonable(self.consistency),
            "decision": _jsonable(self.decision),
            "losses": _jsonable(self.losses),
            "memory": _jsonable(self.memory),
            "warnings": list(self.warnings),
            "stages": [stage.to_dict() for stage in self.stages],
            "sparse_moe": _jsonable(self.sparse_moe),
        }
