"""将模型结构化结果转换为固定中文描述和 CSV 行的轻量工具。"""

from __future__ import annotations

import csv
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence


SCENE_CN = {
    "air": "天空",
    "sea": "海洋",
    "urban": "城市",
    "forest": "森林",
    "uncertain": "不确定",
}
MODALITY_CN = {"visible": "可见光", "ir": "红外", "sar": "SAR"}
TARGET_CN = {
    "soldier": "士兵",
    "small_aircraft": "战斗机/小型飞机",
    "warship": "轮船/舰船",
    "tank": "坦克",
    "patrol_boat": "巡逻艇",
    "armored_vehicle": "装甲车辆",
    "unknown": "未知目标",
}
CSV_FIELDS = (
    "image", "scene", "modality", "target_type_count", "target_total_count",
    "target_details", "max_confidence", "description",
)


def _field(value: object, *names: str, default: object = None) -> object:
    if isinstance(value, Mapping):
        for name in names:
            if name in value:
                return value[name]
        return default
    for name in names:
        if hasattr(value, name):
            return getattr(value, name)
    return default


def _confidence(value: object) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def build_image_summary(
    detections: Iterable[object],
    *,
    scene_label: Optional[str] = None,
    modality_label: Optional[str] = None,
    include_modality: bool = False,
) -> dict[str, Any]:
    """按类别统计检测框，并生成不依赖语言模型的固定中文描述。"""

    grouped: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
    for detection in detections:
        class_name = str(_field(detection, "class_name", "name", default="unknown") or "unknown")
        label_cn = str(_field(detection, "label_cn", default=None) or TARGET_CN.get(class_name, class_name))
        confidence = _confidence(_field(detection, "confidence", default=0.0))
        item = grouped.setdefault(
            class_name,
            {"class_name": class_name, "label_cn": label_cn, "count": 0, "confidences": []},
        )
        item["count"] += 1
        item["confidences"].append(confidence)

    targets = []
    all_confidences = []
    for item in grouped.values():
        confidences = [round(float(value), 6) for value in item["confidences"]]
        all_confidences.extend(confidences)
        targets.append(
            {
                "class_name": item["class_name"],
                "label_cn": item["label_cn"],
                "count": int(item["count"]),
                "confidences": confidences,
                "max_confidence": round(max(confidences), 6),
                "mean_confidence": round(sum(confidences) / len(confidences), 6),
            }
        )

    scene_name = SCENE_CN.get(scene_label or "", scene_label or "未提供")
    modality_name = MODALITY_CN.get(modality_label or "", modality_label or "")
    prefix = "图像场景分类：%s。" % scene_name
    if include_modality and modality_name:
        prefix += "图像模态：%s。" % modality_name
    if not targets:
        description, target_details = prefix + "未检测到目标。", "未检测到目标"
    else:
        parts = []
        for target in targets:
            confidence_text = "、".join("%.2f" % value for value in target["confidences"])
            parts.append("%s %d 个（置信度：%s）" % (target["label_cn"], target["count"], confidence_text))
        target_details = "；".join(parts)
        description = "%s检测到 %d 类目标，共 %d 个：%s。" % (
            prefix, len(targets), sum(target["count"] for target in targets), target_details
        )
    return {
        "scene_label": scene_label,
        "scene_name": scene_name,
        "modality_label": modality_label,
        "modality_name": modality_name or None,
        "target_type_count": len(targets),
        "target_total_count": sum(target["count"] for target in targets),
        "targets": targets,
        "max_confidence": round(max(all_confidences), 6) if all_confidences else 0.0,
        "target_details": target_details,
        "description": description,
    }


def summary_csv_row(image: object, summary: Mapping[str, object]) -> dict[str, object]:
    """将单图摘要转换为可直接写入批量 CSV 的固定列。"""

    return {
        "image": str(image),
        "scene": summary.get("scene_name") or "",
        "modality": summary.get("modality_name") or "",
        "target_type_count": summary.get("target_type_count", 0),
        "target_total_count": summary.get("target_total_count", 0),
        "target_details": summary.get("target_details", ""),
        "max_confidence": summary.get("max_confidence", 0.0),
        "description": summary.get("description", ""),
    }


def write_summary_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    """以 UTF-8 BOM 写入摘要 CSV，便于 Windows Excel 直接打开。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in CSV_FIELDS})


def summarize_prediction_payload(payload: object) -> list[dict[str, Any]]:
    """汇总 310B ONNX/OM 推理 JSON；该 JSON 当前不包含场景分类。"""

    records = payload if isinstance(payload, list) else [payload]
    summaries = []
    for record in records:
        if not isinstance(record, Mapping):
            continue
        detections = record.get("detections", [])
        summary = build_image_summary(detections if isinstance(detections, list) else [])
        summaries.append({"image": str(record.get("image", "")), **summary})
    return summaries
