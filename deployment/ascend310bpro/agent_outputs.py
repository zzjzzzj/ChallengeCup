#!/usr/bin/env python3
"""Build Agent-style reports from Ascend 310B prediction JSONL files.

This module is intentionally lightweight: it only uses the Python standard
library so it can run on the board after NPU inference without importing torch,
ultralytics, or the desktop Agent package.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]

SCENE_LABELS = ("air", "sea", "urban", "forest")
ROUTED_SCENE_ORDER = ("air", "forest", "sea", "urban")
MODALITY_LABELS = ("visible", "ir", "sar")
DEFAULT_CLASS_NAMES = (
    "soldier",
    "small_aircraft",
    "warship",
    "tank",
    "patrol_boat",
    "armored_vehicle",
)

SCENE_CN = {
    "air": "天空",
    "sea": "海洋",
    "urban": "城市",
    "forest": "森林",
    "uncertain": "不确定",
    "unknown": "未知场景",
}

MODALITY_CN = {
    "visible": "可见光",
    "ir": "红外",
    "sar": "SAR",
    "unknown": "未知模态",
}

TARGET_CN = {
    "soldier": "士兵",
    "small_aircraft": "小型飞机",
    "warship": "军舰/船舶",
    "tank": "坦克",
    "patrol_boat": "巡逻艇",
    "armored_vehicle": "装甲车辆",
    "unknown": "未知目标",
}

ALLOWED_TARGETS_BY_SCENE = {
    "air": {"small_aircraft"},
    "sea": {"warship", "patrol_boat"},
    "urban": {"soldier", "tank", "armored_vehicle"},
    "forest": {"soldier", "tank", "armored_vehicle"},
}

SCENES_BY_TARGET = {
    "small_aircraft": {"air"},
    "warship": {"sea"},
    "patrol_boat": {"sea"},
    "soldier": {"urban", "forest"},
    "tank": {"urban", "forest"},
    "armored_vehicle": {"urban", "forest"},
}

SCENE_POLICY = {
    "air": {
        "detector_profile": "single_6class_air_focus",
        "confidence_threshold": 0.30,
        "priority_classes": ["small_aircraft"],
        "enhancement": "keep small high-contrast flying targets",
    },
    "sea": {
        "detector_profile": "single_6class_sea_focus",
        "confidence_threshold": 0.28,
        "priority_classes": ["warship", "patrol_boat"],
        "enhancement": "keep horizontal ship texture and wake edges",
    },
    "urban": {
        "detector_profile": "single_6class_land_focus",
        "confidence_threshold": 0.35,
        "priority_classes": ["soldier", "tank", "armored_vehicle"],
        "enhancement": "keep small land targets and hard edges",
    },
    "forest": {
        "detector_profile": "single_6class_land_focus",
        "confidence_threshold": 0.35,
        "priority_classes": ["soldier", "tank", "armored_vehicle"],
        "enhancement": "raise contrast for low-visibility land targets",
    },
    "uncertain": {
        "detector_profile": "single_6class_general",
        "confidence_threshold": 0.30,
        "priority_classes": list(DEFAULT_CLASS_NAMES),
        "enhancement": "use conservative all-class detection",
    },
}


def compact_source_json(value: Any, depth: int = 0) -> Any:
    if isinstance(value, list):
        preview: Dict[str, Any] = {"type": "list", "items": len(value)}
        if value and isinstance(value[0], dict):
            preview["first_item_keys"] = sorted(str(key) for key in value[0].keys())
        return preview
    if isinstance(value, dict):
        compact: Dict[str, Any] = {}
        for key, item in value.items():
            if isinstance(item, list):
                compact[str(key)] = compact_source_json(item, depth + 1)
            elif isinstance(item, dict) and depth < 1:
                compact[str(key)] = compact_source_json(item, depth + 1)
            elif isinstance(item, (str, int, float, bool)) or item is None:
                compact[str(key)] = item
            else:
                compact[str(key)] = {"type": type(item).__name__}
        return compact
    return value


def load_json(path: Optional[Path]) -> Dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError:
        return {}
    compact = compact_source_json(data)
    return compact if isinstance(compact, dict) else {"payload": compact}


def load_prediction_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError("predictions not found: %s" % path)
    text = path.read_text(encoding="utf-8-sig").strip()
    if not text:
        return []
    if text[0] == "[":
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError("prediction JSON array expected: %s" % path)
        return [item for item in data if isinstance(item, dict)]

    rows: List[Dict[str, Any]] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError("invalid JSONL at %s:%d: %s" % (path, line_no, exc)) from exc
        if isinstance(item, dict):
            rows.append(item)
    return rows


def read_class_names(path: Optional[Path]) -> List[str]:
    if path is None or not path.is_file():
        return list(DEFAULT_CLASS_NAMES)
    names = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    return names or list(DEFAULT_CLASS_NAMES)


def safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int = -1) -> int:
    if value is None:
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def round_float(value: Any, digits: int = 6) -> float:
    return round(safe_float(value), digits)


def tokenize_path(path: str) -> List[str]:
    lower = path.replace("\\", "/").lower()
    return [item for item in re.split(r"[^a-z0-9]+", lower) if item]


def confidence_probabilities(label: str, labels: Sequence[str], confidence: float) -> Dict[str, float]:
    if label not in labels:
        return {item: round(1.0 / len(labels), 6) for item in labels}
    confidence = max(0.0, min(1.0, confidence))
    remaining = max(0.0, 1.0 - confidence)
    other = remaining / max(1, len(labels) - 1)
    values = {item: other for item in labels}
    values[label] = confidence
    return {key: round(value, 6) for key, value in values.items()}


def normalize_scores(scores: Dict[str, float]) -> Dict[str, float]:
    total = sum(max(0.0, float(value)) for value in scores.values())
    if total <= 0.0:
        return {item: round(1.0 / len(scores), 6) for item in scores}
    return {key: round(max(0.0, float(value)) / total, 6) for key, value in scores.items()}


def probability_result(
    label: str,
    confidence: float,
    probabilities: Dict[str, float],
    source: str,
    details: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "label": label,
        "confidence": round_float(confidence),
        "probabilities": {key: round_float(value) for key, value in probabilities.items()},
        "source": source,
        "details": details or {},
    }


def infer_modality(image_path: str, default_modality: Optional[str]) -> Dict[str, Any]:
    tokens = tokenize_path(image_path)
    label = "unknown"
    confidence = 0.34
    source = "unknown"

    if any(token in {"sar", "radar"} for token in tokens):
        label, confidence, source = "sar", 0.95, "filename_rule"
    elif any(token in {"ir", "infrared", "thermal"} for token in tokens):
        label, confidence, source = "ir", 0.95, "filename_rule"
    elif any(token in {"rgb", "vis", "visible", "optical"} for token in tokens):
        label, confidence, source = "visible", 0.90, "filename_rule"
    elif default_modality:
        label, confidence, source = default_modality, 0.75, "cli_default"

    probabilities = confidence_probabilities(label, MODALITY_LABELS, confidence)
    if label == "unknown":
        probabilities = {"visible": 0.34, "ir": 0.33, "sar": 0.33}
    return probability_result(label, confidence, probabilities, source, {"tokens": tokens[:12]})


def infer_scene_from_filename(image_path: str, default_scene: Optional[str]) -> Optional[Dict[str, Any]]:
    tokens = tokenize_path(image_path)
    for label in SCENE_LABELS:
        if label in tokens:
            return probability_result(
                label,
                0.90,
                confidence_probabilities(label, SCENE_LABELS, 0.90),
                "filename_rule",
                {"tokens": tokens[:12]},
            )
    if default_scene:
        return probability_result(
            default_scene,
            0.75,
            confidence_probabilities(default_scene, SCENE_LABELS, 0.75),
            "cli_default",
            {},
        )
    return None


def detection_class_name(raw: Dict[str, Any], class_names: Sequence[str]) -> str:
    name = raw.get("class_name") or raw.get("name") or raw.get("label")
    if isinstance(name, str) and name:
        match = re.fullmatch(r"class[_-]?(\d+)", name.strip())
        if match:
            class_id = int(match.group(1))
            if 0 <= class_id < len(class_names):
                return class_names[class_id]
        return name.strip()
    class_id = safe_int(raw.get("class_id"), -1)
    if 0 <= class_id < len(class_names):
        return class_names[class_id]
    return "unknown"


def detection_confidence(raw: Dict[str, Any]) -> float:
    for key in ("confidence", "score", "conf", "probability"):
        if key in raw:
            return round_float(raw.get(key))
    return 0.0


def image_size_from_row(row: Dict[str, Any], project_root: Path) -> Optional[Tuple[float, float]]:
    raw_size = row.get("image_size") or row.get("size")
    if isinstance(raw_size, (list, tuple)) and len(raw_size) >= 2:
        width = safe_float(raw_size[0], 0.0)
        height = safe_float(raw_size[1], 0.0)
        if width > 0 and height > 0:
            return width, height

    image_value = str(row.get("image") or "")
    if not image_value:
        return None
    image_path = Path(image_value)
    if not image_path.is_absolute():
        image_path = project_root / image_path
    if not image_path.is_file():
        return None
    try:
        from PIL import Image  # type: ignore

        with Image.open(image_path) as image:
            return float(image.size[0]), float(image.size[1])
    except Exception:
        return None


def box_from_detection(raw: Dict[str, Any]) -> Tuple[List[float], List[float]]:
    xyxy = raw.get("box") or raw.get("bbox_xyxy") or raw.get("xyxy")
    if isinstance(xyxy, (list, tuple)) and len(xyxy) >= 4:
        x1, y1, x2, y2 = [safe_float(item) for item in xyxy[:4]]
        return [x1, y1, x2, y2], [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]

    xywh = raw.get("bbox_xywh") or raw.get("xywh")
    if isinstance(xywh, (list, tuple)) and len(xywh) >= 4:
        x, y, width, height = [safe_float(item) for item in xywh[:4]]
        return [x, y, x + width, y + height], [x, y, width, height]

    return [], []


def select_raw_detections(row: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], str]:
    final = row.get("final")
    if isinstance(final, dict) and isinstance(final.get("detections"), list):
        return [item for item in final["detections"] if isinstance(item, dict)], "final"
    if isinstance(row.get("detections"), list):
        return [item for item in row["detections"] if isinstance(item, dict)], "detections"
    main = row.get("main")
    if isinstance(main, dict) and isinstance(main.get("detections"), list):
        return [item for item in main["detections"] if isinstance(item, dict)], "main"
    return [], "none"


def normalize_detections(
    row: Dict[str, Any],
    class_names: Sequence[str],
    project_root: Path,
) -> List[Dict[str, Any]]:
    raw_detections, source_group = select_raw_detections(row)
    size = image_size_from_row(row, project_root)
    image_width, image_height = size if size is not None else (0.0, 0.0)

    detections: List[Dict[str, Any]] = []
    for index, raw in enumerate(raw_detections, start=1):
        class_name = detection_class_name(raw, class_names)
        class_id = safe_int(raw.get("class_id"), -1)
        confidence = detection_confidence(raw)
        xyxy, xywh = box_from_detection(raw)
        area = 0.0
        xyxy_norm: List[float] = []
        if xyxy:
            area = max(0.0, xyxy[2] - xyxy[0]) * max(0.0, xyxy[3] - xyxy[1])
            if image_width > 0 and image_height > 0:
                xyxy_norm = [
                    max(0.0, min(1.0, xyxy[0] / image_width)),
                    max(0.0, min(1.0, xyxy[1] / image_height)),
                    max(0.0, min(1.0, xyxy[2] / image_width)),
                    max(0.0, min(1.0, xyxy[3] / image_height)),
                ]
        detections.append(
            {
                "track_id": str(raw.get("track_id") or "det_%04d" % index),
                "class_id": class_id,
                "class_name": class_name,
                "label_cn": TARGET_CN.get(class_name, class_name),
                "confidence": confidence,
                "bbox_xyxy": [round_float(item, 3) for item in xyxy],
                "bbox_xywh": [round_float(item, 3) for item in xywh],
                "xyxy_norm": [round_float(item, 8) for item in xyxy_norm],
                "area": round_float(area, 3),
                "source": str(raw.get("source") or source_group),
                "branch": str(raw.get("branch") or raw.get("route") or source_group),
                "notes": list(raw.get("notes") or []),
            }
        )
    return detections


def scene_from_prediction(
    row: Dict[str, Any],
    image_path: str,
    default_scene: Optional[str],
) -> Dict[str, Any]:
    raw_scene = row.get("scene")
    if isinstance(raw_scene, dict):
        label = str(raw_scene.get("name") or raw_scene.get("scene_name") or raw_scene.get("label") or "")
        if label in SCENE_LABELS:
            confidence = safe_float(raw_scene.get("confidence"), 0.0)
            raw_scores = raw_scene.get("scores") or raw_scene.get("probabilities")
            if isinstance(raw_scores, dict):
                probabilities = {key: safe_float(raw_scores.get(key), 0.0) for key in SCENE_LABELS}
                probabilities = normalize_scores(probabilities)
            elif isinstance(raw_scores, (list, tuple)) and len(raw_scores) == len(ROUTED_SCENE_ORDER):
                probabilities = normalize_scores(
                    {scene: safe_float(raw_scores[index]) for index, scene in enumerate(ROUTED_SCENE_ORDER)}
                )
            else:
                probabilities = confidence_probabilities(label, SCENE_LABELS, confidence or 0.75)
            return probability_result(label, confidence or probabilities.get(label, 0.0), probabilities, "npu_scene_router")

    filename_scene = infer_scene_from_filename(image_path, default_scene)
    if filename_scene is not None:
        return filename_scene
    return probability_result(
        "uncertain",
        0.25,
        {"air": 0.25, "sea": 0.25, "urban": 0.25, "forest": 0.25},
        "unavailable",
    )


def target_scene_scores(detections: Sequence[Dict[str, Any]], image_path: str) -> Dict[str, float]:
    scores = {scene: 0.0 for scene in SCENE_LABELS}
    filename_tokens = set(tokenize_path(image_path))
    for detection in detections:
        class_name = str(detection.get("class_name") or "unknown")
        confidence = max(0.15, safe_float(detection.get("confidence"), 0.0))
        compatible = set(SCENES_BY_TARGET.get(class_name, set()))
        if compatible == {"urban", "forest"}:
            if "forest" in filename_tokens:
                compatible = {"forest"}
            elif "urban" in filename_tokens:
                compatible = {"urban"}
        for scene in compatible:
            scores[scene] += confidence
    return scores


def resolve_final_scene(
    scene: Dict[str, Any],
    modality: Dict[str, Any],
    detections: Sequence[Dict[str, Any]],
    image_path: str,
) -> Dict[str, Any]:
    raw_label = str(scene.get("label") or "uncertain")
    raw_probs = scene.get("probabilities") if isinstance(scene.get("probabilities"), dict) else {}
    scores = {label: safe_float(raw_probs.get(label), 0.0) * 0.72 for label in SCENE_LABELS}

    target_votes = target_scene_scores(detections, image_path)
    if detections:
        total_vote = sum(target_votes.values()) or 1.0
        for label in SCENE_LABELS:
            scores[label] += 0.24 * (target_votes[label] / total_vote)

    modality_label = str(modality.get("label") or "unknown")
    if modality_label == "sar":
        scores["sea"] += 0.02
        scores["urban"] += 0.01
    elif modality_label == "ir":
        scores["forest"] += 0.015
        scores["urban"] += 0.015

    if raw_label not in SCENE_LABELS and not detections:
        return probability_result(
            "uncertain",
            0.25,
            {"air": 0.25, "sea": 0.25, "urban": 0.25, "forest": 0.25},
            "scene_target_consistency_fusion",
            {"raw_scene": raw_label, "target_vote": target_votes},
        )

    probabilities = normalize_scores(scores)
    label = max(probabilities, key=probabilities.get)
    invalid_after = invalid_combinations(label, detections)
    return probability_result(
        label,
        probabilities[label],
        probabilities,
        "scene_target_consistency_fusion",
        {
            "raw_scene": raw_label,
            "target_vote": {key: round_float(value) for key, value in target_votes.items()},
            "invalid_after_fusion": len(invalid_after),
        },
    )


def invalid_combinations(scene_label: str, detections: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    allowed = ALLOWED_TARGETS_BY_SCENE.get(scene_label)
    if not allowed:
        return []
    invalid: List[Dict[str, Any]] = []
    for detection in detections:
        class_name = str(detection.get("class_name") or "unknown")
        if class_name == "unknown" or class_name in allowed:
            continue
        invalid.append(
            {
                "track_id": detection.get("track_id"),
                "scene": scene_label,
                "scene_cn": SCENE_CN.get(scene_label, scene_label),
                "target": class_name,
                "target_cn": TARGET_CN.get(class_name, class_name),
                "confidence": detection.get("confidence", 0.0),
                "reason": "目标类别与当前场景先验不匹配",
            }
        )
    return invalid


def build_consistency_report(
    scene: Dict[str, Any],
    final_scene: Dict[str, Any],
    detections: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    original_invalid = invalid_combinations(str(scene.get("label") or "uncertain"), detections)
    final_invalid = invalid_combinations(str(final_scene.get("label") or "uncertain"), detections)
    target_counts = Counter(str(item.get("class_name") or "unknown") for item in detections)
    status = "consistent"
    if final_invalid:
        status = "invalid_combination"
    elif original_invalid:
        status = "repaired_by_target_scene_fusion"
    return {
        "status": status,
        "target_counts": dict(sorted(target_counts.items())),
        "original_invalid_count": len(original_invalid),
        "final_invalid_count": len(final_invalid),
        "original_invalid": original_invalid,
        "final_invalid": final_invalid,
        "rule": "场景与目标类别按数据集真实组合做一致性检查；冲突会进入运行损失代理值。",
    }


def active_experts(modality_label: str, detections: Sequence[Dict[str, Any]], final_scene: str) -> List[str]:
    targets = sorted({str(item.get("class_name")) for item in detections if item.get("class_name") != "unknown"})
    if not targets:
        targets = list(SCENE_POLICY.get(final_scene, SCENE_POLICY["uncertain"])["priority_classes"])
    label = modality_label if modality_label in MODALITY_LABELS else "unknown"
    return ["%s_%s_expert" % (label, target) for target in targets] + ["cross_modal_adapter"]


def build_decision(
    modality: Dict[str, Any],
    final_scene: Dict[str, Any],
    detections: Sequence[Dict[str, Any]],
    row: Dict[str, Any],
) -> Dict[str, Any]:
    scene_label = str(final_scene.get("label") or "uncertain")
    modality_label = str(modality.get("label") or "unknown")
    policy = dict(SCENE_POLICY.get(scene_label, SCENE_POLICY["uncertain"]))
    detector = row.get("detector") if isinstance(row.get("detector"), dict) else {}
    route = str(detector.get("route") or "single")
    selected_targets = sorted({str(item.get("class_name")) for item in detections if item.get("class_name") != "unknown"})
    priority_classes = sorted(set(policy["priority_classes"]) | set(selected_targets))
    threshold = safe_float(policy.get("confidence_threshold"), 0.30)
    if safe_float(final_scene.get("confidence"), 0.0) < 0.55:
        threshold = max(0.20, threshold - 0.05)
    return {
        "detector_profile": policy["detector_profile"],
        "confidence_threshold": round_float(threshold, 4),
        "priority_classes": priority_classes,
        "route": route,
        "route_reason": detector.get("route_reason") or "single_6class_best_model",
        "enhancement": policy["enhancement"],
        "expert_routing": {
            "active_experts": active_experts(modality_label, detections, scene_label),
            "adapter": "cross_modal_adapter",
            "note": "部署侧记录专家选择建议；实际 NPU 推理由当前 OM 模型完成。",
        },
        "model_management": {
            "edge_target": "Ascend 310B / 310B4",
            "preferred_export": "ONNX -> ATC -> OM",
            "load_strategy": "优先复用同尺寸同 SOC 的 .om 缓存文件。",
        },
    }


def build_image_summary(
    detections: Sequence[Dict[str, Any]],
    scene_label: str,
    modality_label: str,
    include_modality: bool,
) -> Dict[str, Any]:
    grouped: Dict[str, List[float]] = {}
    for detection in detections:
        class_name = str(detection.get("class_name") or "unknown")
        grouped.setdefault(class_name, []).append(safe_float(detection.get("confidence"), 0.0))

    targets = []
    for class_name, confidences in grouped.items():
        confidences = sorted(confidences, reverse=True)
        targets.append(
            {
                "label": class_name,
                "label_cn": TARGET_CN.get(class_name, class_name),
                "count": len(confidences),
                "max_confidence": round_float(confidences[0] if confidences else 0.0),
                "mean_confidence": round_float(sum(confidences) / max(1, len(confidences))),
                "confidences": [round_float(item) for item in confidences],
            }
        )
    targets.sort(key=lambda item: (-int(item["count"]), -float(item["max_confidence"]), str(item["label"])))

    all_confidences = [safe_float(item.get("confidence"), 0.0) for item in detections]
    target_details = "；".join(
        "%s %d 个（置信度：%s）"
        % (
            item["label_cn"],
            item["count"],
            "、".join("%.2f" % value for value in item["confidences"][:5]),
        )
        for item in targets
    )
    parts = []
    if include_modality:
        parts.append("图像模态：%s。" % MODALITY_CN.get(modality_label, modality_label))
    parts.append("图像场景分类：%s。" % SCENE_CN.get(scene_label, scene_label))
    if detections:
        parts.append(
            "检测到 %d 个目标，包含 %d 类：%s。"
            % (len(detections), len(targets), target_details)
        )
    else:
        parts.append("未检测到目标。")
    return {
        "target_type_count": len(targets),
        "target_total_count": len(detections),
        "targets": targets,
        "target_details": target_details,
        "max_confidence": round_float(max(all_confidences) if all_confidences else 0.0),
        "mean_confidence": round_float(sum(all_confidences) / max(1, len(all_confidences))),
        "description": "".join(parts),
    }


def estimate_runtime_losses(
    modality: Dict[str, Any],
    final_scene: Dict[str, Any],
    detections: Sequence[Dict[str, Any]],
    consistency: Dict[str, Any],
) -> Dict[str, Any]:
    confidences = [safe_float(item.get("confidence"), 0.0) for item in detections]
    known_confidences = [
        safe_float(item.get("confidence"), 0.0)
        for item in detections
        if item.get("class_name") and item.get("class_name") != "unknown"
    ]
    mean_det_conf = sum(confidences) / len(confidences) if confidences else 0.0
    mean_cls_conf = sum(known_confidences) / len(known_confidences) if known_confidences else 0.0
    invalid_rate = safe_float(consistency.get("final_invalid_count"), 0.0) / max(1, len(detections))
    components = {
        "L_moti": round_float(1.0 - safe_float(modality.get("confidence"), 0.0)),
        "L_env": round_float(1.0 - safe_float(final_scene.get("confidence"), 0.0)),
        "L_box": round_float(1.0 - mean_det_conf if detections else 0.0),
        "L_cls": round_float(1.0 - mean_cls_conf if detections else 0.0),
        "L_detail": 0.0,
        "L_proto": round_float(invalid_rate),
    }
    weights = {"modality": 0.30, "scene": 0.60, "box": 1.00, "cls": 1.00, "detail": 0.20, "proto": 0.40}
    total = (
        weights["modality"] * components["L_moti"]
        + weights["scene"] * components["L_env"]
        + weights["box"] * components["L_box"]
        + weights["cls"] * components["L_cls"]
        + weights["detail"] * components["L_detail"]
        + weights["proto"] * components["L_proto"]
    )
    return {
        "type": "runtime_proxy",
        "formula": "L = w_moti*L_moti + w_env*L_env + w_box*L_box + w_cls*L_cls + w_detail*L_detail + w_proto*L_proto",
        "components": components,
        "weights": weights,
        "total": round_float(total),
        "note": "这是推理阶段的置信度/一致性代理损失；训练时仍以真实训练损失为准。",
    }


def build_stages(row: Dict[str, Any], scene: Dict[str, Any], detections: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    stages = []
    raw_scene = row.get("scene") if isinstance(row.get("scene"), dict) else {}
    detector = row.get("detector") if isinstance(row.get("detector"), dict) else {}
    if raw_scene:
        stages.append(
            {
                "name": "scene_classification",
                "status": "ok",
                "seconds": round_float(safe_float(raw_scene.get("elapsed_ms")) / 1000.0, 6),
                "message": str(scene.get("source")),
                "details": {"scene": scene.get("label"), "confidence": scene.get("confidence")},
            }
        )
    else:
        stages.append(
            {
                "name": "scene_classification",
                "status": "inferred",
                "seconds": 0.0,
                "message": str(scene.get("source")),
                "details": {"scene": scene.get("label"), "confidence": scene.get("confidence")},
            }
        )
    stages.append(
        {
            "name": "target_detection",
            "status": "ok",
            "seconds": round_float(safe_float(detector.get("elapsed_ms")) / 1000.0, 6),
            "message": "%d target boxes" % len(detections),
            "details": {
                "route": detector.get("route") or "single",
                "input_size": detector.get("input_size"),
                "output_count": detector.get("output_count"),
            },
        }
    )
    stages.append(
        {
            "name": "scene_target_reasoning",
            "status": "ok",
            "seconds": 0.0,
            "message": "scene-target consistency checked",
            "details": {"detection_count": len(detections)},
        }
    )
    stages.append(
        {
            "name": "agent_report_formatting",
            "status": "ok",
            "seconds": 0.0,
            "message": "Agent-compatible JSON and CSV output generated",
            "details": {},
        }
    )
    return stages


def memory_summary(path: Optional[Path], limit: int = 200) -> Dict[str, Any]:
    if path is None or not path.is_file():
        return {
            "memory_path": str(path) if path else "",
            "recent_records": 0,
            "recent_reports": 0,
            "recent_feedback": 0,
            "scene_counts": {},
            "modality_counts": {},
            "target_counts": {},
        }
    rows = []
    for line in path.read_text(encoding="utf-8-sig").splitlines()[-limit:]:
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    report_rows = [item for item in rows if item.get("type") == "agent_report"]
    feedback_rows = [item for item in rows if item.get("type") == "feedback"]
    return {
        "memory_path": str(path),
        "recent_records": len(rows),
        "recent_reports": len(report_rows),
        "recent_feedback": len(feedback_rows),
        "scene_counts": dict(Counter(item.get("scene", "unknown") for item in report_rows)),
        "modality_counts": dict(Counter(item.get("modality", "unknown") for item in report_rows)),
        "target_counts": dict(
            Counter(
                target.get("class_name", "unknown")
                for row in report_rows
                for target in row.get("targets", [])
            )
        ),
        "usage": "反馈记录可作为后续增量训练的样本筛选和回放依据。",
    }


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_report(
    row: Dict[str, Any],
    class_names: Sequence[str],
    project_root: Path,
    default_modality: Optional[str],
    default_scene: Optional[str],
    include_modality: bool,
    memory_path: Optional[Path],
) -> Dict[str, Any]:
    image_path = str(row.get("image") or "")
    detections = normalize_detections(row, class_names, project_root)
    modality = infer_modality(image_path, default_modality)
    scene = scene_from_prediction(row, image_path, default_scene)
    final_scene = resolve_final_scene(scene, modality, detections, image_path)
    consistency = build_consistency_report(scene, final_scene, detections)
    output_summary = build_image_summary(
        detections,
        str(final_scene.get("label") or "uncertain"),
        str(modality.get("label") or "unknown"),
        include_modality,
    )
    decision = build_decision(modality, final_scene, detections, row)
    decision["description"] = output_summary["description"]
    losses = estimate_runtime_losses(modality, final_scene, detections, consistency)
    return {
        "image": image_path,
        "image_size": row.get("image_size") or row.get("size"),
        "input_modalities": {str(modality["label"]): image_path} if modality.get("label") != "unknown" else {},
        "modality": modality,
        "scene": scene,
        "final_scene": final_scene,
        "environment": {
            "source": "prediction_metadata",
            "noise_level": "unknown",
            "clarity_level": "unknown",
            "target_count": len(detections),
            "route": (row.get("detector") or {}).get("route") if isinstance(row.get("detector"), dict) else "single",
        },
        "preprocessing": {
            "augmentation_plan": {
                "status": "offline_dataset_augmentation",
                "note": "训练数据增广由 run_augment_yolo.sh / run_full_pipeline.sh 完成，推理阶段不改写原图。",
            },
            "aligned_modalities": {str(modality["label"]): image_path} if modality.get("label") != "unknown" else {},
        },
        "detections": detections,
        "output_summary": output_summary,
        "consistency": consistency,
        "decision": decision,
        "losses": losses,
        "memory": {
            "enabled": memory_path is not None,
            "memory_path": str(memory_path) if memory_path else "",
        },
        "warnings": [],
        "stages": build_stages(row, scene, detections),
        "sparse_moe": {
            "deployment_note": "若使用 class-il-yolo --sparse-moe 训练得到新 best.pt，请重新 export ONNX 后再由 ATC 转 OM。",
        },
        "source_prediction": row,
    }


def report_filename(image_path: str, index: int) -> str:
    stem = Path(image_path).stem or "image"
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("_") or "image"
    return "%s_%06d.json" % (safe[:80], index)


def write_summary_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image",
        "report",
        "modality",
        "scene",
        "route",
        "detector_ms",
        "scene_ms",
        "target_count",
        "target_type_count",
        "target_details",
        "max_confidence",
        "mean_confidence",
        "description",
        "consistency",
        "loss_total",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def memory_record_from_report(report: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "type": "agent_report",
        "image": report["image"],
        "modality": report["modality"]["label"],
        "scene": report["final_scene"]["label"],
        "targets": [
            {
                "class_name": item["class_name"],
                "confidence": item["confidence"],
                "xyxy_norm": item.get("xyxy_norm", []),
            }
            for item in report.get("detections", [])
        ],
        "consistency_status": report.get("consistency", {}).get("status"),
        "loss_total": report.get("losses", {}).get("total"),
    }


def format_reports(args: argparse.Namespace) -> int:
    class_names = read_class_names(args.classes)
    rows = load_prediction_rows(args.predictions)
    if args.limit is not None:
        rows = rows[: args.limit]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = args.output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    memory_path = None if args.no_memory else args.memory
    if memory_path and args.rewrite_memory and memory_path.exists():
        memory_path.unlink()

    csv_rows: List[Dict[str, Any]] = []
    class_counts: Counter = Counter()
    scene_counts: Counter = Counter()
    modality_counts: Counter = Counter()
    consistency_counts: Counter = Counter()

    for index, row in enumerate(rows, start=1):
        report = build_report(
            row,
            class_names,
            args.project_root,
            args.default_modality,
            args.default_scene,
            args.include_modality,
            memory_path,
        )
        report_path = reports_dir / report_filename(str(report["image"]), index)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if memory_path:
            append_jsonl(memory_path, memory_record_from_report(report))

        detector = row.get("detector") if isinstance(row.get("detector"), dict) else {}
        scene = row.get("scene") if isinstance(row.get("scene"), dict) else {}
        summary = report["output_summary"]
        csv_rows.append(
            {
                "image": report["image"],
                "report": str(report_path),
                "modality": report["modality"]["label"],
                "scene": report["final_scene"]["label"],
                "route": detector.get("route") or report.get("environment", {}).get("route") or "single",
                "detector_ms": detector.get("elapsed_ms", ""),
                "scene_ms": scene.get("elapsed_ms", ""),
                "target_count": summary["target_total_count"],
                "target_type_count": summary["target_type_count"],
                "target_details": summary["target_details"],
                "max_confidence": summary["max_confidence"],
                "mean_confidence": summary["mean_confidence"],
                "description": summary["description"],
                "consistency": report["consistency"]["status"],
                "loss_total": report["losses"]["total"],
            }
        )
        for detection in report.get("detections", []):
            class_counts[detection.get("class_name", "unknown")] += 1
        scene_counts[report["final_scene"]["label"]] += 1
        modality_counts[report["modality"]["label"]] += 1
        consistency_counts[report["consistency"]["status"]] += 1

    csv_path = args.output_dir / "batch_summary.csv"
    write_summary_csv(csv_path, csv_rows)

    source_summary = load_json(args.summary)
    agent_summary = {
        "images": len(rows),
        "total_detections": int(sum(class_counts.values())),
        "class_counts": dict(sorted(class_counts.items())),
        "scene_counts": dict(sorted(scene_counts.items())),
        "modality_counts": dict(sorted(modality_counts.items())),
        "consistency_counts": dict(sorted(consistency_counts.items())),
        "source_predictions": str(args.predictions),
        "source_summary": str(args.summary) if args.summary else "",
        "source_inference_summary": source_summary,
        "report_dir": str(reports_dir),
        "batch_summary_csv": str(csv_path),
        "memory": memory_summary(memory_path),
    }
    summary_path = args.output_dir / "agent_summary.json"
    summary_path.write_text(json.dumps(agent_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("[INFO] Agent reports: %s" % reports_dir, flush=True)
    print("[INFO] Agent CSV    : %s" % csv_path, flush=True)
    print("[INFO] Agent summary: %s" % summary_path, flush=True)
    if memory_path:
        print("[INFO] Agent memory : %s" % memory_path, flush=True)
    return 0


def append_feedback(args: argparse.Namespace) -> int:
    targets = [item.strip() for item in (args.targets or "").split(",") if item.strip()]
    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "type": "feedback",
        "image": str(args.image),
        "corrected_scene": args.scene,
        "corrected_modality": args.modality,
        "corrected_targets": targets,
        "note": args.note,
    }
    append_jsonl(args.memory, record)
    print(json.dumps({"status": "ok", "memory": memory_summary(args.memory)}, ensure_ascii=False, indent=2))
    return 0


def combine_losses(args: argparse.Namespace) -> int:
    total = (
        args.l_box
        + args.l_cls
        + args.l_dfl
        + args.lambda_detail * args.l_detail
        + args.lambda_scene * args.l_scene
        + args.lambda_proto * args.l_proto
        + args.lambda_moti * args.l_moti
    )
    payload = {
        "L_box": float(args.l_box),
        "L_cls": float(args.l_cls),
        "L_dfl": float(args.l_dfl),
        "L_detail": float(args.l_detail),
        "L_scene": float(args.l_scene),
        "L_proto": float(args.l_proto),
        "L_moti": float(args.l_moti),
        "lambda_detail": float(args.lambda_detail),
        "lambda_scene": float(args.lambda_scene),
        "lambda_proto": float(args.lambda_proto),
        "lambda_moti": float(args.lambda_moti),
        "L_total": round(float(total), 8),
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


def print_memory_summary(args: argparse.Namespace) -> int:
    print(json.dumps(memory_summary(args.memory), ensure_ascii=False, indent=2))
    return 0


def add_format_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--predictions", type=Path, required=True, help="predictions.jsonl or JSON array.")
    parser.add_argument("--summary", type=Path, help="Optional inference summary.json.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for reports, CSV, and agent_summary.json.")
    parser.add_argument("--classes", type=Path, default=SCRIPT_DIR / "classes_6.txt")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--memory", type=Path, help="Optional JSONL memory path.")
    parser.add_argument("--rewrite-memory", action="store_true", help="Replace existing memory before appending this run.")
    parser.add_argument("--no-memory", action="store_true")
    parser.add_argument("--default-modality", choices=MODALITY_LABELS)
    parser.add_argument("--default-scene", choices=SCENE_LABELS)
    parser.add_argument("--include-modality", action="store_true", help="Include modality text in Chinese descriptions.")
    parser.add_argument("--limit", type=int, help="Only format the first N rows.")
    parser.set_defaults(func=format_reports)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ascend 310B pro Agent output tools.")
    sub = parser.add_subparsers(dest="command")

    format_parser = sub.add_parser("format", help="build Agent-style reports from predictions.jsonl")
    add_format_args(format_parser)

    feedback_parser = sub.add_parser("feedback", help="append human correction to memory JSONL")
    feedback_parser.add_argument("--image", required=True)
    feedback_parser.add_argument("--scene", choices=SCENE_LABELS)
    feedback_parser.add_argument("--modality", choices=MODALITY_LABELS)
    feedback_parser.add_argument("--targets", help="comma-separated corrected target labels")
    feedback_parser.add_argument("--note")
    feedback_parser.add_argument("--memory", type=Path, required=True)
    feedback_parser.set_defaults(func=append_feedback)

    memory_parser = sub.add_parser("memory-summary", help="summarize memory JSONL")
    memory_parser.add_argument("--memory", type=Path, required=True)
    memory_parser.set_defaults(func=print_memory_summary)

    loss_parser = sub.add_parser("loss", help="combine training losses with Agent formula")
    loss_parser.add_argument("--l-box", type=float, required=True)
    loss_parser.add_argument("--l-cls", type=float, required=True)
    loss_parser.add_argument("--l-dfl", type=float, default=0.0)
    loss_parser.add_argument("--l-detail", type=float, default=0.0)
    loss_parser.add_argument("--l-scene", type=float, default=0.0)
    loss_parser.add_argument("--l-proto", type=float, default=0.0)
    loss_parser.add_argument("--l-moti", type=float, default=0.0)
    loss_parser.add_argument("--lambda-detail", type=float, default=0.2)
    loss_parser.add_argument("--lambda-scene", type=float, default=0.6)
    loss_parser.add_argument("--lambda-proto", type=float, default=0.4)
    loss_parser.add_argument("--lambda-moti", type=float, default=0.3)
    loss_parser.add_argument("--output", type=Path)
    loss_parser.set_defaults(func=combine_losses)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0].startswith("-"):
        argv.insert(0, "format")
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 2
    return int(args.func(args))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError) as exc:
        print("agent_outputs.py: %s" % exc, file=sys.stderr)
        raise SystemExit(1)
