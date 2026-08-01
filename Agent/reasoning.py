from __future__ import annotations

from statistics import mean
from typing import Any

from .image_ops import normalize_scores
from .schemas import (
    MODALITY_CN,
    SCENE_CN,
    SCENE_LABELS,
    TARGET_CN,
    DetectionBox,
    ProbabilityResult,
)


# 场景-目标先验约束。
# 这组规则来自当前数据集和流程图中的“唯一确定场景”设计：
# 天空主要对应小型飞机，海洋主要对应舰船，城市/森林主要对应士兵和坦克。
# 它不是替代模型的硬分类器，而是在模型输出之后做一致性校验和冲突提示。
ALLOWED_TARGETS_BY_SCENE = {
    "air": {"small_aircraft"},
    "sea": {"warship"},
    "urban": {"soldier", "tank"},
    "forest": {"soldier", "tank"},
}

# 反向索引：由目标类别反推可能场景。
# resolve_final_scene会用它给场景概率加权投票，解决场景模型置信度不足时
# “目标结果可以帮助确定场景”的问题。
SCENES_BY_TARGET = {
    "small_aircraft": {"air"},
    "warship": {"sea"},
    "soldier": {"urban", "forest"},
    "tank": {"urban", "forest"},
}

# 场景决策策略表。
# 每个场景会对应一组检测器配置、优先关注目标、模态权重和特征权重。
# 这些内容会进入最终报告的decision字段，可直接解释为“智能体下发的指令”。
SCENE_POLICY = {
    "air": {
        "detector_profile": "air_small_target",
        "confidence_threshold": 0.30,
        "priority_classes": ["small_aircraft"],
        "sensor_weights": {"visible": 0.35, "ir": 0.45, "sar": 0.20},
        "feature_weights": {"intensity": 0.25, "texture": 0.20, "frequency": 0.20, "detail": 0.35},
    },
    "sea": {
        "detector_profile": "sea_ship",
        "confidence_threshold": 0.35,
        "priority_classes": ["warship"],
        "sensor_weights": {"visible": 0.20, "ir": 0.25, "sar": 0.55},
        "feature_weights": {"intensity": 0.20, "texture": 0.20, "frequency": 0.35, "detail": 0.25},
    },
    "urban": {
        "detector_profile": "urban_occlusion",
        "confidence_threshold": 0.40,
        "priority_classes": ["soldier", "tank"],
        "sensor_weights": {"visible": 0.35, "ir": 0.45, "sar": 0.20},
        "feature_weights": {"intensity": 0.20, "texture": 0.40, "frequency": 0.15, "detail": 0.25},
    },
    "forest": {
        "detector_profile": "forest_complex_background",
        "confidence_threshold": 0.30,
        "priority_classes": ["soldier", "tank"],
        "sensor_weights": {"visible": 0.20, "ir": 0.55, "sar": 0.25},
        "feature_weights": {"intensity": 0.30, "texture": 0.35, "frequency": 0.10, "detail": 0.25},
    },
}


def invalid_combinations(scene_label: str, detections: list[DetectionBox]) -> list[dict[str, Any]]:
    """检查目标类别是否符合当前场景。

    例如海洋场景里出现坦克、天空场景里出现舰船，都会被记录为不合理组合。
    这些不合理组合会进入报告，也会提高L_proto代理损失。
    """

    allowed = ALLOWED_TARGETS_BY_SCENE.get(scene_label, set())
    invalid = []
    for box in detections:
        if box.class_name == "unknown":
            continue
        if box.class_name not in allowed:
            invalid.append(
                {
                    "track_id": box.track_id,
                    "scene": scene_label,
                    "scene_cn": SCENE_CN.get(scene_label, scene_label),
                    "target": box.class_name,
                    "target_cn": TARGET_CN.get(box.class_name, box.class_name),
                    "confidence": box.confidence,
                    "reason": f"{TARGET_CN.get(box.class_name, box.class_name)} is not expected in {SCENE_CN.get(scene_label, scene_label)} scene",
                }
            )
    return invalid


def resolve_final_scene(
    scene_result: ProbabilityResult,
    modality_result: ProbabilityResult,
    detections: list[DetectionBox],
) -> ProbabilityResult:
    """融合场景模型、模态先验和目标类别，得到最终场景。

    融合思想：
    1. 场景模型概率占主导，保留图像整体语义判断；
    2. 检出的目标类别给兼容场景投票，例如warship支持sea；
    3. 模态只给很小的偏置，例如SAR更常用于海面/城市结构观察。

    这样既不让规则盖过模型，也能在模型不稳定时利用目标结果进行修正。
    """

    scores = {scene: float(scene_result.probabilities.get(scene, 0.0)) * 0.72 for scene in SCENE_LABELS}
    if detections:
        vote_weight = 0.24 / max(1, len(detections))
        for box in detections:
            compatible = SCENES_BY_TARGET.get(box.class_name, set())
            if not compatible:
                continue
            for scene in compatible:
                scores[scene] += vote_weight * max(box.confidence, 0.15)
    if modality_result.label == "sar":
        scores["sea"] += 0.02
        scores["urban"] += 0.01
    elif modality_result.label == "ir":
        scores["forest"] += 0.015
        scores["urban"] += 0.015
    probabilities = normalize_scores(scores)
    label = max(probabilities, key=probabilities.get)
    invalid = invalid_combinations(label, detections)
    source = "scene_target_consistency_fusion"
    details: dict[str, Any] = {"raw_scene": scene_result.label, "invalid_after_fusion": len(invalid)}
    if invalid:
        details["status"] = "conflict"
    else:
        details["status"] = "consistent"
    return ProbabilityResult(
        label=label,
        confidence=round(probabilities[label], 6),
        probabilities={key: round(value, 6) for key, value in probabilities.items()},
        source=source,
        details=details,
    )


def build_consistency_report(
    scene_result: ProbabilityResult,
    final_scene: ProbabilityResult,
    detections: list[DetectionBox],
) -> dict[str, Any]:
    """生成一致性报告，说明模型判断是否被目标结果修正。"""

    original_invalid = invalid_combinations(scene_result.label, detections)
    final_invalid = invalid_combinations(final_scene.label, detections)
    target_counts: dict[str, int] = {}
    for box in detections:
        target_counts[box.class_name] = target_counts.get(box.class_name, 0) + 1
    status = "consistent"
    if final_invalid:
        status = "invalid_combination"
    elif original_invalid:
        status = "repaired_by_target_scene_fusion"
    return {
        "status": status,
        "target_counts": target_counts,
        "original_invalid_count": len(original_invalid),
        "final_invalid_count": len(final_invalid),
        "original_invalid": original_invalid,
        "final_invalid": final_invalid,
        "rule": "scene + target must match dataset-realistic combinations; impossible pairs are counted as L_proto/L_cls penalties.",
    }


def active_experts(modality: str, detections: list[DetectionBox], final_scene: str) -> list[str]:
    """根据模态和目标选择专家。

    流程图里提到“12个专家+跨模态适配器”：12个专家可以理解为
    3种模态 * 4类目标。这里先输出路由名称，后续接真实专家模型时
    可以按这个名字加载对应权重。
    """

    targets = [box.class_name for box in detections if box.class_name != "unknown"]
    if not targets:
        targets = list(SCENE_POLICY.get(final_scene, SCENE_POLICY["urban"])["priority_classes"])
    experts = []
    for target in sorted(set(targets)):
        experts.append(f"{modality}_{target}_expert")
    experts.append("cross_modal_adapter")
    return experts


def build_decision(
    modality_result: ProbabilityResult,
    final_scene: ProbabilityResult,
    environment: dict[str, Any],
    detections: list[DetectionBox],
) -> dict[str, Any]:
    """根据最终场景和环境质量生成检测/部署决策。

    输出包含：
    - detector_profile: 应该使用哪类检测配置；
    - confidence_threshold: 当前帧的检测阈值；
    - priority_classes: 当前帧重点关注的目标；
    - sensor_weights/feature_weights: 多模态或特征融合时的建议权重；
    - expert_routing: 多专家模型路由；
    - model_management: 端侧部署策略说明。
    """

    policy = dict(SCENE_POLICY.get(final_scene.label, SCENE_POLICY["urban"]))
    threshold = float(policy["confidence_threshold"])
    if environment.get("noise_level") == "high":
        threshold += 0.05
    if final_scene.confidence < 0.55:
        threshold -= 0.05
    threshold = max(0.20, min(0.60, threshold))

    selected_targets = [box.class_name for box in detections if box.class_name != "unknown"]
    if selected_targets:
        priority_classes = sorted(set(policy["priority_classes"]) | set(selected_targets))
    else:
        priority_classes = list(policy["priority_classes"])

    return {
        "detector_profile": policy["detector_profile"],
        "confidence_threshold": round(threshold, 4),
        "priority_classes": priority_classes,
        "sensor_weights": policy["sensor_weights"],
        "feature_weights": policy["feature_weights"],
        "expert_routing": {
            "expert_bank": "12 target-modality experts + cross-modal adapter",
            "active_experts": active_experts(modality_result.label, detections, final_scene.label),
            "adapter": "cross_modal_adapter",
        },
        "model_management": {
            "edge_target": "Ascend 310B",
            "preferred_export": "ONNX -> ATC -> OM",
            "runtime_goal": "FPS >= 30",
            "load_strategy": "load scene-specific detector profile and target experts on demand",
        },
    }


def describe_scene(
    modality_result: ProbabilityResult,
    final_scene: ProbabilityResult,
    detections: list[DetectionBox],
    consistency: dict[str, Any],
) -> str:
    """生成一段可以直接放进报告或演示页面的中文场景描述。"""

    modality = MODALITY_CN.get(modality_result.label, modality_result.label)
    scene = SCENE_CN.get(final_scene.label, final_scene.label)
    if detections:
        target_text = "、".join(
            f"{TARGET_CN.get(box.class_name, box.class_name)}({box.confidence:.2f})"
            for box in detections
        )
    else:
        target_text = "未检出目标"
    status = "组合合理" if consistency["status"] == "consistent" else "存在组合冲突"
    return (
        f"输入被识别为{modality}图像，最终场景为{scene}"
        f"（置信度{final_scene.confidence:.3f}），目标结果：{target_text}；{status}。"
    )


def summarize_detection_confidence(detections: list[DetectionBox]) -> dict[str, Any]:
    """给阶段耗时日志附带一个简短目标置信度摘要。"""

    if not detections:
        return {"count": 0, "mean_confidence": 0.0, "min_confidence": 0.0}
    values = [float(box.confidence) for box in detections]
    return {
        "count": len(detections),
        "mean_confidence": round(mean(values), 6),
        "min_confidence": round(min(values), 6),
    }
