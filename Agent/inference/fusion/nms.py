from __future__ import annotations

from collections import defaultdict

from Agent.common.schemas import DetectionPrediction


def box_iou(first: tuple[float, float, float, float], second: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def nms(detections: list[DetectionPrediction], iou_threshold: float = 0.55) -> list[DetectionPrediction]:
    """Class-wise non-maximum suppression."""

    kept: list[DetectionPrediction] = []
    grouped: dict[str, list[DetectionPrediction]] = defaultdict(list)
    for detection in detections:
        grouped[detection.class_name].append(detection)
    for class_detections in grouped.values():
        remaining = sorted(class_detections, key=lambda item: item.confidence, reverse=True)
        while remaining:
            current = remaining.pop(0)
            kept.append(current)
            remaining = [
                item
                for item in remaining
                if box_iou(current.bbox_xyxy, item.bbox_xyxy) < iou_threshold
            ]
    return kept


def weighted_box_fusion(
    detections: list[DetectionPrediction],
    iou_threshold: float = 0.55,
) -> list[DetectionPrediction]:
    """Simple class-wise weighted fusion for first-pass and tile detections."""

    output: list[DetectionPrediction] = []
    grouped: dict[str, list[DetectionPrediction]] = defaultdict(list)
    for detection in detections:
        grouped[detection.class_name].append(detection)
    for class_name, class_detections in grouped.items():
        remaining = sorted(class_detections, key=lambda item: item.confidence, reverse=True)
        while remaining:
            seed = remaining.pop(0)
            cluster = [seed]
            kept_remaining = []
            for item in remaining:
                if box_iou(seed.bbox_xyxy, item.bbox_xyxy) >= iou_threshold:
                    cluster.append(item)
                else:
                    kept_remaining.append(item)
            remaining = kept_remaining
            weight_sum = sum(max(item.confidence, 1e-6) for item in cluster)
            fused = tuple(
                sum(item.bbox_xyxy[idx] * max(item.confidence, 1e-6) for item in cluster) / weight_sum
                for idx in range(4)
            )
            best = max(cluster, key=lambda item: item.confidence)
            expert_ids = sorted({expert_id for item in cluster for expert_id in item.expert_ids})
            output.append(
                DetectionPrediction(
                    class_name=class_name,
                    confidence=max(item.confidence for item in cluster),
                    bbox_xyxy=fused,
                    expert_ids=expert_ids,
                    pass_id=max(item.pass_id for item in cluster),
                    source="weighted_box_fusion",
                    metadata={"cluster_size": len(cluster), "best_source": best.source},
                )
            )
    return output
