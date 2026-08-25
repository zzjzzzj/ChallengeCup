from __future__ import annotations

import numpy as np


def _softmax(values: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32) / max(temperature, 1e-6)
    values = values - values.max(axis=-1, keepdims=True)
    exp = np.exp(values)
    return exp / np.maximum(exp.sum(axis=-1, keepdims=True), 1e-12)


def _kl_divergence(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    p = np.maximum(p, 1e-12)
    q = np.maximum(q, 1e-12)
    return (p * (np.log(p) - np.log(q))).sum(axis=-1)


def box_iou_numpy(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    first = np.asarray(first, dtype=np.float32)
    second = np.asarray(second, dtype=np.float32)
    ix1 = np.maximum(first[..., 0], second[..., 0])
    iy1 = np.maximum(first[..., 1], second[..., 1])
    ix2 = np.minimum(first[..., 2], second[..., 2])
    iy2 = np.minimum(first[..., 3], second[..., 3])
    inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
    area_first = np.maximum(0.0, first[..., 2] - first[..., 0]) * np.maximum(0.0, first[..., 3] - first[..., 1])
    area_second = np.maximum(0.0, second[..., 2] - second[..., 0]) * np.maximum(0.0, second[..., 3] - second[..., 1])
    union = area_first + area_second - inter
    return np.where(union > 0, inter / union, 0.0)


def detection_response_kd_numpy(
    teacher_logits: np.ndarray,
    student_logits: np.ndarray,
    teacher_boxes: np.ndarray,
    student_boxes: np.ndarray,
    reliable_mask: np.ndarray,
    *,
    temperature: float = 2.0,
    eta_box: float = 1.0,
) -> dict[str, float]:
    """Classification + localization response distillation for old detections."""

    mask = np.asarray(reliable_mask, dtype=bool)
    if mask.sum() == 0:
        return {"loss_kd_cls": 0.0, "loss_kd_box": 0.0, "loss_kd": 0.0, "reliable_count": 0}
    teacher_prob = _softmax(np.asarray(teacher_logits)[mask], temperature)
    student_prob = _softmax(np.asarray(student_logits)[mask], temperature)
    cls_loss = float((temperature**2) * _kl_divergence(teacher_prob, student_prob).mean())
    iou = box_iou_numpy(np.asarray(teacher_boxes)[mask], np.asarray(student_boxes)[mask])
    box_loss = float((1.0 - iou).mean())
    return {
        "loss_kd_cls": cls_loss,
        "loss_kd_box": box_loss,
        "loss_kd": cls_loss + eta_box * box_loss,
        "reliable_count": int(mask.sum()),
    }
