"""Detection-adapted Dark Experience Replay for Ultralytics YOLO.

DER keeps supervised examples in the same fixed replay buffer as ER and, for
those replay examples only, matches the previous-stage detector's raw class and
box-distribution responses.  The previous checkpoint is frozen and its dark
targets are recomputed online, avoiding a very large per-image logit cache.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import torch
from torch import nn

try:  # Keep protocol/data unit tests importable without the optional YOLO package.
    from ultralytics.nn.distill_model import DistillationModel as _DistillationModel
    from ultralytics.nn.tasks import load_checkpoint as _load_checkpoint
    from ultralytics.utils.torch_utils import copy_attr as _copy_attr
except ModuleNotFoundError:  # pragma: no cover - exercised only in minimal environments
    _load_checkpoint = None
    _copy_attr = None

    class _DistillationModel(nn.Module):
        pass


def normalize_path(path: str | Path) -> str:
    """Normalize paths for case-insensitive replay membership checks on Windows."""

    return os.path.normcase(os.path.abspath(os.fspath(path)))


def replay_batch_mask(im_files: Sequence[str | Path], replay_paths: set[str]) -> torch.Tensor:
    """Return a CPU boolean mask indicating which batch images came from replay."""

    return torch.tensor(
        [normalize_path(path) in replay_paths for path in im_files],
        dtype=torch.bool,
    )


def _raw_detection_levels(predictions) -> list[torch.Tensor]:
    """Extract raw ``[B, C, H, W]`` detection levels from common YOLO outputs."""

    if isinstance(predictions, dict):
        for branch in ("one2many", "one2one"):
            if branch in predictions:
                return _raw_detection_levels(predictions[branch])
    if isinstance(predictions, tuple):
        # Evaluation-mode YOLOv8 returns (decoded_predictions, raw_levels).
        for value in reversed(predictions):
            try:
                return _raw_detection_levels(value)
            except (TypeError, ValueError):
                continue
    if isinstance(predictions, list) and predictions and all(
        isinstance(value, torch.Tensor) and value.ndim == 4 for value in predictions
    ):
        return predictions
    raise TypeError(f"不支持的 YOLO 检测头输出类型: {type(predictions).__name__}")


def _decoupled_detection_output(predictions) -> dict[str, torch.Tensor] | None:
    """Extract YOLO26-style separate ``boxes`` and ``scores`` tensors."""

    if isinstance(predictions, dict):
        if {"boxes", "scores"}.issubset(predictions):
            boxes = predictions["boxes"]
            scores = predictions["scores"]
            if isinstance(boxes, torch.Tensor) and isinstance(scores, torch.Tensor):
                return {"boxes": boxes, "scores": scores}
        for branch in ("one2many", "one2one"):
            if branch in predictions:
                output = _decoupled_detection_output(predictions[branch])
                if output is not None:
                    return output
    if isinstance(predictions, tuple):
        for value in reversed(predictions):
            output = _decoupled_detection_output(value)
            if output is not None:
                return output
    return None


def dark_replay_decoupled_loss(
    student_output: dict[str, torch.Tensor],
    teacher_output: dict[str, torch.Tensor],
    *,
    old_class_count: int,
    cls_weight: float = 1.0,
    box_weight: float = 0.25,
    min_confidence: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Match YOLO26-style separate box and class response tensors."""

    student_boxes = student_output["boxes"].float()
    student_scores = student_output["scores"].float()
    teacher_boxes = teacher_output["boxes"].detach().float()
    teacher_scores = teacher_output["scores"].detach().float()
    if student_boxes.shape != teacher_boxes.shape:
        raise ValueError(
            f"DER 教师/学生 box 形状不一致: {tuple(teacher_boxes.shape)} vs {tuple(student_boxes.shape)}"
        )
    if student_scores.shape[0] != teacher_scores.shape[0] or student_scores.shape[2:] != teacher_scores.shape[2:]:
        raise ValueError("DER 教师/学生 score 空间形状不一致")
    if teacher_scores.shape[1] != old_class_count or student_scores.shape[1] < old_class_count:
        raise ValueError("DER 教师/学生旧类 score 通道不一致")
    teacher_old_scores = teacher_scores[:, :old_class_count]
    student_old_scores = student_scores[:, :old_class_count]
    confidence = teacher_old_scores.sigmoid().amax(dim=1, keepdim=True)
    weights = confidence * (confidence >= min_confidence)
    weight_sum = weights.sum()
    if float(weight_sum.detach()) <= 0:
        zero = student_scores.sum() * 0.0
        return zero, {"cls": zero, "box": zero}
    cls_loss = (
        (student_old_scores - teacher_old_scores).square() * weights
    ).sum() / (weight_sum * old_class_count)
    box_channels = student_boxes.shape[1]
    box_loss = ((student_boxes - teacher_boxes).square() * weights).sum() / (
        weight_sum * box_channels
    )
    total = cls_loss * cls_weight + box_loss * box_weight
    return total, {"cls": cls_loss, "box": box_loss}


def dark_replay_response_loss(
    student_levels: Sequence[torch.Tensor],
    teacher_levels: Sequence[torch.Tensor],
    *,
    old_class_count: int,
    reg_max: int,
    cls_weight: float = 1.0,
    box_weight: float = 0.25,
    min_confidence: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute confidence-weighted dark class/box response matching.

    Student class channels may include a newly appended class.  Only the first
    ``old_class_count`` channels are matched to the teacher; future/current
    class channels remain unconstrained.
    """

    if old_class_count <= 0:
        raise ValueError("old_class_count 必须为正整数")
    if reg_max <= 0:
        raise ValueError("reg_max 必须为正整数")
    if cls_weight < 0 or box_weight < 0 or cls_weight + box_weight <= 0:
        raise ValueError("DER 分类/框权重必须非负且至少一个大于 0")
    if not 0 <= min_confidence <= 1:
        raise ValueError("min_confidence 必须位于 [0, 1]")
    if len(student_levels) != len(teacher_levels) or not student_levels:
        raise ValueError("教师与学生检测层数量必须一致且非空")

    box_channels = reg_max * 4
    cls_losses: list[torch.Tensor] = []
    box_losses: list[torch.Tensor] = []
    for level_index, (student, teacher) in enumerate(zip(student_levels, teacher_levels)):
        if student.ndim != 4 or teacher.ndim != 4:
            raise ValueError("DER 只支持四维原始检测层")
        if student.shape[0] != teacher.shape[0] or student.shape[2:] != teacher.shape[2:]:
            raise ValueError(
                f"第 {level_index} 层教师/学生形状不兼容: "
                f"student={tuple(student.shape)}, teacher={tuple(teacher.shape)}"
            )
        if teacher.shape[1] != box_channels + old_class_count:
            raise ValueError(
                f"教师通道数应为 {box_channels + old_class_count}，"
                f"实际为 {teacher.shape[1]}"
            )
        if student.shape[1] < box_channels + old_class_count:
            raise ValueError("学生检测头缺少旧类别通道")

        student_fp32 = student.float()
        teacher_fp32 = teacher.detach().float()
        teacher_cls = teacher_fp32[:, box_channels : box_channels + old_class_count]
        student_cls = student_fp32[:, box_channels : box_channels + old_class_count]
        confidence = teacher_cls.sigmoid().amax(dim=1, keepdim=True)
        weights = confidence * (confidence >= min_confidence)
        weight_sum = weights.sum()
        if float(weight_sum.detach()) <= 0:
            # Keep a differentiable zero so a replay batch with no confident old
            # response remains valid instead of producing NaN.
            zero = student_fp32.sum() * 0.0
            cls_losses.append(zero)
            box_losses.append(zero)
            continue
        cls_mse = (student_cls - teacher_cls).square()
        cls_losses.append((cls_mse * weights).sum() / (weight_sum * old_class_count))
        student_box = student_fp32[:, :box_channels]
        teacher_box = teacher_fp32[:, :box_channels]
        box_mse = (student_box - teacher_box).square()
        box_losses.append((box_mse * weights).sum() / (weight_sum * box_channels))

    cls_loss = torch.stack(cls_losses).mean()
    box_loss = torch.stack(box_losses).mean()
    total = cls_loss * cls_weight + box_loss * box_weight
    return total, {"cls": cls_loss, "box": box_loss}


def _normalize_names(names: object) -> list[str]:
    if isinstance(names, dict):
        return [str(names[index] if index in names else names[str(index)]) for index in range(len(names))]
    if isinstance(names, (list, tuple)):
        return [str(value) for value in names]
    return []


class DarkReplayModel(_DistillationModel):
    """Teacher-student wrapper that applies DER only to replay-buffer images."""

    def __init__(
        self,
        teacher_checkpoint: str | Path,
        student_model: nn.Module,
        replay_paths: Sequence[str | Path],
        *,
        der_weight: float = 1.0,
        cls_weight: float = 1.0,
        box_weight: float = 0.25,
        min_confidence: float = 0.0,
    ) -> None:
        if _load_checkpoint is None or _copy_attr is None:
            raise ModuleNotFoundError("DER 训练需要本地安装 ultralytics>=8.4,<8.5")
        nn.Module.__init__(self)
        if der_weight <= 0:
            raise ValueError("der_weight 必须大于 0")
        teacher_model = _load_checkpoint(teacher_checkpoint)[0]
        self.teacher_model = teacher_model.to(next(student_model.parameters()).device)
        self.student_model = student_model
        self.projector = nn.ModuleList()  # compatibility with Ultralytics checkpoint recovery
        self.feats_idx: list[int] = []
        self._teacher_feats: dict = {}
        self._student_feats: dict = {}
        self._teacher_hooks: list = []
        self._student_hooks: list = []
        self.replay_paths = {normalize_path(path) for path in replay_paths}
        if not self.replay_paths:
            raise ValueError("DER 阶段必须包含非空 replay manifest")
        self.der_weight = float(der_weight)
        self.der_cls_weight = float(cls_weight)
        self.der_box_weight = float(box_weight)
        self.der_min_confidence = float(min_confidence)

        teacher_names = _normalize_names(getattr(self.teacher_model, "names", None))
        student_names = _normalize_names(getattr(student_model, "names", None))
        self.old_class_count = len(teacher_names)
        if not teacher_names or student_names[: self.old_class_count] != teacher_names:
            raise ValueError(
                "DER 教师类别必须是学生类别的有序前缀；"
                f"teacher={teacher_names}, student={student_names}"
            )
        teacher_head = self.teacher_model.model[-1]
        student_head = student_model.model[-1]
        self.reg_max = int(getattr(teacher_head, "reg_max", 0))
        if self.reg_max != int(getattr(student_head, "reg_max", -1)):
            raise ValueError("DER 教师与学生 reg_max 不一致")
        _copy_attr(self, student_model)
        self._freeze_teacher()

    def forward(self, value, *args, **kwargs):
        if isinstance(value, dict):
            return self.loss(value, *args, **kwargs)
        return self.student_model.predict(value, *args, **kwargs)

    def loss(self, batch, preds=None):
        zero = torch.zeros(1, device=batch["img"].device)
        if not self.training:
            if preds is None:
                preds = self.student_model(batch["img"])
            regular_loss, regular_items = self.student_model.loss(batch, preds)
            return torch.cat([regular_loss.reshape(-1), zero]), torch.cat(
                [regular_items.reshape(-1), zero]
            )

        student_predictions = self.student_model(batch["img"])
        regular_loss, regular_items = self.student_model.loss(batch, student_predictions)
        files = batch.get("im_file")
        if files is None:
            raise ValueError("DER 需要 batch['im_file'] 来识别回放样本")
        mask = replay_batch_mask(files, self.replay_paths).to(batch["img"].device)
        if not bool(mask.any()):
            der_loss = student_predictions[0].sum() * 0.0 if isinstance(student_predictions, list) else zero[0]
        else:
            replay_images = batch["img"][mask]
            with torch.no_grad():
                teacher_predictions = self.teacher_model(replay_images)
            student_decoupled = _decoupled_detection_output(student_predictions)
            teacher_decoupled = _decoupled_detection_output(teacher_predictions)
            if student_decoupled is not None and teacher_decoupled is not None:
                response_loss, _ = dark_replay_decoupled_loss(
                    {
                        "boxes": student_decoupled["boxes"][mask],
                        "scores": student_decoupled["scores"][mask],
                    },
                    teacher_decoupled,
                    old_class_count=self.old_class_count,
                    cls_weight=self.der_cls_weight,
                    box_weight=self.der_box_weight,
                    min_confidence=self.der_min_confidence,
                )
            else:
                student_levels = [level[mask] for level in _raw_detection_levels(student_predictions)]
                teacher_levels = _raw_detection_levels(teacher_predictions)
                response_loss, _ = dark_replay_response_loss(
                    student_levels,
                    teacher_levels,
                    old_class_count=self.old_class_count,
                    reg_max=self.reg_max,
                    cls_weight=self.der_cls_weight,
                    box_weight=self.der_box_weight,
                    min_confidence=self.der_min_confidence,
                )
            der_loss = response_loss * self.der_weight
        scaled_der_loss = der_loss.reshape(1) * batch["img"].shape[0]
        return torch.cat([regular_loss.reshape(-1), scaled_der_loss]), torch.cat(
            [regular_items.reshape(-1), der_loss.detach().reshape(1)]
        )
