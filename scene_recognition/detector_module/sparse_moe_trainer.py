"""Ultralytics trainer adapters for Sparse-MoE ER and DER training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from scene_recognition.detector_module.context_metadata import (
    UNKNOWN,
    normalize_context_path,
    read_context_index,
    resolve_context_metadata,
)
from scene_recognition.detector_module.sparse_moe_model import (
    SparseMoEConfig,
    SparseMoEDetectionModel,
    load_sparse_weights,
)


try:
    from ultralytics.models.yolo.detect import DetectionTrainer
except ModuleNotFoundError:  # pragma: no cover - protocol utilities work without Ultralytics
    DetectionTrainer = object  # type: ignore[assignment,misc]


def _target_index(row: dict[str, str], field: str, names: tuple[str, ...]) -> int:
    value = str(row.get(field, "unknown")).strip().casefold()
    try:
        return names.index(value)
    except ValueError:
        return -1


class SparseMoETrainer(DetectionTrainer):
    """DetectionTrainer that adds metadata-aware MoE losses to YOLO loss."""

    sparse_moe_config: SparseMoEConfig = SparseMoEConfig()
    sparse_context_index: Path | None = None
    der_context: Any | None = None

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._context_rows: dict[str, dict[str, str]] = {}
        if self.sparse_context_index is not None and self.sparse_context_index.is_file():
            self._context_rows = read_context_index(self.sparse_context_index)

    def _build_sparse_student(self, cfg=None, weights=None, verbose: bool = True):
        model = SparseMoEDetectionModel(
            cfg,
            nc=self.data["nc"],
            ch=self.data.get("channels", 3),
            verbose=verbose,
            sparse_moe_config=self.sparse_moe_config,
        )
        model.names = self.data["names"]
        model.args = self.args
        if weights is not None:
            load_sparse_weights(model, weights, verbose=verbose)
        return model

    def get_model(self, cfg=None, weights=None, verbose=True):
        student = self._build_sparse_student(cfg=cfg, weights=weights, verbose=verbose)
        if self.der_context is None:
            return student
        from scene_recognition.detector_module.dark_experience_replay import DarkReplayModel

        return DarkReplayModel(
            self.der_context.teacher_checkpoint,
            student,
            self.der_context.replay_paths,
            der_weight=self.der_context.der_weight,
            cls_weight=self.der_context.cls_weight,
            box_weight=self.der_context.box_weight,
            min_confidence=self.der_context.min_confidence,
        )

    def set_model_attributes(self):
        super().set_model_attributes()
        model = self.model
        student = getattr(model, "student_model", model)
        if isinstance(student, SparseMoEDetectionModel):
            student.nc = self.data["nc"]
            student.names = self.data["names"]
            student.args = self.args

    def get_validator(self):
        validator = super().get_validator()
        base_names = ("box_loss", "cls_loss", "dfl_loss")
        self.loss_names = base_names + (
            "modality_loss",
            "scene_loss",
            "balance_loss",
            "router_z_loss",
            "anchor_loss",
        )
        if self.der_context is not None:
            self.loss_names += ("der_loss",)
        return validator

    def build_optimizer(
        self,
        model,
        name="auto",
        lr=0.001,
        momentum=0.9,
        decay=1e-5,
        iterations=1e5,
    ):
        student = getattr(model, "student_model", model)
        return super().build_optimizer(
            student,
            name=name,
            lr=lr,
            momentum=momentum,
            decay=decay,
            iterations=iterations,
        )

    def _update_router_temperature(self) -> None:
        model = getattr(self.model, "student_model", self.model)
        adapter = getattr(model, "sparse_moe", None)
        if adapter is not None:
            adapter.set_temperature(self.sparse_moe_config.temperature_for_epoch(self.epoch))

    def preprocess_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        batch = super().preprocess_batch(batch)
        self._update_router_temperature()
        paths = batch.get("im_file", [])
        if isinstance(paths, (str, Path)):
            paths = [paths]
        modality_targets = []
        scene_targets = []
        for path in paths:
            row = self._context_rows.get(normalize_context_path(path), {})
            filename_context = resolve_context_metadata(path)
            if row.get("sensor", UNKNOWN) == UNKNOWN and filename_context["sensor"] != UNKNOWN:
                row = {**row, "sensor": filename_context["sensor"]}
            if row.get("scene", UNKNOWN) == UNKNOWN and filename_context["scene"] != UNKNOWN:
                row = {**row, "scene": filename_context["scene"]}
            modality_targets.append(_target_index(row, "sensor", self.sparse_moe_config.modality_names))
            scene_targets.append(_target_index(row, "scene", self.sparse_moe_config.scene_names))
        batch_size = int(batch["img"].shape[0])
        if len(modality_targets) != batch_size:
            modality_targets = [-1] * batch_size
            scene_targets = [-1] * batch_size
        batch["modality_targets"] = torch.tensor(modality_targets, dtype=torch.long, device=self.device)
        batch["scene_targets"] = torch.tensor(scene_targets, dtype=torch.long, device=self.device)
        batch["modality_mask"] = batch["modality_targets"].ge(0)
        batch["scene_mask"] = batch["scene_targets"].ge(0)
        return batch


def make_sparse_moe_trainer(
    config: SparseMoEConfig,
    *,
    context_index: Path | None = None,
    der_context: Any | None = None,
):
    """Return a trainer class configured for one Class-IL stage."""

    configured_der_context = der_context

    class ConfiguredSparseMoETrainer(SparseMoETrainer):
        sparse_moe_config = config
        sparse_context_index = context_index
        der_context = configured_der_context

    ConfiguredSparseMoETrainer.__name__ = "ConfiguredSparseMoETrainer"
    return ConfiguredSparseMoETrainer


__all__ = ["SparseMoETrainer", "make_sparse_moe_trainer"]
