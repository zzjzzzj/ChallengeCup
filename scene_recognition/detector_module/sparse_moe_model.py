"""Sparse-MoE v1 integration without modifying Ultralytics source code."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from Agent.continual.anchors.torch_expert_anchor import TorchExpertAnchorBank
from Agent.models.aux_heads.heads import masked_auxiliary_loss
from Agent.models.experts.sparse_adapter import SparseExpertAdapterBank
from Agent.models.experts.torch_router import TorchExpertRoute, TorchExpertUsageTracker, TorchSparseExpertRouter

try:  # Keep data/protocol imports usable without the optional YOLO package.
    from ultralytics.nn.modules import Detect as _UltralyticsDetect
    from ultralytics.nn.tasks import DetectionModel as _UltralyticsDetectionModel
except ModuleNotFoundError:  # pragma: no cover - only used in minimal environments
    _UltralyticsDetect = nn.Module
    _UltralyticsDetectionModel = None


@dataclass(frozen=True)
class SparseMoEConfig:
    """Serializable configuration for the first sparse expert experiment."""

    expert_count: int = 5
    top_k: int = 2
    expert_bottleneck: float = 0.25
    router_hidden: int = 128
    aux_hidden: int = 128
    modality_loss_weight: float = 0.10
    scene_loss_weight: float = 0.10
    balance_loss_weight: float = 0.01
    router_z_loss_weight: float = 0.001
    anchor_loss_weight: float = 0.001
    anchor_rho: float = 0.95
    router_temperature_start: float = 2.0
    router_temperature_end: float = 1.0
    router_temperature_warmup_epochs: int = 3
    modality_names: tuple[str, ...] = ("ir", "sar")
    scene_names: tuple[str, ...] = ("air", "sea", "urban", "forest")

    def __post_init__(self) -> None:
        if self.expert_count <= 0:
            raise ValueError("expert_count must be positive")
        if self.top_k <= 0 or self.top_k > self.expert_count:
            raise ValueError("top_k must be in [1, expert_count]")
        if not 0.0 < self.expert_bottleneck <= 1.0:
            raise ValueError("expert_bottleneck must be in (0, 1]")
        if self.router_hidden <= 0 or self.aux_hidden <= 0:
            raise ValueError("router_hidden and aux_hidden must be positive")
        if self.anchor_rho < 0.0 or self.anchor_rho >= 1.0:
            raise ValueError("anchor_rho must be in [0, 1)")
        if self.router_temperature_start <= 0 or self.router_temperature_end <= 0:
            raise ValueError("router temperatures must be positive")
        if self.router_temperature_warmup_epochs < 0:
            raise ValueError("router_temperature_warmup_epochs must be non-negative")
        for name in (
            "modality_loss_weight",
            "scene_loss_weight",
            "balance_loss_weight",
            "router_z_loss_weight",
            "anchor_loss_weight",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")

    @property
    def expert_names(self) -> tuple[str, ...]:
        return tuple(f"expert_{index}" for index in range(self.expert_count))

    def temperature_for_epoch(self, epoch: int) -> float:
        if self.router_temperature_warmup_epochs <= 0:
            return float(self.router_temperature_end)
        progress = min(max(float(epoch) / self.router_temperature_warmup_epochs, 0.0), 1.0)
        return float(
            self.router_temperature_start
            + progress * (self.router_temperature_end - self.router_temperature_start)
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["expert_names"] = list(self.expert_names)
        payload["modality_names"] = list(self.modality_names)
        payload["scene_names"] = list(self.scene_names)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SparseMoEConfig":
        known = {
            key: value
            for key, value in payload.items()
            if key in cls.__dataclass_fields__
        }
        for key in ("modality_names", "scene_names"):
            if key in known:
                known[key] = tuple(known[key])
        return cls(**known)


def compute_input_quality_stats(images: torch.Tensor) -> torch.Tensor:
    """Compute four cheap image statistics without labels or detections."""

    if images.ndim != 4:
        raise ValueError(f"images must be BCHW, got {tuple(images.shape)}")
    values = images.float()
    gray = values.mean(dim=1, keepdim=True)
    dx = gray[..., :, 1:] - gray[..., :, :-1]
    dy = gray[..., 1:, :] - gray[..., :-1, :]
    dx_energy = (
        dx.square().mean(dim=(1, 2, 3))
        if dx.shape[-1]
        else values.new_zeros(values.shape[0])
    )
    dy_energy = (
        dy.square().mean(dim=(1, 2, 3))
        if dy.shape[-2]
        else values.new_zeros(values.shape[0])
    )
    gradient_energy = 0.5 * (dx_energy + dy_energy)
    blurred = F.avg_pool2d(gray, kernel_size=3, stride=1, padding=1)
    high_frequency = (gray - blurred).abs().mean(dim=(1, 2, 3))
    return torch.stack(
        [
            values.mean(dim=(1, 2, 3)),
            values.std(dim=(1, 2, 3), unbiased=False),
            gradient_energy,
            high_frequency,
        ],
        dim=1,
    )


def _first_conv_in_channels(module: nn.Module) -> int | None:
    for child in module.modules():
        if isinstance(child, nn.Conv2d):
            return int(child.in_channels)
    return None


def detect_input_channels(detect_head: nn.Module) -> list[int]:
    """Read all Detect input channels from its live convolution branches."""

    branches = getattr(detect_head, "cv2", None)
    if branches is None:
        branches = getattr(detect_head, "cv3", None)
    channels = []
    if branches is not None:
        for branch in branches:
            channel_count = _first_conv_in_channels(branch)
            if channel_count is not None:
                channels.append(channel_count)
    if channels:
        return channels
    declared = getattr(detect_head, "ch", None)
    if declared is not None:
        values = [int(value) for value in declared]
        if values:
            return values
    return []


class SparseMoEDetectAdapter(_UltralyticsDetect):
    """Detect-compatible wrapper that adapts every incoming scale dynamically."""

    def __init__(
        self,
        detect_head: nn.Module,
        config: SparseMoEConfig | None = None,
        input_channels: Sequence[int] | None = None,
    ) -> None:
        nn.Module.__init__(self)
        self.detect_head = detect_head
        self.sparse_moe_config = config or SparseMoEConfig()
        self._feature_channels = list(input_channels or detect_input_channels(detect_head))
        self._temperature = self.sparse_moe_config.router_temperature_start
        self.input_quality: torch.Tensor | None = None
        self.last_route: TorchExpertRoute | None = None
        self.last_auxiliary: Any | None = None
        self.usage_tracker = TorchExpertUsageTracker(
            self.sparse_moe_config.expert_count,
            list(self.sparse_moe_config.expert_names),
        )
        self.anchor_bank = TorchExpertAnchorBank(self.sparse_moe_config.anchor_rho)
        self.expert_pools = nn.ModuleList()
        self.aux_heads: nn.Module | None = None
        self.router: TorchSparseExpertRouter | None = None
        if self._feature_channels:
            self._build_sparse_modules(self._feature_channels)

        # BaseModel uses these attributes while walking the graph and when
        # producing model summaries. They remain attached to the wrapper.
        for name in ("f", "i", "type", "np"):
            if hasattr(detect_head, name):
                setattr(self, name, getattr(detect_head, name))
        self.f = getattr(self, "f", -1)
        self.i = getattr(self, "i", -1)
        self.type = getattr(self, "type", type(detect_head).__name__)
        self.np = getattr(self, "np", sum(value.numel() for value in self.parameters()))

    def _build_sparse_modules(self, channels: Sequence[int]) -> None:
        channels = [int(value) for value in channels]
        if not channels or any(value <= 0 for value in channels):
            raise ValueError("Detect input channels must be positive")
        self._feature_channels = channels
        self.expert_pools = nn.ModuleList(
            [
                SparseExpertAdapterBank(
                    channel_count,
                    self.sparse_moe_config.expert_count,
                    self.sparse_moe_config.expert_bottleneck,
                )
                for channel_count in channels
            ]
        )
        self.aux_heads = __import__(
            "Agent.models.aux_heads.heads", fromlist=["ModalitySceneAuxiliaryHeads"]
        ).ModalitySceneAuxiliaryHeads(
            channels,
            hidden_channels=self.sparse_moe_config.aux_hidden,
            num_modalities=len(self.sparse_moe_config.modality_names),
            num_scenes=len(self.sparse_moe_config.scene_names),
        )
        router_input_dim = self.sparse_moe_config.aux_hidden + len(self.sparse_moe_config.modality_names) + len(
            self.sparse_moe_config.scene_names
        ) + 4
        self.router = TorchSparseExpertRouter(
            router_input_dim,
            expert_count=self.sparse_moe_config.expert_count,
            top_k=self.sparse_moe_config.top_k,
            hidden_dim=self.sparse_moe_config.router_hidden,
            temperature=self._temperature,
        )

    # Detect-compatible delegated properties. These let DetectionModel's
    # stride, class-remapping, validator and export code keep working.
    @property
    def nc(self) -> int:
        return int(getattr(self.detect_head, "nc"))

    @nc.setter
    def nc(self, value: int) -> None:
        self.detect_head.nc = int(value)
        if hasattr(self.detect_head, "no"):
            self.detect_head.no = int(value) + 4 * int(self.reg_max)

    @property
    def nl(self) -> int:
        return int(getattr(self.detect_head, "nl", len(self.expert_pools)))

    @property
    def reg_max(self) -> int:
        return int(getattr(self.detect_head, "reg_max"))

    @property
    def no(self) -> int:
        return int(getattr(self.detect_head, "no", self.nc + self.reg_max * 4))

    @property
    def stride(self):
        return self.detect_head.stride

    @stride.setter
    def stride(self, value) -> None:
        self.detect_head.stride = value

    @property
    def anchors(self):
        return self.detect_head.anchors

    @anchors.setter
    def anchors(self, value) -> None:
        self.detect_head.anchors = value

    @property
    def strides(self):
        return self.detect_head.strides

    @strides.setter
    def strides(self, value) -> None:
        self.detect_head.strides = value

    @property
    def shape(self):
        return self.detect_head.shape

    @shape.setter
    def shape(self, value) -> None:
        self.detect_head.shape = value

    @property
    def dynamic(self) -> bool:
        return bool(getattr(self.detect_head, "dynamic", False))

    @dynamic.setter
    def dynamic(self, value: bool) -> None:
        self.detect_head.dynamic = value

    @property
    def export(self) -> bool:
        return bool(getattr(self.detect_head, "export", False))

    @export.setter
    def export(self, value: bool) -> None:
        self.detect_head.export = value

    @property
    def end2end(self) -> bool:
        return bool(getattr(self.detect_head, "end2end", False))

    @end2end.setter
    def end2end(self, value: bool) -> None:
        self.detect_head.end2end = value

    @property
    def max_det(self) -> int:
        return int(getattr(self.detect_head, "max_det", 300))

    @max_det.setter
    def max_det(self, value: int) -> None:
        self.detect_head.max_det = int(value)

    @property
    def inplace(self) -> bool:
        return bool(getattr(self.detect_head, "inplace", True))

    @inplace.setter
    def inplace(self, value: bool) -> None:
        self.detect_head.inplace = value

    @property
    def agnostic_nms(self) -> bool:
        return bool(getattr(self.detect_head, "agnostic_nms", False))

    @agnostic_nms.setter
    def agnostic_nms(self, value: bool) -> None:
        self.detect_head.agnostic_nms = value

    @property
    def xyxy(self) -> bool:
        return bool(getattr(self.detect_head, "xyxy", False))

    @xyxy.setter
    def xyxy(self, value: bool) -> None:
        self.detect_head.xyxy = value

    @property
    def one2many(self):
        return self.detect_head.one2many

    @property
    def one2one(self):
        return self.detect_head.one2one

    @property
    def cv2(self):
        return self.detect_head.cv2

    @cv2.setter
    def cv2(self, value) -> None:
        self.detect_head.cv2 = value

    @property
    def cv3(self):
        return self.detect_head.cv3

    @cv3.setter
    def cv3(self, value) -> None:
        self.detect_head.cv3 = value

    @property
    def one2one_cv2(self):
        return getattr(self.detect_head, "one2one_cv2", None)

    @one2one_cv2.setter
    def one2one_cv2(self, value) -> None:
        self.detect_head.one2one_cv2 = value

    @property
    def one2one_cv3(self):
        return getattr(self.detect_head, "one2one_cv3", None)

    @one2one_cv3.setter
    def one2one_cv3(self, value) -> None:
        self.detect_head.one2one_cv3 = value

    def set_input_quality(self, quality_stats: torch.Tensor | None) -> None:
        self.input_quality = quality_stats

    def set_temperature(self, temperature: float) -> None:
        if temperature <= 0:
            raise ValueError("router temperature must be positive")
        self._temperature = float(temperature)
        if self.router is not None:
            self.router.temperature = float(temperature)

    def _ensure_sparse_modules(self, features: Sequence[torch.Tensor]) -> None:
        channels = [int(feature.shape[1]) for feature in features]
        if not self._feature_channels:
            self._build_sparse_modules(channels)
        if channels != self._feature_channels:
            raise ValueError(
                "Detect feature channels changed after Sparse-MoE initialization: "
                f"expected {self._feature_channels}, got {channels}"
            )

    def _quality_for_features(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
        if self.input_quality is not None:
            quality = self.input_quality
            if quality.shape[0] != features[0].shape[0]:
                raise ValueError("input quality batch size does not match Detect features")
            return quality.to(device=features[0].device, dtype=features[0].dtype)
        # A safe fallback for direct head unit tests: it still uses input
        # statistics only and never accesses targets or decoded detections.
        pooled = torch.cat(
            [F.adaptive_avg_pool2d(feature.float(), 1).flatten(1) for feature in features], dim=1
        )
        return torch.stack(
            [
                pooled.mean(dim=1),
                pooled.std(dim=1, unbiased=False),
                features[0].float().abs().mean(dim=(1, 2, 3)),
                features[-1].float().abs().mean(dim=(1, 2, 3)),
            ],
            dim=1,
        ).to(dtype=features[0].dtype)

    def _route_features(self, features: Sequence[torch.Tensor]) -> TorchExpertRoute:
        assert self.aux_heads is not None and self.router is not None
        auxiliary = self.aux_heads(features)
        quality = self._quality_for_features(features)
        query = torch.cat(
            [
                auxiliary.embedding,
                auxiliary.modality_probabilities.detach(),
                auxiliary.scene_probabilities.detach(),
                quality,
            ],
            dim=1,
        )
        route = self.router(query, temperature=self._temperature)
        self.last_auxiliary = auxiliary
        self.last_route = route
        if self.training:
            self.usage_tracker.update(route)
        return route

    def forward(self, features: list[torch.Tensor] | tuple[torch.Tensor, ...]):
        if not isinstance(features, (list, tuple)) or not features:
            raise ValueError("Sparse-MoE Detect adapter expects a non-empty feature list")
        self._ensure_sparse_modules(features)
        route = self._route_features(features)
        adapted = [
            pool(feature, route.expert_ids, route.expert_weights)
            for pool, feature in zip(self.expert_pools, features)
        ]
        return self.detect_head(adapted)

    def expert_parameter_vectors(self) -> dict[str, list[torch.Tensor]]:
        vectors: dict[str, list[torch.Tensor]] = {
            name: [] for name in self.sparse_moe_config.expert_names
        }
        for pool in self.expert_pools:
            for expert_index, adapter in enumerate(pool.adapters):
                vectors[f"expert_{expert_index}"].extend(list(adapter.parameters()))
        return vectors

    def loss_components(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Return raw auxiliary, balance, z and anchor losses for the last pass."""

        if self.last_route is None or self.last_auxiliary is None:
            reference = next(self.parameters())
            zero = reference.sum() * 0.0
            return {
                "modality": zero,
                "scene": zero,
                "balance": zero,
                "router_z": zero,
                "anchor": zero,
                "total": zero,
            }
        modality_targets = batch.get("modality_targets", torch.full(
            (self.last_route.probabilities.shape[0],), -1, device=self.last_route.probabilities.device
        ))
        scene_targets = batch.get("scene_targets", torch.full(
            (self.last_route.probabilities.shape[0],), -1, device=self.last_route.probabilities.device
        ))
        aux = masked_auxiliary_loss(
            self.last_auxiliary,
            modality_targets,
            scene_targets,
            batch.get("modality_mask"),
            batch.get("scene_mask"),
        )
        balance = self.router.load_balance_loss(self.last_route) if self.router is not None else aux["total"] * 0.0
        router_z = self.router.router_z_loss(self.last_route) if self.router is not None else aux["total"] * 0.0
        anchor = self.anchor_bank.penalty_from_experts(self.expert_parameter_vectors())
        total = (
            self.sparse_moe_config.modality_loss_weight * aux["modality"]
            + self.sparse_moe_config.scene_loss_weight * aux["scene"]
            + self.sparse_moe_config.balance_loss_weight * balance
            + self.sparse_moe_config.router_z_loss_weight * router_z
            + self.sparse_moe_config.anchor_loss_weight * anchor
        )
        return {
            "modality": aux["modality"],
            "scene": aux["scene"],
            "balance": balance,
            "router_z": router_z,
            "anchor": anchor,
            "total": total,
        }

    def update_anchors(self) -> dict[str, float]:
        """Update EMA anchors from a stage-best model and return importances."""

        return self.anchor_bank.update_from_experts(
            self.expert_parameter_vectors(),
            self.usage_tracker.importance(),
        )

    def diagnostics(self, index: int = 0) -> dict[str, Any]:
        if self.last_route is None or self.last_auxiliary is None:
            return {}
        route = self.last_route.to_dict(index=index)
        modality_prob = self.last_auxiliary.modality_probabilities[index].detach().cpu()
        scene_prob = self.last_auxiliary.scene_probabilities[index].detach().cpu()
        modality_id = int(modality_prob.argmax().item())
        scene_id = int(scene_prob.argmax().item())
        route.update(
            {
                "aux_modality": {
                    "label": self.sparse_moe_config.modality_names[modality_id],
                    "confidence": float(modality_prob[modality_id].item()),
                    "probabilities": {
                        name: float(value)
                        for name, value in zip(self.sparse_moe_config.modality_names, modality_prob.tolist())
                    },
                },
                "aux_scene": {
                    "label": self.sparse_moe_config.scene_names[scene_id],
                    "confidence": float(scene_prob[scene_id].item()),
                    "probabilities": {
                        name: float(value)
                        for name, value in zip(self.sparse_moe_config.scene_names, scene_prob.tolist())
                    },
                },
            }
        )
        return route

    def metadata(self) -> dict[str, Any]:
        return {
            "config": self.sparse_moe_config.to_dict(),
            "feature_channels": list(self._feature_channels),
            "usage": self.usage_tracker.to_dict(),
            "anchors": self.anchor_bank.summary(),
        }


if _UltralyticsDetectionModel is not None:

    class SparseMoEDetectionModel(_UltralyticsDetectionModel):
        """Ultralytics DetectionModel subclass with a Detect-input MoE block."""

        def __init__(
            self,
            cfg: str | dict = "yolo26n.yaml",
            ch: int = 3,
            nc: int | None = None,
            verbose: bool = True,
            sparse_moe_config: SparseMoEConfig | None = None,
        ) -> None:
            super().__init__(cfg, ch=ch, nc=nc, verbose=verbose)
            self.attach_sparse_moe(sparse_moe_config or SparseMoEConfig())

        def attach_sparse_moe(self, config: SparseMoEConfig) -> SparseMoEDetectAdapter:
            for index in range(len(self.model) - 1, -1, -1):
                head = self.model[index]
                if isinstance(head, SparseMoEDetectAdapter):
                    return head
                if isinstance(head, _UltralyticsDetect):
                    wrapped = SparseMoEDetectAdapter(head, config)
                    self.model[index] = wrapped
                    return wrapped
            raise TypeError("Sparse-MoE v1 requires a Detect-compatible model head")

        @property
        def sparse_moe(self) -> SparseMoEDetectAdapter:
            head = get_sparse_moe_adapter(self)
            if head is None:
                raise AttributeError("model has no Sparse-MoE adapter")
            return head

        def forward(self, x, *args, **kwargs):
            if isinstance(x, torch.Tensor):
                adapter = get_sparse_moe_adapter(self)
                if adapter is not None:
                    adapter.set_input_quality(compute_input_quality_stats(x))
            return super().forward(x, *args, **kwargs)

        def loss(self, batch, preds=None):
            if preds is None:
                preds = self.forward(batch["img"])
            regular_loss, regular_items = super().loss(batch, preds)
            components = self.sparse_moe.loss_components(batch)
            values = torch.stack(
                [
                    self.sparse_moe.sparse_moe_config.modality_loss_weight * components["modality"],
                    self.sparse_moe.sparse_moe_config.scene_loss_weight * components["scene"],
                    self.sparse_moe.sparse_moe_config.balance_loss_weight * components["balance"],
                    self.sparse_moe.sparse_moe_config.router_z_loss_weight * components["router_z"],
                    self.sparse_moe.sparse_moe_config.anchor_loss_weight * components["anchor"],
                ]
            )
            batch_size = batch["img"].shape[0]
            return torch.cat([regular_loss.reshape(-1), values * batch_size]), torch.cat(
                [regular_items.reshape(-1), values.detach()]
            )


else:  # pragma: no cover - only used when Ultralytics is not installed

    class SparseMoEDetectionModel(nn.Module):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__()
            raise ModuleNotFoundError("Sparse-MoE DetectionModel requires ultralytics>=8.4,<8.5")


def get_sparse_moe_adapter(model: nn.Module) -> SparseMoEDetectAdapter | None:
    candidate = model
    if not isinstance(candidate, nn.Module) and hasattr(candidate, "model"):
        candidate = candidate.model
    if hasattr(candidate, "student_model"):
        candidate = candidate.student_model
    layers = getattr(candidate, "model", None)
    if layers is None or not len(layers):
        return None
    for head in reversed(list(layers)):
        if isinstance(head, SparseMoEDetectAdapter):
            return head
    return None


def _normalise_class_names(value: object) -> list[str]:
    """Return class names from the list/dict forms used by YOLO checkpoints."""

    if isinstance(value, dict):
        names: list[str] = []
        for index in range(len(value)):
            if index in value:
                item = value[index]
            elif str(index) in value:
                item = value[str(index)]
            else:
                return []
            names.append(str(item))
        return names
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return []


def _iter_module_attribute(value: object) -> tuple[nn.Module, ...]:
    """Safely iterate an optional Ultralytics head branch.

    Ultralytics 8.4.x has optional ``one2one_cv3``/``cv3`` attributes.  Some
    model variants install the attribute with ``None`` rather than omitting
    it, while ``DetectionModel._remap_cls_by_names`` assumes it is iterable.
    Keeping this guard local lets us load checkpoints without mutating the
    live model or relying on that private, version-specific implementation.
    """

    if value is None:
        return ()
    if isinstance(value, (nn.ModuleList, nn.Sequential, list, tuple)):
        return tuple(item for item in value if isinstance(item, nn.Module))
    try:
        return tuple(item for item in value if isinstance(item, nn.Module))  # type: ignore[union-attr]
    except TypeError:
        return ()


def _classification_state_keys(model: nn.Module, state_dict: dict[str, torch.Tensor], class_count: int) -> set[str]:
    """Find Detect class-logit tensors while tolerating optional ``None`` heads."""

    keys: set[str] = set()
    for module_name, module in model.named_modules():
        for attribute in ("cv3", "one2one_cv3"):
            branches = _iter_module_attribute(getattr(module, attribute, None))
            for branch_index, branch in enumerate(branches):
                layers = _iter_module_attribute(branch)
                if not layers:
                    continue
                last = layers[-1]
                if getattr(last, "out_channels", None) != class_count:
                    continue
                prefix = f"{module_name}.{attribute}.{branch_index}.{len(layers) - 1}"
                for parameter_name in ("weight", "bias"):
                    key = f"{prefix}.{parameter_name}"
                    if key in state_dict:
                        keys.add(key)
    return keys


def _source_key_aliases(key: str) -> tuple[str, ...]:
    """Map plain Detect paths to the wrapped ``detect_head`` paths."""

    candidates = [key]
    parts = key.split(".")
    if len(parts) >= 3 and parts[0] == "model" and parts[1].isdigit():
        layer_prefix = ".".join(parts[:2])
        suffix = ".".join(parts[2:])
        candidates.append(f"{layer_prefix}.detect_head.{suffix}")
    return tuple(candidates)


def _remap_classification_rows(
    aligned: dict[str, torch.Tensor],
    source_by_target_key: dict[str, torch.Tensor],
    target_state: dict[str, torch.Tensor],
    source_model: nn.Module,
    target_model: nn.Module,
) -> int:
    """Copy overlapping class rows by name and leave new rows initialized.

    This is intentionally implemented here instead of calling Ultralytics'
    private ``_remap_cls_by_names``.  The latter changed across releases and
    8.4.100 iterates an optional head attribute whose value may be ``None``.
    """

    source_names = _normalise_class_names(getattr(source_model, "names", None))
    target_names = _normalise_class_names(getattr(target_model, "names", None))
    if not source_names or not target_names:
        return 0
    source_lookup = {name.strip().casefold(): index for index, name in enumerate(source_names)}
    mapping = [source_lookup.get(name.strip().casefold(), -1) for name in target_names]
    valid = [(target_index, source_index) for target_index, source_index in enumerate(mapping) if source_index >= 0]
    if not valid:
        return 0

    remapped = 0
    class_keys = _classification_state_keys(target_model, target_state, len(target_names))
    for key in class_keys:
        source_value = source_by_target_key.get(key)
        target_value = target_state[key]
        if source_value is None or source_value.ndim == 0:
            continue
        if source_value.shape[0] != len(source_names) or source_value.shape[1:] != target_value.shape[1:]:
            continue
        value = target_value.detach().clone()
        for target_index, source_index in valid:
            value[target_index] = source_value[source_index].to(device=value.device, dtype=value.dtype)
        aligned[key] = value
        remapped += 1
    return remapped


def load_sparse_weights(model: nn.Module, weights: object, verbose: bool = True) -> int:
    """Load plain or sparse Ultralytics weights into a freshly built sparse model."""

    source_model = weights.get("model") if isinstance(weights, dict) else weights
    if not isinstance(source_model, nn.Module) and hasattr(source_model, "model"):
        source_model = source_model.model
    if not isinstance(source_model, nn.Module):
        raise TypeError("weights must contain a torch.nn.Module")
    source_state = source_model.float().state_dict()
    target_state = model.state_dict()
    # Keep shape-mismatched class tensors available for the explicit row
    # remapper below.  A four-class checkpoint cannot pass the ordinary
    # ``shape ==`` intersection for a six-class student head.
    source_by_target_key: dict[str, torch.Tensor] = {}
    for key, value in source_state.items():
        for candidate in _source_key_aliases(key):
            if candidate in target_state:
                source_by_target_key.setdefault(candidate, value)
    aligned = {
        key: value
        for key, value in source_by_target_key.items()
        if target_state[key].shape == value.shape
    }
    remapped = _remap_classification_rows(
        aligned,
        source_by_target_key,
        target_state,
        source_model,
        model,
    )
    model.load_state_dict(aligned, strict=False)
    source_adapter = get_sparse_moe_adapter(source_model)
    target_adapter = get_sparse_moe_adapter(model)
    if source_adapter is not None and target_adapter is not None:
        # The anchor bank is intentionally non-parameter state, so it is not
        # included in ``state_dict``. Carry it explicitly across Class-IL
        # stages while starting a fresh usage counter for the current stage.
        target_adapter.anchor_bank.load_state_dict(source_adapter.anchor_bank.state_dict())
    if verbose:
        print(f"Transferred {len(aligned)}/{len(target_state)} items into Sparse-MoE model ({remapped} class tensors remapped)")
    return len(aligned)


def build_sparse_moe_model(
    cfg: str | dict,
    *,
    channels: int = 3,
    class_count: int | None = None,
    config: SparseMoEConfig | None = None,
    verbose: bool = True,
) -> SparseMoEDetectionModel:
    return SparseMoEDetectionModel(
        cfg,
        ch=channels,
        nc=class_count,
        verbose=verbose,
        sparse_moe_config=config,
    )


__all__ = [
    "SparseMoEConfig",
    "SparseMoEDetectAdapter",
    "SparseMoEDetectionModel",
    "build_sparse_moe_model",
    "compute_input_quality_stats",
    "detect_input_channels",
    "get_sparse_moe_adapter",
    "load_sparse_weights",
]
