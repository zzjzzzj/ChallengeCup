from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from Agent.continual.anchors.torch_expert_anchor import TorchExpertAnchorBank
from Agent.models.aux_heads.heads import (
    ModalitySceneAuxiliaryHeads,
    masked_auxiliary_loss,
)
from Agent.models.experts.sparse_adapter import SparseExpertAdapter, SparseExpertAdapterBank
from Agent.models.experts.torch_router import TorchExpertUsageTracker, TorchSparseExpertRouter
from scene_recognition.detector_module.sparse_moe_checkpoint import (
    load_sparse_moe_checkpoint,
    restore_sparse_moe_usage,
    save_sparse_moe_checkpoint,
    sparse_moe_usage_state,
)
from scene_recognition.detector_module.sparse_moe_model import (
    SparseMoEConfig,
    SparseMoEDetectAdapter,
    compute_input_quality_stats,
)


class _FakeDetect(nn.Module):
    """Minimal Detect-shaped module for checkpoint round-trip coverage."""

    def __init__(self) -> None:
        super().__init__()
        self.nc = 1
        self.reg_max = 1
        self.no = 5
        self.stride = torch.tensor([8.0])
        self.f = -1
        self.i = 0
        self.type = "FakeDetect"

    def forward(self, features):  # pragma: no cover - checkpoint test does not infer
        return features


class _FakeDetectionModel(nn.Module):
    def __init__(self, adapter: SparseMoEDetectAdapter) -> None:
        super().__init__()
        self.model = nn.ModuleList([adapter])


class SparseMoEComponentTests(unittest.TestCase):
    def test_adapter_is_identity_initialized_and_bank_is_sparse(self) -> None:
        torch.manual_seed(3)
        features = torch.randn(2, 8, 6, 6, requires_grad=True)
        adapter = SparseExpertAdapter(8, bottleneck_ratio=0.25)
        self.assertEqual(adapter.bottleneck_channels, 2)
        self.assertTrue(torch.equal(adapter(features), features))

        bank = SparseExpertAdapterBank(8, expert_count=5, bottleneck_ratio=0.25)
        ids = torch.tensor([[0, 1], [1, 3]])
        weights = torch.tensor([[0.7, 0.3], [0.2, 0.8]])
        output = bank(features.detach(), ids, weights)
        self.assertEqual(output.shape, features.shape)
        self.assertEqual(bank.execution_counts(), [1, 1, 0, 1, 0])

    def test_router_is_true_top_two_and_balancing_is_finite(self) -> None:
        router = TorchSparseExpertRouter(6, expert_count=5, top_k=2, hidden_dim=10)
        route = router(torch.randn(4, 6), temperature=1.5)
        self.assertEqual(route.expert_ids.shape, (4, 2))
        self.assertTrue(torch.all(route.expert_weights > 0))
        self.assertTrue(torch.allclose(route.expert_weights.sum(dim=1), torch.ones(4)))
        self.assertTrue(torch.all(route.expert_ids[:, 0] != route.expert_ids[:, 1]))
        self.assertTrue(torch.isfinite(router.load_balance_loss(route)))
        self.assertTrue(torch.isfinite(router.router_z_loss(route)))

        tracker = TorchExpertUsageTracker(5)
        tracker.update(route)
        report = tracker.to_dict()
        self.assertEqual(report["total_images"], 4)
        self.assertEqual(sum(report["top_k_activations"].values()), 8)
        self.assertAlmostEqual(sum(report["mean_probability"].values()), 1.0, places=5)
        restored = TorchExpertUsageTracker(5)
        restored.load_state_dict(tracker.state_dict())
        self.assertEqual(restored.to_dict(), report)

    def test_quality_stats_are_finite_for_degenerate_image_shapes(self) -> None:
        stats = compute_input_quality_stats(torch.ones(2, 1, 1, 1))
        self.assertEqual(stats.shape, (2, 4))
        self.assertTrue(torch.isfinite(stats).all())

    def test_adapter_bank_mixed_dtype_is_safe_under_cpu_autocast(self) -> None:
        torch.manual_seed(17)
        bank = SparseExpertAdapterBank(4, expert_count=2, bottleneck_ratio=0.5)
        features = torch.randn(2, 4, 6, 6, requires_grad=True)
        expert_ids = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        expert_weights = torch.tensor(
            [[0.75, 0.25], [0.25, 0.75]], dtype=torch.float32, requires_grad=True
        )
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            output = bank(features, expert_ids, expert_weights)
            loss = output.square().mean()
        self.assertEqual(output.dtype, features.dtype)
        self.assertTrue(torch.isfinite(output).all())
        loss.backward()
        self.assertIsNotNone(features.grad)
        self.assertIsNotNone(expert_weights.grad)

    def test_masked_auxiliary_loss_skips_unknown_metadata(self) -> None:
        heads = ModalitySceneAuxiliaryHeads([4, 8], hidden_channels=12)
        outputs = heads((torch.randn(3, 4, 5, 5), torch.randn(3, 8, 3, 3)))
        losses = masked_auxiliary_loss(
            outputs,
            modality_targets=["ir", "unknown", "sar"],
            scene_targets=["air", "unknown", "forest"],
        )
        self.assertTrue(torch.isfinite(losses["total"]))
        unknown = masked_auxiliary_loss(outputs, [-1, -1, -1], [-1, -1, -1])
        self.assertEqual(float(unknown["total"]), 0.0)
        self.assertTrue(unknown["total"].requires_grad)

    def test_anchor_is_zero_at_t1_and_differentiable_after_update(self) -> None:
        bank = TorchExpertAnchorBank(rho=0.95)
        first = torch.tensor([1.0, 2.0], requires_grad=True)
        t1 = bank.penalty("expert_0", first)
        self.assertEqual(float(t1), 0.0)
        self.assertTrue(t1.requires_grad)
        bank.update_anchor("expert_0", first, activation_frequency=1.0)
        current = (first.detach() + 1.0).requires_grad_()
        penalty = bank.penalty("expert_0", current)
        self.assertGreater(float(penalty), 0.0)
        penalty.backward()
        self.assertIsNotNone(current.grad)

    def test_sparse_checkpoint_round_trip_keeps_config_and_anchor(self) -> None:
        config = SparseMoEConfig(expert_count=3, top_k=2, aux_hidden=16, router_hidden=16)
        adapter = SparseMoEDetectAdapter(_FakeDetect(), config, input_channels=[4])
        model = _FakeDetectionModel(adapter)
        parameter = next(adapter.expert_pools[0].adapters[0].parameters())
        adapter.anchor_bank.update_anchor("expert_0", parameter, activation_frequency=0.5)
        with tempfile.TemporaryDirectory() as tmp:
            path = save_sparse_moe_checkpoint(model, Path(tmp) / "sparse.pt", extra={"stage": 1})
            restored = load_sparse_moe_checkpoint(path)
        restored_adapter = restored.model[-1]
        self.assertEqual(restored_adapter.sparse_moe_config.expert_count, 3)
        self.assertTrue(restored_adapter.anchor_bank.has_anchors)
        self.assertEqual(restored_adapter.anchor_bank.importance["expert_0"], 0.5)

    def test_sparse_usage_snapshot_survives_best_model_reload(self) -> None:
        config = SparseMoEConfig(expert_count=3, top_k=2, aux_hidden=16, router_hidden=16)
        source_adapter = SparseMoEDetectAdapter(_FakeDetect(), config, input_channels=[4])
        source_model = _FakeDetectionModel(source_adapter)
        source_state = source_adapter.usage_tracker.state_dict()
        source_state["total_images"] = 7
        source_state["top_counts"] = [4, 5, 0]
        source_state["soft_probability_sum"] = [3.0, 4.0, 0.0]
        source_adapter.usage_tracker.load_state_dict(source_state)
        captured = sparse_moe_usage_state(source_model)
        self.assertIsNotNone(captured)
        self.assertEqual(captured["total_images"], 7)

        reloaded_adapter = SparseMoEDetectAdapter(_FakeDetect(), config, input_channels=[4])
        reloaded_model = _FakeDetectionModel(reloaded_adapter)
        self.assertTrue(restore_sparse_moe_usage(reloaded_model, captured))
        self.assertEqual(reloaded_adapter.usage_tracker.to_dict()["total_images"], 7)
        self.assertGreater(reloaded_adapter.usage_tracker.to_dict()["max_occupancy"], 0.0)


if __name__ == "__main__":
    unittest.main()
