from __future__ import annotations

import unittest

import torch
from torch import nn

from scene_recognition.detector_module import ALL_CLASS_NAMES, BASE_CLASS_NAMES
from scene_recognition.detector_module.sparse_moe_model import load_sparse_weights


class _DetectWithOptionalHead(nn.Module):
    """Small Detect-shaped module matching Ultralytics' class branches."""

    def __init__(self, class_count: int) -> None:
        super().__init__()
        self.nc = class_count
        self.cv3 = nn.ModuleList([nn.Sequential(nn.Conv2d(2, class_count, 1))])
        # Ultralytics 8.4.100 may expose this optional branch as None.  Its
        # private remapper assumes every such attribute is iterable.
        self.one2one_cv3 = None


class _FakeDetectionModel(nn.Module):
    def __init__(self, class_count: int, names: object) -> None:
        super().__init__()
        self.model = nn.ModuleList([_DetectWithOptionalHead(class_count)])
        self.names = names

    def _remap_cls_by_names(self, *args, **kwargs):  # pragma: no cover - must not be called
        # Reproduce the 8.4.100 failure mode if load_sparse_weights delegates
        # to the private helper instead of using its guarded implementation.
        for module in self.modules():
            for attribute in ("cv3", "one2one_cv3"):
                for _ in getattr(module, attribute, ()):
                    pass
        return 0


class _FakeSparseDetectionModel(_FakeDetectionModel):
    def __init__(self, class_count: int, names: object) -> None:
        nn.Module.__init__(self)
        self.model = nn.ModuleList(
            [nn.ModuleDict({"detect_head": _DetectWithOptionalHead(class_count)})]
        )
        self.names = names


class SparseMoEWeightTransferTests(unittest.TestCase):
    def test_four_class_rows_transfer_to_six_class_head_with_none_optional_head(self) -> None:
        source = _FakeDetectionModel(
            len(BASE_CLASS_NAMES),
            {index: name for index, name in enumerate(BASE_CLASS_NAMES)},
        )
        # The target mirrors SparseMoEDetectionModel: the plain Detect head is
        # wrapped below ``detect_head``, so source keys need path aliasing.
        target = _FakeSparseDetectionModel(len(ALL_CLASS_NAMES), list(ALL_CLASS_NAMES))
        with torch.no_grad():
            for index in range(len(BASE_CLASS_NAMES)):
                source.model[0].cv3[0][0].weight[index].fill_(float(index + 1))
                source.model[0].cv3[0][0].bias[index].fill_(float(index + 11))
            target.model[0]["detect_head"].cv3[0][0].weight.fill_(-9.0)
            target.model[0]["detect_head"].cv3[0][0].bias.fill_(-9.0)

        # Calling the simulated upstream helper directly proves the fixture
        # contains the reported None-attribute failure, while the loader must
        # remain independent of that private implementation.
        with self.assertRaises(TypeError):
            target._remap_cls_by_names({}, source)
        transferred = load_sparse_weights(target, source, verbose=False)

        self.assertGreaterEqual(transferred, 2)
        target_weight = target.model[0]["detect_head"].cv3[0][0].weight.detach()
        target_bias = target.model[0]["detect_head"].cv3[0][0].bias.detach()
        source_weight = source.model[0].cv3[0][0].weight.detach()
        source_bias = source.model[0].cv3[0][0].bias.detach()
        self.assertTrue(torch.equal(target_weight[:4], source_weight))
        self.assertTrue(torch.equal(target_bias[:4], source_bias))
        self.assertTrue(torch.all(target_weight[4:] == -9.0))
        self.assertTrue(torch.all(target_bias[4:] == -9.0))


if __name__ == "__main__":
    unittest.main()
