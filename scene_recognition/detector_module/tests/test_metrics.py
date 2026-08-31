from __future__ import annotations

import unittest
from types import SimpleNamespace

from scene_recognition.detector_module import ALL_CLASS_NAMES
from scene_recognition.detector_module.metrics import detection_metrics_to_dict


class _FakeBox:
    def __init__(self, present: list[int], rows: list[tuple[float, float, float, float]]) -> None:
        self.ap_class_index = present
        self._rows = rows
        # This intentionally contains an overall value for absent classes,
        # matching Ultralytics Metric.maps semantics.
        self.maps = [0.199 for _ in ALL_CLASS_NAMES]
        self.mp = 0.61
        self.mr = 0.52
        self.map50 = 0.41
        self.map = 0.31
        self.map75 = 0.35

    def class_result(self, row_index: int) -> tuple[float, float, float, float]:
        return self._rows[row_index]


def _metrics(box: _FakeBox) -> SimpleNamespace:
    return SimpleNamespace(
        box=box,
        fitness=0.29,
        speed={"preprocess": 1.0},
        results_dict={"metrics/mAP50(B)": 0.41},
    )


class DetectionMetricsClassMappingTests(unittest.TestCase):
    def test_absent_class_is_none_in_all_per_class_metrics(self) -> None:
        rows = [(0.10 + index, 0.20 + index, 0.30 + index, 0.40 + index) for index in range(5)]
        result = detection_metrics_to_dict(_metrics(_FakeBox([0, 1, 2, 3, 4], rows)), ALL_CLASS_NAMES)

        self.assertEqual(result["precision"], 0.61)
        self.assertEqual(result["map50"], 0.41)
        self.assertEqual(result["map50_95"], 0.31)
        self.assertEqual(result["per_class"][ALL_CLASS_NAMES[4]]["map50"], 4.3)
        self.assertEqual(result["per_class"][ALL_CLASS_NAMES[4]]["map50_95"], 4.4)
        self.assertEqual(result["per_class"][ALL_CLASS_NAMES[5]], {
            "precision": None,
            "recall": None,
            "map50": None,
            "map50_95": None,
        })

    def test_sparse_present_ids_use_compact_result_rows_without_shift(self) -> None:
        rows = [
            (0.11, 0.12, 0.13, 0.14),
            (0.21, 0.22, 0.23, 0.24),
            (0.51, 0.52, 0.53, 0.54),
        ]
        result = detection_metrics_to_dict(_metrics(_FakeBox([0, 2, 5], rows)), ALL_CLASS_NAMES)
        per_class = result["per_class"]

        self.assertEqual(per_class[ALL_CLASS_NAMES[0]]["map50"], 0.13)
        self.assertEqual(per_class[ALL_CLASS_NAMES[2]]["map50"], 0.23)
        self.assertEqual(per_class[ALL_CLASS_NAMES[5]]["map50"], 0.53)
        self.assertEqual(per_class[ALL_CLASS_NAMES[0]]["map50_95"], 0.14)
        self.assertIsNone(per_class[ALL_CLASS_NAMES[1]]["map50"])
        self.assertIsNone(per_class[ALL_CLASS_NAMES[3]]["map50_95"])


if __name__ == "__main__":
    unittest.main()
