from __future__ import annotations

import unittest

from scene_recognition.detector_module import ALL_CLASS_NAMES
from scene_recognition.detector_module.train_class_incremental_yolo import (
    build_class_incremental_metrics,
)


class ClassIncrementalMetricTests(unittest.TestCase):
    def make_stage(self, stage: int) -> dict:
        learned = ALL_CLASS_NAMES[:stage]
        per_class = {
            name: {
                "map50": 0.9 - 0.02 * (stage - class_index - 1),
                "map50_95": 0.5 - 0.01 * (stage - class_index - 1),
            }
            for class_index, name in enumerate(learned)
        }
        return {"learned_classes": learned, "validation": {"per_class": per_class}}

    def test_builds_six_by_six_matrix_and_forgetting_metrics(self) -> None:
        report = build_class_incremental_metrics(
            [self.make_stage(stage) for stage in range(1, 7)],
            ALL_CLASS_NAMES,
        )
        rows = report["map50"]["rows"]
        self.assertEqual(len(rows), 6)
        self.assertIsNone(rows[0]["small_aircraft"])
        self.assertAlmostEqual(rows[-1]["armored_vehicle"], 0.9)
        self.assertGreater(report["map50"]["average_forgetting"], 0.0)
        self.assertLess(report["map50"]["backward_transfer"], 0.0)
        self.assertFalse(report["official"])

    def test_rejects_missing_stage(self) -> None:
        with self.assertRaisesRegex(ValueError, "每个类别"):
            build_class_incremental_metrics([], ALL_CLASS_NAMES)


if __name__ == "__main__":
    unittest.main()
