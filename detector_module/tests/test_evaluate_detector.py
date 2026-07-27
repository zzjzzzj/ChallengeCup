from __future__ import annotations

import unittest
from pathlib import Path

from detector_module.dataset import DetectionSample
from detector_module.evaluate_detector import build_evaluation_slices


class DetectorEvaluationTests(unittest.TestCase):
    def test_overall_only_avoids_sensor_and_scene_slice_jobs(self):
        samples = [
            DetectionSample(Path("ir.png"), "ir.png", "ir", "air", "test", 1),
            DetectionSample(Path("sar.png"), "sar.png", "sar", "sea", "test", 2),
        ]

        slices = build_evaluation_slices(samples, overall_only=True)

        self.assertEqual([(group, value) for group, value, _ in slices], [("overall", "all")])


if __name__ == "__main__":
    unittest.main()
