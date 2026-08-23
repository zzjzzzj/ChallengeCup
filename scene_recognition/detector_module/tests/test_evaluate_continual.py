from __future__ import annotations

import unittest

from scene_recognition.detector_module.evaluate_continual import build_continual_metrics


def metrics(values: dict[str, tuple[int, float, float]]) -> dict:
    return {
        "per_class": {
            name: {"support": support, "map50": map50, "map50_95": map50_95}
            for name, (support, map50, map50_95) in values.items()
        }
    }


class ContinualMetricTests(unittest.TestCase):
    def test_reports_new_map_and_knowledge_retention_ratio(self) -> None:
        old = ["soldier", "tank"]
        new = ["patrol_boat", "armored_vehicle"]
        before = metrics(
            {
                "soldier": (10, 0.8, 0.4),
                "tank": (10, 0.6, 0.3),
                "patrol_boat": (10, 0.0, 0.0),
                "armored_vehicle": (10, 0.0, 0.0),
            }
        )
        after = metrics(
            {
                "soldier": (10, 0.72, 0.36),
                "tank": (10, 0.54, 0.27),
                "patrol_boat": (10, 0.7, 0.35),
                "armored_vehicle": (10, 0.5, 0.25),
            }
        )

        result = build_continual_metrics(before, after, old, new)

        self.assertAlmostEqual(result["map50"]["old_map_before"]["value"], 0.7)
        self.assertAlmostEqual(result["map50"]["old_map_after"]["value"], 0.63)
        self.assertAlmostEqual(result["map50"]["new_map"]["value"], 0.6)
        self.assertAlmostEqual(result["map50"]["krr"], 0.9)
        self.assertTrue(result["evaluation_ready"])


if __name__ == "__main__":
    unittest.main()
