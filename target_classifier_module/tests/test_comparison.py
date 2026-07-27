from __future__ import annotations

import json
import unittest

from target_classifier_module.compare_baselines import write_baseline_comparison
from test_support import workspace_test_directory


class BaselineComparisonTests(unittest.TestCase):
    def test_report_keeps_crop_classification_and_detection_metrics_separate(self):
        with workspace_test_directory("baseline-comparison") as root:
            classifier = root / "classifier.json"
            detector = root / "detector.json"
            whole_image = root / "whole_image.json"
            output = root / "comparison.md"
            classifier.write_text(
                json.dumps(
                    {
                        "test": {"accuracy": 0.99, "macro_f1": 0.98, "sample_count": 100},
                        "scope_warning": "known boxes only",
                    }
                ),
                encoding="utf-8",
            )
            detector.write_text(
                json.dumps(
                    {
                        "test_image_count": 20,
                        "slices": [
                            {
                                "group": "overall",
                                "value": "all",
                                "precision": 0.8,
                                "recall": 0.7,
                                "map50": 0.75,
                                "map50_95": 0.4,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            whole_image.write_text(
                json.dumps(
                    {
                        "test": {
                            "exact_match_accuracy": 0.91,
                            "macro_f1": 0.92,
                            "sample_count": 20,
                        }
                    }
                ),
                encoding="utf-8",
            )

            write_baseline_comparison(
                classifier,
                detector,
                output,
                whole_image_metrics_path=whole_image,
            )

            report = output.read_text(encoding="utf-8")
            self.assertIn("真实框裁剪分类", report)
            self.assertIn("整图多标签识别", report)
            self.assertIn("完整目标检测", report)
            self.assertIn("不能直接比较", report)


if __name__ == "__main__":
    unittest.main()
