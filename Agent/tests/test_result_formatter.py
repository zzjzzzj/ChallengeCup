from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from result_formatter import build_image_summary, summarize_prediction_payload, write_summary_csv


class ResultFormatterTests(unittest.TestCase):
    def test_groups_targets_and_generates_fixed_chinese_description(self) -> None:
        summary = build_image_summary(
            [
                {"name": "warship", "confidence": 0.91},
                {"name": "warship", "confidence": 0.88},
                {"name": "patrol_boat", "confidence": 0.82},
            ],
            scene_label="sea",
            modality_label="sar",
        )

        self.assertEqual(summary["target_type_count"], 2)
        self.assertEqual(summary["target_total_count"], 3)
        self.assertEqual(summary["targets"][0]["label_cn"], "轮船/舰船")
        self.assertIn("图像场景分类：海洋", summary["description"])
        self.assertIn("轮船/舰船 2 个（置信度：0.91、0.88）", summary["description"])
        self.assertNotIn("图像模态", summary["description"])

    def test_no_detection_and_310b_payload_are_supported(self) -> None:
        summary = build_image_summary([], scene_label="forest")
        self.assertEqual(summary["target_total_count"], 0)
        self.assertEqual(summary["description"], "图像场景分类：森林。未检测到目标。")

        rows = summarize_prediction_payload(
            {"image": "demo.png", "detections": [{"name": "tank", "confidence": 0.75}]}
        )
        self.assertEqual(rows[0]["scene_name"], "未提供")
        self.assertEqual(rows[0]["target_total_count"], 1)

    def test_csv_uses_utf8_bom(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "summary.csv"
            write_summary_csv(output, [{"image": "demo.png", "description": "未检测到目标"}])
            content = output.read_bytes()
        self.assertTrue(content.startswith(b"\xef\xbb\xbf"))
        self.assertIn("description".encode("utf-8"), content)


if __name__ == "__main__":
    unittest.main()
