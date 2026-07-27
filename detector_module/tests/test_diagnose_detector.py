from __future__ import annotations

import unittest
from pathlib import Path

from detector_module.boxes import YoloBox, box_iou, parse_yolo_boxes, size_bucket
from detector_module.dataset import DetectionSample
from detector_module.diagnose_detector import (
    build_diagnostics,
)
from test_support import workspace_test_directory


class DetectorDiagnosisTests(unittest.TestCase):
    def test_box_iou_for_equal_boxes_is_one(self):
        box = YoloBox(0, 0.5, 0.5, 0.2, 0.2)
        self.assertAlmostEqual(box_iou(box, box), 1.0)

    def test_parse_yolo_boxes_reads_normalized_values(self):
        with workspace_test_directory("parse-box") as root:
            label = root / "sample.txt"
            label.write_text("2 0.4 0.5 0.1 0.2\n", encoding="utf-8")

            boxes = parse_yolo_boxes(label, 4)

            self.assertEqual(boxes, [YoloBox(2, 0.4, 0.5, 0.1, 0.2)])

    def test_size_bucket_uses_normalized_area(self):
        self.assertEqual(size_bucket(YoloBox(0, 0.5, 0.5, 0.02, 0.02)), "tiny(<0.25%)")
        self.assertEqual(size_bucket(YoloBox(0, 0.5, 0.5, 0.05, 0.1)), "small(0.25%-1%)")
        self.assertEqual(size_bucket(YoloBox(0, 0.5, 0.5, 0.2, 0.1)), "medium(1%-4%)")
        self.assertEqual(size_bucket(YoloBox(0, 0.5, 0.5, 0.3, 0.3)), "large(>=4%)")

    def test_build_diagnostics_counts_hits_misses_and_false_positives(self):
        with workspace_test_directory("detector-diagnosis") as root:
            image = root / "case.png"
            image.write_bytes(b"image")
            sample = DetectionSample(
                image_path=image,
                image_name=image.name,
                sensor="sar",
                scene="urban",
                split="test",
                sequence_index=1,
            )
            key = str(image.resolve())
            ground_truth = {
                key: [
                    YoloBox(0, 0.5, 0.5, 0.2, 0.2),
                    YoloBox(3, 0.2, 0.2, 0.1, 0.1),
                ]
            }
            predictions = {
                key: [
                    YoloBox(0, 0.5, 0.5, 0.2, 0.2, 0.9),
                    YoloBox(1, 0.8, 0.8, 0.1, 0.1, 0.8),
                ]
            }

            report = build_diagnostics(
                [sample],
                ground_truth,
                predictions,
                ["soldier", "small_aircraft", "warship", "tank"],
                0.5,
            )

            by_class = {row["class"]: row for row in report["class_summary"]}
            self.assertEqual(by_class["soldier"]["matched"], 1)
            self.assertEqual(by_class["tank"]["missed"], 1)
            self.assertEqual(by_class["small_aircraft"]["false_positive"], 1)
            self.assertEqual(report["top_missed_samples"][0]["missed_classes"], "tank")


if __name__ == "__main__":
    unittest.main()
