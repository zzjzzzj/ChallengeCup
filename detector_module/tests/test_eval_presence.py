from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from detector_module.boxes import YoloBox
from detector_module.eval_presence import (
    apply_thresholds,
    extract_detections,
    max_confidence_scores,
    presence_vector_from_boxes,
    read_presence_manifest,
    resolve_ultralytics_device,
)
from target_classifier_module.whole_image import (
    compute_multilabel_metrics,
    optimize_thresholds,
)


class FakeBoxes:
    """Stand-in for ultralytics Results.boxes so tests need no weights or images."""

    def __init__(self, class_ids, confidences):
        self.cls = np.asarray(class_ids, dtype=np.float32)
        self.conf = np.asarray(confidences, dtype=np.float32)


class FakeResult:
    def __init__(self, class_ids, confidences, path="ir_r1_base_air_000001.png"):
        self.boxes = FakeBoxes(class_ids, confidences)
        self.path = path


class PresenceAggregationTests(unittest.TestCase):
    def test_ground_truth_presence_is_one_when_any_box_carries_the_class(self):
        boxes = [
            YoloBox(0, 0.5, 0.5, 0.1, 0.1),
            YoloBox(3, 0.2, 0.2, 0.2, 0.2),
            YoloBox(0, 0.8, 0.8, 0.05, 0.05),
        ]

        vector = presence_vector_from_boxes(boxes, 4)

        self.assertEqual(vector.tolist(), [1, 0, 0, 1])

    def test_empty_label_file_produces_all_zero_presence_vector(self):
        self.assertEqual(presence_vector_from_boxes([], 4).tolist(), [0, 0, 0, 0])

    def test_class_score_is_the_highest_confidence_and_missing_class_is_zero(self):
        detections = [(0, 0.31), (0, 0.87), (2, 0.44), (0, 0.12)]

        scores = max_confidence_scores(detections, 4)

        self.assertAlmostEqual(float(scores[0]), 0.87, places=6)
        self.assertAlmostEqual(float(scores[1]), 0.0, places=6)
        self.assertAlmostEqual(float(scores[2]), 0.44, places=6)
        self.assertAlmostEqual(float(scores[3]), 0.0, places=6)

    def test_extract_detections_reads_class_and_confidence_from_a_result(self):
        result = FakeResult([1.0, 1.0, 2.0], [0.9, 0.2, 0.55])

        detections = extract_detections(result)

        self.assertEqual([class_id for class_id, _ in detections], [1, 1, 2])
        for actual, expected in zip(
            [confidence for _, confidence in detections], [0.9, 0.2, 0.55]
        ):
            self.assertAlmostEqual(actual, expected, places=6)
        self.assertAlmostEqual(
            float(max_confidence_scores(detections, 4)[1]), 0.9, places=6
        )

    def test_extract_detections_on_an_image_without_boxes_is_empty(self):
        self.assertEqual(extract_detections(FakeResult([], [])), [])
        self.assertEqual(
            max_confidence_scores(extract_detections(FakeResult([], [])), 4).tolist(),
            [0.0, 0.0, 0.0, 0.0],
        )

    def test_out_of_range_class_id_is_rejected(self):
        with self.assertRaises(ValueError):
            max_confidence_scores([(4, 0.9)], 4)


class ThresholdApplicationTests(unittest.TestCase):
    def test_thresholds_are_applied_per_class_with_greater_or_equal(self):
        scores = np.array(
            [[0.90, 0.05, 0.00, 0.60], [0.10, 0.30, 0.95, 0.20]], dtype=np.float32
        )
        thresholds = [0.50, 0.25, 0.50, 0.60]

        predicted = apply_thresholds(scores, thresholds)

        self.assertEqual(predicted.tolist(), [[1, 0, 0, 1], [0, 1, 1, 0]])

    def test_apply_thresholds_matches_compute_multilabel_metrics_binarisation(self):
        scores = np.array(
            [[0.9, 0.1, 0.0, 0.8], [0.0, 0.7, 0.0, 0.0], [0.2, 0.0, 0.6, 0.1]],
            dtype=np.float32,
        )
        true = np.array([[1, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=np.int64)
        thresholds = [0.5, 0.5, 0.5, 0.5]
        names = ["soldier", "small_aircraft", "warship", "tank"]

        predicted = apply_thresholds(scores, thresholds)
        metrics = compute_multilabel_metrics(
            true,
            scores,
            thresholds,
            ["ir", "ir", "sar"],
            ["urban", "air", "sea"],
            names,
        )

        self.assertEqual(predicted.tolist(), true.tolist())
        self.assertAlmostEqual(metrics["exact_match_accuracy"], 1.0, places=6)
        self.assertEqual(metrics["sample_count"], 3)
        self.assertIn("sensor_ir", metrics)
        self.assertIn("scene_air", metrics)

    def test_wrong_threshold_count_is_rejected(self):
        with self.assertRaises(ValueError):
            apply_thresholds(np.zeros((2, 4), dtype=np.float32), [0.5, 0.5])

    def test_validation_only_thresholds_split_yolo_scores(self):
        # 低分正例 + 零分负例：val 上搜出的阈值必须落在两者之间。
        true = np.array([[1, 0, 0, 0]] * 4 + [[0, 0, 0, 0]] * 4, dtype=np.int64)
        scores = np.zeros((8, 4), dtype=np.float32)
        scores[:4, 0] = 0.25

        thresholds = optimize_thresholds(true, scores, 4)

        self.assertLessEqual(thresholds[0], 0.25)
        predicted = apply_thresholds(scores, thresholds)
        self.assertEqual(predicted[:, 0].tolist(), [1, 1, 1, 1, 0, 0, 0, 0])


class ManifestReadingTests(unittest.TestCase):
    def test_manifest_rows_carry_sensor_scene_and_presence_from_sibling_txt(self):
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            image_path = directory / "ir_r1_base_urban_000007.png"
            image_path.write_bytes(b"not-a-real-png-but-read_presence_manifest-never-opens-it")
            image_path.with_suffix(".txt").write_text(
                "0 0.5 0.5 0.1 0.1\n3 0.3 0.3 0.2 0.2\n0 0.7 0.7 0.05 0.05\n",
                encoding="utf-8",
            )
            manifest_path = directory / "test.txt"
            manifest_path.write_text(image_path.as_posix() + "\n", encoding="utf-8")

            rows = read_presence_manifest(manifest_path, 4)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["sensor"], "ir")
        self.assertEqual(rows[0]["scene"], "urban")
        self.assertEqual(rows[0]["target"].tolist(), [1, 0, 0, 1])

    def test_duplicate_manifest_entries_are_rejected(self):
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            image_path = directory / "sar_r1_base_sea_000002.png"
            image_path.write_bytes(b"placeholder")
            image_path.with_suffix(".txt").write_text("2 0.5 0.5 0.2 0.2\n", encoding="utf-8")
            manifest_path = directory / "val.txt"
            manifest_path.write_text(
                image_path.as_posix() + "\n" + image_path.as_posix() + "\n", encoding="utf-8"
            )

            with self.assertRaises(ValueError):
                read_presence_manifest(manifest_path, 4)


class DeviceResolutionTests(unittest.TestCase):
    def test_explicit_device_is_passed_through_unchanged(self):
        self.assertEqual(resolve_ultralytics_device("cpu"), "cpu")
        self.assertEqual(resolve_ultralytics_device("0"), "0")

    def test_auto_resolves_to_a_concrete_ultralytics_device_string(self):
        self.assertIn(resolve_ultralytics_device("auto"), {"cpu", "0"})


if __name__ == "__main__":
    unittest.main()
