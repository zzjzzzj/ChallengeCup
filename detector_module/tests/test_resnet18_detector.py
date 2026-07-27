from __future__ import annotations

import unittest

import torch

from detector_module.resnet18_detector import (
    build_resnet18_detector,
    detection_metrics,
    yolo_rows_to_target,
)


class ResNet18DetectorTests(unittest.TestCase):
    def test_yolo_rows_are_converted_to_pixel_targets(self):
        target = yolo_rows_to_target(
            [(2, 0.5, 0.5, 0.4, 0.2)], image_width=100, image_height=80
        )

        self.assertEqual(target["labels"].tolist(), [3])
        self.assertTrue(torch.allclose(target["boxes"], torch.tensor([[30.0, 32.0, 70.0, 48.0]])))

    def test_perfect_prediction_has_perfect_map(self):
        targets = [
            {
                "boxes": torch.tensor([[10.0, 10.0, 30.0, 30.0]]),
                "labels": torch.tensor([1]),
            }
        ]
        predictions = [
            {
                "boxes": torch.tensor([[10.0, 10.0, 30.0, 30.0]]),
                "labels": torch.tensor([1]),
                "scores": torch.tensor([0.9]),
            }
        ]

        metrics = detection_metrics(predictions, targets, ["soldier"])

        self.assertAlmostEqual(metrics["map50"], 1.0)
        self.assertAlmostEqual(metrics["map50_95"], 1.0)
        self.assertAlmostEqual(metrics["precision"], 1.0)
        self.assertAlmostEqual(metrics["recall"], 1.0)

    def test_detector_accepts_a_full_image_and_returns_detection_fields(self):
        model = build_resnet18_detector(
            class_count=4,
            pretrained=False,
            min_size=64,
            max_size=64,
        )
        model.eval()

        with torch.no_grad():
            prediction = model([torch.rand(3, 64, 64)])[0]

        self.assertEqual(set(prediction), {"boxes", "labels", "scores"})
        self.assertEqual(prediction["boxes"].shape[1], 4)
        self.assertEqual(prediction["labels"].ndim, 1)


if __name__ == "__main__":
    unittest.main()
