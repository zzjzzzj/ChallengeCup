from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from scene_recognition.target_classifier_module.training import build_resnet18, build_transforms
from scene_recognition.target_classifier_module.whole_image import (
    WholeImageDataset,
    WholeImageTrainingConfig,
    compute_multilabel_metrics,
    optimize_thresholds,
    predict_whole_image,
    train_whole_image_classifier,
)
from test_support import workspace_test_directory


class WholeImageClassifierTests(unittest.TestCase):
    def test_dataset_converts_yolo_labels_to_presence_vector(self):
        with workspace_test_directory("whole-loader") as root:
            image_path = root / "ir_r1_base_urban_000001.png"
            label_path = image_path.with_suffix(".txt")
            Image.new("RGB", (40, 20), color=(20, 40, 60)).save(image_path)
            label_path.write_text(
                "0 0.2 0.2 0.1 0.1\n3 0.8 0.8 0.1 0.1\n3 0.5 0.5 0.1 0.1\n",
                encoding="utf-8",
            )
            manifest = root / "train.txt"
            manifest.write_text(str(image_path) + "\n", encoding="utf-8")
            _, transform = build_transforms(32, "none")
            dataset = WholeImageDataset(manifest, transform)
            tensor, target, sensor, scene, returned_path = dataset[0]
            self.assertEqual(tuple(tensor.shape), (3, 32, 32))
            self.assertTrue(torch.equal(target, torch.tensor([1.0, 0.0, 0.0, 1.0])))
            self.assertEqual((sensor, scene), ("ir", "urban"))
            self.assertEqual(returned_path, str(image_path))

    def test_metrics_keep_multi_label_presence_separate_from_detection(self):
        true = np.array([[1, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0]])
        probabilities = np.array([[0.9, 0.1, 0.2, 0.8], [0.2, 0.8, 0.1, 0.2], [0.1, 0.2, 0.9, 0.1]])
        metrics = compute_multilabel_metrics(
            true,
            probabilities,
            [0.5] * 4,
            ["ir", "ir", "sar"],
            ["urban", "air", "sea"],
            ["soldier", "small_aircraft", "warship", "tank"],
        )
        self.assertEqual(metrics["exact_match_accuracy"], 1.0)
        self.assertEqual(metrics["sensor_ir"]["sample_count"], 2)
        self.assertNotIn("map50", metrics)

    def test_thresholds_are_optimized_per_class(self):
        true = np.array([[1], [1], [0], [0]])
        probabilities = np.array([[0.3], [0.4], [0.2], [0.1]])
        self.assertEqual(optimize_thresholds(true, probabilities, 1), [0.3])

    def test_training_rejects_source_image_leakage_between_splits(self):
        with workspace_test_directory("whole-leak") as root:
            image_path = root / "ir_r1_base_air_000001.png"
            Image.new("RGB", (32, 32)).save(image_path)
            image_path.with_suffix(".txt").write_text(
                "1 0.5 0.5 0.2 0.2\n", encoding="utf-8"
            )
            manifest_dir = root / "manifests"
            manifest_dir.mkdir()
            for split in ("train", "val", "test"):
                (manifest_dir / f"{split}.txt").write_text(
                    str(image_path) + "\n", encoding="utf-8"
                )
            with self.assertRaisesRegex(ValueError, "原图泄漏"):
                train_whole_image_classifier(
                    manifest_dir,
                    root / "run",
                    WholeImageTrainingConfig(
                        epochs=1,
                        batch_size=1,
                        image_size=32,
                        pretrained=False,
                        device="cpu",
                    ),
                )

    def test_one_epoch_whole_image_training_and_inference(self):
        with workspace_test_directory("whole-train") as root:
            manifest_dir = root / "manifests"
            manifest_dir.mkdir()
            for split in ("train", "val", "test"):
                paths = []
                for index, classes in enumerate(((0, 3), (1,), (2,), (0, 3))):
                    image_path = root / f"{split}_{index}" / f"ir_r1_base_urban_{index:06d}.png"
                    image_path.parent.mkdir(parents=True, exist_ok=True)
                    Image.new("RGB", (40, 32), color=(30 + index * 20, 50, 70)).save(image_path)
                    label_path = image_path.with_suffix(".txt")
                    label_path.write_text(
                        "\n".join(f"{class_id} 0.5 0.5 0.2 0.2" for class_id in classes),
                        encoding="utf-8",
                    )
                    paths.append(str(image_path))
                (manifest_dir / f"{split}.txt").write_text("\n".join(paths) + "\n", encoding="utf-8")
            result = train_whole_image_classifier(
                manifest_dir,
                root / "run",
                WholeImageTrainingConfig(
                    epochs=1,
                    batch_size=2,
                    image_size=32,
                    pretrained=False,
                    device="cpu",
                ),
            )
            self.assertEqual(result["model"], "resnet18_whole_image_multilabel")
            self.assertEqual(result["test"]["sample_count"], 4)
            prediction = predict_whole_image(
                root / "train_0" / "ir_r1_base_urban_000000.png",
                root / "run" / "best.pt",
                "cpu",
            )
            self.assertEqual(len(prediction["probabilities"]), 4)
            self.assertIn("warning", prediction)


if __name__ == "__main__":
    unittest.main()
