from __future__ import annotations

import unittest
import csv

from PIL import Image

import torch

from target_classifier_module.training import (
    TargetCropDataset,
    TrainingConfig,
    build_resnet18,
    build_transforms,
    compute_classification_metrics,
    train_target_classifier,
)
from target_classifier_module.infer import predict_target_crop
from test_support import workspace_test_directory


class TargetClassifierTrainingTests(unittest.TestCase):
    def test_resnet18_outputs_four_target_scores_per_crop(self):
        model = build_resnet18(class_count=4, pretrained=False)

        logits = model(torch.zeros(2, 3, 64, 64))

        self.assertEqual(tuple(logits.shape), (2, 4))

    def test_manifest_dataset_returns_square_tensor_and_trace_metadata(self):
        with workspace_test_directory("target-loader") as root:
            crop_path = root / "tank.png"
            Image.new("RGB", (40, 20), color=(20, 40, 60)).save(crop_path)
            manifest_path = root / "manifest.csv"
            row = {
                "crop_path": str(crop_path),
                "source_image_path": str(root / "source.png"),
                "source_image_name": "source.png",
                "label_path": str(root / "source.txt"),
                "split": "train",
                "sensor": "sar",
                "scene": "sea",
                "box_index": 0,
                "class_id": 3,
                "class_name": "tank",
            }
            with manifest_path.open("w", newline="", encoding="utf-8-sig") as handle:
                writer = csv.DictWriter(handle, fieldnames=row.keys())
                writer.writeheader()
                writer.writerow(row)

            _, evaluation_transform = build_transforms(image_size=32, augmentation="none")
            dataset = TargetCropDataset(manifest_path, "train", evaluation_transform)
            tensor, label, sensor, scene, returned_path, source_path = dataset[0]

            self.assertEqual(tuple(tensor.shape), (3, 32, 32))
            self.assertEqual(label, 3)
            self.assertEqual((sensor, scene), ("sar", "sea"))
            self.assertEqual(returned_path, str(crop_path))
            self.assertEqual(source_path, str(root / "source.png"))

    def test_manifest_rejects_disagreeing_class_id_and_name(self):
        with workspace_test_directory("target-invalid-manifest") as root:
            crop_path = root / "bad.png"
            Image.new("RGB", (20, 20)).save(crop_path)
            manifest_path = root / "manifest.csv"
            row = {
                "crop_path": str(crop_path),
                "source_image_path": str(root / "source.png"),
                "split": "train",
                "sensor": "ir",
                "scene": "urban",
                "class_id": 3,
                "class_name": "soldier",
            }
            with manifest_path.open("w", newline="", encoding="utf-8-sig") as handle:
                writer = csv.DictWriter(handle, fieldnames=row.keys())
                writer.writeheader()
                writer.writerow(row)

            _, transform = build_transforms(32, "none")
            with self.assertRaisesRegex(ValueError, "类别编号与名称不一致"):
                TargetCropDataset(manifest_path, "train", transform)

    def test_metrics_report_overall_per_class_and_sensor_results(self):
        metrics = compute_classification_metrics(
            true=[0, 1, 2, 3],
            predicted=[0, 1, 2, 2],
            sensors=["ir", "ir", "sar", "sar"],
            scenes=["air", "sea", "urban", "forest"],
            class_names=["soldier", "small_aircraft", "warship", "tank"],
        )

        self.assertEqual(metrics["accuracy"], 0.75)
        self.assertEqual(metrics["per_class_recall"]["tank"], 0.0)
        self.assertEqual(metrics["ir_accuracy"], 1.0)
        self.assertEqual(metrics["sar_accuracy"], 0.5)

    def test_one_epoch_training_writes_reproducible_artifacts(self):
        with workspace_test_directory("target-train") as root:
            manifest_path = root / "manifest.csv"
            rows = []
            for split_index, split in enumerate(("train", "val", "test")):
                for class_id, class_name in enumerate(
                    ("soldier", "small_aircraft", "warship", "tank")
                ):
                    crop_path = root / f"{split}_{class_name}.png"
                    Image.new(
                        "RGB", (24, 16), color=(30 + class_id * 40, 40 + split_index, 70)
                    ).save(crop_path)
                    rows.append(
                        {
                            "crop_path": str(crop_path),
                            "source_image_path": str(root / f"source_{split}.png"),
                            "source_image_name": f"source_{split}.png",
                            "label_path": str(root / f"source_{split}.txt"),
                            "split": split,
                            "sensor": "ir" if class_id % 2 == 0 else "sar",
                            "scene": "urban",
                            "box_index": class_id,
                            "class_id": class_id,
                            "class_name": class_name,
                        }
                    )
            with manifest_path.open("w", newline="", encoding="utf-8-sig") as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)

            result = train_target_classifier(
                manifest_path,
                root / "run",
                TrainingConfig(
                    epochs=1,
                    batch_size=4,
                    image_size=32,
                    pretrained=False,
                    device="cpu",
                ),
            )

            self.assertEqual(result["model"], "resnet18")
            self.assertEqual(result["test"]["sample_count"], 4)
            for name in (
                "best.pt",
                "metrics.json",
                "history.csv",
                "confusion_matrix.csv",
                "test_predictions.csv",
            ):
                self.assertTrue((root / "run" / name).is_file(), name)

    def test_single_crop_inference_returns_four_probabilities(self):
        with workspace_test_directory("target-infer") as root:
            image_path = root / "unknown.png"
            Image.new("RGB", (30, 18), color=(30, 60, 90)).save(image_path)
            model = build_resnet18(class_count=4, pretrained=False)
            checkpoint_path = root / "checkpoint.pt"
            torch.save(
                {
                    "model_name": "resnet18",
                    "state_dict": model.state_dict(),
                    "class_names": ["soldier", "small_aircraft", "warship", "tank"],
                    "image_size": 32,
                },
                checkpoint_path,
            )

            result = predict_target_crop(image_path, checkpoint_path, device_name="cpu")

            self.assertIn(result["predicted"], result["probabilities"])
            self.assertEqual(len(result["probabilities"]), 4)
            self.assertAlmostEqual(sum(result["probabilities"].values()), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
