from __future__ import annotations

import csv
import unittest
from pathlib import Path

import yaml

from scene_recognition.detector_module.dataset import parse_yolo_label, prepare_detection_dataset
from test_support import workspace_test_directory


class DatasetPreparationTests(unittest.TestCase):
    def make_index(self, root: Path) -> Path:
        index_path = root / "scene_index.csv"
        rows = []
        for split_index, split in enumerate(("train", "val", "test")):
            for class_id in range(4):
                image = root / f"ir_r1_base_scene_{split_index}_{class_id}.png"
                image.write_bytes(b"test-image-placeholder")
                image.with_suffix(".txt").write_text(
                    f"{class_id} 0.5 0.5 0.2 0.2\n", encoding="utf-8"
                )
                rows.append(
                    {
                        "image_path": str(image),
                        "image_name": image.name,
                        "sensor": "ir",
                        "scene": "urban",
                        "split": split,
                        "sequence_index": str(class_id),
                    }
                )
        with index_path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        return index_path

    def test_prepare_writes_three_manifests_and_yaml(self):
        with workspace_test_directory("detection-dataset") as root:
            stats = prepare_detection_dataset(self.make_index(root), root / "output")

            self.assertEqual(stats["image_count"], 12)
            self.assertEqual(stats["object_count"], 12)
            config = yaml.safe_load((root / "output" / "dataset.yaml").read_text(encoding="utf-8"))
            self.assertEqual(config["nc"], 4)
            self.assertEqual(config["names"][2], "warship")
            for split in ("train", "val", "test"):
                manifest = Path(config[split])
                self.assertTrue(manifest.is_file())
                self.assertEqual(len(manifest.read_text(encoding="utf-8").splitlines()), 4)

    def test_invalid_normalized_box_is_rejected(self):
        with workspace_test_directory("invalid-box") as root:
            label = root / "bad.txt"
            label.write_text("0 1.2 0.5 0.2 0.2\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "中心坐标"):
                parse_yolo_label(label, 4)

    def test_out_of_range_class_is_rejected(self):
        with workspace_test_directory("invalid-class") as root:
            label = root / "bad.txt"
            label.write_text("4 0.5 0.5 0.2 0.2\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "类别编号越界"):
                parse_yolo_label(label, 4)


if __name__ == "__main__":
    unittest.main()
