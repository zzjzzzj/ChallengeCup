from __future__ import annotations

import csv
import unittest
from pathlib import Path

from PIL import Image

from scene_recognition.detector_module.boxes import YoloBox
from scene_recognition.target_classifier_module.dataset import build_target_crop_dataset, normalized_box_to_pixels
from test_support import workspace_test_directory


class TargetCropDatasetTests(unittest.TestCase):
    def test_normalized_box_is_converted_to_pixel_bounds(self):
        box = YoloBox(class_id=3, x_center=0.5, y_center=0.5, width=0.25, height=0.5)

        bounds = normalized_box_to_pixels(box, image_size=(640, 512))

        self.assertEqual(bounds, (240, 128, 400, 384))

    def test_build_dataset_crops_objects_and_preserves_source_splits(self):
        with workspace_test_directory("target-crops") as root:
            rows = []
            boxes = [
                "0 0.20 0.25 0.20 0.20",
                "1 0.70 0.25 0.20 0.20",
                "2 0.20 0.75 0.20 0.20",
                "3 0.70 0.75 0.20 0.20",
            ]
            for split_index, split in enumerate(("train", "val", "test"), start=1):
                image_path = root / f"ir_r1_base_urban_{split_index:06d}.png"
                Image.new("RGB", (100, 80), color=(split_index * 30, 80, 120)).save(image_path)
                image_path.with_suffix(".txt").write_text("\n".join(boxes) + "\n", encoding="utf-8")
                rows.append(
                    {
                        "image_path": str(image_path),
                        "image_name": image_path.name,
                        "sensor": "ir",
                        "scene": "urban",
                        "split": split,
                        "sequence_index": split_index,
                    }
                )

            index_path = root / "scene_index.csv"
            with index_path.open("w", newline="", encoding="utf-8-sig") as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)

            summary = build_target_crop_dataset(index_path, root / "output", padding_ratio=0.0)

            self.assertEqual(summary["source_images"], 3)
            self.assertEqual(summary["crops"], 12)
            self.assertEqual(summary["splits"]["train"]["crops"], 4)
            with (root / "output" / "manifest.csv").open(
                "r", newline="", encoding="utf-8-sig"
            ) as handle:
                manifest = list(csv.DictReader(handle))
            self.assertEqual(len(manifest), 12)
            self.assertEqual({row["split"] for row in manifest}, {"train", "val", "test"})
            self.assertTrue((root / "output" / "train" / "soldier").is_dir())
            self.assertTrue(Path(manifest[0]["crop_path"]).is_file())


if __name__ == "__main__":
    unittest.main()
