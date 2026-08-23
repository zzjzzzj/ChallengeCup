from __future__ import annotations

import csv
import unittest
from pathlib import Path

from PIL import Image
import yaml

from scene_recognition.detector_module.prepare_continual_dataset import (
    prepare_continual_dataset,
)
from test_support import workspace_test_directory


CLASSES = [
    "soldier",
    "small_aircraft",
    "warship",
    "tank",
    "patrol_boat",
    "armored_vehicle",
]


class ContinualDatasetPreparationTests(unittest.TestCase):
    def make_image(self, path: Path, class_id: int) -> None:
        Image.new("RGB", (32, 24), color=(40, 40, 40)).save(path)
        path.with_suffix(".txt").write_text(
            f"{class_id} 0.5 0.5 0.2 0.2\n",
            encoding="utf-8",
        )

    def make_base_index(self, root: Path) -> Path:
        rows = []
        for index, split in enumerate(("train", "val", "test"), start=1):
            image = root / f"ir_r1_base_sea_{index:06d}.png"
            self.make_image(image, 2)
            rows.append(
                {
                    "image_path": str(image),
                    "sensor": "ir",
                    "scene": "sea",
                    "split": split,
                }
            )
        index_path = root / "base_index.csv"
        with index_path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        return index_path

    def test_builds_increment_and_replay_manifests_without_copying_images(self) -> None:
        with workspace_test_directory("continual-preparation") as root:
            increment = root / "r2"
            increment.mkdir()
            (increment / "classes.txt").write_text("\n".join(CLASSES) + "\n", encoding="utf-8")
            for index in range(1, 5):
                self.make_image(increment / f"ir_r2_inc_sea_{index:06d}.png", 4)
                self.make_image(increment / f"sar_r2_inc_urban_{index:06d}.png", 5)
            output = root / "output"

            report = prepare_continual_dataset(
                increment,
                output,
                base_index=self.make_base_index(root),
                replay_limit=1,
            )

            self.assertEqual(report["new_classes"], ["patrol_boat", "armored_vehicle"])
            self.assertEqual(report["statistics"]["increment_train"]["images"], 8)
            self.assertEqual(report["statistics"]["replay_train"]["images"], 1)
            self.assertEqual(report["statistics"]["mixed_train"]["images"], 9)
            self.assertFalse(report["privacy"]["source_images_copied"])
            config = yaml.safe_load(Path(report["yamls"]["replay"]).read_text(encoding="utf-8"))
            self.assertEqual(config["names"][4], "patrol_boat")
            self.assertEqual(config["names"][5], "armored_vehicle")
            self.assertFalse(any(output.rglob("*.png")))


if __name__ == "__main__":
    unittest.main()
