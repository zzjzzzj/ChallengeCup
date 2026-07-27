from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from detector_module.prepare_comparison_dataset import prepare_comparison_dataset


class PrepareComparisonDatasetTests(unittest.TestCase):
    def test_builds_original_augmented_and_holdout_manifests(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "dataset"
            for relative in ("images/train", "images/val", "labels/train", "labels/val"):
                (root / relative).mkdir(parents=True)

            for scene in ("air", "forest", "sea", "urban"):
                original = f"ir_r1_base_{scene}_000001"
                augmented = f"{original}__aug-flip"
                for stem in (original, augmented):
                    (root / "images/train" / f"{stem}.png").write_bytes(b"image")
                    (root / "labels/train" / f"{stem}.txt").write_text(
                        "0 0.5 0.5 0.1 0.1\n", encoding="utf-8"
                    )
                for index in range(4):
                    stem = f"ir_r1_base_{scene}_{index + 10:06d}"
                    (root / "images/val" / f"{stem}.png").write_bytes(b"image")
                    (root / "labels/val" / f"{stem}.txt").write_text(
                        "0 0.5 0.5 0.1 0.1\n", encoding="utf-8"
                    )

            output = Path(temporary_directory) / "output"
            stats = prepare_comparison_dataset(root, output)

            self.assertEqual(stats["original_train_images"], 4)
            self.assertEqual(stats["augmented_train_images"], 8)
            self.assertEqual(stats["validation_images"], 8)
            self.assertEqual(stats["test_images"], 8)
            self.assertEqual(len((output / "train_noaug.txt").read_text().splitlines()), 4)
            self.assertEqual(len((output / "train_aug.txt").read_text().splitlines()), 8)

            config = yaml.safe_load((output / "data_aug.yaml").read_text(encoding="utf-8"))
            self.assertEqual(config["nc"], 4)
            self.assertTrue(Path(config["train"]).is_file())
            self.assertTrue(Path(config["test"]).is_file())


if __name__ == "__main__":
    unittest.main()
