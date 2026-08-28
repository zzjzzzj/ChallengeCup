from __future__ import annotations

import csv
import unittest
from pathlib import Path

from PIL import Image
import yaml

from scene_recognition.detector_module import ALL_CLASS_NAMES
from scene_recognition.detector_module.split_yolo_dataset import split_yolo_dataset
from test_support import workspace_test_directory


class YoloDatasetSplitTests(unittest.TestCase):
    def make_dataset(self, root: Path) -> Path:
        for split in ("train", "val"):
            (root / "images" / split).mkdir(parents=True)
            (root / "labels" / split).mkdir(parents=True)
        manifest_rows: list[dict[str, str]] = []
        for class_id in range(len(ALL_CLASS_NAMES)):
            train_name = f"train_c{class_id}.png"
            Image.new("RGB", (16, 16), color=(class_id + 1, 0, 0)).save(
                root / "images" / "train" / train_name
            )
            (root / "labels" / "train" / f"train_c{class_id}.txt").write_text(
                f"{class_id} 0.5 0.5 0.2 0.2\n", encoding="utf-8"
            )
            manifest_rows.append(
                {
                    "split": "train",
                    "provenance": "base",
                    "source_image": train_name,
                    "output_image": train_name,
                    "operation": "original",
                }
            )
            for index in range(2):
                source_name = f"holdout_c{class_id}_{index}.png"
                for suffix, operation in (("", "original"), ("__aug-flip", "flip")):
                    output_name = f"holdout_c{class_id}_{index}{suffix}.png"
                    Image.new("RGB", (16, 16), color=(class_id + 1, index + 1, 0)).save(
                        root / "images" / "val" / output_name
                    )
                    (root / "labels" / "val" / f"{Path(output_name).stem}.txt").write_text(
                        f"{class_id} 0.5 0.5 0.2 0.2\n", encoding="utf-8"
                    )
                    manifest_rows.append(
                        {
                            "split": "val",
                            "provenance": "base",
                            "source_image": source_name,
                            "output_image": output_name,
                            "operation": operation,
                        }
                    )
        with (root / "dataset_manifest.csv").open(
            "w", encoding="utf-8-sig", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
            writer.writeheader()
            writer.writerows(manifest_rows)
        data_yaml = root / "data.yaml"
        data_yaml.write_text(
            yaml.safe_dump(
                {
                    "path": str(root),
                    "train": "images/train",
                    "val": "images/val",
                    "nc": len(ALL_CLASS_NAMES),
                    "names": {
                        index: name for index, name in enumerate(ALL_CLASS_NAMES)
                    },
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        return data_yaml

    def test_keeps_source_variants_together_and_balances_all_classes(self) -> None:
        with workspace_test_directory("yolo-tvt") as root:
            output = root / "output"
            report = split_yolo_dataset(
                self.make_dataset(root / "source"),
                output,
                test_fraction=0.5,
                seed=7,
                search_trials=500,
            )
            self.assertEqual(report["splits"]["train"]["images"], 6)
            self.assertEqual(report["splits"]["val"]["source_groups"], 6)
            self.assertEqual(report["splits"]["test"]["source_groups"], 6)
            self.assertEqual(report["leakage_audit"]["val_test_source_overlap"], 0)
            self.assertTrue((output / "data.yaml").is_file())
            for class_name in ALL_CLASS_NAMES:
                self.assertEqual(
                    report["splits"]["val"]["class_image_presence"][class_name], 2
                )
                self.assertEqual(
                    report["splits"]["test"]["class_image_presence"][class_name], 2
                )

            with (output / "dataset_manifest.csv").open(
                "r", encoding="utf-8-sig", newline=""
            ) as handle:
                rows = list(csv.DictReader(handle))
            source_splits: dict[str, set[str]] = {}
            for row in rows:
                source_splits.setdefault(row["source_image"], set()).add(row["split"])
            self.assertTrue(all(len(splits) == 1 for splits in source_splits.values()))


if __name__ == "__main__":
    unittest.main()
