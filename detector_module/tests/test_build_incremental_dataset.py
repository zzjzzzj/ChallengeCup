from __future__ import annotations

import csv
import unittest
from pathlib import Path

import yaml

from detector_module.build_incremental_dataset import build_incremental_dataset
from detector_module.create_incremental_protocol import build_protocol
from test_support import workspace_test_directory


class IncrementalDatasetBuildTests(unittest.TestCase):
    def make_sample(
        self,
        root: Path,
        rows: list[dict],
        split: str,
        name: str,
        labels: list[str],
    ) -> None:
        image = root / f"{name}.png"
        image.write_bytes(b"image")
        image.with_suffix(".txt").write_text("\n".join(labels) + "\n", encoding="utf-8")
        rows.append(
            {
                "image_path": str(image),
                "image_name": image.name,
                "sensor": "ir",
                "scene": "urban",
                "split": split,
                "sequence_index": str(len(rows)),
            }
        )

    def make_index(self, root: Path) -> Path:
        rows: list[dict] = []
        for split in ("train", "val", "test"):
            self.make_sample(root, rows, split, f"{split}_soldier", ["0 0.5 0.5 0.2 0.2"])
            self.make_sample(root, rows, split, f"{split}_tank", ["3 0.5 0.5 0.2 0.2"])
            self.make_sample(root, rows, split, f"{split}_aircraft", ["1 0.5 0.5 0.2 0.2"])
            self.make_sample(
                root,
                rows,
                split,
                f"{split}_mixed",
                ["0 0.5 0.5 0.2 0.2", "1 0.3 0.3 0.1 0.1"],
            )
        index = root / "scene_index.csv"
        with index.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        return index

    def test_builds_stage_views_with_remapped_labels(self):
        with workspace_test_directory("incremental-dataset") as root:
            protocol = build_protocol(["soldier", "tank"], [["small_aircraft"], ["warship"]])

            report = build_incremental_dataset(self.make_index(root), protocol, root / "incremental")

            base = report["stages"][0]
            self.assertEqual(base["roles"]["train_new"]["images"], 3)
            self.assertEqual(base["roles"]["train_new"]["objects_by_class"], {"soldier": 2, "tank": 1})
            tank_label = (
                root
                / "incremental"
                / "stage_0_base"
                / "train_new"
                / "labels"
                / "train_tank.txt"
            )
            self.assertTrue(tank_label.read_text(encoding="utf-8").startswith("1 "))

            increment = report["stages"][1]
            self.assertEqual(increment["roles"]["train_new"]["images"], 2)
            self.assertEqual(increment["roles"]["train_replay"]["images"], 4)
            config = yaml.safe_load(Path(increment["yamls"]["train_new"]).read_text(encoding="utf-8"))
            self.assertEqual(config["names"], {0: "soldier", 1: "tank", 2: "small_aircraft"})
            aircraft_label = (
                root
                / "incremental"
                / "stage_1_increment_1"
                / "train_new"
                / "labels"
                / "train_aircraft.txt"
            )
            self.assertTrue(aircraft_label.read_text(encoding="utf-8").startswith("2 "))


if __name__ == "__main__":
    unittest.main()
