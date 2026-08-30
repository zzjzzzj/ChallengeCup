from __future__ import annotations

import unittest
from pathlib import Path

from PIL import Image
import yaml

from scene_recognition.detector_module import ALL_CLASS_NAMES
from scene_recognition.detector_module.boxes import parse_yolo_boxes, resolve_label_path
from scene_recognition.detector_module.context_metadata import CONTEXT_INDEX_FIELDS, read_context_rows
from scene_recognition.detector_module.prepare_class_incremental_dataset import (
    prepare_class_incremental_dataset,
)
from test_support import workspace_test_directory


class ClassIncrementalPreparationTests(unittest.TestCase):
    def make_dataset(self, root: Path, *, include_test: bool = False) -> Path:
        splits = ("train", "val", "test") if include_test else ("train", "val")
        for split in splits:
            (root / "images" / split).mkdir(parents=True)
            (root / "labels" / split).mkdir(parents=True)
        for class_id in range(len(ALL_CLASS_NAMES)):
            for index in range(3):
                image = root / "images" / "train" / f"train_c{class_id}_{index}.png"
                Image.new("RGB", (32, 24), color=(20 + class_id, 20, 20)).save(image)
                (root / "labels" / "train" / f"train_c{class_id}_{index}.txt").write_text(
                    f"{class_id} 0.5 0.5 0.2 0.2\n",
                    encoding="utf-8",
                )
            image = root / "images" / "val" / f"val_c{class_id}.png"
            Image.new("RGB", (32, 24), color=(20 + class_id, 20, 20)).save(image)
            (root / "labels" / "val" / f"val_c{class_id}.txt").write_text(
                f"{class_id} 0.5 0.5 0.2 0.2\n",
                encoding="utf-8",
            )
            if include_test:
                image = root / "images" / "test" / f"test_c{class_id}.png"
                Image.new("RGB", (32, 24), color=(20 + class_id, 30, 20)).save(image)
                (root / "labels" / "test" / f"test_c{class_id}.txt").write_text(
                    f"{class_id} 0.5 0.5 0.2 0.2\n",
                    encoding="utf-8",
                )
        data_config = {
            "path": str(root),
            "train": "images/train",
            "val": "images/val",
            "nc": len(ALL_CLASS_NAMES),
            "names": {index: name for index, name in enumerate(ALL_CLASS_NAMES)},
        }
        if include_test:
            data_config["test"] = "images/test"
        data_yaml = root / "data.yaml"
        data_yaml.write_text(
            yaml.safe_dump(data_config, sort_keys=False),
            encoding="utf-8",
        )
        return data_yaml

    def test_builds_six_singleton_stages_and_bounded_balanced_buffers(self) -> None:
        with workspace_test_directory("class-il-prepare") as root:
            report = prepare_class_incremental_dataset(
                self.make_dataset(root / "source"),
                root / "prepared",
                buffer_sizes=(2, 5),
                seed=7,
            )
            self.assertEqual(report["scenario"], "class_incremental")
            self.assertEqual(len(report["stages"]), 6)
            self.assertEqual(report["stages"][0]["new_classes"], ["soldier"])
            self.assertEqual(report["stages"][5]["all_learned_classes"], ALL_CLASS_NAMES)
            self.assertEqual(
                report["stages"][1]["buffers"]["2"]["replay_before"]["classes"],
                {"soldier": 2},
            )
            stage_three_buffer = report["stages"][2]["buffers"]["5"]["replay_before"]
            self.assertEqual(stage_three_buffer["images"], 5)
            self.assertEqual(set(stage_three_buffer["classes"]), {"soldier", "small_aircraft"})

    def test_training_labels_never_include_future_classes(self) -> None:
        with workspace_test_directory("class-il-no-future") as root:
            report = prepare_class_incremental_dataset(
                self.make_dataset(root / "source"),
                root / "prepared",
                buffer_sizes=(2,),
            )
            for stage in report["stages"]:
                learned_count = stage["stage"]
                manifest = Path(stage["buffers"]["2"]["training"]["manifest"])
                for line in manifest.read_text(encoding="utf-8").splitlines():
                    boxes = parse_yolo_boxes(resolve_label_path(Path(line)), learned_count)
                    self.assertTrue(boxes)
                    self.assertTrue(all(box.class_id < learned_count for box in boxes))

    def test_current_task_examples_keep_only_the_new_class_label(self) -> None:
        with workspace_test_directory("class-il-current-label") as root:
            source = root / "source"
            data_yaml = self.make_dataset(source)
            # Put a known old-class object in an aircraft task image.  The T2
            # current view must keep only aircraft, not silently replay soldier.
            mixed_label = source / "labels" / "train" / "train_c1_0.txt"
            mixed_label.write_text(
                "0 0.2 0.2 0.1 0.1\n1 0.5 0.5 0.2 0.2\n",
                encoding="utf-8",
            )
            report = prepare_class_incremental_dataset(
                data_yaml,
                root / "prepared",
                buffer_sizes=(2,),
            )
            stage_two = report["stages"][1]["buffers"]["2"]["training"]
            replay_paths = set(
                Path(stage_two["replay_manifest"]).read_text(encoding="utf-8").splitlines()
            )
            manifest = Path(stage_two["manifest"]).read_text(encoding="utf-8").splitlines()
            current_paths = [Path(path) for path in manifest if path not in replay_paths]
            self.assertTrue(current_paths)
            for image_path in current_paths:
                boxes = parse_yolo_boxes(resolve_label_path(image_path), 2)
                self.assertEqual({box.class_id for box in boxes}, {1})

    def test_materializes_independent_test_views_for_each_stage(self) -> None:
        with workspace_test_directory("class-il-test") as root:
            report = prepare_class_incremental_dataset(
                self.make_dataset(root / "source", include_test=True),
                root / "prepared",
                buffer_sizes=(2,),
            )
            self.assertEqual(report["evaluation"]["split"], "test")
            self.assertTrue(report["evaluation"]["official"])
            self.assertFalse(report["evaluation"]["competition_official"])
            for stage in report["stages"]:
                self.assertIsNotNone(stage["test"])
                data_yaml = Path(stage["buffers"]["2"]["data_yaml"])
                data_config = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
                self.assertIn("test", data_config)
                context_rows = read_context_rows(Path(stage["context_index"]))
                self.assertTrue(context_rows)
                self.assertEqual(set(context_rows[0]), set(CONTEXT_INDEX_FIELDS))
                self.assertIn("val", {row["sample_role"] for row in context_rows})
                self.assertIn("test", {row["sample_role"] for row in context_rows})


if __name__ == "__main__":
    unittest.main()
