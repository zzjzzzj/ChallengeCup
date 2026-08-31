from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml
from PIL import Image

from scene_recognition.detector_module import ALL_CLASS_NAMES, BASE_CLASS_NAMES
from scene_recognition.detector_module.augment_yolo_dataset import augment_yolo_dataset
from scene_recognition.detector_module.prepare_batch_incremental_dataset import (
    _normalise_plan,
    prepare_batch_incremental_dataset,
)
from scene_recognition.detector_module.run_four_to_six_pipeline import build_pipeline_plan, run_pipeline
from scene_recognition.detector_module.train_batch_incremental_yolo import run_batch_incremental
from test_support import workspace_test_directory


class BatchIncrementalProtocolTests(unittest.TestCase):
    def _dataset(self, root: Path, names: list[str]) -> Path:
        for split in ("train", "val", "test"):
            (root / "images" / split).mkdir(parents=True)
            (root / "labels" / split).mkdir(parents=True)
        for split in ("train", "val", "test"):
            for class_id in range(len(names)):
                image = root / "images" / split / f"ir_{split}_{class_id}.png"
                Image.new("RGB", (24, 20), (20 + class_id, 30, 40)).save(image)
                (root / "labels" / split / f"{image.stem}.txt").write_text(
                    f"{class_id} 0.5 0.5 0.2 0.2\n", encoding="utf-8"
                )
        config = {
            "path": str(root),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "nc": len(names),
            "names": {index: name for index, name in enumerate(names)},
        }
        data_yaml = root / "data.yaml"
        data_yaml.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        return data_yaml

    def test_augmentation_and_batch_protocol(self) -> None:
        with workspace_test_directory("batch-il-protocol") as root:
            base = self._dataset(root / "base", BASE_CLASS_NAMES)
            increment = self._dataset(root / "increment", ALL_CLASS_NAMES)
            base_aug = augment_yolo_dataset(base, root / "base_aug", default_modality="ir")
            increment_aug = augment_yolo_dataset(increment, root / "increment_aug", default_modality="ir")
            self.assertEqual(base_aug["output_images"]["train"], 16)
            self.assertEqual(increment_aug["output_images"]["val"], 6)
            self.assertEqual(increment_aug["operation_counts"]["rot180"], 6)
            output_yaml = yaml.safe_load((root / "increment_aug" / "data.yaml").read_text(encoding="utf-8"))
            self.assertEqual(output_yaml["names"], {index: name for index, name in enumerate(ALL_CLASS_NAMES)})

            plan_path = root / "plan.json"
            plan_path.write_text(
                json.dumps(
                    {
                        "batches": [
                            {"id": "batch_01", "classes": ["soldier", "patrol_boat"]},
                            {"id": "batch_02", "classes": ["patrol_boat"]},
                            {"id": "batch_03", "classes": ["armored_vehicle"]},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            report = prepare_batch_incremental_dataset(
                root / "base_aug" / "data.yaml",
                root / "increment_aug" / "data.yaml",
                root / "prepared",
                3,
                batch_plan=plan_path,
                buffer_sizes=(200, 500),
                max_current_images_per_class=1,
            )
            first, second, third = report["batches"]
            self.assertEqual(first["current"]["by_class"], {"soldier": 1, "patrol_boat": 1})
            self.assertEqual(second["present"], [])
            self.assertEqual(second["missing"], ["patrol_boat"])
            self.assertGreater(first["buffers"]["200"]["replay_before"]["images"], 0)
            self.assertEqual(
                first["buffers"]["200"]["buffer_after"]["images"],
                second["buffers"]["200"]["replay_before"]["images"],
            )
            self.assertEqual(report["first_arrival_batch"], {"patrol_boat": 1, "armored_vehicle": 3})
            self.assertEqual(report["privacy"]["dataset_upload"], False)

            batch_one_val = Path(first["buffers"]["200"]["validation"]["manifest"])
            visible_ids = set()
            for image_path in batch_one_val.read_text(encoding="utf-8").splitlines():
                label = Path(image_path).with_suffix(".txt").as_posix().replace("/images/", "/labels/")
                visible_ids.update(int(line.split()[0]) for line in Path(label).read_text(encoding="utf-8").splitlines() if line.strip())
            self.assertNotIn(5, visible_ids)

            checkpoint = root / "base.pt"
            checkpoint.write_bytes(b"local checkpoint placeholder")
            args = __import__(
                "scene_recognition.detector_module.train_batch_incremental_yolo",
                fromlist=["parse_args"],
            ).parse_args(
                [
                    "--prepared", str(root / "prepared"), "--initial-checkpoint", str(checkpoint),
                    "--method", "der", "--buffer-size", "200", "--output", str(root / "run"), "--dry-run",
                ]
            )
            dry = run_batch_incremental(args)
            self.assertEqual(dry["status"], "dry_run_ok")
            self.assertTrue(dry["der_first_batch_enabled"])

    def test_plan_validation_and_pipeline_dry_plan(self) -> None:
        with self.assertRaises(ValueError):
            _normalise_plan({"batches": [{"id": "b1", "classes": ["soldier"]}]}, None, 1)
        with self.assertRaises(ValueError):
            _normalise_plan({"batches": [{"id": "b1", "classes": ["patrol_boat", "patrol_boat"]}, {"id": "b2", "classes": ["armored_vehicle"]}]}, None, 1)
        with workspace_test_directory("batch-il-pipeline") as root:
            base = self._dataset(root / "base", BASE_CLASS_NAMES)
            increment = self._dataset(root / "increment", ALL_CLASS_NAMES)
            generic = root / "generic.pt"
            generic.write_bytes(b"local model")
            plan = build_pipeline_plan(
                base,
                increment,
                generic,
                root / "workspace",
                num_batches=2,
                buffer_sizes=(200,),
                sparse_moe=True,
            )
            self.assertTrue(plan["offline"])
            self.assertIn("--sparse-moe", plan["training_commands"]["200"])
            self.assertIn("--no-amp", plan["commands"][1])
            self.assertIn("--no-amp", plan["training_commands"]["200"])
            self.assertEqual(plan["audit"]["amp"], "disabled_to_prevent_Ultralytics_networked_AMP_probe")
            self.assertEqual(plan["audit"]["test"], "after_final_batch_only")

    def test_missing_actual_new_class_fails_before_preparation(self) -> None:
        with workspace_test_directory("batch-il-missing-arrival") as root:
            base = self._dataset(root / "base", BASE_CLASS_NAMES)
            increment = self._dataset(root / "increment", ALL_CLASS_NAMES)
            base_out = root / "base_aug"
            increment_out = root / "increment_aug"
            augment_yolo_dataset(base, base_out, default_modality="ir")
            augment_yolo_dataset(increment, increment_out, default_modality="ir")
            for image in (increment_out / "images" / "train").glob("*train_5*"):
                image.unlink()
                (increment_out / "labels" / "train" / f"{image.stem}.txt").unlink()
            with self.assertRaisesRegex(ValueError, "实际到达"):
                prepare_batch_incremental_dataset(
                    base_out / "data.yaml",
                    increment_out / "data.yaml",
                    root / "prepared",
                    batch_plan={"batches": [{"id": "b1", "classes": ["patrol_boat"]}, {"id": "b2", "classes": ["armored_vehicle"]}]},
                    buffer_sizes=(200,),
                )

    def test_augmentation_force_never_deletes_existing_directory(self) -> None:
        with workspace_test_directory("batch-il-force-safety") as root:
            source = self._dataset(root / "source", BASE_CLASS_NAMES)
            output = root / "existing"
            output.mkdir()
            marker = output / "keep.txt"
            marker.write_text("keep", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "force 已禁用"):
                augment_yolo_dataset(source, output, default_modality="ir", force=True)
            self.assertEqual(marker.read_text(encoding="utf-8"), "keep")

    def test_pipeline_completion_records_final_models_and_training_summaries(self) -> None:
        with workspace_test_directory("batch-il-pipeline-complete") as root:
            base = self._dataset(root / "base", BASE_CLASS_NAMES)
            increment = self._dataset(root / "increment", ALL_CLASS_NAMES)
            generic = root / "generic.pt"
            generic.write_bytes(b"local model")
            workspace = root / "workspace"
            args = __import__(
                "scene_recognition.detector_module.run_four_to_six_pipeline",
                fromlist=["parse_args"],
            ).parse_args(
                [
                    "--base-data", str(base), "--increment-data", str(increment),
                    "--generic-model", str(generic), "--workspace", str(workspace),
                    "--num-batches", "2", "--buffer-size", "200", "--base-epochs", "1",
                    "--increment-epochs", "1",
                ]
            )

            def fake_run(command, **_kwargs):
                if any("train_detector" in part and "train_batch" not in part for part in command):
                    checkpoint = workspace / "runs" / "base_four" / "weights" / "best.pt"
                    checkpoint.parent.mkdir(parents=True, exist_ok=True)
                    checkpoint.write_bytes(b"checkpoint")
                if any("train_batch_incremental_yolo" in part for part in command):
                    output = Path(command[command.index("--output") + 1])
                    output.mkdir(parents=True, exist_ok=True)
                    final = output / "batch_02" / "weights" / "best.pt"
                    final.parent.mkdir(parents=True, exist_ok=True)
                    final.write_bytes(b"final")
                    (output / "batch_incremental_training_summary.json").write_text(
                        json.dumps({"scenario": "batch_class_incremental", "final_model": str(final)}),
                        encoding="utf-8",
                    )

            with patch("scene_recognition.detector_module.run_four_to_six_pipeline.subprocess.run", side_effect=fake_run):
                result = run_pipeline(args)
            self.assertEqual(result["status"], "complete")
            self.assertIn("200", result["final_models"])
            self.assertEqual(result["training_summaries"]["200"]["final_model"], result["final_models"]["200"])


if __name__ == "__main__":
    unittest.main()
