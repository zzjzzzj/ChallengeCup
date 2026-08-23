from __future__ import annotations

import json
import unittest

from scene_recognition.detector_module.train_continual_yolo import validate_strategy_data
from test_support import workspace_test_directory


class ContinualTrainingGuardTests(unittest.TestCase):
    def test_replay_strategy_rejects_empty_replay_manifest(self) -> None:
        with workspace_test_directory("continual-train-guard") as root:
            data = root / "data_replay.yaml"
            data.write_text("names: {}\n", encoding="utf-8")
            (root / "continual_dataset_summary.json").write_text(
                json.dumps({"statistics": {"replay_train": {"images": 0}}}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "回放图像为 0"):
                validate_strategy_data(data, "replay")

    def test_strategy_requires_matching_yaml(self) -> None:
        with workspace_test_directory("continual-train-yaml") as root:
            data = root / "data_increment_only.yaml"
            data.write_text("names: {}\n", encoding="utf-8")
            (root / "continual_dataset_summary.json").write_text(
                json.dumps({"statistics": {"replay_train": {"images": 2}}}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "data_replay.yaml"):
                validate_strategy_data(data, "replay")


if __name__ == "__main__":
    unittest.main()
