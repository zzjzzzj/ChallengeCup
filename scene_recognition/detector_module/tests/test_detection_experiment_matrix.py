from __future__ import annotations

import unittest
from pathlib import Path

from scene_recognition.experiments.run_detection_experiments import build_experiments


class DetectionExperimentMatrixTests(unittest.TestCase):
    def test_matrix_has_eight_unique_groups(self) -> None:
        experiments = build_experiments(Path("data"), Path("runs"))

        self.assertEqual(len(experiments), 8)
        self.assertEqual(len({experiment.name for experiment in experiments}), 8)
        self.assertEqual({experiment.model for experiment in experiments}, {"yolo", "resnet"})
        self.assertEqual(
            sum("--no-pretrained" in experiment.command for experiment in experiments),
            4,
        )

    def test_augmented_resnet_uses_balanced_epoch_budget(self) -> None:
        experiments = build_experiments(
            Path("data"),
            Path("runs"),
            resnet_noaug_epochs=40,
            resnet_aug_epochs=6,
        )
        augmented = next(
            experiment
            for experiment in experiments
            if experiment.name == "cmp8_resnet18det_aug_pretrained"
        )
        epoch_index = augmented.command.index("--epochs") + 1
        self.assertEqual(augmented.command[epoch_index], "6")


if __name__ == "__main__":
    unittest.main()
