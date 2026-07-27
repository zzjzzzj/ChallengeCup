from __future__ import annotations

import unittest

from scene_recognition.cli import COMMANDS, usage


class SceneRecognitionCliTests(unittest.TestCase):
    def test_exposes_training_inference_and_dashboard(self) -> None:
        self.assertEqual(
            set(COMMANDS),
            {"train-features", "train-cnn", "infer", "dashboard"},
        )

    def test_usage_names_the_module(self) -> None:
        self.assertIn("scene_recognition.cli", usage())


if __name__ == "__main__":
    unittest.main()
