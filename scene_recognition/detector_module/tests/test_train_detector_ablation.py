from __future__ import annotations

import unittest

from scene_recognition.detector_module.train_detector import (
    BUILTIN_AUGMENTATION,
    DISABLED_AUGMENTATION,
)
from scene_recognition.detector_module.train_detector_ablation import resolve_augmentation


class ResolveAblationAugmentationTest(unittest.TestCase):
    def test_default_keeps_baseline_online_augmentation(self) -> None:
        self.assertEqual(resolve_augmentation(False), BUILTIN_AUGMENTATION)

    def test_switch_disables_every_online_augmentation(self) -> None:
        recipe = resolve_augmentation(True)

        self.assertEqual(recipe, DISABLED_AUGMENTATION)
        self.assertTrue(all(value == 0.0 for value in recipe.values()))

    def test_returns_copy_of_shared_recipe(self) -> None:
        recipe = resolve_augmentation(True)
        recipe["mosaic"] = 1.0

        self.assertEqual(DISABLED_AUGMENTATION["mosaic"], 0.0)


if __name__ == "__main__":
    unittest.main()
