from __future__ import annotations

import unittest

from image_processing.cli import COMMANDS, usage


class ImageProcessingCliTests(unittest.TestCase):
    def test_exposes_all_preprocessing_stages(self) -> None:
        self.assertEqual(
            set(COMMANDS),
            {"audit", "features", "crops", "detection", "comparison"},
        )

    def test_usage_names_the_module(self) -> None:
        self.assertIn("image_processing.cli", usage())


if __name__ == "__main__":
    unittest.main()
