from __future__ import annotations

import unittest

from PIL import Image

from scene_recognition.target_classifier_module.diagnose_shortcut import (
    check_label_degeneracy,
    read_manifest_samples,
    trivial_baselines,
)
from test_support import workspace_test_directory


def _write_split(root, split, specs):
    """specs: list of (sensor, scene, index, [(class_id, w_ratio, h_ratio), ...])"""

    paths = []
    for sensor, scene, index, boxes in specs:
        image_path = root / f"{sensor}_r1_base_{scene}_{index:06d}.png"
        Image.new("RGB", (64, 64), color=(40, 60, 80)).save(image_path)
        image_path.with_suffix(".txt").write_text(
            "\n".join(
                f"{class_id} 0.5 0.5 {width:.4f} {height:.4f}"
                for class_id, width, height in boxes
            )
            + "\n",
            encoding="utf-8",
        )
        paths.append(str(image_path))
    (root / "manifests" / f"{split}.txt").write_text("\n".join(paths) + "\n", encoding="utf-8")


class LabelDegeneracyTests(unittest.TestCase):
    def test_detects_scene_bound_labels_as_degenerate(self):
        with workspace_test_directory("shortcut-degenerate") as root:
            (root / "manifests").mkdir()
            # 每个场景只出现一种存在向量 —— 与真实数据集的情况一致
            for split in ("train", "val", "test"):
                _write_split(
                    root,
                    split,
                    [
                        ("ir", "air", 1, [(1, 0.1, 0.1)]),
                        ("ir", "sea", 2, [(2, 0.1, 0.1)]),
                        ("ir", "urban", 3, [(0, 0.02, 0.02), (3, 0.2, 0.2)]),
                    ],
                )
            samples = read_manifest_samples(root / "manifests")
            report = check_label_degeneracy(samples)
            self.assertTrue(report["label_is_deterministic_function_of_scene"])
            self.assertEqual(report["distinct_presence_vectors"], 3)
            self.assertEqual(report["theoretical_maximum_vectors"], 16)
            self.assertEqual(
                report["scene_to_vectors"]["air"][0]["vector"], [0, 1, 0, 0]
            )
            self.assertIn("场景分类", report["interpretation"])

    def test_scene_with_two_label_vectors_is_not_degenerate(self):
        with workspace_test_directory("shortcut-varied") as root:
            (root / "manifests").mkdir()
            for split in ("train", "val", "test"):
                _write_split(
                    root,
                    split,
                    [
                        # 同为 urban，但存在向量不同 -> 不退化
                        ("ir", "urban", 1, [(0, 0.02, 0.02)]),
                        ("ir", "urban", 2, [(3, 0.2, 0.2)]),
                        ("ir", "air", 3, [(1, 0.1, 0.1)]),
                    ],
                )
            samples = read_manifest_samples(root / "manifests")
            report = check_label_degeneracy(samples)
            self.assertFalse(report["label_is_deterministic_function_of_scene"])
            self.assertIsNone(
                report["trivial_baselines_by_split"]["test"][
                    "perfect_scene_oracle_exact_match"
                ]
            )

    def test_constant_prediction_baseline_matches_majority_vector(self):
        with workspace_test_directory("shortcut-constant") as root:
            (root / "manifests").mkdir()
            for split in ("train", "val", "test"):
                _write_split(
                    root,
                    split,
                    [
                        ("ir", "urban", 1, [(0, 0.02, 0.02), (3, 0.2, 0.2)]),
                        ("ir", "forest", 2, [(0, 0.02, 0.02), (3, 0.2, 0.2)]),
                        ("ir", "air", 3, [(1, 0.1, 0.1)]),
                    ],
                )
            samples = read_manifest_samples(root / "manifests")
            report = check_label_degeneracy(samples)
            baseline = report["trivial_baselines_by_split"]["test"]
            self.assertEqual(baseline["constant_prediction"], [1, 0, 0, 1])
            self.assertAlmostEqual(baseline["constant_exact_match"], 2 / 3)
            self.assertEqual(baseline["perfect_scene_oracle_exact_match"], 1.0)


class TrivialBaselineTests(unittest.TestCase):
    def test_pixel_free_baselines_report_scene_and_size_accuracy(self):
        with workspace_test_directory("shortcut-trivial") as root:
            (root / "manifests").mkdir()
            # soldier 框远小于 tank 框，且类别与场景绑定 -> 平凡基线应接近满分
            specs = [
                ("ir", "air", 1, [(1, 0.15, 0.15)]),
                ("ir", "sea", 2, [(2, 0.15, 0.15)]),
                ("ir", "urban", 3, [(0, 0.02, 0.02), (3, 0.30, 0.30)]),
                ("ir", "forest", 4, [(0, 0.02, 0.02), (3, 0.30, 0.30)]),
            ]
            for split in ("train", "val", "test"):
                _write_split(root, split, specs)
            samples = read_manifest_samples(root / "manifests")
            report = trivial_baselines(samples)
            self.assertEqual(report["scene_lookup"]["air"], "small_aircraft")
            self.assertEqual(report["scene_lookup"]["sea"], "warship")
            self.assertEqual(report["scene_and_size"]["accuracy"], 1.0)
            self.assertEqual(report["best_pixel_free_accuracy"], 1.0)
            self.assertEqual(
                report["soldier_vs_tank_size_stump"]["accuracy"], 1.0
            )
            self.assertIn("不看像素", report["interpretation"])

    def test_reports_low_pixel_free_accuracy_when_cues_are_absent(self):
        with workspace_test_directory("shortcut-nocue") as root:
            (root / "manifests").mkdir()
            # 同一场景、同一尺寸下混合类别 -> 平凡线索失效
            specs = [
                ("ir", "urban", 1, [(0, 0.1, 0.1)]),
                ("ir", "urban", 2, [(3, 0.1, 0.1)]),
                ("ir", "urban", 3, [(0, 0.1, 0.1)]),
                ("ir", "urban", 4, [(3, 0.1, 0.1)]),
            ]
            for split in ("train", "val", "test"):
                _write_split(root, split, specs)
            samples = read_manifest_samples(root / "manifests")
            report = trivial_baselines(samples)
            self.assertLessEqual(report["best_pixel_free_accuracy"], 0.75)


if __name__ == "__main__":
    unittest.main()
