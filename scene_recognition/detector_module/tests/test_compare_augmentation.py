from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scene_recognition.detector_module.compare_augmentation import build_comparison, load_run


def make_summary(
    directory: Path,
    *,
    split: str = "val",
    epochs: int = 60,
    disabled: bool = True,
    map50: float = 0.8,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    summary = {
        "configuration": {
            "epochs": epochs,
            "batch_size": 16,
            "data": "d.yaml",
            "builtin_augmentation_disabled": disabled,
        },
        "evaluation_split": split,
        split: {
            "precision": 0.9,
            "recall": 0.7,
            "map50": map50,
            "map50_95": 0.4,
            "per_class": {
                "soldier": {"map50": 0.81},
                "small_aircraft": {"map50": 0.82},
                "warship": {"map50": 0.83},
                "tank": {"map50": 0.84},
            },
        },
    }
    path = directory / "baseline_summary.json"
    path.write_text(json.dumps(summary, ensure_ascii=False), encoding="utf-8")
    return path


class CompareAugmentationTests(unittest.TestCase):
    def test_load_run_reads_declared_split(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = make_summary(Path(tmp) / "a", split="val")
            run = load_run(path)
            self.assertEqual(run["evaluation_split"], "val")
            self.assertAlmostEqual(run["metrics"]["map50"], 0.8)
            self.assertTrue(run["builtin_augmentation_disabled"])

    def test_load_run_falls_back_to_legacy_test_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "legacy"
            directory.mkdir()
            path = directory / "baseline_summary.json"
            path.write_text(
                json.dumps({"configuration": {"epochs": 100}, "test": {"map50": 0.5}}),
                encoding="utf-8",
            )
            self.assertEqual(load_run(path)["evaluation_split"], "test")

    def test_load_run_rejects_missing_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "broken"
            directory.mkdir()
            path = directory / "baseline_summary.json"
            path.write_text(
                json.dumps({"configuration": {}, "evaluation_split": "test"}),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_run(path)

    def test_comparison_reports_image_presentations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runs = {
                "aug": load_run(make_summary(Path(tmp) / "aug", epochs=60)),
                "noaug": load_run(make_summary(Path(tmp) / "noaug", epochs=444)),
            }
            markdown = build_comparison(runs, {"aug": 4400, "noaug": 595})
            # 4400*60 = 264,000 与 595*444 = 264,180，预算基本对齐。
            self.assertIn("264,000", markdown)
            self.assertIn("264,180", markdown)

    def test_comparison_warns_on_mismatched_split(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runs = {
                "a": load_run(make_summary(Path(tmp) / "a", split="val")),
                "b": load_run(make_summary(Path(tmp) / "b", split="test")),
            }
            self.assertIn("指标不可比", build_comparison(runs))

    def test_comparison_warns_when_builtin_augmentation_left_on(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runs = {
                "a": load_run(make_summary(Path(tmp) / "a", disabled=False)),
                "b": load_run(make_summary(Path(tmp) / "b", disabled=False)),
            }
            markdown = build_comparison(runs)
            self.assertIn("不能单独归因", markdown)

    def test_comparison_requires_two_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runs = {"only": load_run(make_summary(Path(tmp) / "only"))}
            with self.assertRaises(ValueError):
                build_comparison(runs)


if __name__ == "__main__":
    unittest.main()
