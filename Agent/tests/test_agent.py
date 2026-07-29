from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from Agent.agent import IntelligentRecognitionAgent
from Agent.config import AgentConfig
from Agent.losses import combine_training_losses


class AgentFlowTests(unittest.TestCase):
    def _make_image(self, path: Path) -> None:
        image = Image.new("L", (64, 48), color=90)
        image.save(path)

    def test_sidecar_label_completes_valid_flow(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "ir_r1_base_sea_000001.png"
            self._make_image(image)
            image.with_suffix(".txt").write_text("2 0.5 0.5 0.25 0.20\n", encoding="utf-8")
            config = AgentConfig.from_values(memory_path=root / "memory.jsonl", remember_runs=False)
            report = IntelligentRecognitionAgent(config).run(image, sensor_hint="ir", remember=False).to_dict()

        self.assertEqual(report["modality"]["label"], "ir")
        self.assertEqual(report["scene"]["label"], "sea")
        self.assertEqual(report["detections"][0]["class_name"], "warship")
        self.assertEqual(report["consistency"]["status"], "consistent")

    def test_invalid_scene_target_combination_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "sar_r1_base_sea_000002.png"
            self._make_image(image)
            image.with_suffix(".txt").write_text("3 0.5 0.5 0.20 0.20\n", encoding="utf-8")
            config = AgentConfig.from_values(memory_path=root / "memory.jsonl", remember_runs=False)
            report = IntelligentRecognitionAgent(config).run(image, sensor_hint="sar", remember=False).to_dict()

        self.assertGreaterEqual(report["consistency"]["original_invalid_count"], 1)
        self.assertIn(report["consistency"]["status"], {"invalid_combination", "repaired_by_target_scene_fusion"})

    def test_loss_formula(self) -> None:
        result = combine_training_losses(
            l_box=1.0,
            l_cls=2.0,
            l_dfl=0.5,
            l_detail=0.5,
            l_scene=0.25,
            l_proto=0.25,
            l_moti=0.25,
        )
        self.assertAlmostEqual(result["L_total"], 3.925)


if __name__ == "__main__":
    unittest.main()
