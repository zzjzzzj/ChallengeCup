from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from Agent.agent import IntelligentRecognitionAgent
from Agent.config import AgentConfig
from Agent.losses import combine_training_losses
from Agent.reasoning import SCENE_POLICY
from scene_recognition.detector_module.boxes import parse_yolo_boxes, resolve_label_path
from scene_recognition.feature_infer import selected_feature_values


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
        self.assertIn("policy_source", report["decision"])
        self.assertEqual(report["decision"]["policy_source"], "image_processing.scene_runtime.DEFAULT_POLICY")

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

    def test_scene_policy_inherits_runtime_defaults(self) -> None:
        self.assertEqual(SCENE_POLICY["sea"]["detector_profile"], "sea_ship")
        self.assertIn("sensor_weights", SCENE_POLICY["sea"])
        self.assertIn("notes", SCENE_POLICY["sea"])


class BoxesApiTests(unittest.TestCase):
    def test_resolve_and_parse_with_confidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "sample.png"
            Image.new("L", (8, 8), color=10).save(image)
            label = image.with_suffix(".txt")
            label.write_text("1 0.5 0.5 0.2 0.2 0.88\n", encoding="utf-8")
            resolved = resolve_label_path(image)
            boxes = parse_yolo_boxes(resolved, 4, allow_confidence=True)
        self.assertEqual(resolved, label)
        self.assertEqual(len(boxes), 1)
        self.assertEqual(boxes[0].class_id, 1)
        self.assertAlmostEqual(float(boxes[0].confidence or 0.0), 0.88)


class FeatureInferApiTests(unittest.TestCase):
    def test_selected_feature_values(self) -> None:
        values = selected_feature_values({"a": 1.23456789, "b": 2.0}, ["b", "a"], precision=3)
        self.assertEqual(list(values.keys()), ["b", "a"])
        self.assertEqual(values["a"], 1.235)

    def test_predict_scene_from_features_features_only(self) -> None:
        from scene_recognition.feature_infer import predict_scene_from_features

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "img.png"
            Image.new("RGB", (32, 32), color=(40, 40, 40)).save(image)
            metadata = {
                "selected_features": ["int_mean"],
                "input_features": ["int_mean"],
                "scene_names": ["air", "sea", "urban", "forest"],
            }
            metadata_path = root / "meta.json"
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            fake_features = {"int_mean": 0.5}
            with patch("scene_recognition.feature_infer.extract_one", return_value=fake_features):
                result = predict_scene_from_features(
                    image,
                    root / "missing.joblib",
                    metadata_path,
                    features_only=True,
                    extracted=fake_features,
                )
        self.assertEqual(result["selected_feature_count"], 1)
        self.assertEqual(result["selected_feature_values"]["int_mean"], 0.5)


if __name__ == "__main__":
    unittest.main()
