"""310B 端到端编排器的无模型单元测试。"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


DEPLOYMENT_DIR = Path(__file__).resolve().parents[1]
if str(DEPLOYMENT_DIR) not in sys.path:
    sys.path.insert(0, str(DEPLOYMENT_DIR))

import run_end_to_end


class RunEndToEndTest(unittest.TestCase):
    def test_auto_backend_uses_model_suffix(self) -> None:
        self.assertEqual(run_end_to_end.resolve_backend(Path("model.onnx"), "auto"), "onnx")
        self.assertEqual(run_end_to_end.resolve_backend(Path("model.om"), "auto"), "om")
        with self.assertRaises(ValueError):
            run_end_to_end.resolve_backend(Path("model.pt"), "auto")

    def test_pipeline_connects_augmentation_and_onnx_inference(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_root = root / "dataset"
            dataset_root.mkdir()
            model_path = root / "detector.onnx"
            model_path.write_bytes(b"test-model")
            output_dir = root / "output"
            commands = []

            def fake_run(command):
                commands.append(command)
                if command[1].endswith("augment_selected_yolo.py"):
                    augmented_dir = Path(command[command.index("--output") + 1])
                    (augmented_dir / "images").mkdir(parents=True)
                    (augmented_dir / "classes.txt").write_text("ship\n", encoding="utf-8")
                    (augmented_dir / "augmentation_summary.json").write_text(
                        json.dumps({"source_images": 1}, ensure_ascii=False), encoding="utf-8"
                    )
                    return
                prediction_path = Path(command[command.index("--output") + 1])
                prediction_path.parent.mkdir(parents=True, exist_ok=True)
                prediction_path.write_text(
                    json.dumps({"image": "sample.png", "detections": [{"name": "ship"}]}),
                    encoding="utf-8",
                )

            with patch.object(run_end_to_end, "run_command", side_effect=fake_run):
                exit_code = run_end_to_end.main(
                    [
                        "--dataset-root",
                        str(dataset_root),
                        "--model",
                        str(model_path),
                        "--output-dir",
                        str(output_dir),
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertEqual(len(commands), 2)
            self.assertTrue(commands[0][1].endswith("augment_selected_yolo.py"))
            self.assertTrue(commands[1][1].endswith("infer_yolov8_onnx.py"))
            self.assertIn(str(output_dir / "augmented_dataset" / "images"), commands[1])
            summary = json.loads((output_dir / "pipeline_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["model"]["backend"], "onnx")
            self.assertEqual(summary["training"]["performed"], False)
            self.assertEqual(summary["inference"]["statistics"], {"images": 1, "detections": 1})
            self.assertTrue((output_dir / "inference" / "runtime_metadata.json").is_file())
            self.assertTrue((output_dir / "inference" / "result_summary.csv").is_file())


if __name__ == "__main__":
    unittest.main()
