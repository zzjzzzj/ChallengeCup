from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from Agent.common.schemas import DetectionPrediction
from Agent.continual.manager import ContinualLearningManager
from Agent.continual.protocols import ContinualProtocol
from Agent.continual.replay import ReplayBuffer, ReplayItem
from Agent.data.copy_paste import CopyPasteObject, paste_objects
from Agent.data.indexing import YoloLabel, parse_yolo_label_file
from Agent.data.sampling import image_sampling_weight
from Agent.inference.cascade import should_trigger_second_evaluation
from Agent.inference.fusion import weighted_box_fusion
from Agent.inference.slicing import generate_tiles, map_tile_box_to_image
from Agent.inference.uncertainty import uncertainty_score
from Agent.models.adapters import SoftContextFusion
from Agent.models.prototypes import PrototypeBank
from Agent.models.yolo_p2 import write_yolov8n_p2_yaml
from PIL import Image


class ArchitectureComponentTests(unittest.TestCase):
    def test_soft_context_fusion_is_additive_not_hard_rule(self) -> None:
        fusion = SoftContextFusion(
            class_names=["soldier", "small_aircraft", "warship", "tank"],
            scene_names=["air", "sea", "urban", "forest"],
            modality_names=["ir", "sar"],
            alpha=0.2,
            beta=0.1,
        )
        det_logits = np.asarray([3.0, 0.2, 0.1, 0.1], dtype=np.float32)
        scene_prob = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        modality_prob = np.asarray([1.0, 0.0], dtype=np.float32)
        fused = fusion.fuse_numpy(det_logits, scene_prob, modality_prob)
        self.assertGreater(fused[0], fused[1])
        self.assertGreater(fused[1], det_logits[1])

    def test_prototype_bank_builds_shared_class_prototype(self) -> None:
        bank = PrototypeBank(momentum=0.5)
        bank.update("tank", "ir", np.asarray([[1.0, 0.0], [0.9, 0.1]], dtype=np.float32))
        bank.update("tank", "sar", np.asarray([[0.8, 0.2]], dtype=np.float32))
        shared = bank.shared_prototype("tank")
        self.assertIsNotNone(shared)
        self.assertAlmostEqual(float(np.linalg.norm(shared)), 1.0, places=5)

    def test_replay_buffer_keeps_highest_scored_items(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            buffer = ReplayBuffer(root, capacity=1)
            low = ReplayItem(image_path=root / "a.png", label_path=None, task_id="t1", classes=["warship"], score=0.1)
            high = ReplayItem(image_path=root / "b.png", label_path=None, task_id="t1", classes=["soldier"], score=0.9)
            buffer.add(low)
            buffer.add(high)
            items = buffer.load()
            self.assertEqual(len(items), 1)
            self.assertEqual(items[0].image_path.name, "b.png")

    def test_protocol_manager_builds_plan_from_scene_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "ir_r1_base_air_000001.png"
            label = root / "ir_r1_base_air_000001.txt"
            image.write_bytes(b"fake")
            label.write_text("1 0.5 0.5 0.1 0.1\n", encoding="utf-8")
            index = root / "scene_index.csv"
            index.write_text(
                "image_path,image_name,sensor,scene,scene_id,sequence_index,split\n"
                f"{image},{image.name},ir,air,0,1,train\n",
                encoding="utf-8-sig",
            )
            protocol = ContinualProtocol.from_dict(
                {
                    "protocol_name": "demo",
                    "stages": [
                        {
                            "task_id": "task_1",
                            "name": "demo",
                            "modalities": ["ir"],
                            "scenes": ["air"],
                            "classes": ["small_aircraft"],
                        }
                    ],
                }
            )
            manager = ContinualLearningManager(root / "workspace", protocol, replay_capacity=4)
            plan = manager.build_training_plan(index, "task_1")
            self.assertEqual(len(plan.current_samples), 1)
            assets = manager.export_training_assets(plan, root / "exported")
            self.assertTrue(Path(assets["data_yaml"]).is_file())
            self.assertEqual(assets["exported_counts"]["train"], 1)
            self.assertIn("small_aircraft", assets["allowed_label_classes"])

    def test_uncertainty_cascade_tiles_and_fusion(self) -> None:
        u = uncertainty_score(np.asarray([0.34, 0.33, 0.33], dtype=np.float32))
        decision = should_trigger_second_evaluation([], u["uncertainty"], threshold=0.1)
        self.assertTrue(decision.triggered)
        tiles = generate_tiles(640, 512, grid=2, overlap=0.2)
        self.assertEqual(len(tiles), 4)
        mapped = map_tile_box_to_image(tiles[0], (0.1, 0.1, 0.2, 0.2))
        self.assertTrue(all(0.0 <= value <= 1.0 for value in mapped))
        detections = [
            DetectionPrediction("soldier", 0.8, (0.10, 0.10, 0.20, 0.20), pass_id=1),
            DetectionPrediction("soldier", 0.7, (0.11, 0.11, 0.21, 0.21), pass_id=2),
        ]
        fused = weighted_box_fusion(detections, iou_threshold=0.3)
        self.assertEqual(len(fused), 1)
        self.assertEqual(fused[0].pass_id, 2)

    def test_yolo_p2_yaml_and_hard_sampling(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            yaml_path = write_yolov8n_p2_yaml(Path(tmp) / "yolov8n_p2.yaml", class_count=4)
            text = yaml_path.read_text(encoding="utf-8")
            self.assertIn("[[18, 21, 24, 27], 1, Detect, [nc]]", text)
        labels = [YoloLabel(0, 0.5, 0.5, 0.01, 0.02), YoloLabel(3, 0.4, 0.4, 0.04, 0.04)]
        self.assertGreater(image_sampling_weight(labels), 1.0)

    def test_copy_paste_preserves_object_label_size(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bg = root / "bg.png"
            src = root / "src.png"
            bg_label = root / "bg.txt"
            out_img = root / "out.png"
            out_label = root / "out.txt"
            Image.new("RGB", (100, 100), color=(20, 20, 20)).save(bg)
            Image.new("RGB", (100, 100), color=(200, 200, 200)).save(src)
            bg_label.write_text("", encoding="utf-8")
            obj = CopyPasteObject(src, YoloLabel(0, 0.5, 0.5, 0.2, 0.2), padding_ratio=0.1)
            paste_objects(bg, bg_label, [obj], out_img, out_label, seed=1)
            labels = parse_yolo_label_file(out_label)
            self.assertEqual(len(labels), 1)
            self.assertAlmostEqual(labels[0].width, 0.2, places=6)


if __name__ == "__main__":
    unittest.main()
