from __future__ import annotations

import unittest

import torch

from scene_recognition.detector_module.dark_experience_replay import (
    dark_replay_decoupled_loss,
    dark_replay_response_loss,
    normalize_path,
    replay_batch_mask,
)


class DarkExperienceReplayTests(unittest.TestCase):
    def test_identical_old_responses_have_zero_loss_despite_new_student_channel(self) -> None:
        teacher = torch.randn(2, 66, 4, 4)
        teacher[:, 64:, :, :] = 4.0
        student = torch.cat([teacher.clone(), torch.randn(2, 1, 4, 4)], dim=1).requires_grad_()
        loss, parts = dark_replay_response_loss(
            [student],
            [teacher],
            old_class_count=2,
            reg_max=16,
        )
        self.assertAlmostEqual(float(loss.detach()), 0.0, places=7)
        self.assertAlmostEqual(float(parts["cls"].detach()), 0.0, places=7)
        loss.backward()
        self.assertIsNotNone(student.grad)

    def test_changed_old_class_and_box_responses_produce_positive_loss(self) -> None:
        teacher = torch.zeros(1, 66, 2, 2)
        teacher[:, 64:, :, :] = 4.0
        student = torch.cat([teacher.clone(), torch.zeros(1, 1, 2, 2)], dim=1)
        student[:, 0, :, :] = 2.0
        student[:, 64, :, :] = 1.0
        loss, parts = dark_replay_response_loss(
            [student],
            [teacher],
            old_class_count=2,
            reg_max=16,
            min_confidence=0.05,
        )
        self.assertGreater(float(loss), 0.0)
        self.assertGreater(float(parts["cls"]), 0.0)
        self.assertGreater(float(parts["box"]), 0.0)

    def test_replay_mask_matches_only_declared_paths(self) -> None:
        paths = ["C:/data/current.png", "C:/data/replay.png"]
        replay = {normalize_path(paths[1])}
        self.assertEqual(replay_batch_mask(paths, replay).tolist(), [False, True])

    def test_yolo26_decoupled_outputs_ignore_the_new_student_class(self) -> None:
        teacher = {
            "boxes": torch.randn(2, 4, 84),
            "scores": torch.full((2, 1, 84), 4.0),
        }
        student = {
            "boxes": teacher["boxes"].clone(),
            "scores": torch.cat([teacher["scores"].clone(), torch.randn(2, 1, 84)], dim=1),
        }
        loss, _ = dark_replay_decoupled_loss(
            student,
            teacher,
            old_class_count=1,
        )
        self.assertAlmostEqual(float(loss), 0.0, places=7)


if __name__ == "__main__":
    unittest.main()
