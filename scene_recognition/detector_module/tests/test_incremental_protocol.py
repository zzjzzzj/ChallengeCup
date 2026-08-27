from __future__ import annotations

import unittest

from scene_recognition.detector_module.create_incremental_protocol import build_protocol, validate_partition


class IncrementalProtocolTests(unittest.TestCase):
    def test_builds_base_and_two_incremental_stages(self):
        protocol = build_protocol([["soldier", "tank"]][0], [["small_aircraft"], ["warship"]])
        self.assertEqual(len(protocol["stages"]), 3)
        self.assertEqual(protocol["stages"][1]["old_classes"], ["soldier", "tank"])
        self.assertEqual(protocol["stages"][2]["all_learned_classes"], [
            "soldier", "tank", "small_aircraft", "warship"
        ])

    def test_duplicate_assignment_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "重复分配"):
            validate_partition(["soldier", "tank"], [["soldier"], ["warship", "small_aircraft"]])

    def test_missing_class_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "未纳入协议"):
            validate_partition(["soldier", "tank"], [["small_aircraft"]])

    def test_supports_official_r2_six_class_protocol(self):
        classes = [
            "soldier",
            "small_aircraft",
            "warship",
            "tank",
            "patrol_boat",
            "armored_vehicle",
        ]
        protocol = build_protocol(
            classes[:4],
            [["patrol_boat", "armored_vehicle"]],
            classes,
        )
        self.assertEqual(protocol["class_order"], classes)
        self.assertEqual(protocol["stages"][1]["old_classes"], classes[:4])
        self.assertEqual(protocol["stages"][1]["all_learned_classes"], classes)

    def test_supports_six_singleton_class_incremental_stages(self):
        classes = [
            "soldier",
            "small_aircraft",
            "warship",
            "tank",
            "patrol_boat",
            "armored_vehicle",
        ]
        protocol = build_protocol([classes[0]], [[name] for name in classes[1:]], classes)
        self.assertEqual(protocol["scenario"], "class_incremental")
        self.assertEqual(len(protocol["stages"]), 6)
        self.assertTrue(all(len(stage["new_classes"]) == 1 for stage in protocol["stages"]))
        self.assertEqual(protocol["stages"][-1]["all_learned_classes"], classes)


if __name__ == "__main__":
    unittest.main()
