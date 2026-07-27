from __future__ import annotations

import unittest

from detector_module.create_incremental_protocol import build_protocol, validate_partition


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


if __name__ == "__main__":
    unittest.main()
