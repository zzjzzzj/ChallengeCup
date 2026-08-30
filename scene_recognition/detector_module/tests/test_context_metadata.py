from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scene_recognition.detector_module.context_metadata import (
    CONTEXT_INDEX_FIELDS,
    build_context_row,
    context_index_summary,
    read_context_index,
    read_context_rows,
    write_context_index,
)


class ContextMetadataTests(unittest.TestCase):
    def test_round_trip_has_explicit_context_fields_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "materialized" / "image.png"
            row = build_context_row(
                materialized_image_path=image,
                source_image="source_without_context.png",
                sensor="sar",
                scene="urban",
                split="train",
                stage=2,
                sample_role="replay",
                augmentation_operation="flip",
            )
            path = write_context_index([row], root / "context_index.csv")
            rows = read_context_rows(path)
            indexed = read_context_index(path)
        self.assertEqual(set(rows[0]), set(CONTEXT_INDEX_FIELDS))
        self.assertEqual(rows[0]["sensor"], "sar")
        self.assertEqual(rows[0]["scene"], "urban")
        self.assertEqual(rows[0]["sample_role"], "replay")
        self.assertEqual(len(indexed), 1)
        self.assertEqual(context_index_summary(rows)["known_sensor_images"], 1)

    def test_empty_or_malformed_index_is_checked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            empty = root / "empty.csv"
            empty.write_text(",".join(CONTEXT_INDEX_FIELDS) + "\n", encoding="utf-8")
            self.assertEqual(read_context_rows(empty), [])
            malformed = root / "malformed.csv"
            malformed.write_text("materialized_image_path\nfoo.png\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                read_context_rows(malformed)


if __name__ == "__main__":
    unittest.main()
