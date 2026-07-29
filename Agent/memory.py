from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


class EpisodeMemory:
    """JSONL memory for replay, feedback, and incremental-learning bookkeeping."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def append(self, record: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            **record,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def append_report(self, report: dict[str, Any]) -> None:
        self.append(
            {
                "type": "agent_report",
                "image": report["image"],
                "modality": report["modality"]["label"],
                "scene": report["final_scene"]["label"],
                "targets": [
                    {
                        "class_name": box["class_name"],
                        "confidence": box["confidence"],
                        "xyxy_norm": box["xyxy_norm"],
                    }
                    for box in report.get("detections", [])
                ],
                "consistency_status": report.get("consistency", {}).get("status"),
                "loss_total": report.get("losses", {}).get("total"),
            }
        )

    def append_feedback(
        self,
        *,
        image: str,
        corrected_scene: str | None = None,
        corrected_modality: str | None = None,
        corrected_targets: list[str] | None = None,
        note: str | None = None,
    ) -> None:
        self.append(
            {
                "type": "feedback",
                "image": image,
                "corrected_scene": corrected_scene,
                "corrected_modality": corrected_modality,
                "corrected_targets": corrected_targets or [],
                "note": note,
            }
        )

    def load_recent(self, limit: int = 100) -> list[dict[str, Any]]:
        if not self.path.is_file():
            return []
        rows = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return rows[-limit:]

    def summary(self, limit: int = 200) -> dict[str, Any]:
        rows = self.load_recent(limit)
        report_rows = [row for row in rows if row.get("type") == "agent_report"]
        feedback_rows = [row for row in rows if row.get("type") == "feedback"]
        scenes = Counter(row.get("scene", "unknown") for row in report_rows)
        modalities = Counter(row.get("modality", "unknown") for row in report_rows)
        targets = Counter(
            target.get("class_name", "unknown")
            for row in report_rows
            for target in row.get("targets", [])
        )
        return {
            "memory_path": str(self.path),
            "recent_records": len(rows),
            "recent_reports": len(report_rows),
            "recent_feedback": len(feedback_rows),
            "scene_counts": dict(scenes),
            "modality_counts": dict(modalities),
            "target_counts": dict(targets),
            "usage": "Use feedback records as replay/prototype seeds for incremental model updates.",
        }
