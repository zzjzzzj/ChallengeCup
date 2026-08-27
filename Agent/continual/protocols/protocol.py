from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from Agent.common.schemas import TaskStage
from Agent.common.utils.jsonio import read_json, write_json


@dataclass
class ContinualProtocol:
    """Editable continual-learning task sequence."""

    protocol_name: str
    stages: list[TaskStage]
    description: str = ""

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ContinualProtocol":
        stages = [TaskStage(**stage) for stage in payload.get("stages", [])]
        if not stages:
            raise ValueError("continual protocol must contain at least one stage")
        task_ids = [stage.task_id for stage in stages]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("task_id values must be unique")
        return cls(
            protocol_name=str(payload.get("protocol_name", "unnamed_protocol")),
            description=str(payload.get("description", "")),
            stages=stages,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol_name": self.protocol_name,
            "description": self.description,
            "stages": [stage.to_dict() for stage in self.stages],
        }

    def save(self, path: Path) -> None:
        write_json(path, self.to_dict())

    def get_stage(self, task_id: str) -> TaskStage:
        for stage in self.stages:
            if stage.task_id == task_id:
                return stage
        raise KeyError(f"unknown task_id: {task_id}")

    def previous_stages(self, task_id: str) -> list[TaskStage]:
        result = []
        for stage in self.stages:
            if stage.task_id == task_id:
                return result
            result.append(stage)
        raise KeyError(f"unknown task_id: {task_id}")


def load_protocol(path: Path) -> ContinualProtocol:
    return ContinualProtocol.from_dict(read_json(path))
