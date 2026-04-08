from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict


@dataclass(frozen=True, slots=True)
class HostTraceContext:
    trace_id: str
    recording_dir: Path
    metadata_file: Path
    conversations_file: Path
    traj_file: Path
    task_description: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "recording_dir": str(self.recording_dir.resolve()),
            "metadata_file": str(self.metadata_file.resolve()),
            "conversations_file": str(self.conversations_file.resolve()),
            "traj_file": str(self.traj_file.resolve()),
            "task_description": self.task_description,
        }
