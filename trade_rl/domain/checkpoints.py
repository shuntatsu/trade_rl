"""Framework-neutral policy checkpoint contracts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class PolicyCheckpoint:
    path: Path
    algorithm: str

    def __post_init__(self) -> None:
        if not self.algorithm:
            raise ValueError("checkpoint algorithm must be non-empty")
        if not self.path.is_file():
            raise FileNotFoundError(f"policy checkpoint is missing: {self.path}")


class PolicyCheckpointLoader(Protocol):
    def load(self, checkpoint: PolicyCheckpoint) -> Any: ...


__all__ = ["PolicyCheckpoint", "PolicyCheckpointLoader"]
