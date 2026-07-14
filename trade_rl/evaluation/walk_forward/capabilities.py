"""Range-only capabilities for nested walk-forward stages."""

from __future__ import annotations

from dataclasses import dataclass

from trade_rl.domain.common import require_non_empty, require_sha256
from trade_rl.evaluation.walk_forward.folds import IndexRange


@dataclass(frozen=True, slots=True)
class RangeCapability:
    """Authorize one stage to request only subranges of an assigned range."""

    dataset_id: str
    stage: str
    allowed: IndexRange

    def __post_init__(self) -> None:
        require_sha256(self.dataset_id, field="dataset_id")
        require_non_empty(self.stage, field="stage")

    def require(self, requested: IndexRange) -> IndexRange:
        if not (
            self.allowed.start <= requested.start
            and requested.stop <= self.allowed.stop
        ):
            raise ValueError("requested range is outside the stage capability")
        return requested


__all__ = ["RangeCapability"]
