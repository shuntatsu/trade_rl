"""Signal artifact state and identity."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from trade_rl.domain.common import (
    require_aware_datetime,
    require_non_empty,
    require_sha256,
)


class SignalStatus(StrEnum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    DISABLED = "disabled"


@dataclass(frozen=True, slots=True)
class SignalArtifactManifest:
    """Immutable metadata for a fitted alpha/signal artifact."""

    digest: str
    dataset_id: str
    model_kind: str
    target: str
    horizon: int
    status: SignalStatus
    alpha_enabled: bool
    created_at: datetime
    schema_version: str = "signal_artifact_v1"

    def __post_init__(self) -> None:
        require_sha256(self.digest, field="digest")
        require_sha256(self.dataset_id, field="dataset_id")
        require_non_empty(self.model_kind, field="model_kind")
        require_non_empty(self.target, field="target")
        if self.horizon <= 0:
            raise ValueError("horizon must be positive")
        require_aware_datetime(self.created_at, field="created_at")
        require_non_empty(self.schema_version, field="schema_version")
        if self.status is SignalStatus.REJECTED and self.alpha_enabled:
            raise ValueError("rejected signal cannot enable alpha")
        if self.status is SignalStatus.DISABLED and self.alpha_enabled:
            raise ValueError("disabled signal cannot enable alpha")
