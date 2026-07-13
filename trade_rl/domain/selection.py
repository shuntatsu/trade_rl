"""Candidate selection state for baseline-only and residual policies."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from trade_rl.domain.common import (
    require_aware_datetime,
    require_non_empty,
    require_sha256,
)


class PolicyMode(StrEnum):
    BASELINE_ONLY = "baseline_only"
    RESIDUAL_POLICY = "residual_policy"


@dataclass(frozen=True, slots=True)
class SelectionDecision:
    """Immutable configuration-selection decision made before sealed holdout use."""

    dataset_id: str
    mode: PolicyMode
    selected_configuration: str
    selected_policy_digest: str | None
    signal_digest: str
    evaluation_digest: str
    selected_at: datetime
    reasons: tuple[str, ...]
    schema_version: str = "selection_decision_v1"

    def __post_init__(self) -> None:
        require_sha256(self.dataset_id, field="dataset_id")
        require_non_empty(
            self.selected_configuration,
            field="selected_configuration",
        )
        require_sha256(self.signal_digest, field="signal_digest")
        require_sha256(self.evaluation_digest, field="evaluation_digest")
        require_aware_datetime(self.selected_at, field="selected_at")
        require_non_empty(self.schema_version, field="schema_version")
        if not self.reasons:
            raise ValueError("selection reasons must not be empty")
        for reason in self.reasons:
            require_non_empty(reason, field="reason")
        if self.mode is PolicyMode.BASELINE_ONLY:
            if self.selected_policy_digest is not None:
                raise ValueError(
                    "baseline_only selection cannot contain a policy digest"
                )
        elif self.selected_policy_digest is None:
            raise ValueError("residual policy selection requires a policy digest")
        else:
            require_sha256(
                self.selected_policy_digest,
                field="selected_policy_digest",
            )
