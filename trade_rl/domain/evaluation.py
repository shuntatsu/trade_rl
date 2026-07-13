"""Evaluation and mandatory gate domain records."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from trade_rl.domain.common import (
    require_aware_datetime,
    require_non_empty,
    require_sha256,
)


@dataclass(frozen=True, slots=True)
class GateCheck:
    """One named evaluation gate check."""

    name: str
    passed: bool
    mandatory: bool = True
    detail: str | None = None

    def __post_init__(self) -> None:
        require_non_empty(self.name, field="name")
        if self.detail is not None:
            require_non_empty(self.detail, field="detail")


@dataclass(frozen=True, slots=True)
class GateDecision:
    """Gate result bound to the evaluated dataset and selected policy identity."""

    dataset_id: str
    selected_policy_digest: str | None
    evaluation_digest: str
    passed: bool
    checks: tuple[GateCheck, ...]
    decided_at: datetime
    schema_version: str = "gate_decision_v2"

    def __post_init__(self) -> None:
        require_sha256(self.dataset_id, field="dataset_id")
        if self.selected_policy_digest is not None:
            require_sha256(
                self.selected_policy_digest,
                field="selected_policy_digest",
            )
        require_sha256(self.evaluation_digest, field="evaluation_digest")
        if not self.checks:
            raise ValueError("checks must not be empty")
        names = tuple(check.name for check in self.checks)
        if len(set(names)) != len(names):
            raise ValueError("gate check names must be unique")
        require_aware_datetime(self.decided_at, field="decided_at")
        require_non_empty(self.schema_version, field="schema_version")
        mandatory_passed = all(check.passed for check in self.checks if check.mandatory)
        if self.passed != mandatory_passed:
            raise ValueError(
                "gate passed flag must equal the conjunction of mandatory checks"
            )

    @property
    def failed_mandatory_checks(self) -> tuple[GateCheck, ...]:
        return tuple(
            check for check in self.checks if check.mandatory and not check.passed
        )
