"""Evaluation and mandatory gate domain records."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from trade_rl.domain.common import require_aware_datetime, require_non_empty


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
    """Resolved gate decision with explicit mandatory-check semantics."""

    passed: bool
    checks: tuple[GateCheck, ...]
    decided_at: datetime
    schema_version: str = "gate_decision_v1"

    def __post_init__(self) -> None:
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
