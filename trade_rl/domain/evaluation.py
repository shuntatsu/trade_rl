"""Evaluation and evidence-bound mandatory gate domain records."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from trade_rl.domain.common import (
    require_aware_datetime,
    require_non_empty,
    require_sha256,
)

Comparator = Literal[">", ">=", "<", "<=", "=="]


def _compare(value: float, comparator: Comparator, threshold: float) -> bool:
    if comparator == ">":
        return value > threshold
    if comparator == ">=":
        return value >= threshold
    if comparator == "<":
        return value < threshold
    if comparator == "<=":
        return value <= threshold
    return value == threshold


@dataclass(frozen=True, slots=True)
class GateCheck:
    """One named gate, optionally derived from immutable metric evidence."""

    name: str
    passed: bool
    mandatory: bool = True
    detail: str | None = None
    metric_name: str | None = None
    observed_value: float | None = None
    comparator: Comparator | None = None
    threshold: float | None = None
    evidence_digest: str | None = None
    implementation_digest: str | None = None

    def __post_init__(self) -> None:
        require_non_empty(self.name, field="name")
        if self.detail is not None:
            require_non_empty(self.detail, field="detail")
        evidence_fields = (
            self.metric_name,
            self.observed_value,
            self.comparator,
            self.threshold,
            self.evidence_digest,
            self.implementation_digest,
        )
        populated = tuple(value is not None for value in evidence_fields)
        if any(populated) and not all(populated):
            raise ValueError("metric gate evidence fields must be provided together")
        if all(populated):
            assert self.metric_name is not None
            assert self.observed_value is not None
            assert self.comparator is not None
            assert self.threshold is not None
            assert self.evidence_digest is not None
            assert self.implementation_digest is not None
            require_non_empty(self.metric_name, field="metric_name")
            if not math.isfinite(self.observed_value) or not math.isfinite(self.threshold):
                raise ValueError("gate metric values must be finite")
            if self.comparator not in {">", ">=", "<", "<=", "=="}:
                raise ValueError("gate comparator is unsupported")
            require_sha256(self.evidence_digest, field="evidence_digest")
            require_sha256(self.implementation_digest, field="implementation_digest")
            expected = _compare(self.observed_value, self.comparator, self.threshold)
            if self.passed != expected:
                raise ValueError("gate passed flag does not match metric comparison")

    @property
    def evidence_bound(self) -> bool:
        return self.evidence_digest is not None

    @classmethod
    def from_metric(
        cls,
        *,
        name: str,
        metric_name: str,
        observed_value: float,
        comparator: Comparator,
        threshold: float,
        evidence_digest: str,
        implementation_digest: str,
        mandatory: bool = True,
        detail: str | None = None,
    ) -> GateCheck:
        return cls(
            name=name,
            passed=_compare(observed_value, comparator, threshold),
            mandatory=mandatory,
            detail=detail,
            metric_name=metric_name,
            observed_value=observed_value,
            comparator=comparator,
            threshold=threshold,
            evidence_digest=evidence_digest,
            implementation_digest=implementation_digest,
        )


@dataclass(frozen=True, slots=True)
class GateDecision:
    """Gate result bound to evaluated dataset and selected policy identity."""

    dataset_id: str
    selected_policy_digest: str | None
    evaluation_digest: str
    passed: bool
    checks: tuple[GateCheck, ...]
    decided_at: datetime
    schema_version: str = "gate_decision_v3"

    def __post_init__(self) -> None:
        require_sha256(self.dataset_id, field="dataset_id")
        if self.selected_policy_digest is not None:
            require_sha256(self.selected_policy_digest, field="selected_policy_digest")
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
