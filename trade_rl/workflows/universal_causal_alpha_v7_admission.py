"""Paired untouched holdout Admission for Causal Alpha V7 research."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.common import require_sha256
from trade_rl.learning.causal_alpha_v7 import CausalAlphaV7Candidate
from trade_rl.workflows.universal_causal_alpha_selection import (
    causal_alpha_unexplained_execution_rejection_count,
)
from trade_rl.workflows.universal_causal_alpha_v7_replay import (
    CausalAlphaV7ReplayMetric,
)
from trade_rl.workflows.universal_causal_alpha_v7_selection import (
    CausalAlphaV7SelectionEvidence,
)
from trade_rl.workflows.universal_causal_alpha_v7_signal import (
    CausalAlphaV7SignalEvidence,
)

_SUMMARY_SCHEMA: Final = "causal_alpha_v7_admission_summary_v1"
_ADMISSION_SCHEMA: Final = "causal_alpha_v7_admission_evidence_v1"
_EXPECTED_SYMBOL_COUNT: Final = 9
_MINIMUM_POSITIVE_SYMBOL_COUNT: Final = 6
_MINIMUM_SYMBOL_NET_RETURN: Final = -0.02
_EPSILON: Final = 1e-12


@dataclass(frozen=True, slots=True)
class CausalAlphaV7AdmissionSummary:
    candidate: CausalAlphaV7Candidate
    record_count: int
    aggregate_gross_return: float
    aggregate_gross_wealth: float
    aggregate_net_return: float
    aggregate_net_wealth: float
    positive_net_symbol_count: int
    worst_symbol_net_return: float
    turnover_p50: float
    turnover_p95: float
    total_execution_cost: float
    meaningful_execution_scope_count: int
    total_target_change_count: int
    total_submitted_change_count: int
    total_executed_change_count: int
    total_closed_trade_count: int
    total_sign_flip_count: int
    hard_risk_violation_count: int
    unexplained_execution_rejection_count: int
    schema_version: str = _SUMMARY_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        candidate = CausalAlphaV7Candidate(self.candidate)
        count_fields = (
            "record_count",
            "positive_net_symbol_count",
            "meaningful_execution_scope_count",
            "total_target_change_count",
            "total_submitted_change_count",
            "total_executed_change_count",
            "total_closed_trade_count",
            "total_sign_flip_count",
            "hard_risk_violation_count",
            "unexplained_execution_rejection_count",
        )
        if any(
            isinstance(getattr(self, name), bool)
            or not isinstance(getattr(self, name), int)
            or getattr(self, name) < 0
            for name in count_fields
        ):
            raise ValueError("V7 Admission summary counts are invalid")
        numeric = (
            self.aggregate_gross_return,
            self.aggregate_gross_wealth,
            self.aggregate_net_return,
            self.aggregate_net_wealth,
            self.worst_symbol_net_return,
            self.turnover_p50,
            self.turnover_p95,
            self.total_execution_cost,
        )
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError("V7 Admission summary must be finite")
        if min(self.aggregate_gross_wealth, self.aggregate_net_wealth) <= 0.0:
            raise ValueError("V7 Admission wealth must be positive")
        if min(self.turnover_p50, self.turnover_p95, self.total_execution_cost) < 0.0:
            raise ValueError("V7 Admission turnover/cost must be non-negative")
        if not math.isclose(
            math.log(self.aggregate_gross_wealth),
            self.aggregate_gross_return,
            abs_tol=1e-12,
        ) or not math.isclose(
            math.log(self.aggregate_net_wealth),
            self.aggregate_net_return,
            abs_tol=1e-12,
        ):
            raise ValueError("V7 Admission wealth is inconsistent")
        if self.schema_version != _SUMMARY_SCHEMA:
            raise ValueError("unsupported V7 Admission summary schema")
        object.__setattr__(self, "candidate", candidate)
        expected = content_digest(self.to_payload(include_digest=False))
        if self.digest and self.digest != expected:
            raise ValueError("V7 Admission summary digest mismatch")
        object.__setattr__(self, "digest", expected)

    def to_payload(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name not in {"candidate", "digest"}
        }
        payload["candidate"] = self.candidate.value
        if include_digest:
            payload["artifact_digest"] = self.digest
        return payload


def _summary(
    records: tuple[CausalAlphaV7ReplayMetric, ...],
) -> CausalAlphaV7AdmissionSummary:
    base = tuple(record.v6_metric for record in records)
    gross = float(sum(record.gross_return for record in base))
    net = float(sum(record.net_return for record in base))
    turnovers = np.asarray([record.turnover_per_day for record in base])
    return CausalAlphaV7AdmissionSummary(
        candidate=records[0].candidate,
        record_count=len(records),
        aggregate_gross_return=gross,
        aggregate_gross_wealth=math.exp(gross),
        aggregate_net_return=net,
        aggregate_net_wealth=math.exp(net),
        positive_net_symbol_count=sum(record.net_return > 0.0 for record in base),
        worst_symbol_net_return=min(record.net_return for record in base),
        turnover_p50=float(np.quantile(turnovers, 0.50)),
        turnover_p95=float(np.quantile(turnovers, 0.95)),
        total_execution_cost=float(sum(record.total_execution_cost for record in base)),
        meaningful_execution_scope_count=sum(record.has_meaningful_execution for record in base),
        total_target_change_count=sum(record.target_change_count for record in base),
        total_submitted_change_count=sum(record.submitted_change_count for record in base),
        total_executed_change_count=sum(record.executed_change_count for record in base),
        total_closed_trade_count=sum(record.closed_trade_count for record in base),
        total_sign_flip_count=sum(record.sign_flip_count for record in base),
        hard_risk_violation_count=sum(record.hard_risk_violation for record in base),
        unexplained_execution_rejection_count=sum(
            causal_alpha_unexplained_execution_rejection_count(
                record.execution_rejection_reason_counts
            )
            for record in base
        ),
    )


def _gate_reasons(
    selected: CausalAlphaV7AdmissionSummary,
    control: CausalAlphaV7AdmissionSummary,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if selected.aggregate_gross_return <= 0.0:
        reasons.append("aggregate_gross_return")
    if selected.aggregate_net_return <= 0.0:
        reasons.append("aggregate_net_return")
    if selected.positive_net_symbol_count < _MINIMUM_POSITIVE_SYMBOL_COUNT:
        reasons.append("positive_net_symbol_count")
    if selected.worst_symbol_net_return < _MINIMUM_SYMBOL_NET_RETURN:
        reasons.append("worst_symbol_net_return")
    if selected.hard_risk_violation_count:
        reasons.append("hard_risk_violation")
    if selected.unexplained_execution_rejection_count:
        reasons.append("unexplained_execution_rejection")
    if (
        selected.candidate is not CausalAlphaV7Candidate.V6_CONTROL
        and selected.aggregate_net_wealth < control.aggregate_net_wealth - _EPSILON
    ):
        reasons.append("selected_underperformed_control")
    return tuple(reasons)


@dataclass(frozen=True, slots=True)
class CausalAlphaV7AdmissionEvidence:
    selected_records: tuple[CausalAlphaV7ReplayMetric, ...]
    control_records: tuple[CausalAlphaV7ReplayMetric, ...]
    selected_summary: CausalAlphaV7AdmissionSummary
    control_summary: CausalAlphaV7AdmissionSummary
    signal_evidence_digest: str
    selection_digest: str
    selected_candidate: CausalAlphaV7Candidate
    selected_config_digest: str
    run_manifest_digest: str
    v4_context_manifest_digest: str
    fit_digest: str
    fit_knowledge_cutoff: int
    holdout_start: int
    paired_holdout_count: int
    passed: bool
    rejection_reasons: tuple[str, ...]
    promotion_eligible: bool = False
    schema_version: str = _ADMISSION_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        selected = tuple(self.selected_records)
        control = tuple(self.control_records)
        candidate = CausalAlphaV7Candidate(self.selected_candidate)
        for name in (
            "signal_evidence_digest",
            "selection_digest",
            "selected_config_digest",
            "run_manifest_digest",
            "v4_context_manifest_digest",
            "fit_digest",
        ):
            require_sha256(getattr(self, name), field=f"V7 Admission {name}")
        if self.fit_knowledge_cutoff != self.holdout_start or self.holdout_start < 0:
            raise ValueError("V7 Admission fit cutoff must equal holdout start")
        if self.paired_holdout_count != _EXPECTED_SYMBOL_COUNT:
            raise ValueError("V7 Admission paired holdout count is invalid")
        if self.selected_summary != _summary(selected) or self.control_summary != _summary(control):
            raise ValueError("V7 Admission summaries are inconsistent")
        if self.selected_summary.candidate is not candidate:
            raise ValueError("V7 Admission selected candidate drifted")
        reasons = tuple(self.rejection_reasons)
        if reasons != _gate_reasons(self.selected_summary, self.control_summary):
            raise ValueError("V7 Admission reasons are inconsistent")
        if self.passed != (not reasons) or self.promotion_eligible:
            raise ValueError("V7 Admission pass/promotion state is invalid")
        if self.schema_version != _ADMISSION_SCHEMA:
            raise ValueError("unsupported V7 Admission schema")
        object.__setattr__(self, "selected_records", selected)
        object.__setattr__(self, "control_records", control)
        object.__setattr__(self, "selected_candidate", candidate)
        object.__setattr__(self, "rejection_reasons", reasons)
        expected = content_digest(self.to_payload(include_digest=False))
        if self.digest and self.digest != expected:
            raise ValueError("V7 Admission digest mismatch")
        object.__setattr__(self, "digest", expected)

    @property
    def aggregate_net_return(self) -> float:
        return self.selected_summary.aggregate_net_return

    @property
    def aggregate_net_wealth(self) -> float:
        return self.selected_summary.aggregate_net_wealth

    @property
    def positive_net_symbol_count(self) -> int:
        return self.selected_summary.positive_net_symbol_count

    @property
    def worst_symbol_net_return(self) -> float:
        return self.selected_summary.worst_symbol_net_return

    def to_payload(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "control_record_digests": tuple(record.digest for record in self.control_records),
            "control_summary": self.control_summary.to_payload(),
            "fit_digest": self.fit_digest,
            "fit_knowledge_cutoff": self.fit_knowledge_cutoff,
            "holdout_start": self.holdout_start,
            "paired_holdout_count": self.paired_holdout_count,
            "passed": self.passed,
            "promotion_eligible": self.promotion_eligible,
            "rejection_reasons": self.rejection_reasons,
            "run_manifest_digest": self.run_manifest_digest,
            "schema_version": self.schema_version,
            "selected_candidate": self.selected_candidate.value,
            "selected_config_digest": self.selected_config_digest,
            "selected_record_digests": tuple(record.digest for record in self.selected_records),
            "selected_summary": self.selected_summary.to_payload(),
            "selection_digest": self.selection_digest,
            "signal_evidence_digest": self.signal_evidence_digest,
            "v4_context_manifest_digest": self.v4_context_manifest_digest,
        }
        if include_digest:
            payload["artifact_digest"] = self.digest
        return payload


def _validate_records(
    records: tuple[CausalAlphaV7ReplayMetric, ...],
    *,
    candidate: CausalAlphaV7Candidate,
    field: str,
) -> None:
    if len(records) != _EXPECTED_SYMBOL_COUNT or len(
        {record.v6_metric.symbol for record in records}
    ) != _EXPECTED_SYMBOL_COUNT:
        raise ValueError(f"V7 Admission {field} requires nine unique symbols")
    if any(record.candidate is not candidate for record in records):
        raise ValueError(f"V7 Admission {field} candidate drifted")
    for name in ("v7_config_digest", "calibration_fit_digest"):
        if len({getattr(record, name) for record in records}) != 1:
            raise ValueError(f"V7 Admission {field} {name} drifted")
    for name in ("run_manifest_digest", "v4_context_manifest_digest", "fit_digest"):
        if len({getattr(record.v6_metric, name) for record in records}) != 1:
            raise ValueError(f"V7 Admission {field} {name} drifted")


def evaluate_causal_alpha_v7_admission(
    selected_records: tuple[CausalAlphaV7ReplayMetric, ...],
    control_records: tuple[CausalAlphaV7ReplayMetric, ...],
    *,
    signal_evidence: CausalAlphaV7SignalEvidence,
    selection_evidence: CausalAlphaV7SelectionEvidence,
    fit_knowledge_cutoff: int,
    holdout_start: int,
) -> CausalAlphaV7AdmissionEvidence:
    """Open the paired nine-symbol holdout exactly once after Selection."""

    if fit_knowledge_cutoff != holdout_start or isinstance(fit_knowledge_cutoff, bool):
        raise ValueError("V7 Admission fit cutoff must equal holdout start")
    if not isinstance(signal_evidence, CausalAlphaV7SignalEvidence) or not signal_evidence.passed:
        raise ValueError("V7 Admission cannot bypass failed Signal")
    if (
        not isinstance(selection_evidence, CausalAlphaV7SelectionEvidence)
        or not selection_evidence.passed
        or selection_evidence.selected_candidate is None
        or selection_evidence.selected_config_digest is None
    ):
        raise ValueError("V7 Admission cannot bypass failed Selection")
    selected = tuple(selected_records)
    control = tuple(control_records)
    candidate = selection_evidence.selected_candidate
    _validate_records(selected, candidate=candidate, field="selected holdout")
    _validate_records(
        control,
        candidate=CausalAlphaV7Candidate.V6_CONTROL,
        field="control holdout",
    )
    if tuple(sorted(record.paired_identity for record in selected)) != tuple(
        sorted(record.paired_identity for record in control)
    ):
        raise ValueError("V7 Admission holdout records are not paired")
    all_records = (*selected, *control)
    run_digest = selected[0].v6_metric.run_manifest_digest
    context_digest = selected[0].v6_metric.v4_context_manifest_digest
    config_digest = selected[0].v7_config_digest
    fit_digest = selected[0].v6_metric.fit_digest
    if any(record.v7_config_digest != config_digest for record in all_records):
        raise ValueError("V7 Admission paired config drifted")
    if selection_evidence.selected_config_digest != config_digest:
        raise ValueError("V7 Admission Selection config drifted")
    selected_upstream = selection_evidence.candidates[
        tuple(CausalAlphaV7Candidate).index(candidate)
    ]
    if (
        selected_upstream.run_manifest_digest != run_digest
        or selected_upstream.v4_context_manifest_digest != context_digest
        or signal_evidence.candidates[0].metrics[0].run_manifest_digest != run_digest
        or signal_evidence.candidates[0].metrics[0].v7_config_digest != config_digest
    ):
        raise ValueError("V7 Admission upstream identity drifted")
    selected_summary = _summary(selected)
    control_summary = _summary(control)
    reasons = _gate_reasons(selected_summary, control_summary)
    return CausalAlphaV7AdmissionEvidence(
        selected_records=selected,
        control_records=control,
        selected_summary=selected_summary,
        control_summary=control_summary,
        signal_evidence_digest=signal_evidence.digest,
        selection_digest=selection_evidence.digest,
        selected_candidate=candidate,
        selected_config_digest=config_digest,
        run_manifest_digest=run_digest,
        v4_context_manifest_digest=context_digest,
        fit_digest=fit_digest,
        fit_knowledge_cutoff=fit_knowledge_cutoff,
        holdout_start=holdout_start,
        paired_holdout_count=len(selected),
        passed=not reasons,
        rejection_reasons=reasons,
    )


__all__ = [
    "CausalAlphaV7AdmissionEvidence",
    "CausalAlphaV7AdmissionSummary",
    "evaluate_causal_alpha_v7_admission",
]
