"""Paired untouched holdout Admission for Causal Alpha V6 research."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.common import require_sha256
from trade_rl.learning.causal_alpha_v6 import CausalAlphaV6Candidate
from trade_rl.workflows.universal_causal_alpha_selection import (
    causal_alpha_unexplained_execution_rejection_count,
)
from trade_rl.workflows.universal_causal_alpha_v6_replay import (
    CausalAlphaV6ReplayMetric,
)
from trade_rl.workflows.universal_causal_alpha_v6_selection import (
    CausalAlphaV6SelectionEvidence,
)
from trade_rl.workflows.universal_causal_alpha_v6_signal import (
    CausalAlphaV6SignalEvidence,
)

CAUSAL_ALPHA_V6_ADMISSION_SUMMARY_SCHEMA: Final = (
    "causal_alpha_v6_admission_summary_v1"
)
CAUSAL_ALPHA_V6_ADMISSION_SCHEMA: Final = "causal_alpha_v6_admission_evidence_v1"
_EXPECTED_SYMBOL_COUNT: Final = 9
_MINIMUM_POSITIVE_SYMBOL_COUNT: Final = 6
_MINIMUM_SYMBOL_NET_RETURN: Final = -0.02
_EPSILON: Final = 1e-12


@dataclass(frozen=True, slots=True)
class CausalAlphaV6AdmissionSummary:
    candidate: CausalAlphaV6Candidate
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
    schema_version: str = CAUSAL_ALPHA_V6_ADMISSION_SUMMARY_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        candidate = CausalAlphaV6Candidate(self.candidate)
        for name in (
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
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"V6 Admission {name} is invalid")
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
            raise ValueError("V6 Admission summary must be finite")
        if min(
            self.aggregate_gross_wealth,
            self.aggregate_net_wealth,
        ) <= 0.0 or min(
            self.turnover_p50,
            self.turnover_p95,
            self.total_execution_cost,
        ) < 0.0:
            raise ValueError("V6 Admission wealth/cost evidence is invalid")
        if not math.isclose(
            self.aggregate_gross_wealth,
            math.exp(self.aggregate_gross_return),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ) or not math.isclose(
            self.aggregate_net_wealth,
            math.exp(self.aggregate_net_return),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("V6 Admission wealth is inconsistent")
        if self.schema_version != CAUSAL_ALPHA_V6_ADMISSION_SUMMARY_SCHEMA:
            raise ValueError("unsupported V6 Admission summary schema")
        object.__setattr__(self, "candidate", candidate)
        expected = content_digest(self.to_payload(include_digest=False))
        if self.digest and self.digest != expected:
            raise ValueError("V6 Admission summary digest mismatch")
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
    records: tuple[CausalAlphaV6ReplayMetric, ...],
) -> CausalAlphaV6AdmissionSummary:
    gross = float(sum(record.gross_return for record in records))
    net = float(sum(record.net_return for record in records))
    turnovers = np.asarray([record.turnover_per_day for record in records])
    return CausalAlphaV6AdmissionSummary(
        candidate=records[0].candidate,
        record_count=len(records),
        aggregate_gross_return=gross,
        aggregate_gross_wealth=math.exp(gross),
        aggregate_net_return=net,
        aggregate_net_wealth=math.exp(net),
        positive_net_symbol_count=sum(record.net_return > 0.0 for record in records),
        worst_symbol_net_return=min(record.net_return for record in records),
        turnover_p50=float(np.quantile(turnovers, 0.50)),
        turnover_p95=float(np.quantile(turnovers, 0.95)),
        total_execution_cost=float(
            sum(record.total_execution_cost for record in records)
        ),
        meaningful_execution_scope_count=sum(
            record.has_meaningful_execution for record in records
        ),
        total_target_change_count=sum(record.target_change_count for record in records),
        total_submitted_change_count=sum(
            record.submitted_change_count for record in records
        ),
        total_executed_change_count=sum(
            record.executed_change_count for record in records
        ),
        total_closed_trade_count=sum(record.closed_trade_count for record in records),
        total_sign_flip_count=sum(record.sign_flip_count for record in records),
        hard_risk_violation_count=sum(
            record.hard_risk_violation for record in records
        ),
        unexplained_execution_rejection_count=sum(
            causal_alpha_unexplained_execution_rejection_count(
                record.execution_rejection_reason_counts
            )
            for record in records
        ),
    )


@dataclass(frozen=True, slots=True)
class CausalAlphaV6AdmissionEvidence:
    selected_records: tuple[CausalAlphaV6ReplayMetric, ...]
    fast_only_records: tuple[CausalAlphaV6ReplayMetric, ...]
    selected_summary: CausalAlphaV6AdmissionSummary
    fast_only_summary: CausalAlphaV6AdmissionSummary
    signal_evidence_digest: str
    selection_digest: str
    selected_candidate: CausalAlphaV6Candidate
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
    schema_version: str = CAUSAL_ALPHA_V6_ADMISSION_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        selected_records = tuple(self.selected_records)
        fast_records = tuple(self.fast_only_records)
        selected = CausalAlphaV6Candidate(self.selected_candidate)
        for name in (
            "signal_evidence_digest",
            "selection_digest",
            "selected_config_digest",
            "run_manifest_digest",
            "v4_context_manifest_digest",
            "fit_digest",
        ):
            require_sha256(getattr(self, name), field=f"V6 Admission {name}")
        if self.fit_knowledge_cutoff != self.holdout_start or self.holdout_start < 0:
            raise ValueError("V6 Admission fit cutoff must equal holdout start")
        if self.paired_holdout_count != _EXPECTED_SYMBOL_COUNT:
            raise ValueError("V6 Admission paired holdout count is invalid")
        if self.selected_summary != _summary(selected_records) or self.fast_only_summary != _summary(fast_records):
            raise ValueError("V6 Admission summaries are inconsistent")
        if self.selected_summary.candidate is not selected:
            raise ValueError("V6 Admission selected summary candidate drifted")
        reasons = tuple(self.rejection_reasons)
        expected_reasons = _gate_reasons(
            self.selected_summary,
            self.fast_only_summary,
        )
        if reasons != expected_reasons or self.passed != (not reasons):
            raise ValueError("V6 Admission state is inconsistent")
        if self.promotion_eligible:
            raise ValueError("V6 research Admission cannot be promotion eligible")
        if self.schema_version != CAUSAL_ALPHA_V6_ADMISSION_SCHEMA:
            raise ValueError("unsupported V6 Admission schema")
        object.__setattr__(self, "selected_records", selected_records)
        object.__setattr__(self, "fast_only_records", fast_records)
        object.__setattr__(self, "selected_candidate", selected)
        object.__setattr__(self, "rejection_reasons", reasons)
        expected = content_digest(self.to_payload(include_digest=False))
        if self.digest and self.digest != expected:
            raise ValueError("V6 Admission digest mismatch")
        object.__setattr__(self, "digest", expected)

    @property
    def aggregate_gross_return(self) -> float:
        return self.selected_summary.aggregate_gross_return

    @property
    def aggregate_gross_wealth(self) -> float:
        return self.selected_summary.aggregate_gross_wealth

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
            "fast_only_record_digests": tuple(
                record.digest for record in self.fast_only_records
            ),
            "fast_only_summary": self.fast_only_summary.to_payload(),
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
            "selected_record_digests": tuple(
                record.digest for record in self.selected_records
            ),
            "selected_summary": self.selected_summary.to_payload(),
            "selection_digest": self.selection_digest,
            "signal_evidence_digest": self.signal_evidence_digest,
            "v4_context_manifest_digest": self.v4_context_manifest_digest,
        }
        if include_digest:
            payload["artifact_digest"] = self.digest
        return payload


def _gate_reasons(
    selected: CausalAlphaV6AdmissionSummary,
    baseline: CausalAlphaV6AdmissionSummary,
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
        selected.candidate is CausalAlphaV6Candidate.FAST_SLOW_RETENTION
        and selected.aggregate_net_wealth
        < baseline.aggregate_net_wealth - _EPSILON
    ):
        reasons.append("retention_underperformed_fast_only")
    return tuple(reasons)


def _validate_records(
    records: tuple[CausalAlphaV6ReplayMetric, ...],
    *,
    candidate: CausalAlphaV6Candidate,
    field: str,
) -> None:
    if len(records) != _EXPECTED_SYMBOL_COUNT or len(
        {record.symbol for record in records}
    ) != _EXPECTED_SYMBOL_COUNT:
        raise ValueError(f"V6 Admission {field} requires nine unique symbols")
    if any(record.candidate is not candidate for record in records):
        raise ValueError(f"V6 Admission {field} candidate drifted")
    for name in (
        "run_manifest_digest",
        "v4_context_manifest_digest",
        "config_digest",
        "fit_digest",
    ):
        if len({getattr(record, name) for record in records}) != 1:
            raise ValueError(f"V6 Admission {field} {name} drifted")


def _paired(
    selected: tuple[CausalAlphaV6ReplayMetric, ...],
    baseline: tuple[CausalAlphaV6ReplayMetric, ...],
) -> bool:
    selected_ids = tuple(sorted(record.paired_identity for record in selected))
    baseline_ids = tuple(sorted(record.paired_identity for record in baseline))
    return selected_ids == baseline_ids


def evaluate_causal_alpha_v6_admission(
    selected_records: tuple[CausalAlphaV6ReplayMetric, ...],
    baseline_records: tuple[CausalAlphaV6ReplayMetric, ...],
    *,
    signal_evidence: CausalAlphaV6SignalEvidence,
    selection_evidence: CausalAlphaV6SelectionEvidence,
    fit_knowledge_cutoff: int,
    holdout_start: int,
) -> CausalAlphaV6AdmissionEvidence:
    """Open and evaluate the paired nine-symbol holdout exactly once."""

    if (
        isinstance(fit_knowledge_cutoff, bool)
        or fit_knowledge_cutoff < 0
        or fit_knowledge_cutoff != holdout_start
    ):
        raise ValueError("V6 Admission fit cutoff must equal holdout start")
    if not isinstance(signal_evidence, CausalAlphaV6SignalEvidence):
        raise TypeError("V6 Admission Signal evidence is invalid")
    if not isinstance(selection_evidence, CausalAlphaV6SelectionEvidence):
        raise TypeError("V6 Admission Selection evidence is invalid")
    if not signal_evidence.passed:
        raise ValueError("V6 Admission cannot bypass failed Signal")
    if not selection_evidence.passed or selection_evidence.selected_candidate is None:
        raise ValueError("V6 Admission cannot bypass failed Selection")
    selected = tuple(selected_records)
    baseline = tuple(baseline_records)
    selected_candidate = CausalAlphaV6Candidate(selection_evidence.selected_candidate)
    _validate_records(selected, candidate=selected_candidate, field="selected holdout")
    _validate_records(
        baseline,
        candidate=CausalAlphaV6Candidate.FAST_ONLY,
        field="fast-only holdout",
    )
    if not _paired(selected, baseline):
        raise ValueError("V6 Admission holdout records are not paired")
    all_records = (*selected, *baseline)
    run_digest = selected[0].run_manifest_digest
    context_digest = selected[0].v4_context_manifest_digest
    config_digest = selected[0].config_digest
    fit_digest = selected[0].fit_digest
    if any(
        len({getattr(record, name) for record in all_records}) != 1
        for name in (
            "run_manifest_digest",
            "v4_context_manifest_digest",
            "config_digest",
            "fit_digest",
        )
    ):
        raise ValueError("V6 Admission paired holdout identity drifted")
    selected_upstream = (
        selection_evidence.fast_only
        if selected_candidate is CausalAlphaV6Candidate.FAST_ONLY
        else selection_evidence.fast_slow_retention
    )
    if (
        selection_evidence.selected_config_digest != config_digest
        or selected_upstream.run_manifest_digest != run_digest
        or selected_upstream.v4_context_manifest_digest != context_digest
        or selected_upstream.config_digest != config_digest
    ):
        raise ValueError("V6 Admission Selection identity drifted")
    signal_metric = signal_evidence.fast_only.metrics[0]
    if (
        signal_metric.run_manifest_digest != run_digest
        or signal_metric.config_digest != config_digest
    ):
        raise ValueError("V6 Admission Signal identity drifted")
    selected_summary = _summary(selected)
    baseline_summary = _summary(baseline)
    reasons = _gate_reasons(selected_summary, baseline_summary)
    return CausalAlphaV6AdmissionEvidence(
        selected_records=selected,
        fast_only_records=baseline,
        selected_summary=selected_summary,
        fast_only_summary=baseline_summary,
        signal_evidence_digest=signal_evidence.digest,
        selection_digest=selection_evidence.digest,
        selected_candidate=selected_candidate,
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
    "CAUSAL_ALPHA_V6_ADMISSION_SCHEMA",
    "CAUSAL_ALPHA_V6_ADMISSION_SUMMARY_SCHEMA",
    "CausalAlphaV6AdmissionEvidence",
    "CausalAlphaV6AdmissionSummary",
    "evaluate_causal_alpha_v6_admission",
]
