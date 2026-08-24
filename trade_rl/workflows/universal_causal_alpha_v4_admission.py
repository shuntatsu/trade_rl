"""Untouched teacher admission for the research-only Causal Alpha V4 lane."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.common import require_sha256
from trade_rl.workflows.universal_causal_alpha_selection import (
    causal_alpha_unexplained_execution_rejection_count,
)
from trade_rl.workflows.universal_causal_alpha_v4_replay import (
    CausalAlphaV4ReplayMetric,
)
from trade_rl.workflows.universal_causal_alpha_v4_selection import (
    CausalAlphaV4SelectionEvidence,
)
from trade_rl.workflows.universal_causal_alpha_v4_signal import (
    CausalAlphaV4SignalEvidence,
)

CAUSAL_ALPHA_V4_ADMISSION_SCHEMA: Final = "causal_alpha_v4_admission_evidence_v1"
_V4_MINIMUM_SYMBOL_NET_RETURN: Final = -0.05


@dataclass(frozen=True, slots=True)
class CausalAlphaV4AdmissionEvidence:
    records: tuple[CausalAlphaV4ReplayMetric, ...]
    signal_evidence_digest: str
    selection_digest: str
    run_manifest_digest: str
    v4_context_manifest_digest: str
    config_digest: str
    fit_digest: str
    fit_knowledge_cutoff: int
    holdout_start: int
    aggregate_gross_return: float
    aggregate_net_return: float
    negative_gross_symbol_count: int
    worst_symbol_net_return: float
    meaningful_execution_scope_count: int
    total_submitted_change_count: int
    total_executed_change_count: int
    total_closed_trade_count: int
    hard_risk_violation_count: int
    unexplained_execution_rejection_count: int
    passed: bool
    rejection_reasons: tuple[str, ...]
    promotion_eligible: bool = False
    schema_version: str = CAUSAL_ALPHA_V4_ADMISSION_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        records = tuple(self.records)
        if not records or len({record.symbol for record in records}) != len(records):
            raise ValueError("V4 admission requires unique symbol holdout records")
        for field_name in (
            "signal_evidence_digest",
            "selection_digest",
            "run_manifest_digest",
            "v4_context_manifest_digest",
            "config_digest",
            "fit_digest",
        ):
            require_sha256(
                getattr(self, field_name), field=f"V4 admission {field_name}"
            )
        if {record.run_manifest_digest for record in records} != {
            self.run_manifest_digest
        }:
            raise ValueError("V4 admission run identity drifted")
        if {record.v4_context_manifest_digest for record in records} != {
            self.v4_context_manifest_digest
        }:
            raise ValueError("V4 admission context identity drifted")
        if {record.config_digest for record in records} != {self.config_digest}:
            raise ValueError("V4 admission config identity drifted")
        if {record.fit_digest for record in records} != {self.fit_digest}:
            raise ValueError("V4 admission fit identity drifted")
        for field_name in ("fit_knowledge_cutoff", "holdout_start"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"V4 admission {field_name} must be non-negative")
        if self.fit_knowledge_cutoff != self.holdout_start:
            raise ValueError("V4 admission fit cutoff must equal holdout start")
        for field_name in (
            "aggregate_gross_return",
            "aggregate_net_return",
            "worst_symbol_net_return",
        ):
            if not math.isfinite(getattr(self, field_name)):
                raise ValueError(f"V4 admission {field_name} must be finite")
        for field_name in (
            "negative_gross_symbol_count",
            "meaningful_execution_scope_count",
            "total_submitted_change_count",
            "total_executed_change_count",
            "total_closed_trade_count",
            "hard_risk_violation_count",
            "unexplained_execution_rejection_count",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"V4 admission {field_name} must be non-negative")
        expected = _admission_summary(records)
        actual = (
            self.aggregate_gross_return,
            self.aggregate_net_return,
            self.negative_gross_symbol_count,
            self.worst_symbol_net_return,
            self.meaningful_execution_scope_count,
            self.total_submitted_change_count,
            self.total_executed_change_count,
            self.total_closed_trade_count,
            self.hard_risk_violation_count,
            self.unexplained_execution_rejection_count,
        )
        if actual != expected[:-1]:
            raise ValueError("V4 admission summary is inconsistent")
        reasons = tuple(self.rejection_reasons)
        if reasons != expected[-1] or self.passed != (not reasons):
            raise ValueError("V4 admission pass state is inconsistent")
        if self.promotion_eligible:
            raise ValueError("V4 admission evidence cannot be promotion eligible")
        if self.schema_version != CAUSAL_ALPHA_V4_ADMISSION_SCHEMA:
            raise ValueError("unsupported V4 admission evidence schema")
        object.__setattr__(self, "records", records)
        object.__setattr__(self, "rejection_reasons", reasons)
        digest = content_digest(self.to_payload(include_digest=False))
        if self.digest and self.digest != digest:
            raise ValueError("V4 admission evidence digest mismatch")
        object.__setattr__(self, "digest", digest)

    def to_payload(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "aggregate_gross_return": self.aggregate_gross_return,
            "aggregate_net_return": self.aggregate_net_return,
            "config_digest": self.config_digest,
            "fit_digest": self.fit_digest,
            "fit_knowledge_cutoff": self.fit_knowledge_cutoff,
            "hard_risk_violation_count": self.hard_risk_violation_count,
            "holdout_start": self.holdout_start,
            "meaningful_execution_scope_count": self.meaningful_execution_scope_count,
            "negative_gross_symbol_count": self.negative_gross_symbol_count,
            "passed": self.passed,
            "promotion_eligible": self.promotion_eligible,
            "record_digests": tuple(record.digest for record in self.records),
            "rejection_reasons": self.rejection_reasons,
            "run_manifest_digest": self.run_manifest_digest,
            "schema_version": self.schema_version,
            "selection_digest": self.selection_digest,
            "signal_evidence_digest": self.signal_evidence_digest,
            "total_closed_trade_count": self.total_closed_trade_count,
            "total_executed_change_count": self.total_executed_change_count,
            "total_submitted_change_count": self.total_submitted_change_count,
            "unexplained_execution_rejection_count": (
                self.unexplained_execution_rejection_count
            ),
            "v4_context_manifest_digest": self.v4_context_manifest_digest,
            "worst_symbol_net_return": self.worst_symbol_net_return,
        }
        if include_digest:
            payload["artifact_digest"] = self.digest
        return payload


def _admission_summary(
    records: tuple[CausalAlphaV4ReplayMetric, ...],
) -> tuple[float, float, int, float, int, int, int, int, int, int, tuple[str, ...]]:
    aggregate_gross = float(sum(record.gross_return for record in records))
    aggregate_net = float(sum(record.net_return for record in records))
    negative_gross = sum(record.gross_return < 0.0 for record in records)
    worst_net = float(min(record.net_return for record in records))
    meaningful = sum(record.has_meaningful_execution for record in records)
    submitted = sum(record.submitted_change_count for record in records)
    executed = sum(record.executed_change_count for record in records)
    closed = sum(record.closed_trade_count for record in records)
    hard_risk = sum(record.hard_risk_violation for record in records)
    unexplained = sum(
        causal_alpha_unexplained_execution_rejection_count(
            record.execution_rejection_reason_counts
        )
        for record in records
    )
    reasons: list[str] = []
    if aggregate_gross < 0.0:
        reasons.append("negative_aggregate_gross_return")
    if aggregate_net < 0.0:
        reasons.append("negative_aggregate_net_return")
    if negative_gross > len(records) // 2:
        reasons.append("majority_negative_gross_holdouts")
    if worst_net < _V4_MINIMUM_SYMBOL_NET_RETURN:
        reasons.append("symbol_net_return_below_floor")
    if hard_risk:
        reasons.append("hard_risk_violation")
    if unexplained:
        reasons.append("unexplained_execution_rejection")
    if meaningful == 0:
        reasons.append("no_meaningful_execution")
    return (
        aggregate_gross,
        aggregate_net,
        negative_gross,
        worst_net,
        meaningful,
        submitted,
        executed,
        closed,
        hard_risk,
        unexplained,
        tuple(reasons),
    )


def evaluate_causal_alpha_v4_admission(
    records: tuple[CausalAlphaV4ReplayMetric, ...],
    *,
    signal_evidence: CausalAlphaV4SignalEvidence,
    selection_evidence: CausalAlphaV4SelectionEvidence,
    fit_knowledge_cutoff: int,
    holdout_start: int,
) -> CausalAlphaV4AdmissionEvidence:
    """Open untouched holdout admission only after all upstream V4 gates pass."""

    if not isinstance(signal_evidence, CausalAlphaV4SignalEvidence):
        raise TypeError("V4 admission signal evidence is invalid")
    if not isinstance(selection_evidence, CausalAlphaV4SelectionEvidence):
        raise TypeError("V4 admission selection evidence is invalid")
    if not signal_evidence.passed:
        raise ValueError("V4 admission cannot bypass a failed signal gate")
    if not selection_evidence.passed:
        raise ValueError("V4 admission cannot bypass failed economic selection")
    if (
        isinstance(fit_knowledge_cutoff, bool)
        or not isinstance(fit_knowledge_cutoff, int)
        or isinstance(holdout_start, bool)
        or not isinstance(holdout_start, int)
        or fit_knowledge_cutoff < 0
        or holdout_start < 0
    ):
        raise ValueError("V4 admission cutoff/start must be non-negative integers")
    if fit_knowledge_cutoff != holdout_start:
        raise ValueError("V4 admission fit cutoff must equal holdout start")

    values = tuple(records)
    if not values or len({record.symbol for record in values}) != len(values):
        raise ValueError("V4 admission requires unique symbol holdout records")
    run_digests = {record.run_manifest_digest for record in values}
    context_digests = {record.v4_context_manifest_digest for record in values}
    config_digests = {record.config_digest for record in values}
    fit_digests = {record.fit_digest for record in values}
    if (
        len(run_digests) != 1
        or len(context_digests) != 1
        or len(config_digests) != 1
        or len(fit_digests) != 1
    ):
        raise ValueError("V4 admission holdout identity drifted")
    run_digest = next(iter(run_digests))
    context_digest = next(iter(context_digests))
    config_digest = next(iter(config_digests))
    if selection_evidence.run_manifest_digest != run_digest:
        raise ValueError("V4 admission selection/run identity drifted")
    if selection_evidence.v4_context_manifest_digest != context_digest:
        raise ValueError("V4 admission selection/context identity drifted")
    if selection_evidence.config_digest != config_digest:
        raise ValueError("V4 admission selection/config identity drifted")
    if (
        signal_evidence.fast_4h.run_manifest_digest != run_digest
        or signal_evidence.slow_fused.run_manifest_digest != run_digest
    ):
        raise ValueError("V4 admission signal/run identity drifted")

    summary = _admission_summary(values)
    return CausalAlphaV4AdmissionEvidence(
        records=values,
        signal_evidence_digest=signal_evidence.digest,
        selection_digest=selection_evidence.digest,
        run_manifest_digest=run_digest,
        v4_context_manifest_digest=context_digest,
        config_digest=config_digest,
        fit_digest=next(iter(fit_digests)),
        fit_knowledge_cutoff=fit_knowledge_cutoff,
        holdout_start=holdout_start,
        aggregate_gross_return=summary[0],
        aggregate_net_return=summary[1],
        negative_gross_symbol_count=summary[2],
        worst_symbol_net_return=summary[3],
        meaningful_execution_scope_count=summary[4],
        total_submitted_change_count=summary[5],
        total_executed_change_count=summary[6],
        total_closed_trade_count=summary[7],
        hard_risk_violation_count=summary[8],
        unexplained_execution_rejection_count=summary[9],
        passed=not summary[10],
        rejection_reasons=summary[10],
    )


__all__ = [
    "CAUSAL_ALPHA_V4_ADMISSION_SCHEMA",
    "CausalAlphaV4AdmissionEvidence",
    "evaluate_causal_alpha_v4_admission",
]
