"""Untouched holdout Admission for the research-only Causal Alpha V5 lane."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.common import require_sha256
from trade_rl.workflows.universal_causal_alpha_selection import (
    causal_alpha_unexplained_execution_rejection_count,
)
from trade_rl.workflows.universal_causal_alpha_v5_replay import (
    CausalAlphaV5ReplayMetric,
)
from trade_rl.workflows.universal_causal_alpha_v5_selection import (
    CausalAlphaV5SelectionEvidence,
)
from trade_rl.workflows.universal_causal_alpha_v5_signal import (
    CausalAlphaV5SignalEvidence,
)

CAUSAL_ALPHA_V5_ADMISSION_SCHEMA: Final = "causal_alpha_v5_admission_evidence_v1"
_MINIMUM_SYMBOL_NET_RETURN: Final = -0.05


def _summary(
    records: tuple[CausalAlphaV5ReplayMetric, ...],
) -> tuple[float, float, int, float, int, int, int, int, int, int, tuple[str, ...]]:
    gross = float(sum(item.gross_return for item in records))
    net = float(sum(item.net_return for item in records))
    negative_gross = sum(item.gross_return < 0.0 for item in records)
    worst_net = float(min(item.net_return for item in records))
    meaningful = sum(item.has_meaningful_execution for item in records)
    submitted = sum(item.submitted_change_count for item in records)
    executed = sum(item.executed_change_count for item in records)
    closed = sum(item.closed_trade_count for item in records)
    hard_risk = sum(item.hard_risk_violation for item in records)
    unexplained = sum(
        causal_alpha_unexplained_execution_rejection_count(
            item.execution_rejection_reason_counts
        )
        for item in records
    )
    reasons: list[str] = []
    if gross < 0.0:
        reasons.append("negative_aggregate_gross_return")
    if net < 0.0:
        reasons.append("negative_aggregate_net_return")
    if negative_gross > len(records) // 2:
        reasons.append("majority_negative_gross_holdouts")
    if worst_net < _MINIMUM_SYMBOL_NET_RETURN:
        reasons.append("symbol_net_return_below_floor")
    if hard_risk:
        reasons.append("hard_risk_violation")
    if unexplained:
        reasons.append("unexplained_execution_rejection")
    if meaningful == 0:
        reasons.append("no_meaningful_execution")
    return (
        gross,
        net,
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


@dataclass(frozen=True, slots=True)
class CausalAlphaV5AdmissionEvidence:
    records: tuple[CausalAlphaV5ReplayMetric, ...]
    signal_evidence_digest: str
    selection_digest: str
    calibration_fit_digest: str
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
    schema_version: str = CAUSAL_ALPHA_V5_ADMISSION_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        records = tuple(self.records)
        if not records or len({item.symbol for item in records}) != len(records):
            raise ValueError("V5 admission requires unique symbol holdout records")
        for name in (
            "signal_evidence_digest",
            "selection_digest",
            "calibration_fit_digest",
            "run_manifest_digest",
            "v4_context_manifest_digest",
            "config_digest",
            "fit_digest",
        ):
            require_sha256(getattr(self, name), field=f"V5 admission {name}")
        if self.fit_knowledge_cutoff != self.holdout_start or self.holdout_start < 0:
            raise ValueError("V5 admission fit cutoff must equal holdout start")
        if {item.run_manifest_digest for item in records} != {self.run_manifest_digest}:
            raise ValueError("V5 admission run identity drifted")
        if {item.v4_context_manifest_digest for item in records} != {
            self.v4_context_manifest_digest
        }:
            raise ValueError("V5 admission context identity drifted")
        if {item.config_digest for item in records} != {self.config_digest}:
            raise ValueError("V5 admission config identity drifted")
        if {item.fit_digest for item in records} != {self.fit_digest}:
            raise ValueError("V5 admission fit identity drifted")
        if {item.calibration_fit_digest for item in records} != {
            self.calibration_fit_digest
        }:
            raise ValueError("V5 admission calibration identity drifted")
        expected = _summary(records)
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
            raise ValueError("V5 admission summary is inconsistent")
        reasons = tuple(self.rejection_reasons)
        if (
            reasons != expected[-1]
            or self.passed != (not reasons)
            or self.promotion_eligible
        ):
            raise ValueError("V5 admission state is inconsistent")
        if self.schema_version != CAUSAL_ALPHA_V5_ADMISSION_SCHEMA:
            raise ValueError("unsupported V5 admission schema")
        object.__setattr__(self, "records", records)
        object.__setattr__(self, "rejection_reasons", reasons)
        expected_digest = content_digest(self.to_payload(include_digest=False))
        if self.digest and self.digest != expected_digest:
            raise ValueError("V5 admission digest mismatch")
        object.__setattr__(self, "digest", expected_digest)

    def to_payload(self, *, include_digest: bool = True) -> dict[str, object]:
        payload = {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name not in {"records", "digest"}
        }
        payload["record_digests"] = tuple(record.digest for record in self.records)
        if include_digest:
            payload["artifact_digest"] = self.digest
        return payload


def evaluate_causal_alpha_v5_admission(
    records: tuple[CausalAlphaV5ReplayMetric, ...],
    *,
    signal_evidence: CausalAlphaV5SignalEvidence,
    selection_evidence: CausalAlphaV5SelectionEvidence,
    fit_knowledge_cutoff: int,
    holdout_start: int,
) -> CausalAlphaV5AdmissionEvidence:
    """Apply unchanged V4 holdout economics only after V5 upstream gates pass."""

    if (
        fit_knowledge_cutoff != holdout_start
        or isinstance(fit_knowledge_cutoff, bool)
        or fit_knowledge_cutoff < 0
    ):
        raise ValueError("V5 admission fit cutoff must equal holdout start")
    if not isinstance(signal_evidence, CausalAlphaV5SignalEvidence):
        raise TypeError("V5 admission signal evidence is invalid")
    if not isinstance(selection_evidence, CausalAlphaV5SelectionEvidence):
        raise TypeError("V5 admission selection evidence is invalid")
    if not signal_evidence.passed:
        raise ValueError("V5 admission cannot bypass failed Signal")
    if not selection_evidence.passed:
        raise ValueError("V5 admission cannot bypass failed Selection")
    values = tuple(records)
    if not values or len({item.symbol for item in values}) != len(values):
        raise ValueError("V5 admission requires unique symbol holdout records")
    run_digests = {item.run_manifest_digest for item in values}
    context_digests = {item.v4_context_manifest_digest for item in values}
    config_digests = {item.config_digest for item in values}
    fit_digests = {item.fit_digest for item in values}
    calibration_digests = {item.calibration_fit_digest for item in values}
    if any(
        len(group) != 1
        for group in (
            run_digests,
            context_digests,
            config_digests,
            fit_digests,
            calibration_digests,
        )
    ):
        raise ValueError("V5 admission holdout identity drifted")
    run_digest = next(iter(run_digests))
    context_digest = next(iter(context_digests))
    config_digest = next(iter(config_digests))
    calibration_digest = next(iter(calibration_digests))
    if (
        selection_evidence.run_manifest_digest != run_digest
        or selection_evidence.v4_context_manifest_digest != context_digest
        or selection_evidence.config_digest != config_digest
    ):
        raise ValueError("V5 admission Selection identity drifted")
    if (
        signal_evidence.slow.run_manifest_digest != run_digest
        or signal_evidence.slow.calibration_config_digest != config_digest
    ):
        raise ValueError("V5 admission Signal identity drifted")
    if {metric.calibration_fit_digest for metric in signal_evidence.slow.metrics} != {
        calibration_digest
    }:
        raise ValueError("V5 admission calibration/Signal identity drifted")
    summary = _summary(values)
    return CausalAlphaV5AdmissionEvidence(
        records=values,
        signal_evidence_digest=signal_evidence.digest,
        selection_digest=selection_evidence.digest,
        calibration_fit_digest=calibration_digest,
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
    "CAUSAL_ALPHA_V5_ADMISSION_SCHEMA",
    "CausalAlphaV5AdmissionEvidence",
    "evaluate_causal_alpha_v5_admission",
]
