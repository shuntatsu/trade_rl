"""Fail-closed admission evidence for the research-only causal alpha V3 lane."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Final, Mapping

from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.common import require_sha256
from trade_rl.learning.causal_alpha_teacher import CausalAlphaTeacherHoldoutMetric

_RECORD_SCHEMA: Final = "causal_alpha_v3_admission_record_v2"
_EVIDENCE_SCHEMA: Final = "causal_alpha_v3_admission_evidence_v2"
_EXPLAINED_REJECTIONS: Final = frozenset(
    {"below_minimum_notional", "zero_quantity_after_rounding"}
)


def _reason_counts(value: Any, *, field: str) -> tuple[tuple[str, int], ...]:
    try:
        resolved = tuple((str(reason), int(count)) for reason, count in value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must contain reason/count pairs") from error
    if (
        any(not reason or count < 0 for reason, count in resolved)
        or len({reason for reason, _ in resolved}) != len(resolved)
        or tuple(sorted(resolved)) != resolved
    ):
        raise ValueError(f"{field} contains invalid reason counts")
    return resolved


def _strict_payload(
    raw: Mapping[str, Any], *, fields: frozenset[str], schema: str, label: str
) -> dict[str, Any]:
    values = dict(raw)
    if set(values) != fields:
        missing = sorted(fields - set(values))
        unknown = sorted(set(values) - fields)
        raise ValueError(
            f"{label} fields mismatch; missing={missing}, unknown={unknown}"
        )
    if values.get("schema_version") != schema:
        raise ValueError(f"{label} schema is unsupported")
    return values


@dataclass(frozen=True, slots=True)
class CausalAlphaV3AdmissionRecordV2:
    run_manifest_digest: str
    freeze_digest: str
    selection_digest: str
    selected_candidate_digest: str
    symbol: str
    contract_digest: str
    gross_return: float
    net_return: float
    turnover_per_day: float
    total_execution_cost: float
    trade_count: int
    maximum_drawdown: float
    execution_rejection_reason_counts: tuple[tuple[str, int], ...] = ()
    risk_projection_reason_counts: tuple[tuple[str, int], ...] = ()
    hard_risk_violation: bool = False
    schema_version: str = _RECORD_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        for name in (
            "run_manifest_digest",
            "freeze_digest",
            "selection_digest",
            "selected_candidate_digest",
            "contract_digest",
        ):
            require_sha256(getattr(self, name), field=f"V3 admission {name}")
        if not self.symbol:
            raise ValueError("V3 admission symbol must be non-empty")
        for name in (
            "gross_return",
            "net_return",
            "turnover_per_day",
            "total_execution_cost",
            "maximum_drawdown",
        ):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"V3 admission {name} must be finite")
        if self.turnover_per_day < 0.0 or self.total_execution_cost < 0.0:
            raise ValueError("V3 admission turnover/cost must be non-negative")
        if (
            isinstance(self.trade_count, bool)
            or not isinstance(self.trade_count, int)
            or self.trade_count < 0
        ):
            raise ValueError("V3 admission trade_count must be non-negative")
        execution = _reason_counts(
            self.execution_rejection_reason_counts,
            field="V3 admission execution rejections",
        )
        risk = _reason_counts(
            self.risk_projection_reason_counts,
            field="V3 admission risk projections",
        )
        if not isinstance(self.hard_risk_violation, bool):
            raise ValueError("V3 admission hard_risk_violation must be boolean")
        if self.schema_version != _RECORD_SCHEMA:
            raise ValueError("unsupported V3 admission record schema")
        object.__setattr__(self, "execution_rejection_reason_counts", execution)
        object.__setattr__(self, "risk_projection_reason_counts", risk)
        expected = content_digest(self.to_payload(include_digest=False))
        if self.digest and self.digest != expected:
            raise ValueError("V3 admission record digest mismatch")
        object.__setattr__(self, "digest", expected)

    @property
    def unexplained_execution_rejection_count(self) -> int:
        return sum(
            count
            for reason, count in self.execution_rejection_reason_counts
            if reason not in _EXPLAINED_REJECTIONS
        )

    def to_holdout_metric(self) -> CausalAlphaTeacherHoldoutMetric:
        return CausalAlphaTeacherHoldoutMetric(
            symbol=self.symbol,
            gross_return=self.gross_return,
            net_return=self.net_return,
            turnover_per_day=self.turnover_per_day,
            total_execution_cost=self.total_execution_cost,
            trade_count=self.trade_count,
            maximum_drawdown=self.maximum_drawdown,
        )

    def to_payload(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "contract_digest": self.contract_digest,
            "execution_rejection_reason_counts": self.execution_rejection_reason_counts,
            "freeze_digest": self.freeze_digest,
            "gross_return": self.gross_return,
            "hard_risk_violation": self.hard_risk_violation,
            "maximum_drawdown": self.maximum_drawdown,
            "net_return": self.net_return,
            "risk_projection_reason_counts": self.risk_projection_reason_counts,
            "run_manifest_digest": self.run_manifest_digest,
            "schema_version": self.schema_version,
            "selected_candidate_digest": self.selected_candidate_digest,
            "selection_digest": self.selection_digest,
            "symbol": self.symbol,
            "total_execution_cost": self.total_execution_cost,
            "trade_count": self.trade_count,
            "turnover_per_day": self.turnover_per_day,
        }
        if include_digest:
            payload["artifact_digest"] = self.digest
        return payload

    @classmethod
    def from_payload(cls, raw: Mapping[str, Any]) -> CausalAlphaV3AdmissionRecordV2:
        fields = frozenset(
            {
                "artifact_digest",
                "contract_digest",
                "execution_rejection_reason_counts",
                "freeze_digest",
                "gross_return",
                "hard_risk_violation",
                "maximum_drawdown",
                "net_return",
                "risk_projection_reason_counts",
                "run_manifest_digest",
                "schema_version",
                "selected_candidate_digest",
                "selection_digest",
                "symbol",
                "total_execution_cost",
                "trade_count",
                "turnover_per_day",
            }
        )
        values = _strict_payload(
            raw, fields=fields, schema=_RECORD_SCHEMA, label="V3 admission record"
        )
        if not isinstance(values["hard_risk_violation"], bool):
            raise ValueError("V3 admission hard_risk_violation must be boolean")
        return cls(
            run_manifest_digest=str(values["run_manifest_digest"]),
            freeze_digest=str(values["freeze_digest"]),
            selection_digest=str(values["selection_digest"]),
            selected_candidate_digest=str(values["selected_candidate_digest"]),
            symbol=str(values["symbol"]),
            contract_digest=str(values["contract_digest"]),
            gross_return=float(values["gross_return"]),
            net_return=float(values["net_return"]),
            turnover_per_day=float(values["turnover_per_day"]),
            total_execution_cost=float(values["total_execution_cost"]),
            trade_count=int(values["trade_count"]),
            maximum_drawdown=float(values["maximum_drawdown"]),
            execution_rejection_reason_counts=_reason_counts(
                values["execution_rejection_reason_counts"],
                field="V3 admission execution rejections",
            ),
            risk_projection_reason_counts=_reason_counts(
                values["risk_projection_reason_counts"],
                field="V3 admission risk projections",
            ),
            hard_risk_violation=values["hard_risk_violation"],
            schema_version=str(values["schema_version"]),
            digest=str(values["artifact_digest"]),
        )


@dataclass(frozen=True, slots=True)
class CausalAlphaV3AdmissionEvidenceV2:
    records: tuple[CausalAlphaV3AdmissionRecordV2, ...]
    aggregate_gross_return: float
    aggregate_net_return: float
    negative_gross_symbol_count: int
    hard_risk_violation_count: int
    unexplained_execution_rejection_count: int
    passed: bool
    rejection_reasons: tuple[str, ...]
    promotion_eligible: bool = False
    schema_version: str = _EVIDENCE_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        records = tuple(self.records)
        if not records or len({item.symbol for item in records}) != len(records):
            raise ValueError("V3 admission evidence requires unique symbol records")
        if not math.isfinite(self.aggregate_gross_return) or not math.isfinite(
            self.aggregate_net_return
        ):
            raise ValueError("V3 admission aggregate returns must be finite")
        for name in (
            "negative_gross_symbol_count",
            "hard_risk_violation_count",
            "unexplained_execution_rejection_count",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"V3 admission evidence {name} is invalid")
        reasons = tuple(self.rejection_reasons)
        if self.passed == bool(reasons):
            raise ValueError("V3 admission pass state and rejection reasons disagree")
        if self.promotion_eligible:
            raise ValueError("V3 admission evidence cannot be promotion eligible")
        if self.schema_version != _EVIDENCE_SCHEMA:
            raise ValueError("unsupported V3 admission evidence schema")
        object.__setattr__(self, "records", records)
        object.__setattr__(self, "rejection_reasons", reasons)
        expected = content_digest(self.to_payload(include_digest=False))
        if self.digest and self.digest != expected:
            raise ValueError("V3 admission evidence digest mismatch")
        object.__setattr__(self, "digest", expected)

    @property
    def metrics(self) -> tuple[CausalAlphaTeacherHoldoutMetric, ...]:
        return tuple(record.to_holdout_metric() for record in self.records)

    def to_payload(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "aggregate_gross_return": self.aggregate_gross_return,
            "aggregate_net_return": self.aggregate_net_return,
            "hard_risk_violation_count": self.hard_risk_violation_count,
            "negative_gross_symbol_count": self.negative_gross_symbol_count,
            "passed": self.passed,
            "promotion_eligible": self.promotion_eligible,
            "record_digests": tuple(record.digest for record in self.records),
            "rejection_reasons": self.rejection_reasons,
            "schema_version": self.schema_version,
            "unexplained_execution_rejection_count": (
                self.unexplained_execution_rejection_count
            ),
        }
        if include_digest:
            payload["artifact_digest"] = self.digest
        return payload


def evaluate_causal_alpha_v3_admission_gate(
    records: tuple[CausalAlphaV3AdmissionRecordV2, ...],
) -> CausalAlphaV3AdmissionEvidenceV2:
    """Apply V3-specific net-economic and hard-risk holdout admission."""

    values = tuple(records)
    if not values or len({item.symbol for item in values}) != len(values):
        raise ValueError("V3 admission requires unique symbol records")
    aggregate_gross = float(sum(item.gross_return for item in values))
    aggregate_net = float(sum(item.net_return for item in values))
    negative_count = sum(item.gross_return < 0.0 for item in values)
    hard_risk_count = sum(item.hard_risk_violation for item in values)
    unexplained = sum(item.unexplained_execution_rejection_count for item in values)
    reasons: list[str] = []
    if aggregate_gross < 0.0:
        reasons.append("negative_aggregate_gross_return")
    if aggregate_net < 0.0:
        reasons.append("negative_aggregate_net_return")
    if negative_count > len(values) // 2:
        reasons.append("majority_negative_gross_holdouts")
    if hard_risk_count:
        reasons.append("hard_risk_violation")
    if unexplained:
        reasons.append("unexplained_execution_rejection")
    return CausalAlphaV3AdmissionEvidenceV2(
        records=values,
        aggregate_gross_return=aggregate_gross,
        aggregate_net_return=aggregate_net,
        negative_gross_symbol_count=negative_count,
        hard_risk_violation_count=hard_risk_count,
        unexplained_execution_rejection_count=unexplained,
        passed=not reasons,
        rejection_reasons=tuple(reasons),
    )


__all__ = [
    "CausalAlphaV3AdmissionEvidenceV2",
    "CausalAlphaV3AdmissionRecordV2",
    "evaluate_causal_alpha_v3_admission_gate",
]
