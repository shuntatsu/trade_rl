"""Fail-closed economic selection for the single authored Causal Alpha V4 hypothesis."""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import fmean
from typing import Final

from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.common import require_sha256
from trade_rl.workflows.universal_causal_alpha_selection import (
    causal_alpha_unexplained_execution_rejection_count,
)
from trade_rl.workflows.universal_causal_alpha_v4_replay import CausalAlphaV4ReplayMetric

CAUSAL_ALPHA_V4_SELECTION_SCHEMA: Final = "causal_alpha_v4_selection_evidence_v1"
_V4_MINIMUM_WORST_NET_RETURN: Final = -0.05
_V4_MINIMUM_POSITIVE_GROSS_FRACTION: Final = 0.5


@dataclass(frozen=True, slots=True)
class CausalAlphaV4SelectionEvidence:
    metrics: tuple[CausalAlphaV4ReplayMetric, ...]
    run_manifest_digest: str
    v4_context_manifest_digest: str
    config_digest: str
    mean_gross_return: float
    mean_net_return: float
    worst_symbol_episode_net_return: float
    positive_gross_episode_fraction: float
    mean_turnover_per_day: float
    total_execution_cost: float
    meaningful_execution_scope_count: int
    total_submitted_change_count: int
    total_executed_change_count: int
    total_closed_trade_count: int
    hard_risk_violation_count: int
    unexplained_execution_rejection_count: int
    passed: bool
    rejection_reasons: tuple[str, ...]
    promotion_eligible: bool = False
    schema_version: str = CAUSAL_ALPHA_V4_SELECTION_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        metrics = tuple(self.metrics)
        if not metrics:
            raise ValueError("V4 selection requires replay metrics")
        identities = tuple((metric.symbol, metric.episode_index) for metric in metrics)
        if len(set(identities)) != len(identities):
            raise ValueError("V4 selection replay scope is duplicated")
        for field_name in (
            "run_manifest_digest",
            "v4_context_manifest_digest",
            "config_digest",
        ):
            require_sha256(getattr(self, field_name), field=f"V4 selection {field_name}")
        if {metric.run_manifest_digest for metric in metrics} != {
            self.run_manifest_digest
        }:
            raise ValueError("V4 selection run manifest identity drifted")
        if {metric.v4_context_manifest_digest for metric in metrics} != {
            self.v4_context_manifest_digest
        }:
            raise ValueError("V4 selection context manifest identity drifted")
        if {metric.config_digest for metric in metrics} != {self.config_digest}:
            raise ValueError("V4 selection config identity drifted")
        for field_name in (
            "mean_gross_return",
            "mean_net_return",
            "worst_symbol_episode_net_return",
            "positive_gross_episode_fraction",
            "mean_turnover_per_day",
            "total_execution_cost",
        ):
            value = getattr(self, field_name)
            if not math.isfinite(value):
                raise ValueError(f"V4 selection {field_name} must be finite")
        if not 0.0 <= self.positive_gross_episode_fraction <= 1.0:
            raise ValueError("V4 selection positive gross fraction is invalid")
        if self.mean_turnover_per_day < 0.0 or self.total_execution_cost < 0.0:
            raise ValueError("V4 selection turnover/cost must be non-negative")
        for field_name in (
            "meaningful_execution_scope_count",
            "total_submitted_change_count",
            "total_executed_change_count",
            "total_closed_trade_count",
            "hard_risk_violation_count",
            "unexplained_execution_rejection_count",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"V4 selection {field_name} must be non-negative")
        expected = _selection_summary(metrics)
        actual = (
            self.mean_gross_return,
            self.mean_net_return,
            self.worst_symbol_episode_net_return,
            self.positive_gross_episode_fraction,
            self.mean_turnover_per_day,
            self.total_execution_cost,
            self.meaningful_execution_scope_count,
            self.total_submitted_change_count,
            self.total_executed_change_count,
            self.total_closed_trade_count,
            self.hard_risk_violation_count,
            self.unexplained_execution_rejection_count,
        )
        if actual != expected[:-1]:
            raise ValueError("V4 selection summary is inconsistent")
        reasons = tuple(self.rejection_reasons)
        if reasons != expected[-1] or self.passed != (not reasons):
            raise ValueError("V4 selection pass state is inconsistent")
        if self.promotion_eligible:
            raise ValueError("V4 selection evidence cannot be promotion eligible")
        if self.schema_version != CAUSAL_ALPHA_V4_SELECTION_SCHEMA:
            raise ValueError("unsupported V4 selection evidence schema")
        object.__setattr__(self, "metrics", metrics)
        object.__setattr__(self, "rejection_reasons", reasons)
        digest = content_digest(self.to_payload(include_digest=False))
        if self.digest and self.digest != digest:
            raise ValueError("V4 selection evidence digest mismatch")
        object.__setattr__(self, "digest", digest)

    @property
    def selected_config_digest(self) -> str | None:
        return self.config_digest if self.passed else None

    def to_payload(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "config_digest": self.config_digest,
            "hard_risk_violation_count": self.hard_risk_violation_count,
            "mean_gross_return": self.mean_gross_return,
            "mean_net_return": self.mean_net_return,
            "mean_turnover_per_day": self.mean_turnover_per_day,
            "meaningful_execution_scope_count": self.meaningful_execution_scope_count,
            "passed": self.passed,
            "positive_gross_episode_fraction": self.positive_gross_episode_fraction,
            "promotion_eligible": self.promotion_eligible,
            "rejection_reasons": self.rejection_reasons,
            "replay_metric_digests": tuple(metric.digest for metric in self.metrics),
            "run_manifest_digest": self.run_manifest_digest,
            "schema_version": self.schema_version,
            "selected_config_digest": self.selected_config_digest,
            "total_closed_trade_count": self.total_closed_trade_count,
            "total_executed_change_count": self.total_executed_change_count,
            "total_execution_cost": self.total_execution_cost,
            "total_submitted_change_count": self.total_submitted_change_count,
            "unexplained_execution_rejection_count": (
                self.unexplained_execution_rejection_count
            ),
            "v4_context_manifest_digest": self.v4_context_manifest_digest,
            "worst_symbol_episode_net_return": self.worst_symbol_episode_net_return,
        }
        if include_digest:
            payload["artifact_digest"] = self.digest
        return payload


def _selection_summary(
    metrics: tuple[CausalAlphaV4ReplayMetric, ...],
) -> tuple[
    float,
    float,
    float,
    float,
    float,
    float,
    int,
    int,
    int,
    int,
    int,
    int,
    tuple[str, ...],
]:
    mean_gross = float(fmean(metric.gross_return for metric in metrics))
    mean_net = float(fmean(metric.net_return for metric in metrics))
    worst_net = float(min(metric.net_return for metric in metrics))
    positive_fraction = sum(metric.gross_return > 0.0 for metric in metrics) / float(
        len(metrics)
    )
    mean_turnover = float(fmean(metric.turnover_per_day for metric in metrics))
    total_cost = float(sum(metric.total_execution_cost for metric in metrics))
    meaningful = sum(metric.has_meaningful_execution for metric in metrics)
    submitted = sum(metric.submitted_change_count for metric in metrics)
    executed = sum(metric.executed_change_count for metric in metrics)
    closed = sum(metric.closed_trade_count for metric in metrics)
    hard_risk = sum(metric.hard_risk_violation for metric in metrics)
    unexplained = sum(
        causal_alpha_unexplained_execution_rejection_count(
            metric.execution_rejection_reason_counts
        )
        for metric in metrics
    )
    reasons: list[str] = []
    if mean_gross < 0.0:
        reasons.append("mean_gross_return_below_minimum")
    if mean_net < 0.0:
        reasons.append("mean_net_return_below_minimum")
    if worst_net < _V4_MINIMUM_WORST_NET_RETURN:
        reasons.append("worst_symbol_episode_net_return_below_floor")
    if positive_fraction < _V4_MINIMUM_POSITIVE_GROSS_FRACTION:
        reasons.append("positive_gross_episode_fraction_below_minimum")
    if unexplained:
        reasons.append("unexplained_execution_rejection")
    if hard_risk:
        reasons.append("hard_risk_violation")
    if meaningful == 0:
        reasons.append("no_meaningful_execution")
    return (
        mean_gross,
        mean_net,
        worst_net,
        positive_fraction,
        mean_turnover,
        total_cost,
        meaningful,
        submitted,
        executed,
        closed,
        hard_risk,
        unexplained,
        tuple(reasons),
    )


def evaluate_causal_alpha_v4_selection(
    metrics: tuple[CausalAlphaV4ReplayMetric, ...],
) -> CausalAlphaV4SelectionEvidence:
    """Evaluate the only authored V4 candidate without introducing a ranking grid."""

    values = tuple(metrics)
    if not values:
        raise ValueError("V4 selection requires replay metrics")
    identities = {(metric.symbol, metric.episode_index) for metric in values}
    if len(identities) != len(values):
        raise ValueError("V4 selection replay scope is duplicated")
    run_digests = {metric.run_manifest_digest for metric in values}
    context_digests = {metric.v4_context_manifest_digest for metric in values}
    config_digests = {metric.config_digest for metric in values}
    if len(run_digests) != 1 or len(context_digests) != 1 or len(config_digests) != 1:
        raise ValueError("V4 selection replay identity drifted")
    summary = _selection_summary(values)
    return CausalAlphaV4SelectionEvidence(
        metrics=values,
        run_manifest_digest=next(iter(run_digests)),
        v4_context_manifest_digest=next(iter(context_digests)),
        config_digest=next(iter(config_digests)),
        mean_gross_return=summary[0],
        mean_net_return=summary[1],
        worst_symbol_episode_net_return=summary[2],
        positive_gross_episode_fraction=summary[3],
        mean_turnover_per_day=summary[4],
        total_execution_cost=summary[5],
        meaningful_execution_scope_count=summary[6],
        total_submitted_change_count=summary[7],
        total_executed_change_count=summary[8],
        total_closed_trade_count=summary[9],
        hard_risk_violation_count=summary[10],
        unexplained_execution_rejection_count=summary[11],
        passed=not summary[12],
        rejection_reasons=summary[12],
    )


__all__ = [
    "CAUSAL_ALPHA_V4_SELECTION_SCHEMA",
    "CausalAlphaV4SelectionEvidence",
    "evaluate_causal_alpha_v4_selection",
]
