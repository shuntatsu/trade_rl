"""Paired symbol-balanced after-cost Selection for Causal Alpha V6."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from statistics import fmean, median
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

CAUSAL_ALPHA_V6_SYMBOL_SELECTION_SCHEMA: Final = (
    "causal_alpha_v6_symbol_selection_summary_v1"
)
CAUSAL_ALPHA_V6_CANDIDATE_SELECTION_SCHEMA: Final = (
    "causal_alpha_v6_candidate_selection_evidence_v1"
)
CAUSAL_ALPHA_V6_SELECTION_SCHEMA: Final = "causal_alpha_v6_selection_evidence_v1"
_TURNOVER_P95_LIMIT: Final = 1.0
_EPSILON: Final = 1e-12


def _finite_exp(value: float, *, field: str) -> float:
    try:
        result = math.exp(value)
    except OverflowError as error:
        raise ValueError(f"V6 Selection {field} overflowed") from error
    if not math.isfinite(result):
        raise ValueError(f"V6 Selection {field} is non-finite")
    return result


@dataclass(frozen=True, slots=True)
class CausalAlphaV6SymbolSelectionSummary:
    symbol: str
    scope_count: int
    gross_log_return: float
    net_log_return: float
    gross_wealth: float
    net_wealth: float
    meaningful_execution_scope_count: int
    schema_version: str = CAUSAL_ALPHA_V6_SYMBOL_SELECTION_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        if not self.symbol or self.scope_count <= 0:
            raise ValueError("V6 symbol Selection identity is invalid")
        if not 0 <= self.meaningful_execution_scope_count <= self.scope_count:
            raise ValueError("V6 symbol Selection execution support is invalid")
        if not all(
            math.isfinite(value)
            for value in (
                self.gross_log_return,
                self.net_log_return,
                self.gross_wealth,
                self.net_wealth,
            )
        ) or min(self.gross_wealth, self.net_wealth) <= 0.0:
            raise ValueError("V6 symbol Selection economics are invalid")
        if not math.isclose(
            math.log(self.gross_wealth), self.gross_log_return, abs_tol=1e-12
        ) or not math.isclose(
            math.log(self.net_wealth), self.net_log_return, abs_tol=1e-12
        ):
            raise ValueError("V6 symbol Selection wealth is inconsistent")
        if self.schema_version != CAUSAL_ALPHA_V6_SYMBOL_SELECTION_SCHEMA:
            raise ValueError("unsupported V6 symbol Selection schema")
        expected = content_digest(self.to_payload(include_digest=False))
        if self.digest and self.digest != expected:
            raise ValueError("V6 symbol Selection digest mismatch")
        object.__setattr__(self, "digest", expected)

    def to_payload(self, *, include_digest: bool = True) -> dict[str, object]:
        payload = {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name != "digest"
        }
        if include_digest:
            payload["artifact_digest"] = self.digest
        return payload


@dataclass(frozen=True, slots=True)
class CausalAlphaV6CandidateSelectionEvidence:
    candidate: CausalAlphaV6Candidate
    metrics: tuple[CausalAlphaV6ReplayMetric, ...]
    symbol_summaries: tuple[CausalAlphaV6SymbolSelectionSummary, ...]
    expected_symbols: tuple[str, ...]
    run_manifest_digest: str
    v4_context_manifest_digest: str
    config_digest: str
    symbol_balanced_gross_wealth: float
    symbol_balanced_net_wealth: float
    median_symbol_net_wealth: float
    minimum_symbol_net_wealth: float
    positive_net_scope_fraction: float
    worst_symbol_episode_net_return: float
    scope_net_return_cvar_10: float
    turnover_p50: float
    turnover_p95: float
    total_execution_cost: float
    net_to_gross_retention: float
    meaningful_execution_scope_count: int
    total_target_change_count: int
    total_submitted_change_count: int
    total_executed_change_count: int
    total_closed_trade_count: int
    total_sign_flip_count: int
    hard_risk_violation_count: int
    unexplained_execution_rejection_count: int
    eligible: bool
    rejection_reasons: tuple[str, ...]
    schema_version: str = CAUSAL_ALPHA_V6_CANDIDATE_SELECTION_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        candidate = CausalAlphaV6Candidate(self.candidate)
        metrics = tuple(self.metrics)
        summaries = tuple(self.symbol_summaries)
        expected_symbols = tuple(self.expected_symbols)
        if not metrics or any(metric.candidate is not candidate for metric in metrics):
            raise ValueError("V6 candidate Selection metrics are invalid")
        summary_symbols = tuple(summary.symbol for summary in summaries)
        if summary_symbols != tuple(sorted(summary_symbols)) or len(
            set(summary_symbols)
        ) != len(summary_symbols):
            raise ValueError("V6 candidate Selection summaries are unordered")
        for name in (
            "run_manifest_digest",
            "v4_context_manifest_digest",
            "config_digest",
        ):
            require_sha256(getattr(self, name), field=f"V6 Selection {name}")
        if {metric.run_manifest_digest for metric in metrics} != {
            self.run_manifest_digest
        } or {metric.v4_context_manifest_digest for metric in metrics} != {
            self.v4_context_manifest_digest
        } or {metric.config_digest for metric in metrics} != {self.config_digest}:
            raise ValueError("V6 candidate Selection identity drifted")
        numeric = (
            self.symbol_balanced_gross_wealth,
            self.symbol_balanced_net_wealth,
            self.median_symbol_net_wealth,
            self.minimum_symbol_net_wealth,
            self.positive_net_scope_fraction,
            self.worst_symbol_episode_net_return,
            self.scope_net_return_cvar_10,
            self.turnover_p50,
            self.turnover_p95,
            self.total_execution_cost,
            self.net_to_gross_retention,
        )
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError("V6 candidate Selection contains non-finite values")
        if not 0.0 <= self.positive_net_scope_fraction <= 1.0:
            raise ValueError("V6 candidate Selection positive fraction is invalid")
        if min(
            self.turnover_p50,
            self.turnover_p95,
            self.total_execution_cost,
            self.net_to_gross_retention,
        ) < 0.0:
            raise ValueError("V6 candidate Selection costs are invalid")
        for name in (
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
                raise ValueError(f"V6 candidate Selection {name} is invalid")
        reasons = tuple(self.rejection_reasons)
        if self.eligible == bool(reasons):
            raise ValueError("V6 candidate Selection eligibility is invalid")
        if self.schema_version != CAUSAL_ALPHA_V6_CANDIDATE_SELECTION_SCHEMA:
            raise ValueError("unsupported V6 candidate Selection schema")
        object.__setattr__(self, "candidate", candidate)
        object.__setattr__(self, "metrics", metrics)
        object.__setattr__(self, "symbol_summaries", summaries)
        object.__setattr__(self, "expected_symbols", expected_symbols)
        object.__setattr__(self, "rejection_reasons", reasons)
        expected = content_digest(self.to_payload(include_digest=False))
        if self.digest and self.digest != expected:
            raise ValueError("V6 candidate Selection digest mismatch")
        object.__setattr__(self, "digest", expected)

    def to_payload(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name not in {"candidate", "metrics", "symbol_summaries", "digest"}
        }
        payload["candidate"] = self.candidate.value
        payload["replay_metric_digests"] = tuple(metric.digest for metric in self.metrics)
        payload["symbol_summaries"] = tuple(
            summary.to_payload() for summary in self.symbol_summaries
        )
        if include_digest:
            payload["artifact_digest"] = self.digest
        return payload


@dataclass(frozen=True, slots=True)
class CausalAlphaV6SelectionEvidence:
    fast_only: CausalAlphaV6CandidateSelectionEvidence
    fast_slow_retention: CausalAlphaV6CandidateSelectionEvidence
    paired_scope_count: int
    selected_candidate: CausalAlphaV6Candidate | None
    selected_config_digest: str | None
    passed: bool
    rejection_reasons: tuple[str, ...]
    promotion_eligible: bool = False
    schema_version: str = CAUSAL_ALPHA_V6_SELECTION_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        if self.fast_only.candidate is not CausalAlphaV6Candidate.FAST_ONLY:
            raise ValueError("V6 Selection fast-only evidence is invalid")
        if (
            self.fast_slow_retention.candidate
            is not CausalAlphaV6Candidate.FAST_SLOW_RETENTION
        ):
            raise ValueError("V6 Selection retention evidence is invalid")
        selected = (
            None
            if self.selected_candidate is None
            else CausalAlphaV6Candidate(self.selected_candidate)
        )
        reasons = tuple(self.rejection_reasons)
        if self.passed != (selected is not None and not reasons):
            raise ValueError("V6 Selection pass state is invalid")
        selected_config_digest = self.selected_config_digest
        if selected is None:
            if selected_config_digest is not None:
                raise ValueError("failed V6 Selection cannot select a config")
        else:
            if selected_config_digest is None:
                raise ValueError("passed V6 Selection must select a config")
            require_sha256(
                selected_config_digest, field="V6 selected config digest"
            )
            selected_evidence = (
                self.fast_only
                if selected is CausalAlphaV6Candidate.FAST_ONLY
                else self.fast_slow_retention
            )
            if (
                not selected_evidence.eligible
                or selected_config_digest != selected_evidence.config_digest
            ):
                raise ValueError("V6 selected candidate/config is invalid")
        if self.promotion_eligible:
            raise ValueError("V6 Selection cannot be promotion eligible")
        if self.schema_version != CAUSAL_ALPHA_V6_SELECTION_SCHEMA:
            raise ValueError("unsupported V6 Selection schema")
        object.__setattr__(self, "selected_candidate", selected)
        object.__setattr__(self, "rejection_reasons", reasons)
        expected = content_digest(self.to_payload(include_digest=False))
        if self.digest and self.digest != expected:
            raise ValueError("V6 Selection digest mismatch")
        object.__setattr__(self, "digest", expected)

    def to_payload(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "fast_only": self.fast_only.to_payload(),
            "fast_slow_retention": self.fast_slow_retention.to_payload(),
            "paired_scope_count": self.paired_scope_count,
            "passed": self.passed,
            "promotion_eligible": self.promotion_eligible,
            "rejection_reasons": self.rejection_reasons,
            "schema_version": self.schema_version,
            "selected_candidate": (
                None if self.selected_candidate is None else self.selected_candidate.value
            ),
            "selected_config_digest": self.selected_config_digest,
        }
        if include_digest:
            payload["artifact_digest"] = self.digest
        return payload


def _symbol_summaries(
    metrics: tuple[CausalAlphaV6ReplayMetric, ...],
    expected_symbols: tuple[str, ...],
) -> tuple[CausalAlphaV6SymbolSelectionSummary, ...]:
    grouped: dict[str, list[CausalAlphaV6ReplayMetric]] = defaultdict(list)
    for metric in metrics:
        grouped[metric.symbol].append(metric)
    summaries: list[CausalAlphaV6SymbolSelectionSummary] = []
    for symbol in sorted(set(expected_symbols) & set(grouped)):
        scopes = grouped[symbol]
        gross = float(sum(metric.gross_return for metric in scopes))
        net = float(sum(metric.net_return for metric in scopes))
        summaries.append(
            CausalAlphaV6SymbolSelectionSummary(
                symbol=symbol,
                scope_count=len(scopes),
                gross_log_return=gross,
                net_log_return=net,
                gross_wealth=_finite_exp(gross, field="symbol gross wealth"),
                net_wealth=_finite_exp(net, field="symbol net wealth"),
                meaningful_execution_scope_count=sum(
                    metric.has_meaningful_execution for metric in scopes
                ),
            )
        )
    return tuple(summaries)


def _candidate_evidence(
    candidate: CausalAlphaV6Candidate,
    metrics: tuple[CausalAlphaV6ReplayMetric, ...],
    expected_symbols: tuple[str, ...],
) -> CausalAlphaV6CandidateSelectionEvidence:
    summaries = _symbol_summaries(metrics, expected_symbols)
    balanced_gross = _finite_exp(
        float(fmean(summary.gross_log_return for summary in summaries)),
        field="balanced gross wealth",
    )
    balanced_net = _finite_exp(
        float(fmean(summary.net_log_return for summary in summaries)),
        field="balanced net wealth",
    )
    symbol_net = tuple(summary.net_wealth for summary in summaries)
    net_returns = np.asarray([metric.net_return for metric in metrics])
    turnovers = np.asarray([metric.turnover_per_day for metric in metrics])
    cvar_count = max(1, math.ceil(0.10 * len(metrics)))
    meaningful = sum(metric.has_meaningful_execution for metric in metrics)
    hard_risk = sum(metric.hard_risk_violation for metric in metrics)
    unexplained = sum(
        causal_alpha_unexplained_execution_rejection_count(
            metric.execution_rejection_reason_counts
        )
        for metric in metrics
    )
    p95 = float(np.quantile(turnovers, 0.95))
    positive_fraction = float(np.mean(net_returns > 0.0))
    reasons: list[str] = []
    if len({metric.identity for metric in metrics}) != len(metrics):
        reasons.append("duplicate_scope_identity")
    if {metric.symbol for metric in metrics} != set(expected_symbols):
        reasons.append("symbol_coverage")
    if balanced_gross <= 1.0:
        reasons.append("symbol_balanced_gross_wealth")
    if balanced_net <= 1.0:
        reasons.append("symbol_balanced_net_wealth")
    if min(symbol_net) < 1.0:
        reasons.append("minimum_symbol_net_wealth")
    if median(symbol_net) < 1.0:
        reasons.append("median_symbol_net_wealth")
    if positive_fraction < 0.5:
        reasons.append("positive_net_scope_fraction")
    if p95 > _TURNOVER_P95_LIMIT:
        reasons.append("turnover_p95")
    if meaningful == 0:
        reasons.append("no_meaningful_execution")
    if hard_risk:
        reasons.append("hard_risk_violation")
    if unexplained:
        reasons.append("unexplained_execution_rejection")
    total_cost = float(sum(metric.total_execution_cost for metric in metrics))
    return CausalAlphaV6CandidateSelectionEvidence(
        candidate=candidate,
        metrics=metrics,
        symbol_summaries=summaries,
        expected_symbols=expected_symbols,
        run_manifest_digest=metrics[0].run_manifest_digest,
        v4_context_manifest_digest=metrics[0].v4_context_manifest_digest,
        config_digest=metrics[0].config_digest,
        symbol_balanced_gross_wealth=balanced_gross,
        symbol_balanced_net_wealth=balanced_net,
        median_symbol_net_wealth=float(median(symbol_net)),
        minimum_symbol_net_wealth=float(min(symbol_net)),
        positive_net_scope_fraction=positive_fraction,
        worst_symbol_episode_net_return=float(np.min(net_returns)),
        scope_net_return_cvar_10=float(np.mean(np.sort(net_returns)[:cvar_count])),
        turnover_p50=float(np.quantile(turnovers, 0.50)),
        turnover_p95=p95,
        total_execution_cost=total_cost,
        net_to_gross_retention=balanced_net / balanced_gross,
        meaningful_execution_scope_count=meaningful,
        total_target_change_count=sum(metric.target_change_count for metric in metrics),
        total_submitted_change_count=sum(
            metric.submitted_change_count for metric in metrics
        ),
        total_executed_change_count=sum(
            metric.executed_change_count for metric in metrics
        ),
        total_closed_trade_count=sum(metric.closed_trade_count for metric in metrics),
        total_sign_flip_count=sum(metric.sign_flip_count for metric in metrics),
        hard_risk_violation_count=hard_risk,
        unexplained_execution_rejection_count=unexplained,
        eligible=not reasons,
        rejection_reasons=tuple(reasons),
    )


def _paired(
    fast: tuple[CausalAlphaV6ReplayMetric, ...],
    retention: tuple[CausalAlphaV6ReplayMetric, ...],
) -> bool:
    fast_map: dict[tuple[object, ...], int] = defaultdict(int)
    retention_map: dict[tuple[object, ...], int] = defaultdict(int)
    for metric in fast:
        fast_map[metric.paired_identity] += 1
    for metric in retention:
        retention_map[metric.paired_identity] += 1
    return fast_map == retention_map and all(count == 1 for count in fast_map.values())


def _retention_dominates(
    fast: CausalAlphaV6CandidateSelectionEvidence,
    retention: CausalAlphaV6CandidateSelectionEvidence,
) -> bool:
    return (
        retention.symbol_balanced_net_wealth
        > fast.symbol_balanced_net_wealth + _EPSILON
        and retention.minimum_symbol_net_wealth
        >= fast.minimum_symbol_net_wealth - _EPSILON
        and retention.turnover_p95 <= fast.turnover_p95 + _EPSILON
        and retention.total_execution_cost
        <= fast.total_execution_cost + _EPSILON
        and retention.total_sign_flip_count <= fast.total_sign_flip_count
    )


def evaluate_causal_alpha_v6_selection(
    metrics: tuple[CausalAlphaV6ReplayMetric, ...],
    *,
    expected_symbols: tuple[str, ...],
) -> CausalAlphaV6SelectionEvidence:
    """Select one candidate only after common economic and paired gates."""

    values = tuple(metrics)
    expected = tuple(expected_symbols)
    if not values:
        raise ValueError("V6 Selection requires replay metrics")
    if len(expected) != 9 or len(set(expected)) != len(expected):
        raise ValueError("V6 Selection requires exactly nine expected symbols")
    grouped = {
        candidate: tuple(metric for metric in values if metric.candidate is candidate)
        for candidate in CausalAlphaV6Candidate
    }
    if any(not grouped[candidate] for candidate in CausalAlphaV6Candidate):
        raise ValueError("V6 Selection requires both candidate replay paths")
    fast = _candidate_evidence(
        CausalAlphaV6Candidate.FAST_ONLY,
        grouped[CausalAlphaV6Candidate.FAST_ONLY],
        expected,
    )
    retention = _candidate_evidence(
        CausalAlphaV6Candidate.FAST_SLOW_RETENTION,
        grouped[CausalAlphaV6Candidate.FAST_SLOW_RETENTION],
        expected,
    )
    reasons: list[str] = []
    if len({metric.run_manifest_digest for metric in values}) != 1:
        reasons.append("run_identity")
    if len({metric.v4_context_manifest_digest for metric in values}) != 1:
        reasons.append("context_identity")
    if len({metric.config_digest for metric in values}) != 1:
        reasons.append("config_identity")
    paired = _paired(
        grouped[CausalAlphaV6Candidate.FAST_ONLY],
        grouped[CausalAlphaV6Candidate.FAST_SLOW_RETENTION],
    )
    if not paired:
        reasons.append("scope_pairing")
    selected: CausalAlphaV6Candidate | None = None
    if not reasons:
        if fast.eligible and retention.eligible:
            selected = (
                CausalAlphaV6Candidate.FAST_SLOW_RETENTION
                if _retention_dominates(fast, retention)
                else CausalAlphaV6Candidate.FAST_ONLY
            )
        elif fast.eligible:
            selected = CausalAlphaV6Candidate.FAST_ONLY
        elif retention.eligible:
            selected = CausalAlphaV6Candidate.FAST_SLOW_RETENTION
        else:
            reasons.append("no_eligible_candidate")
    return CausalAlphaV6SelectionEvidence(
        fast_only=fast,
        fast_slow_retention=retention,
        paired_scope_count=(len(grouped[CausalAlphaV6Candidate.FAST_ONLY]) if paired else 0),
        selected_candidate=selected,
        selected_config_digest=(None if selected is None else fast.config_digest),
        passed=selected is not None and not reasons,
        rejection_reasons=tuple(reasons),
    )


__all__ = [
    "CAUSAL_ALPHA_V6_CANDIDATE_SELECTION_SCHEMA",
    "CAUSAL_ALPHA_V6_SELECTION_SCHEMA",
    "CAUSAL_ALPHA_V6_SYMBOL_SELECTION_SCHEMA",
    "CausalAlphaV6CandidateSelectionEvidence",
    "CausalAlphaV6SelectionEvidence",
    "CausalAlphaV6SymbolSelectionSummary",
    "evaluate_causal_alpha_v6_selection",
]
