"""Three-way symbol-balanced after-cost Selection for Causal Alpha V7."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from statistics import fmean, median
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

_SYMBOL_SCHEMA: Final = "causal_alpha_v7_symbol_selection_summary_v1"
_CANDIDATE_SCHEMA: Final = "causal_alpha_v7_candidate_selection_evidence_v1"
_SELECTION_SCHEMA: Final = "causal_alpha_v7_selection_evidence_v1"
_TURNOVER_P95_LIMIT: Final = 1.0


def _wealth(log_return: float, *, field: str) -> float:
    try:
        value = math.exp(log_return)
    except OverflowError as error:
        raise ValueError(f"V7 Selection {field} overflowed") from error
    if not math.isfinite(value):
        raise ValueError(f"V7 Selection {field} is non-finite")
    return value


@dataclass(frozen=True, slots=True)
class CausalAlphaV7SymbolSelectionSummary:
    symbol: str
    scope_count: int
    gross_log_return: float
    net_log_return: float
    gross_wealth: float
    net_wealth: float
    meaningful_execution_scope_count: int
    schema_version: str = _SYMBOL_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        if not self.symbol or self.scope_count <= 0:
            raise ValueError("V7 symbol Selection identity is invalid")
        if not 0 <= self.meaningful_execution_scope_count <= self.scope_count:
            raise ValueError("V7 symbol Selection execution support is invalid")
        if not math.isclose(math.log(self.gross_wealth), self.gross_log_return, abs_tol=1e-12):
            raise ValueError("V7 symbol gross wealth is inconsistent")
        if not math.isclose(math.log(self.net_wealth), self.net_log_return, abs_tol=1e-12):
            raise ValueError("V7 symbol net wealth is inconsistent")
        if self.schema_version != _SYMBOL_SCHEMA:
            raise ValueError("unsupported V7 symbol Selection schema")
        expected = content_digest(self.to_payload(include_digest=False))
        if self.digest and self.digest != expected:
            raise ValueError("V7 symbol Selection digest mismatch")
        object.__setattr__(self, "digest", expected)

    def to_payload(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "gross_log_return": self.gross_log_return,
            "gross_wealth": self.gross_wealth,
            "meaningful_execution_scope_count": self.meaningful_execution_scope_count,
            "net_log_return": self.net_log_return,
            "net_wealth": self.net_wealth,
            "schema_version": self.schema_version,
            "scope_count": self.scope_count,
            "symbol": self.symbol,
        }
        if include_digest:
            payload["artifact_digest"] = self.digest
        return payload


@dataclass(frozen=True, slots=True)
class CausalAlphaV7CandidateSelectionEvidence:
    candidate: CausalAlphaV7Candidate
    metrics: tuple[CausalAlphaV7ReplayMetric, ...]
    symbol_summaries: tuple[CausalAlphaV7SymbolSelectionSummary, ...]
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
    schema_version: str = _CANDIDATE_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        candidate = CausalAlphaV7Candidate(self.candidate)
        metrics = tuple(self.metrics)
        summaries = tuple(self.symbol_summaries)
        if not metrics or any(metric.candidate is not candidate for metric in metrics):
            raise ValueError("V7 candidate Selection metrics are invalid")
        for name in ("run_manifest_digest", "v4_context_manifest_digest", "config_digest"):
            require_sha256(getattr(self, name), field=f"V7 Selection {name}")
        if any(metric.v7_config_digest != self.config_digest for metric in metrics):
            raise ValueError("V7 candidate Selection config identity drifted")
        if tuple(summary.symbol for summary in summaries) != tuple(
            sorted(summary.symbol for summary in summaries)
        ):
            raise ValueError("V7 candidate Selection summaries are unordered")
        reasons = tuple(self.rejection_reasons)
        if self.eligible == bool(reasons):
            raise ValueError("V7 candidate Selection eligibility is invalid")
        if self.schema_version != _CANDIDATE_SCHEMA:
            raise ValueError("unsupported V7 candidate Selection schema")
        object.__setattr__(self, "candidate", candidate)
        object.__setattr__(self, "metrics", metrics)
        object.__setattr__(self, "symbol_summaries", summaries)
        object.__setattr__(self, "rejection_reasons", reasons)
        expected = content_digest(self.to_payload(include_digest=False))
        if self.digest and self.digest != expected:
            raise ValueError("V7 candidate Selection digest mismatch")
        object.__setattr__(self, "digest", expected)

    def to_payload(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name not in {"candidate", "metrics", "symbol_summaries", "digest"}
        }
        payload["candidate"] = self.candidate.value
        payload["replay_metric_digests"] = tuple(metric.digest for metric in self.metrics)
        payload["attribution_digests"] = tuple(
            metric.attribution.digest for metric in self.metrics
        )
        payload["symbol_summaries"] = tuple(
            summary.to_payload() for summary in self.symbol_summaries
        )
        if include_digest:
            payload["artifact_digest"] = self.digest
        return payload


@dataclass(frozen=True, slots=True)
class CausalAlphaV7SelectionEvidence:
    candidates: tuple[CausalAlphaV7CandidateSelectionEvidence, ...]
    paired_scope_count: int
    selected_candidate: CausalAlphaV7Candidate | None
    selected_config_digest: str | None
    passed: bool
    rejection_reasons: tuple[str, ...]
    promotion_eligible: bool = False
    schema_version: str = _SELECTION_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        candidates = tuple(self.candidates)
        if tuple(item.candidate for item in candidates) != tuple(CausalAlphaV7Candidate):
            raise ValueError("V7 Selection candidate evidence is not canonical")
        selected = None if self.selected_candidate is None else CausalAlphaV7Candidate(self.selected_candidate)
        reasons = tuple(self.rejection_reasons)
        if self.passed != (selected is not None and not reasons):
            raise ValueError("V7 Selection pass state is invalid")
        if selected is None:
            if self.selected_config_digest is not None:
                raise ValueError("failed V7 Selection cannot select a config")
        else:
            if self.selected_config_digest is None:
                raise ValueError("passed V7 Selection must select a config")
            require_sha256(self.selected_config_digest, field="V7 selected config digest")
            selected_evidence = candidates[tuple(CausalAlphaV7Candidate).index(selected)]
            if not selected_evidence.eligible or selected_evidence.config_digest != self.selected_config_digest:
                raise ValueError("V7 selected candidate/config is invalid")
        if self.promotion_eligible:
            raise ValueError("V7 Selection cannot be promotion eligible")
        if self.schema_version != _SELECTION_SCHEMA:
            raise ValueError("unsupported V7 Selection schema")
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "selected_candidate", selected)
        object.__setattr__(self, "rejection_reasons", reasons)
        expected = content_digest(self.to_payload(include_digest=False))
        if self.digest and self.digest != expected:
            raise ValueError("V7 Selection digest mismatch")
        object.__setattr__(self, "digest", expected)

    def to_payload(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "candidates": tuple(candidate.to_payload() for candidate in self.candidates),
            "paired_scope_count": self.paired_scope_count,
            "passed": self.passed,
            "promotion_eligible": self.promotion_eligible,
            "rejection_reasons": self.rejection_reasons,
            "schema_version": self.schema_version,
            "selected_candidate": None if self.selected_candidate is None else self.selected_candidate.value,
            "selected_config_digest": self.selected_config_digest,
        }
        if include_digest:
            payload["artifact_digest"] = self.digest
        return payload


def _summaries(
    metrics: tuple[CausalAlphaV7ReplayMetric, ...],
    expected_symbols: tuple[str, ...],
) -> tuple[CausalAlphaV7SymbolSelectionSummary, ...]:
    grouped: dict[str, list[CausalAlphaV7ReplayMetric]] = defaultdict(list)
    for metric in metrics:
        grouped[metric.v6_metric.symbol].append(metric)
    return tuple(
        CausalAlphaV7SymbolSelectionSummary(
            symbol=symbol,
            scope_count=len(grouped[symbol]),
            gross_log_return=float(sum(item.v6_metric.gross_return for item in grouped[symbol])),
            net_log_return=float(sum(item.v6_metric.net_return for item in grouped[symbol])),
            gross_wealth=_wealth(
                float(sum(item.v6_metric.gross_return for item in grouped[symbol])),
                field="symbol gross wealth",
            ),
            net_wealth=_wealth(
                float(sum(item.v6_metric.net_return for item in grouped[symbol])),
                field="symbol net wealth",
            ),
            meaningful_execution_scope_count=sum(
                item.v6_metric.has_meaningful_execution for item in grouped[symbol]
            ),
        )
        for symbol in sorted(set(expected_symbols) & set(grouped))
    )


def _candidate_evidence(
    candidate: CausalAlphaV7Candidate,
    metrics: tuple[CausalAlphaV7ReplayMetric, ...],
    expected_symbols: tuple[str, ...],
) -> CausalAlphaV7CandidateSelectionEvidence:
    summaries = _summaries(metrics, expected_symbols)
    balanced_gross = _wealth(
        float(fmean(summary.gross_log_return for summary in summaries)),
        field="balanced gross wealth",
    )
    balanced_net = _wealth(
        float(fmean(summary.net_log_return for summary in summaries)),
        field="balanced net wealth",
    )
    symbol_net = tuple(summary.net_wealth for summary in summaries)
    net_returns = np.asarray([metric.v6_metric.net_return for metric in metrics])
    turnovers = np.asarray([metric.v6_metric.turnover_per_day for metric in metrics])
    meaningful = sum(metric.v6_metric.has_meaningful_execution for metric in metrics)
    hard_risk = sum(metric.v6_metric.hard_risk_violation for metric in metrics)
    unexplained = sum(
        causal_alpha_unexplained_execution_rejection_count(
            metric.v6_metric.execution_rejection_reason_counts
        )
        for metric in metrics
    )
    p95 = float(np.quantile(turnovers, 0.95))
    positive_fraction = float(np.mean(net_returns > 0.0))
    reasons: list[str] = []
    if len({metric.identity for metric in metrics}) != len(metrics):
        reasons.append("duplicate_scope_identity")
    if {metric.v6_metric.symbol for metric in metrics} != set(expected_symbols):
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
    cvar_count = max(1, math.ceil(0.10 * len(metrics)))
    base = tuple(metric.v6_metric for metric in metrics)
    return CausalAlphaV7CandidateSelectionEvidence(
        candidate=candidate,
        metrics=metrics,
        symbol_summaries=summaries,
        expected_symbols=expected_symbols,
        run_manifest_digest=base[0].run_manifest_digest,
        v4_context_manifest_digest=base[0].v4_context_manifest_digest,
        config_digest=metrics[0].v7_config_digest,
        symbol_balanced_gross_wealth=balanced_gross,
        symbol_balanced_net_wealth=balanced_net,
        median_symbol_net_wealth=float(median(symbol_net)),
        minimum_symbol_net_wealth=float(min(symbol_net)),
        positive_net_scope_fraction=positive_fraction,
        worst_symbol_episode_net_return=float(np.min(net_returns)),
        scope_net_return_cvar_10=float(np.mean(np.sort(net_returns)[:cvar_count])),
        turnover_p50=float(np.quantile(turnovers, 0.50)),
        turnover_p95=p95,
        total_execution_cost=float(sum(item.total_execution_cost for item in base)),
        net_to_gross_retention=balanced_net / balanced_gross,
        meaningful_execution_scope_count=meaningful,
        total_target_change_count=sum(item.target_change_count for item in base),
        total_submitted_change_count=sum(item.submitted_change_count for item in base),
        total_executed_change_count=sum(item.executed_change_count for item in base),
        total_closed_trade_count=sum(item.closed_trade_count for item in base),
        total_sign_flip_count=sum(item.sign_flip_count for item in base),
        hard_risk_violation_count=hard_risk,
        unexplained_execution_rejection_count=unexplained,
        eligible=not reasons,
        rejection_reasons=tuple(reasons),
    )


def _paired(grouped: dict[CausalAlphaV7Candidate, tuple[CausalAlphaV7ReplayMetric, ...]]) -> bool:
    maps: list[dict[tuple[object, ...], int]] = []
    for candidate in CausalAlphaV7Candidate:
        counts: dict[tuple[object, ...], int] = defaultdict(int)
        for metric in grouped[candidate]:
            counts[metric.paired_identity] += 1
        maps.append(counts)
    return all(mapping == maps[0] for mapping in maps[1:]) and all(
        count == 1 for count in maps[0].values()
    )


def evaluate_causal_alpha_v7_selection(
    metrics: tuple[CausalAlphaV7ReplayMetric, ...],
    *,
    expected_symbols: tuple[str, ...],
) -> CausalAlphaV7SelectionEvidence:
    """Evaluate all fixed V7 candidates under unchanged universal gates."""

    values = tuple(metrics)
    expected = tuple(expected_symbols)
    if not values:
        raise ValueError("V7 Selection requires replay metrics")
    if len(expected) != 9 or len(set(expected)) != 9:
        raise ValueError("V7 Selection requires exactly nine expected symbols")
    grouped = {
        candidate: tuple(metric for metric in values if metric.candidate is candidate)
        for candidate in CausalAlphaV7Candidate
    }
    if any(not grouped[candidate] for candidate in CausalAlphaV7Candidate):
        raise ValueError("V7 Selection requires every fixed candidate")
    candidate_evidence = tuple(
        _candidate_evidence(candidate, grouped[candidate], expected)
        for candidate in CausalAlphaV7Candidate
    )
    reasons: list[str] = []
    paired = _paired(grouped)
    if not paired:
        reasons.append("scope_pairing")
    eligible = tuple(item for item in candidate_evidence if item.eligible)
    selected: CausalAlphaV7CandidateSelectionEvidence | None = None
    if paired and eligible:
        candidate_order = {candidate: index for index, candidate in enumerate(CausalAlphaV7Candidate)}
        selected = max(
            eligible,
            key=lambda item: (
                item.symbol_balanced_net_wealth,
                -item.turnover_p95,
                -item.total_execution_cost,
                -candidate_order[item.candidate],
            ),
        )
    if paired and selected is None:
        reasons.append("no_eligible_candidate")
    paired_count = len(grouped[CausalAlphaV7Candidate.V6_CONTROL]) if paired else 0
    return CausalAlphaV7SelectionEvidence(
        candidates=candidate_evidence,
        paired_scope_count=paired_count,
        selected_candidate=None if selected is None else selected.candidate,
        selected_config_digest=None if selected is None else selected.config_digest,
        passed=selected is not None and not reasons,
        rejection_reasons=tuple(reasons),
    )


__all__ = [
    "CausalAlphaV7CandidateSelectionEvidence",
    "CausalAlphaV7SelectionEvidence",
    "CausalAlphaV7SymbolSelectionSummary",
    "evaluate_causal_alpha_v7_selection",
]
