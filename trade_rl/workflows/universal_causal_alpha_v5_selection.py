"""Symbol-balanced after-cost wealth Selection for Causal Alpha V5."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from statistics import fmean, median
from typing import Final

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.common import require_sha256
from trade_rl.workflows.universal_causal_alpha_selection import (
    causal_alpha_unexplained_execution_rejection_count,
)
from trade_rl.workflows.universal_causal_alpha_v5_replay import (
    CausalAlphaV5ReplayMetric,
)

CAUSAL_ALPHA_V5_SYMBOL_SELECTION_SCHEMA: Final = (
    "causal_alpha_v5_symbol_selection_summary_v1"
)
CAUSAL_ALPHA_V5_SELECTION_SCHEMA: Final = "causal_alpha_v5_selection_evidence_v1"


def _finite_exp(value: float, *, field: str) -> float:
    try:
        result = math.exp(value)
    except OverflowError as error:
        raise ValueError(f"V5 selection {field} overflowed") from error
    if not math.isfinite(result):
        raise ValueError(f"V5 selection {field} is non-finite")
    return result


@dataclass(frozen=True, slots=True)
class CausalAlphaV5SymbolSelectionSummary:
    symbol: str
    scope_count: int
    gross_log_return: float
    net_log_return: float
    gross_wealth: float
    net_wealth: float
    meaningful_execution_scope_count: int
    schema_version: str = CAUSAL_ALPHA_V5_SYMBOL_SELECTION_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        if not self.symbol or self.scope_count <= 0:
            raise ValueError("V5 symbol selection identity is invalid")
        if not 0 <= self.meaningful_execution_scope_count <= self.scope_count:
            raise ValueError("V5 symbol selection execution support is invalid")
        if not all(
            math.isfinite(value) and value > 0.0
            for value in (self.gross_wealth, self.net_wealth)
        ):
            raise ValueError("V5 symbol selection wealth is invalid")
        if not all(
            math.isfinite(value)
            for value in (self.gross_log_return, self.net_log_return)
        ):
            raise ValueError("V5 symbol selection return is invalid")
        if not math.isclose(
            math.log(self.gross_wealth),
            self.gross_log_return,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("V5 symbol gross wealth is inconsistent")
        if not math.isclose(
            math.log(self.net_wealth), self.net_log_return, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError("V5 symbol net wealth is inconsistent")
        if self.schema_version != CAUSAL_ALPHA_V5_SYMBOL_SELECTION_SCHEMA:
            raise ValueError("unsupported V5 symbol selection schema")
        expected = content_digest(self.to_payload(include_digest=False))
        if self.digest and self.digest != expected:
            raise ValueError("V5 symbol selection digest mismatch")
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
class CausalAlphaV5SelectionEvidence:
    metrics: tuple[CausalAlphaV5ReplayMetric, ...]
    symbol_summaries: tuple[CausalAlphaV5SymbolSelectionSummary, ...]
    expected_symbols: tuple[str, ...]
    run_manifest_digest: str
    v4_context_manifest_digest: str
    config_digest: str
    symbol_balanced_gross_wealth: float
    symbol_balanced_net_wealth: float
    median_symbol_net_wealth: float
    positive_net_scope_fraction: float
    worst_symbol_episode_net_return: float
    scope_net_return_cvar_10: float
    turnover_p50: float
    turnover_p95: float
    total_execution_cost: float
    net_to_gross_retention: float
    meaningful_execution_scope_count: int
    total_submitted_change_count: int
    total_executed_change_count: int
    total_closed_trade_count: int
    hard_risk_violation_count: int
    unexplained_execution_rejection_count: int
    passed: bool
    rejection_reasons: tuple[str, ...]
    promotion_eligible: bool = False
    schema_version: str = CAUSAL_ALPHA_V5_SELECTION_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        metrics = tuple(self.metrics)
        summaries = tuple(self.symbol_summaries)
        expected_symbols = tuple(self.expected_symbols)
        if not metrics or len({metric.identity for metric in metrics}) != len(metrics):
            raise ValueError("V5 selection replay scope is duplicated or empty")
        summary_symbols = tuple(summary.symbol for summary in summaries)
        if (
            summary_symbols != tuple(sorted(summary_symbols))
            or len(set(summary_symbols)) != len(summary_symbols)
            or not set(summary_symbols) <= set(expected_symbols)
        ):
            raise ValueError("V5 selection symbol summaries are invalid or unordered")
        for name in (
            "run_manifest_digest",
            "v4_context_manifest_digest",
            "config_digest",
        ):
            require_sha256(getattr(self, name), field=f"V5 selection {name}")
        if {metric.run_manifest_digest for metric in metrics} != {
            self.run_manifest_digest
        }:
            raise ValueError("V5 selection run identity drifted")
        if {metric.v4_context_manifest_digest for metric in metrics} != {
            self.v4_context_manifest_digest
        }:
            raise ValueError("V5 selection context identity drifted")
        if {metric.config_digest for metric in metrics} != {self.config_digest}:
            raise ValueError("V5 selection config identity drifted")
        numeric = (
            self.symbol_balanced_gross_wealth,
            self.symbol_balanced_net_wealth,
            self.median_symbol_net_wealth,
            self.positive_net_scope_fraction,
            self.worst_symbol_episode_net_return,
            self.scope_net_return_cvar_10,
            self.turnover_p50,
            self.turnover_p95,
            self.total_execution_cost,
            self.net_to_gross_retention,
        )
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError("V5 selection summary contains non-finite values")
        if not 0.0 <= self.positive_net_scope_fraction <= 1.0:
            raise ValueError("V5 selection positive fraction is invalid")
        if (
            min(
                self.turnover_p50,
                self.turnover_p95,
                self.total_execution_cost,
                self.net_to_gross_retention,
            )
            < 0.0
        ):
            raise ValueError("V5 selection cost/turnover/retention is invalid")
        for name in (
            "meaningful_execution_scope_count",
            "total_submitted_change_count",
            "total_executed_change_count",
            "total_closed_trade_count",
            "hard_risk_violation_count",
            "unexplained_execution_rejection_count",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"V5 selection {name} is invalid")
        reasons = tuple(self.rejection_reasons)
        if self.passed != (not reasons) or self.promotion_eligible:
            raise ValueError("V5 selection pass/promotion state is invalid")
        if self.schema_version != CAUSAL_ALPHA_V5_SELECTION_SCHEMA:
            raise ValueError("unsupported V5 selection schema")
        object.__setattr__(self, "metrics", metrics)
        object.__setattr__(self, "symbol_summaries", summaries)
        object.__setattr__(self, "expected_symbols", expected_symbols)
        object.__setattr__(self, "rejection_reasons", reasons)
        expected = content_digest(self.to_payload(include_digest=False))
        if self.digest and self.digest != expected:
            raise ValueError("V5 selection digest mismatch")
        object.__setattr__(self, "digest", expected)

    @property
    def selected_config_digest(self) -> str | None:
        return self.config_digest if self.passed else None

    def to_payload(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name not in {"metrics", "symbol_summaries", "digest"}
        }
        payload["replay_metric_digests"] = tuple(
            metric.digest for metric in self.metrics
        )
        payload["symbol_summary_digests"] = tuple(
            summary.digest for summary in self.symbol_summaries
        )
        payload["selected_config_digest"] = self.selected_config_digest
        if include_digest:
            payload["artifact_digest"] = self.digest
        return payload


def evaluate_causal_alpha_v5_selection(
    metrics: tuple[CausalAlphaV5ReplayMetric, ...], *, expected_symbols: tuple[str, ...]
) -> CausalAlphaV5SelectionEvidence:
    """Evaluate one V5 candidate by balanced after-cost wealth, not average PnL."""

    values = tuple(metrics)
    if not values or len({metric.identity for metric in values}) != len(values):
        raise ValueError("V5 selection replay scope is duplicated or empty")
    expected = tuple(expected_symbols)
    if not expected or len(set(expected)) != len(expected):
        raise ValueError("V5 selection expected symbols are invalid")
    run_digests = {metric.run_manifest_digest for metric in values}
    context_digests = {metric.v4_context_manifest_digest for metric in values}
    config_digests = {metric.config_digest for metric in values}
    if len(run_digests) != 1 or len(context_digests) != 1 or len(config_digests) != 1:
        raise ValueError("V5 selection replay identity drifted")
    grouped: dict[str, list[CausalAlphaV5ReplayMetric]] = defaultdict(list)
    for metric in values:
        grouped[metric.symbol].append(metric)
    summaries: list[CausalAlphaV5SymbolSelectionSummary] = []
    for symbol in sorted(set(expected) & set(grouped)):
        scopes = grouped[symbol]
        gross_log = float(sum(metric.gross_return for metric in scopes))
        net_log = float(sum(metric.net_return for metric in scopes))
        summaries.append(
            CausalAlphaV5SymbolSelectionSummary(
                symbol=symbol,
                scope_count=len(scopes),
                gross_log_return=gross_log,
                net_log_return=net_log,
                gross_wealth=_finite_exp(gross_log, field="symbol gross wealth"),
                net_wealth=_finite_exp(net_log, field="symbol net wealth"),
                meaningful_execution_scope_count=sum(
                    metric.has_meaningful_execution for metric in scopes
                ),
            )
        )
    if summaries:
        balanced_gross = _finite_exp(
            float(fmean(item.gross_log_return for item in summaries)),
            field="balanced gross wealth",
        )
        balanced_net = _finite_exp(
            float(fmean(item.net_log_return for item in summaries)),
            field="balanced net wealth",
        )
        median_net = float(median(item.net_wealth for item in summaries))
    else:
        balanced_gross = balanced_net = median_net = 1.0
    net_returns = np.asarray([metric.net_return for metric in values], dtype=np.float64)
    cvar_count = max(1, math.ceil(0.10 * len(values)))
    cvar = float(np.mean(np.sort(net_returns)[:cvar_count]))
    turnovers = np.asarray(
        [metric.turnover_per_day for metric in values], dtype=np.float64
    )
    meaningful = sum(metric.has_meaningful_execution for metric in values)
    hard_risk = sum(metric.hard_risk_violation for metric in values)
    unexplained = sum(
        causal_alpha_unexplained_execution_rejection_count(
            metric.execution_rejection_reason_counts
        )
        for metric in values
    )
    reasons: list[str] = []
    if balanced_net <= 1.0:
        reasons.append("symbol_balanced_net_wealth")
    if median_net < 1.0:
        reasons.append("median_symbol_net_wealth")
    positive_fraction = float(np.mean(net_returns > 0.0))
    if positive_fraction < 0.5:
        reasons.append("positive_net_scope_fraction")
    if set(grouped) != set(expected):
        reasons.append("symbol_coverage")
    if meaningful == 0:
        reasons.append("no_meaningful_execution")
    if hard_risk:
        reasons.append("hard_risk_violation")
    if unexplained:
        reasons.append("unexplained_execution_rejection")
    return CausalAlphaV5SelectionEvidence(
        metrics=values,
        symbol_summaries=tuple(summaries),
        expected_symbols=expected,
        run_manifest_digest=next(iter(run_digests)),
        v4_context_manifest_digest=next(iter(context_digests)),
        config_digest=next(iter(config_digests)),
        symbol_balanced_gross_wealth=balanced_gross,
        symbol_balanced_net_wealth=balanced_net,
        median_symbol_net_wealth=median_net,
        positive_net_scope_fraction=positive_fraction,
        worst_symbol_episode_net_return=float(np.min(net_returns)),
        scope_net_return_cvar_10=cvar,
        turnover_p50=float(np.quantile(turnovers, 0.50)),
        turnover_p95=float(np.quantile(turnovers, 0.95)),
        total_execution_cost=float(
            sum(metric.total_execution_cost for metric in values)
        ),
        net_to_gross_retention=balanced_net / balanced_gross,
        meaningful_execution_scope_count=meaningful,
        total_submitted_change_count=sum(
            metric.submitted_change_count for metric in values
        ),
        total_executed_change_count=sum(
            metric.executed_change_count for metric in values
        ),
        total_closed_trade_count=sum(metric.closed_trade_count for metric in values),
        hard_risk_violation_count=hard_risk,
        unexplained_execution_rejection_count=unexplained,
        passed=not reasons,
        rejection_reasons=tuple(reasons),
    )


__all__ = [
    "CAUSAL_ALPHA_V5_SELECTION_SCHEMA",
    "CAUSAL_ALPHA_V5_SYMBOL_SELECTION_SCHEMA",
    "CausalAlphaV5SelectionEvidence",
    "CausalAlphaV5SymbolSelectionSummary",
    "evaluate_causal_alpha_v5_selection",
]
