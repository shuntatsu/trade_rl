"""Seed-complete and unused-period robustness summaries."""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import median

import numpy as np

from trade_rl.evaluation.bootstrap import moving_block_mean_test
from trade_rl.evaluation.metrics import compound_return, evaluate_performance
from trade_rl.evaluation.series import ReturnSeries


@dataclass(frozen=True, slots=True)
class SeedEvaluation:
    seed: int
    returns: ReturnSeries
    turnover_total: float
    total_cost: float = 0.0

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        for field_name, value in (
            ("turnover_total", self.turnover_total),
            ("total_cost", self.total_cost),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{field_name} must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class SeedResult:
    seed: int
    total_return: float
    maximum_drawdown: float
    turnover_total: float
    total_cost: float
    baseline_difference: float


@dataclass(frozen=True, slots=True)
class SeedRobustnessSummary:
    evaluation_label: str
    seed_results: tuple[SeedResult, ...]
    median_return: float
    worst_seed: int
    worst_return: float
    maximum_drawdown: float
    median_turnover: float
    baseline_return: float
    median_baseline_difference: float
    bootstrap_lower_ci: float
    bootstrap_upper_ci: float
    bootstrap_p_value: float
    bootstrap_block_size: int


def summarize_seed_robustness(
    *,
    evaluation_label: str,
    seeds: tuple[SeedEvaluation, ...],
    baseline: ReturnSeries,
    n_bootstrap: int = 1_000,
    bootstrap_seed: int = 0,
) -> SeedRobustnessSummary:
    if not evaluation_label:
        raise ValueError("evaluation_label must not be empty")
    if len(seeds) < 2:
        raise ValueError("seed robustness requires at least two seeds")
    identifiers = tuple(item.seed for item in seeds)
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("seed robustness requires unique seeds")
    if any(item.returns.kind != baseline.kind for item in seeds):
        raise ValueError("seed and baseline return kinds must match")
    if any(len(item.returns.values) != len(baseline.values) for item in seeds):
        raise ValueError("seed and baseline ranges must have equal length")
    if any(item.returns.periods_per_year != baseline.periods_per_year for item in seeds):
        raise ValueError("seed and baseline annualization must match")

    baseline_return = compound_return(baseline.values)
    resolved: list[SeedResult] = []
    for item in seeds:
        metrics = evaluate_performance(
            item.returns,
            turnover_total=item.turnover_total,
            total_cost=item.total_cost,
        )
        resolved.append(
            SeedResult(
                seed=item.seed,
                total_return=metrics.total_return,
                maximum_drawdown=metrics.max_drawdown,
                turnover_total=item.turnover_total,
                total_cost=item.total_cost,
                baseline_difference=metrics.total_return - baseline_return,
            )
        )
    seed_matrix = np.asarray([item.returns.values for item in seeds], dtype=np.float64)
    period_median = np.median(seed_matrix, axis=0)
    paired = tuple(
        float(value)
        for value in period_median - np.asarray(baseline.values, dtype=np.float64)
    )
    bootstrap = moving_block_mean_test(
        paired,
        n_bootstrap=n_bootstrap,
        seed=bootstrap_seed,
    )
    worst = min(resolved, key=lambda item: (item.total_return, item.seed))
    return SeedRobustnessSummary(
        evaluation_label=evaluation_label,
        seed_results=tuple(resolved),
        median_return=float(median(item.total_return for item in resolved)),
        worst_seed=worst.seed,
        worst_return=worst.total_return,
        maximum_drawdown=max(item.maximum_drawdown for item in resolved),
        median_turnover=float(median(item.turnover_total for item in resolved)),
        baseline_return=baseline_return,
        median_baseline_difference=float(
            median(item.baseline_difference for item in resolved)
        ),
        bootstrap_lower_ci=bootstrap.lower_ci,
        bootstrap_upper_ci=bootstrap.upper_ci,
        bootstrap_p_value=bootstrap.p_value,
        bootstrap_block_size=bootstrap.block_size,
    )


__all__ = [
    "SeedEvaluation",
    "SeedResult",
    "SeedRobustnessSummary",
    "summarize_seed_robustness",
]
