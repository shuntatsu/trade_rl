"""Paired candidate-versus-benchmark evaluation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import fmean

from trade_rl.evaluation.bootstrap import moving_block_mean_test
from trade_rl.evaluation.metrics import compound_return
from trade_rl.evaluation.series import ReturnSeries


@dataclass(frozen=True, slots=True)
class PairedComparison:
    """Arithmetic, logarithmic, and bootstrap paired excess evidence."""

    excess_total_return: float
    excess_log_return: float
    mean_period_excess: float
    p_value: float
    lower_ci: float
    upper_ci: float
    block_size: int


def _validate_compatible(candidate: ReturnSeries, benchmark: ReturnSeries) -> None:
    if len(candidate.values) != len(benchmark.values):
        raise ValueError("paired return series length mismatch")
    if candidate.kind is not benchmark.kind:
        raise ValueError("paired return series kind mismatch")
    if candidate.periods_per_year != benchmark.periods_per_year:
        raise ValueError("paired return series annualization mismatch")


def compare_paired_returns(
    candidate: ReturnSeries,
    benchmark: ReturnSeries,
    *,
    n_bootstrap: int = 1_000,
    seed: int = 0,
) -> PairedComparison:
    """Compare chronologically paired candidate and benchmark returns."""

    _validate_compatible(candidate, benchmark)
    differences = tuple(
        candidate_value - benchmark_value
        for candidate_value, benchmark_value in zip(
            candidate.values,
            benchmark.values,
            strict=True,
        )
    )
    bootstrap = moving_block_mean_test(
        differences,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    candidate_log_return = sum(math.log1p(value) for value in candidate.values)
    benchmark_log_return = sum(math.log1p(value) for value in benchmark.values)

    return PairedComparison(
        excess_total_return=compound_return(candidate.values)
        - compound_return(benchmark.values),
        excess_log_return=candidate_log_return - benchmark_log_return,
        mean_period_excess=fmean(differences),
        p_value=bootstrap.p_value,
        lower_ci=bootstrap.lower_ci,
        upper_ci=bootstrap.upper_ci,
        block_size=bootstrap.block_size,
    )
