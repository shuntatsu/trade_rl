from __future__ import annotations

import math

import pytest

from trade_rl.evaluation.comparisons import compare_paired_returns
from trade_rl.evaluation.series import ReturnKind, ReturnSeries


def series(
    values: tuple[float, ...],
    *,
    kind: ReturnKind = ReturnKind.BASE_BAR,
    periods_per_year: int = 8_760,
) -> ReturnSeries:
    return ReturnSeries(
        values=values,
        kind=kind,
        periods_per_year=periods_per_year,
    )


def test_paired_comparison_keeps_arithmetic_and_log_excess_separate() -> None:
    candidate = series((0.02, -0.01, 0.03, 0.0))
    benchmark = series((0.01, -0.01, 0.01, 0.0))

    result = compare_paired_returns(candidate, benchmark, n_bootstrap=250, seed=17)

    candidate_total = 1.02 * 0.99 * 1.03 - 1.0
    benchmark_total = 1.01 * 0.99 * 1.01 - 1.0
    assert result.excess_total_return == pytest.approx(
        candidate_total - benchmark_total
    )
    assert result.excess_log_return == pytest.approx(
        sum(math.log1p(value) for value in candidate.values)
        - sum(math.log1p(value) for value in benchmark.values)
    )
    assert result.mean_period_excess == pytest.approx(0.01)
    assert 0.0 <= result.p_value <= 1.0
    assert result.block_size >= 1


def test_identity_comparison_is_exact_and_non_significant() -> None:
    values = series((0.01, -0.02, 0.03))

    result = compare_paired_returns(values, values, n_bootstrap=100, seed=0)

    assert result.excess_total_return == 0.0
    assert result.excess_log_return == 0.0
    assert result.mean_period_excess == 0.0
    assert result.p_value == 1.0
    assert result.lower_ci == 0.0
    assert result.upper_ci == 0.0
    assert result.block_size == 1


def test_bootstrap_is_deterministic_for_an_explicit_seed() -> None:
    candidate = series((0.03, -0.01, 0.02, 0.01, -0.005))
    benchmark = series((0.01, -0.01, 0.01, 0.0, -0.005))

    left = compare_paired_returns(candidate, benchmark, n_bootstrap=300, seed=9)
    right = compare_paired_returns(candidate, benchmark, n_bootstrap=300, seed=9)

    assert left == right


@pytest.mark.parametrize(
    ("candidate", "benchmark", "message"),
    [
        (series((0.01,)), series((0.01, 0.02)), "length"),
        (
            series((0.01,), kind=ReturnKind.BASE_BAR),
            series((0.01,), kind=ReturnKind.DECISION_STEP),
            "kind",
        ),
        (
            series((0.01,), periods_per_year=8_760),
            series((0.01,), periods_per_year=365),
            "annualization",
        ),
    ],
)
def test_paired_comparison_rejects_incompatible_series(
    candidate: ReturnSeries,
    benchmark: ReturnSeries,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        compare_paired_returns(candidate, benchmark)
