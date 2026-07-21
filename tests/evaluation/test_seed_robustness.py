from __future__ import annotations

import pytest

from trade_rl.evaluation.seed_robustness import (
    SeedEvaluation,
    summarize_seed_robustness,
)
from trade_rl.evaluation.series import ReturnKind, ReturnSeries


def _series(values: tuple[float, ...]) -> ReturnSeries:
    return ReturnSeries(
        values=values,
        kind=ReturnKind.DECISION_STEP,
        periods_per_year=365,
    )


def test_unused_period_summary_reports_every_seed_and_uncertainty() -> None:
    baseline = _series((0.001, -0.002, 0.001, 0.0, 0.002, -0.001))
    seeds = (
        SeedEvaluation(
            seed=0,
            returns=_series((0.003, -0.001, 0.002, 0.001, 0.002, 0.0)),
            turnover_total=1.2,
            total_cost=0.002,
        ),
        SeedEvaluation(
            seed=1,
            returns=_series((0.001, -0.003, 0.0, 0.002, 0.001, -0.001)),
            turnover_total=1.8,
            total_cost=0.003,
        ),
        SeedEvaluation(
            seed=2,
            returns=_series((-0.002, -0.004, 0.001, -0.001, 0.0, -0.002)),
            turnover_total=0.9,
            total_cost=0.001,
        ),
    )

    summary = summarize_seed_robustness(
        evaluation_label="unused_2026_06",
        seeds=seeds,
        baseline=baseline,
        n_bootstrap=500,
        bootstrap_seed=17,
    )

    assert summary.evaluation_label == "unused_2026_06"
    assert tuple(item.seed for item in summary.seed_results) == (0, 1, 2)
    assert summary.worst_seed == 2
    assert summary.worst_return == min(
        item.total_return for item in summary.seed_results
    )
    assert summary.maximum_drawdown == max(
        item.maximum_drawdown for item in summary.seed_results
    )
    assert summary.median_turnover == pytest.approx(1.2)
    assert summary.median_return == pytest.approx(
        sorted(item.total_return for item in summary.seed_results)[1]
    )
    assert summary.median_baseline_difference == pytest.approx(
        sorted(item.baseline_difference for item in summary.seed_results)[1]
    )
    assert summary.bootstrap_lower_ci <= summary.bootstrap_upper_ci
    assert 0.0 <= summary.bootstrap_p_value <= 1.0
    assert summary.bootstrap_block_size >= 1


def test_seed_summary_rejects_best_seed_only_reporting() -> None:
    with pytest.raises(ValueError, match="at least two seeds"):
        summarize_seed_robustness(
            evaluation_label="unused",
            seeds=(
                SeedEvaluation(
                    seed=0,
                    returns=_series((0.01, 0.02)),
                    turnover_total=1.0,
                ),
            ),
            baseline=_series((0.0, 0.0)),
        )
