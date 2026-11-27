from __future__ import annotations

import math

import pytest

from trade_rl.evaluation.comparison.seed_robustness import (
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


def _elapsed_series(years: float) -> ReturnSeries:
    return ReturnSeries(
        values=(0.01, 0.02),
        kind=ReturnKind.DECISION_STEP,
        periods_per_year=365,
        elapsed_years=years,
    )


@pytest.mark.parametrize("wrong_seed", [0, 1])
def test_seed_summary_rejects_different_effective_elapsed_clock(
    wrong_seed: int,
) -> None:
    baseline = _elapsed_series(0.01)
    seeds = tuple(
        SeedEvaluation(
            seed=seed,
            returns=_elapsed_series(0.02 if seed == wrong_seed else 0.01),
            turnover_total=0.0,
        )
        for seed in (0, 1)
    )
    with pytest.raises(ValueError, match="annualization"):
        summarize_seed_robustness(
            evaluation_label="synthetic",
            seeds=seeds,
            baseline=baseline,
            n_bootstrap=5,
        )


def test_seed_summary_preserves_matching_elapsed_clock() -> None:
    baseline = _elapsed_series(0.01)
    summary = summarize_seed_robustness(
        evaluation_label="synthetic",
        seeds=tuple(SeedEvaluation(seed, baseline, 0.0) for seed in (0, 1)),
        baseline=baseline,
        n_bootstrap=5,
    )
    assert summary.median_baseline_difference == 0.0
    assert summary.bootstrap_p_value == 1.0


@pytest.mark.parametrize("delta, compatible", [(5e-13, True), (2e-12, False)])
def test_effective_clock_uses_the_existing_paired_tolerance(delta, compatible) -> None:
    baseline = _elapsed_series(2.0 / 200.0)
    candidate = _elapsed_series(2.0 / (200.0 + delta))
    difference = abs(
        candidate.annualization_periods_per_year
        - baseline.annualization_periods_per_year
    )
    assert (difference <= 1e-12) is compatible
    seeds = (SeedEvaluation(0, candidate, 0.0), SeedEvaluation(1, baseline, 0.0))
    if compatible:
        assert (
            summarize_seed_robustness(
                evaluation_label="synthetic",
                seeds=seeds,
                baseline=baseline,
                n_bootstrap=5,
            ).median_baseline_difference
            == 0.0
        )
    else:
        with pytest.raises(ValueError, match="annualization"):
            summarize_seed_robustness(
                evaluation_label="synthetic",
                seeds=seeds,
                baseline=baseline,
                n_bootstrap=5,
            )


@pytest.mark.parametrize("years, compatible", [(2.0 / 365.0, True), (0.01, False)])
def test_effective_clock_checks_mixed_explicit_and_implicit_elapsed_time(
    years, compatible
) -> None:
    baseline = _series((0.01, 0.02))
    candidate = _elapsed_series(years)
    seeds = (SeedEvaluation(0, candidate, 0.0), SeedEvaluation(1, baseline, 0.0))
    if compatible:
        assert (
            summarize_seed_robustness(
                evaluation_label="synthetic",
                seeds=seeds,
                baseline=baseline,
                n_bootstrap=5,
            ).median_baseline_difference
            == 0.0
        )
    else:
        with pytest.raises(ValueError, match="annualization"):
            summarize_seed_robustness(
                evaluation_label="synthetic",
                seeds=seeds,
                baseline=baseline,
                n_bootstrap=5,
            )


@pytest.mark.parametrize("invalid_baseline", [False, True])
def test_effective_clock_rejects_unrepresentable_annualization(
    invalid_baseline,
) -> None:
    finite = _elapsed_series(0.01)
    unrepresentable = _elapsed_series(5e-324)
    assert math.isinf(unrepresentable.annualization_periods_per_year)
    baseline = unrepresentable if invalid_baseline else finite
    seeds = (SeedEvaluation(0, unrepresentable, 0.0), SeedEvaluation(1, finite, 0.0))
    with pytest.raises(ValueError, match="annualization"):
        summarize_seed_robustness(
            evaluation_label="synthetic", seeds=seeds, baseline=baseline, n_bootstrap=5
        )


def test_matching_effective_clock_does_not_remove_legacy_nominal_check() -> None:
    baseline = _elapsed_series(0.01)
    candidate = ReturnSeries(
        (0.01, 0.02), ReturnKind.DECISION_STEP, 200, elapsed_years=0.01
    )
    assert (
        candidate.annualization_periods_per_year
        == baseline.annualization_periods_per_year
    )
    with pytest.raises(ValueError, match="annualization"):
        summarize_seed_robustness(
            evaluation_label="synthetic",
            seeds=(SeedEvaluation(0, candidate, 0.0), SeedEvaluation(1, baseline, 0.0)),
            baseline=baseline,
            n_bootstrap=5,
        )


def test_identical_infinite_effective_clocks_are_rejected_before_metrics(
    monkeypatch,
) -> None:
    from trade_rl.evaluation.comparison import seed_robustness as module

    unrepresentable = _elapsed_series(5e-324)
    assert math.isinf(unrepresentable.annualization_periods_per_year)

    def unexpected(*args, **kwargs):
        pytest.fail("invalid baseline frequency reached performance calculation")

    monkeypatch.setattr(module, "compound_return", unexpected)
    monkeypatch.setattr(module, "evaluate_performance", unexpected)
    with pytest.raises(ValueError, match="baseline annualization must be finite"):
        summarize_seed_robustness(
            evaluation_label="synthetic",
            seeds=(
                SeedEvaluation(0, unrepresentable, 0.0),
                SeedEvaluation(1, unrepresentable, 0.0),
            ),
            baseline=unrepresentable,
            n_bootstrap=5,
        )


def test_explicit_baseline_also_rejects_mismatched_implicit_seed_clock() -> None:
    baseline = _elapsed_series(0.01)
    with pytest.raises(ValueError, match="annualization"):
        summarize_seed_robustness(
            evaluation_label="synthetic",
            seeds=(
                SeedEvaluation(0, _series((0.01, 0.02)), 0.0),
                SeedEvaluation(1, baseline, 0.0),
            ),
            baseline=baseline,
            n_bootstrap=5,
        )
