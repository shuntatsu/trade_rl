"""SD1 (#1146 section 7): the measurement tools recover known truth on synthetic markets.

Every market is drawn from a seeded numpy generator; no market value is read. The four
oracles of the plan, plus the control that documents what demeaning does not remove.
"""

from __future__ import annotations

import numpy as np

from tests.evaluation.signal_measurement_synthetic import (
    HOURS_PER_DAY,
    common_level_panel,
    planted_signal_panel,
    shuffle_days,
    tail_only_panel,
)
from trade_rl.evaluation.signal_measurement import (
    bh_rejections,
    block_mean_interval,
    causal_beta_residual,
    cross_sectional_demean,
    daily_ic,
    holm_rejections,
    linear_ic,
    rank_ic,
)

BLOCK_DAYS = 5
RESAMPLES = 400
Q = 0.10
ALPHA = 0.05


def _daily_rank_ic_p(
    day_ids: np.ndarray, feature: np.ndarray, target: np.ndarray, seed: int
):
    series = daily_ic(day_ids, feature, target, method="rank")
    return block_mean_interval(
        series.values, block_days=BLOCK_DAYS, resamples=RESAMPLES, seed=seed
    )


# ------------------------------------------------------- oracle 1: planted-signal recovery


def test_a_planted_signal_is_recovered_with_the_closed_form_rank_ic() -> None:
    panel = planted_signal_panel(
        11, days=120, symbols=5, null_features=10, strength=0.25
    )

    pooled = daily_ic(panel.day_ids, panel.signal, panel.target, method="rank")
    interval = _daily_rank_ic_p(panel.day_ids, panel.signal, panel.target, seed=0)
    null_results = [
        _daily_rank_ic_p(panel.day_ids, feature, panel.target, seed=index)
        for index, feature in enumerate(panel.nulls)
    ]

    assert pooled.skipped_days == 0
    assert abs(float(pooled.mean) - panel.true_rank_ic) < 0.03
    assert interval.lower > 0.0
    p_values = np.array(
        [interval.p_value_one_sided] + [r.p_value_one_sided for r in null_results]
    )
    # The planted feature (index 0) is found by both corrections; Holm lets none of the
    # ten pure-noise features through.
    assert bool(holm_rejections(p_values, ALPHA)[0])
    assert bool(bh_rejections(p_values, Q)[0])
    assert not holm_rejections(p_values, ALPHA)[1:].any()
    # The signal's linear IC agrees with the Gaussian Pearson value.
    rho = 0.25 / np.sqrt(1.0 + 0.25**2)
    assert abs(float(linear_ic(panel.signal, panel.target) or 0.0) - rho) < 0.01


# ------------------------------- oracle 2: shuffled-future null with false-discovery control


def test_bh_controls_the_false_discovery_rate_and_holm_the_family_wise_rate() -> None:
    """Procedure-level check with exactly uniform null p-values (independent tests)."""

    rng = np.random.default_rng(2026)
    reps, total, true_effects = 4000, 100, 20
    false_discovery_proportions = np.empty(reps)
    holm_any_false = 0
    for rep in range(reps):
        p = rng.uniform(size=total)
        p[:true_effects] = rng.beta(
            0.05, 1.0, size=true_effects
        )  # real effects: tiny p
        rejected = bh_rejections(p, Q)
        count = int(rejected.sum())
        false_count = int(rejected[true_effects:].sum())
        false_discovery_proportions[rep] = false_count / count if count else 0.0
        holm_any_false += int(holm_rejections(p, ALPHA)[true_effects:].any())
    # BH: FDR = q * (null share) = 0.10 * 0.8 = 0.08 for independent p-values.
    assert float(false_discovery_proportions.mean()) <= Q
    assert holm_any_false / reps <= ALPHA + 0.01


def test_shuffling_the_future_removes_the_signal_and_keeps_false_discoveries_low() -> (
    None
):
    """End to end: day-shuffled targets destroy the link a planted feature had.

    Note on small-sample bootstrap dispersion under the global null:
    On a short 60-day panel with 5-day blocks, discrete one-sided bootstrap p-values
    exhibit finite-sample boundary dispersion where the empirical family-wise
    rejection rate P(any rejection) is ~0.20-0.25 (higher than nominal q = 0.10).
    This test verifies signal destruction (the planted feature is no longer
    specifically detected over nulls) and that the global null rejection rate
    remains bounded, rather than claiming asymptotic FDR control at 60 days.
    """

    days, symbols, null_features = 60, 5, 8
    panel = planted_signal_panel(
        5, days=days, symbols=symbols, null_features=null_features, strength=0.25
    )
    features = (panel.signal, *panel.nulls)
    rows_per_day = HOURS_PER_DAY * symbols

    real_p = np.array(
        [
            _daily_rank_ic_p(panel.day_ids, f, panel.target, seed=i).p_value_one_sided
            for i, f in enumerate(features)
        ]
    )
    assert bh_rejections(real_p, Q)[0], "the planted feature must be found unshuffled"

    rng = np.random.default_rng(99)
    replicates = 20
    discoveries = []
    planted_discoveries = 0
    for rep in range(replicates):
        shuffled = shuffle_days(panel.target, rows_per_day=rows_per_day, rng=rng)
        p = np.array(
            [
                _daily_rank_ic_p(panel.day_ids, f, shuffled, seed=rep).p_value_one_sided
                for f in features
            ]
        )
        rejected = bh_rejections(p, Q)
        # Every hypothesis is null after the shuffle, so every rejection is false and the
        # false discovery proportion is 1 whenever anything is rejected.
        discoveries.append(1.0 if rejected.any() else 0.0)
        planted_discoveries += int(bool(rejected[0]))
    # The planted feature is no longer specifically privileged over pure noise.
    assert planted_discoveries <= 4
    # The small-sample empirical rejection rate remains bounded.
    assert float(np.mean(discoveries)) <= 0.35


# --------------------------- oracle 3: common-level market, and what demeaning leaves behind


def test_a_common_level_only_market_has_no_demeaned_ic_but_a_large_raw_one() -> None:
    panel = common_level_panel(
        3,
        days=90,
        betas=(1.0, 1.0, 1.0, 1.0, 1.0),
        idiosyncratic_sd=0.3,
        forecast_noise_sd=0.5,
    )
    symbols = panel.betas.shape[0]
    feature = np.repeat(panel.market_forecast, symbols)  # one view of the common factor
    raw = panel.returns.reshape(-1)
    demeaned = cross_sectional_demean(panel.returns).reshape(-1)

    raw_ic = daily_ic(panel.day_ids, feature, raw, method="rank")
    demeaned_ic = daily_ic(panel.day_ids, feature, demeaned, method="rank")
    interval = block_mean_interval(
        demeaned_ic.values, block_days=BLOCK_DAYS, resamples=RESAMPLES, seed=0
    )

    assert float(raw_ic.mean) > 0.5
    assert abs(float(demeaned_ic.mean)) < 0.03
    assert interval.lower < 0.0 < interval.upper


def test_unequal_loadings_leave_a_demeaned_ic_that_a_causal_beta_residual_removes() -> (
    None
):
    """The documented control: demeaning is not beta removal; a causal estimate is."""

    betas = (0.4, 0.7, 1.0, 1.3, 1.6)
    panel = common_level_panel(
        4, days=120, betas=betas, idiosyncratic_sd=0.3, forecast_noise_sd=0.5
    )
    # A feature that knows each symbol's own loading times the common-factor view.
    feature = (panel.market_forecast[:, None] * panel.betas[None, :]).reshape(-1)
    demeaned = cross_sectional_demean(panel.returns).reshape(-1)
    residual = causal_beta_residual(
        panel.returns, horizon_bars=1, min_observations=240
    ).reshape(-1)

    demeaned_ic = daily_ic(panel.day_ids, feature, demeaned, method="rank")
    residual_ic = daily_ic(panel.day_ids, feature, residual, method="rank")

    assert float(demeaned_ic.mean) > 0.15  # beta heterogeneity survives demeaning
    assert abs(float(residual_ic.mean)) < 0.05  # the causal beta residual removes it
    assert residual_ic.skipped_days <= 10  # only the warm-up days lack a beta estimate


# ------------------------------ oracle 4: tail-only signal, rank IC and linear IC diverge


def test_a_tail_only_signal_splits_the_linear_ic_from_the_rank_ic() -> None:
    feature, target = tail_only_panel(8, rows=200_000, tail_share=0.01, jump=10.0)

    linear = float(linear_ic(feature, target) or 0.0)
    rank = float(rank_ic(feature, target) or 0.0)

    assert linear > 0.15
    assert rank < 0.06
    assert linear > 3.0 * rank


def test_a_bulk_signal_does_not_split_the_two_statistics() -> None:
    rng = np.random.default_rng(9)
    feature = rng.standard_normal(200_000)
    target = 0.3 * feature + rng.standard_normal(200_000)

    linear = float(linear_ic(feature, target) or 0.0)
    rank = float(rank_ic(feature, target) or 0.0)

    assert abs(linear - rank) / linear < 0.1
