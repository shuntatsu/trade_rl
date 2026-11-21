"""Oracles (G2) of ``xs_lowvol_funding_v1`` computed by hand, not through production sizing.

Pure policy arithmetic, the point-in-time universe, the decision rule and the protocol
bindings. The engine-level oracles (forced close, funding sign, causality, embargo, INVALID)
live in ``test_xs_lowvol_funding_engine.py``.
"""

from __future__ import annotations

import math
import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from trade_rl.evaluation.trend_oos.decision import OosCell
from trade_rl.evaluation.xs_lowvol_funding import (
    data,
    decision,
    policy,
    protocol,
    universe,
)

ROOT = Path(__file__).resolve().parents[2]

# ================================================================ scores and legs


def test_zr_of_distinct_values_is_the_centred_rank_over_the_population_std() -> None:
    # ranks of [0.3, 0.1, 0.2] are [3, 1, 2]; mean 2, population std sqrt(2/3)
    expected = np.array([1.0, -1.0, 0.0]) / math.sqrt(2.0 / 3.0)
    assert np.allclose(policy.zr(np.array([0.3, 0.1, 0.2])), expected, atol=1e-12)


def test_zr_gives_tied_values_the_mean_of_their_ranks_and_a_constant_panel_zero() -> (
    None
):
    # [1, 1, 2]: ranks [1.5, 1.5, 3], mean 2, deviations [-0.5, -0.5, 1]
    std = math.sqrt((0.25 + 0.25 + 1.0) / 3.0)
    assert np.allclose(
        policy.zr(np.array([1.0, 1.0, 2.0])),
        np.array([-0.5, -0.5, 1.0]) / std,
        atol=1e-12,
    )
    assert np.all(policy.zr(np.array([5.0, 5.0, 5.0])) == 0.0)


def test_six_name_panel_scores_legs_and_inverse_volatility_weights_by_hand() -> None:
    sigma = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06])
    funding = np.array([3.0, -1.0, 2.0, 1.0, -2.0, 4.0]) * 1e-4
    # zr(-sigma): ranks of -sigma are [6, 5, 4, 3, 2, 1];
    # zr(-f): -f = [-3, 1, -2, -1, 2, -4] -> ranks [2, 5, 3, 4, 6, 1]; both over sqrt(35/12)
    spread = math.sqrt(35.0 / 12.0)
    z_vol = (np.array([6, 5, 4, 3, 2, 1]) - 3.5) / spread
    z_fund = (np.array([2, 5, 3, 4, 6, 1]) - 3.5) / spread
    assert np.allclose(policy.zr(-sigma), z_vol, atol=1e-12)
    assert np.allclose(policy.zr(-funding), z_fund, atol=1e-12)
    # score = (rank sum - 7) / (2 spread) = [1, 3, 0, 0, 1, -5] / (2 spread)
    score = policy.score_of(protocol.CANDIDATE, sigma, funding)
    hand = np.array([1.0, 3.0, 0.0, 0.0, 1.0, -5.0]) / (2.0 * spread)
    assert np.allclose(score, hand, atol=1e-12)
    # k = floor(6 / 3) = 2; the ties 0 / 4 (score 1) and 2 / 3 (score 0) break by position
    long_index, short_index = policy.select_legs(score, 6 // protocol.LEG_DIVISOR)
    assert list(long_index) == [1, 0]
    assert list(short_index) == [5, 2]
    # w = +-1 / (k sigma): longs 1/(2*0.01) = 50, 1/(2*0.02) = 25; shorts -1/(2*0.06), -1/(2*0.03)
    weights = policy.leg_weights(sigma, long_index, short_index)
    expected = np.array([50.0, 25.0, -1.0 / 0.06, 0.0, 0.0, -1.0 / 0.12])
    assert np.allclose(weights, expected, atol=1e-9)
    # each leg has sum |w| * sigma = 1
    assert math.isclose(float(np.sum(np.abs(weights[[0, 1]]) * sigma[[0, 1]])), 1.0)
    assert math.isclose(float(np.sum(np.abs(weights[[2, 5]]) * sigma[[2, 5]])), 1.0)


def test_ablation_scores_use_one_factor_each() -> None:
    sigma = np.array([0.03, 0.01, 0.02])
    funding = np.array([1.0, 2.0, 3.0]) * 1e-4
    spread = math.sqrt(2.0 / 3.0)
    assert np.allclose(
        policy.score_of(protocol.ARM_LOWVOL, sigma, funding),
        np.array([-1.0, 1.0, 0.0]) / spread,
    )
    assert np.allclose(
        policy.score_of(protocol.ARM_FUNDING, sigma, funding),
        np.array([1.0, 0.0, -1.0]) / spread,
    )
    with pytest.raises(ValueError):
        policy.score_of(protocol.ARM_EW_LONG, sigma, funding)


def test_select_legs_needs_two_k_names() -> None:
    with pytest.raises(ValueError):
        policy.select_legs(np.array([1.0, 2.0, 3.0]), 2)
    with pytest.raises(ValueError):
        policy.select_legs(np.array([1.0, 2.0, 3.0]), 0)


# ======================================================= volatility target and caps


def test_volatility_target_scale_matches_an_exact_two_name_covariance() -> None:
    a = np.array([0.01, -0.01, 0.01, -0.01])
    b = np.array([0.02, 0.02, -0.01, -0.03])
    returns = np.column_stack([a, b])
    # means are 0, n - 1 = 3: var(a) = 4e-4 / 3, var(b) = 18e-4 / 3 = 6e-4,
    # cov = (2e-4 - 2e-4 - 1e-4 + 3e-4) / 3 = 2e-4 / 3.
    var_a, var_b, cov = 4e-4 / 3.0, 6e-4, 2e-4 / 3.0
    # shrink 50/50 to the diagonal halves the covariance; annualise with 2190 bars
    annual = np.array([[var_a, cov / 2.0], [cov / 2.0, var_b]]) * 2190.0
    assert np.allclose(annual, [[0.292, 0.073], [0.073, 1.314]], atol=1e-12)
    weights = np.array([1.5, -0.5])
    variance = 1.5**2 * 0.292 + 0.5**2 * 1.314 + 2.0 * 1.5 * (-0.5) * 0.073
    assert math.isclose(variance, 0.876)
    expected = 0.20 / math.sqrt(0.876)
    assert math.isclose(
        policy.vol_target_scale(weights, returns), expected, rel_tol=1e-12
    )
    assert np.allclose(policy.shrunk_annual_covariance(returns), annual, atol=1e-12)
    # the scaled book has exactly the target volatility
    scaled = weights * policy.vol_target_scale(weights, returns)
    assert math.isclose(math.sqrt(scaled @ annual @ scaled), 0.20, rel_tol=1e-12)


def test_vol_target_of_a_riskless_book_is_zero() -> None:
    assert policy.vol_target_scale(np.array([1.0, -1.0]), np.zeros((5, 2))) == 0.0


def test_caps_apply_per_name_first_then_gross_proportionally() -> None:
    capped = policy.cap_weights(np.array([0.4, -0.4, 0.3, -0.1]))
    # per-name 0.25 -> [0.25, -0.25, 0.25, -0.1]; gross 0.85 is under 2.0
    assert np.allclose(capped, [0.25, -0.25, 0.25, -0.1])
    many = policy.cap_weights(np.full(12, 0.25))
    # per-name leaves 0.25 each (gross 3.0 > 2.0): every weight is scaled by 2/3
    assert np.allclose(many, 0.25 * 2.0 / 3.0)
    assert math.isclose(float(np.abs(many).sum()), 2.0)


def test_a_microscopic_real_buy_at_cash_limit_is_not_treated_as_a_hold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default NumPy relative allclose must not hide executable cap violations."""
    inputs = _alternating_inputs()
    current = np.full(6, protocol.NET_EXPOSURE_LIMIT / 6)
    proposed = current + 2e-7
    monkeypatch.setattr(
        policy,
        "book_target",
        lambda *args, **kwargs: (proposed.copy(), 6, 0),
    )
    snapshot = SimpleNamespace(
        symbols=inputs.legs,
        index=399,
        account=SimpleNamespace(
            weights=tuple(current),
            portfolio_value=100_000.0,
            peak_value=100_000.0,
            max_drawdown=0.0,
        ),
        risk=SimpleNamespace(drawdown_start=0.125, drawdown_stop=0.25),
    )
    candidate = policy.XsPolicy(
        arm=protocol.ARM_EW_LONG,
        inputs=inputs,
        start_index=399,
    )
    result = candidate.decide_portfolio(snapshot)  # type: ignore[arg-type]
    assert candidate.stats.bound_by_net_exposure == 1
    assert result.hold_current_quantity_symbols == ()
    assert sum(result.target_weights) <= protocol.NET_EXPOSURE_LIMIT + 1e-9


def test_cash_account_scales_by_the_tighter_of_margin_and_net_exposure() -> None:
    # equal-weight long, gross 1.0: margin 0.4 (fine), net 1.0 -> 0.97 / 1.0
    factor, margin, net = policy.cash_factors(np.full(6, 1.0 / 6.0))
    assert margin == 1.0 and math.isclose(net, 0.97) and math.isclose(factor, 0.97)
    # long 1.5 and 1.5: gross 3.0 -> margin 1.2 > 1 -> 1/1.2; net 3.0 -> 0.97/3.0 tighter
    factor, margin, net = policy.cash_factors(np.array([1.5, 1.5]))
    assert math.isclose(margin, 1.0 / 1.2) and math.isclose(net, 0.97 / 3.0)
    assert math.isclose(factor, 0.97 / 3.0)
    # a long-short book with net 0.1 and gross 2.0: margin 0.8, nothing binds
    assert policy.cash_factors(np.array([1.05, -0.95]))[0] == 1.0
    # a net short book is not bound by the net limit
    assert policy.cash_factors(np.array([0.2, -1.0]))[0] == 1.0
    assert math.isclose(policy.cash_measure(np.array([1.0, -1.0])), 0.8)


# ================================================================== funding mean


def test_funding_mean_is_the_mean_of_the_last_nine_events_up_to_the_row() -> None:
    rate = np.array([0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    count = (rate != 0.0).astype(np.int32)
    # events sit at rows 1, 3, 5, 6, ..., 12 (values 1, 2, 3, ..., 10); the last nine up to
    # row 12 are the values 2..10
    assert math.isclose(policy.funding_mean(rate, count, upto=12), 54.0 / 9.0)
    # up to row 11 the events are the nine values 1..9
    assert math.isclose(policy.funding_mean(rate, count, upto=11), 5.0)
    # up to row 10 only eight events exist
    assert math.isnan(policy.funding_mean(rate, count, upto=10))
    # a row after ``upto`` is never read
    changed = rate.copy()
    changed[12] = 1e6
    assert policy.funding_mean(changed, count, upto=11) == 5.0


def test_funding_mean_splits_a_bar_with_several_events_into_its_mean_rate() -> None:
    # five single events of 0.1 and one bar of four events summing to 0.4
    rate = np.array([0.1] * 5 + [0.0, 0.0, 0.4])
    count = np.array([1] * 5 + [0, 0, 4], dtype=np.int32)
    # the last nine events: the four of the last bar and rows 4, 3, 2, 1, 0
    assert math.isclose(policy.funding_mean(rate, count, upto=7), 0.1)
    # with an eight-event history the mean is NaN
    assert math.isnan(policy.funding_mean(rate[:7], count[:7], upto=6))


# ======================================================== end-to-end book on a panel


def _alternating_inputs(n_rows: int = 400) -> policy.XsInputs:
    """Six legs whose 4h log returns are ``s_i * e(r)`` and whose funding is constant."""
    step = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06])
    rate = np.array([3.0, -1.0, 2.0, 1.0, -2.0, 4.0]) * 1e-4
    sign = np.where(np.arange(n_rows) % 2 == 0, 1.0, -1.0)
    close = 100.0 * np.exp(np.cumsum(sign[:, None] * step[None, :], axis=0))
    ones = np.ones((n_rows, 6), dtype=np.bool_)
    return policy.XsInputs(
        legs=tuple(f"S{i}USDT" for i in range(6)),
        close=close,
        active=ones.copy(),
        tradable=ones.copy(),
        funding_rate=np.tile(rate, (n_rows, 1)),
        funding_count=np.ones((n_rows, 6), dtype=np.int64),
        member=ones.copy(),
        forced=np.zeros((n_rows, 6), dtype=np.bool_),
        last_bar=np.zeros((n_rows, 6), dtype=np.bool_),
    )


def test_candidate_book_equals_the_hand_formula_for_the_alternating_panel() -> None:
    inputs = _alternating_inputs()
    target, ranked, zeroed = policy.book_target(
        inputs, arm=protocol.CANDIDATE, decision_row=399, signal_row=399
    )
    assert (ranked, zeroed) == (6, 0)
    # the scores are those of the six-name panel above: long {1, 0}, short {5, 2}.
    # sigma_i = s_i sqrt(180 / 179) (90 up and 90 down returns, ddof 1), so
    # w_i = +-1 / (2 sigma_i) and w_i * s_i = +-0.5 sqrt(179 / 180) =: +-c.
    c = 0.5 * math.sqrt(179.0 / 180.0)
    # every leg's returns are s_i e(r): covariance = s s' * v with v = 360 / 359, the shrink
    # halves the off-diagonal, so  w' C w = 2190 v (0.5 (sum w s)^2 + 0.5 sum (w s)^2),
    # and sum w s = c (1 + 1 - 1 - 1) = 0, sum (w s)^2 = 4 c^2.
    variance = 2190.0 * (360.0 / 359.0) * 0.5 * 4.0 * c * c
    scale = 0.20 / math.sqrt(variance)
    sigma = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06]) * math.sqrt(180.0 / 179.0)
    raw = np.zeros(6)
    raw[[0, 1]] = 1.0 / (2.0 * sigma[[0, 1]])
    raw[[5, 2]] = -1.0 / (2.0 * sigma[[5, 2]])
    expected = np.clip(raw * scale, -0.25, 0.25)  # 0.3018 for leg 0 is clipped to 0.25
    assert abs(raw[0] * scale) > 0.25  # the per-name cap binds on leg 0
    assert np.allclose(target, expected, atol=1e-12)
    assert target[3] == 0.0 and target[4] == 0.0
    assert float(np.abs(target).sum()) < protocol.GROSS_CAP


def test_equal_weight_control_and_the_cash_arm() -> None:
    inputs = _alternating_inputs()
    target, ranked, _ = policy.book_target(
        inputs, arm=protocol.ARM_EW_LONG, decision_row=399, signal_row=399
    )
    assert ranked == 6 and np.allclose(target, 1.0 / 6.0)
    flat, ranked, _ = policy.book_target(
        inputs, arm=protocol.CONTROL_CASH, decision_row=399, signal_row=399
    )
    assert not flat.any() and ranked == 0


def test_a_name_without_nine_funding_events_or_history_is_excluded() -> None:
    inputs = _alternating_inputs()
    inputs.funding_count[:, 2] = 0  # leg 2 never settled a funding event
    inputs.active[100, 4] = False  # leg 4 has a hole inside the 361-close window
    target, ranked, _ = policy.book_target(
        inputs, arm=protocol.CANDIDATE, decision_row=399, signal_row=399
    )
    assert ranked == 4  # k = floor(4 / 3) = 1: one long, one short
    assert target[2] == 0.0 and target[4] == 0.0
    assert np.count_nonzero(target) == 2
    # fewer than three usable names leave the book flat
    inputs.funding_count[:, 1] = 0
    flat, ranked, _ = policy.book_target(
        inputs, arm=protocol.CANDIDATE, decision_row=399, signal_row=399
    )
    assert ranked == 3 and np.count_nonzero(flat) == 2
    inputs.funding_count[:, 0] = 0
    flat, ranked, _ = policy.book_target(
        inputs, arm=protocol.CANDIDATE, decision_row=399, signal_row=399
    )
    assert ranked == 2 and not flat.any()


def test_forced_close_rows_start_42_bars_before_the_last_available_bar() -> None:
    active = np.ones((100, 2), dtype=np.bool_)
    active[61:, 0] = False  # leg 0's last available bar is row 60
    forced = policy.XsInputs.forced_rows(active)
    assert list(np.flatnonzero(forced[:, 0])) == list(range(18, 61))
    assert list(np.flatnonzero(policy.XsInputs.last_bar_rows(active)[:, 0])) == [60]
    assert not forced[:, 1].any()  # a leg cut off by the dataset end is never flagged
    # a forced name is zeroed after sizing, wherever it ranks
    inputs = _alternating_inputs()
    inputs.forced[399, 1] = True
    target, _, zeroed = policy.book_target(
        inputs, arm=protocol.CANDIDATE, decision_row=399, signal_row=399
    )
    assert target[1] == 0.0 and zeroed == 1


# ===================================================================== universe


def _series(
    n_hours: int,
    volume: float,
    *,
    hole: int | None = None,
    events: int = 1_000,
    spike_from: int | None = None,
) -> universe.UniverseSeries:
    complete = np.ones(n_hours, dtype=np.bool_)
    if hole is not None:
        complete[hole] = False
    quote = np.full(n_hours, volume)
    if spike_from is not None:
        quote[spike_from:] = 1e9
    grid_ms = int(GRID.timestamp() * 1_000)
    stamps = grid_ms + np.arange(0, n_hours, 8, dtype=np.int64) * 3_600_000
    return universe.UniverseSeries(
        complete=complete, quote_volume=quote, funding_ms=stamps[:events]
    )


GRID = datetime(2021, 1, 1, tzinfo=UTC)
MONTH = datetime(2021, 4, 1, tzinfo=UTC)  # 90 days after the grid start
MAY = datetime(2021, 5, 1, tzinfo=UTC)
HOURS = 24 * 150


def test_the_top_30_is_by_trailing_volume_and_a_spike_inside_the_month_is_ignored() -> (
    None
):
    panel = {f"S{i:02d}USDT": _series(HOURS, float(i + 5)) for i in range(33)}
    # the spike symbol has the smallest volume before April and a huge one from April on
    month_hour = universe.hour_index(MONTH, grid_start=GRID)
    panel["SPIKEUSDT"] = _series(HOURS, 0.5, spike_from=month_hour)
    result = universe.select_universe(panel, month_start=MONTH, grid_start=GRID)
    assert result.book and len(result.members) == 30
    assert "SPIKEUSDT" not in result.members
    # the 30 largest of volumes 5..37 are 8..37, i.e. symbols S03..S32
    assert result.members == tuple(
        sorted((f"S{i:02d}USDT" for i in range(3, 33)), key=lambda s: (-int(s[1:3]), s))
    )
    # the next month sees April's rows: the spike symbol now leads
    next_start = datetime(2021, 5, 1, tzinfo=UTC)
    later = universe.select_universe(panel, month_start=next_start, grid_start=GRID)
    assert later.members[0] == "SPIKEUSDT"


def test_eligibility_needs_a_complete_90_days_and_nine_funding_events() -> None:
    month_hour = universe.hour_index(MAY, grid_start=GRID)  # 2880
    panel = {f"S{i:02d}USDT": _series(HOURS, 10.0 + i) for i in range(13)}
    panel["HOLEUSDT"] = _series(HOURS, 99.0, hole=month_hour - 100)
    # the hour just before the 90-day window does not count
    panel["OLDHOLEUSDT"] = _series(HOURS, 98.0, hole=month_hour - 24 * 90 - 1)
    # events every 8 hours: the last nine before May 1 are hours 2808..2872; dropping the
    # last one leaves eight in the three days before the month
    panel["FEWFUNDUSDT"] = _series(HOURS, 97.0, events=month_hour // 8 - 1)
    result = universe.select_universe(panel, month_start=MAY, grid_start=GRID)
    assert "HOLEUSDT" not in result.eligible  # a hole inside the 90 days
    assert "OLDHOLEUSDT" in result.eligible
    assert "FEWFUNDUSDT" not in result.eligible  # fewer than nine events in three days
    assert result.book and len(result.eligible) == 14


def test_a_month_with_fewer_than_twelve_eligible_symbols_holds_no_book() -> None:
    panel = {f"S{i:02d}USDT": _series(HOURS, 10.0 + i) for i in range(11)}
    result = universe.select_universe(panel, month_start=MONTH, grid_start=GRID)
    assert len(result.eligible) == 11 and not result.book and result.members == ()
    panel["S11USDT"] = _series(HOURS, 50.0)
    assert universe.select_universe(panel, month_start=MONTH, grid_start=GRID).book


def test_ties_in_volume_break_by_symbol_name() -> None:
    panel = {f"S{i:02d}USDT": _series(HOURS, 7.0) for i in range(32)}
    result = universe.select_universe(panel, month_start=MONTH, grid_start=GRID)
    assert result.members == tuple(f"S{i:02d}USDT" for i in range(30))


def test_a_month_that_starts_before_the_grid_or_after_the_series_is_refused() -> None:
    panel = {"AUSDT": _series(HOURS, 1.0)}
    with pytest.raises(universe.UniverseDataError):
        universe.select_universe(
            panel, month_start=datetime(2021, 2, 1, tzinfo=UTC), grid_start=GRID
        )
    with pytest.raises(universe.UniverseDataError):
        universe.select_universe(
            panel, month_start=datetime(2022, 1, 1, tzinfo=UTC), grid_start=GRID
        )


def test_membership_mask_follows_the_month_of_each_row_close() -> None:
    april = universe.MonthUniverse(MONTH, ("A", "B"), ("A", "B"), True)
    may = universe.MonthUniverse(
        datetime(2021, 5, 1, tzinfo=UTC), ("B", "C"), ("B", "C"), True
    )
    clock = np.array(
        ["2021-04-30T20:00", "2021-05-01T00:00", "2021-05-01T04:00"],
        dtype="datetime64[ns]",
    )
    mask = universe.membership_mask(
        {MONTH: april, may.month_start: may}, clock, ("A", "B", "C")
    )
    assert mask.tolist() == [
        [True, True, False],
        [False, True, True],
        [False, True, True],
    ]
    assert universe.ever_members({MONTH: april, may.month_start: may}) == (
        "A",
        "B",
        "C",
    )


def test_stage_months_cover_every_month_with_a_weekly_decision() -> None:
    months = universe.months_of_stage(
        datetime(2021, 4, 5, tzinfo=UTC), datetime(2021, 7, 26, tzinfo=UTC)
    )
    assert [m.month for m in months] == [4, 5, 6, 7]


# ================================================================== decision rule


def _cell(
    arm: str,
    scenario: str,
    *,
    sharpe: float = 1.0,
    ret: float = 0.1,
    dd: float = 0.1,
    returns: tuple[float, ...] | None = None,
    halves: tuple[float, float] = (1.0, 1.0),
    valid: bool = True,
) -> OosCell:
    series = returns if returns is not None else tuple([0.002, 0.001] * 200)
    return OosCell(
        arm=arm,
        scenario=scenario,
        total_return=ret,
        annual_sharpe=sharpe,
        max_drawdown=dd,
        total_cost=10.0,
        funding_pnl=1.0,
        turnover_total=5.0,
        complete=True,
        terminal_flat=True,
        terminal_dust_only=False,
        execution_valid=True,
        causality_passed=True,
        ledger_reconciled=True,
        stopped=False,
        stop_reason=None,
        invalid_reasons=() if valid else ("negative_cash_borrowing",),
        n_returns=len(series),
        calendar_year_returns={"2021": 0.1},
        returns=series,
        cap_statistics={"half_sharpes": {"H1": halves[0], "H2": halves[1]}},
    )


def _grid(**candidate: object) -> dict[tuple[str, str], OosCell]:
    cells: dict[tuple[str, str], OosCell] = {}
    for arm in protocol.ARMS:
        for scenario in protocol.SCENARIOS:
            cells[(arm, scenario)] = _cell(arm, scenario)
    base = {
        (protocol.CANDIDATE, protocol.BASE_SCENARIO): candidate.get("base"),
        (protocol.CANDIDATE, protocol.HIGH_COST_SCENARIO): candidate.get("high"),
        (protocol.CANDIDATE, protocol.DELAY_SCENARIO): candidate.get("delay"),
    }
    for key, cell in base.items():
        if cell is not None:
            assert isinstance(cell, OosCell)
            cells[key] = cell
    return cells


def _base(**kwargs: object) -> OosCell:
    return _cell(protocol.CANDIDATE, protocol.BASE_SCENARIO, **kwargs)  # type: ignore[arg-type]


def test_all_six_conditions_hold_for_a_strong_candidate() -> None:
    outcome = decision.decide_disjoint(_grid())
    assert outcome["label"] == protocol.LABEL_SUPPORTED and outcome["supported"]
    assert all(outcome["conditions"].values()) and outcome["failed_conditions"] == []
    assert outcome["bootstrap"]["lower_bound"] > 0.0


@pytest.mark.parametrize(
    ("override", "failed"),
    [
        ({"base": _base(sharpe=0.59)}, "C2_base_sharpe_at_least_min"),
        ({"base": _base(dd=0.25)}, "C4_base_max_drawdown_below_limit"),
        ({"base": _base(halves=(1.0, 0.0))}, "C5_base_sharpe_positive_in_both_halves"),
        ({"base": _base(halves=(-0.1, 1.0))}, "C5_base_sharpe_positive_in_both_halves"),
        (
            {"base": _base(returns=tuple([-0.001, 0.0005] * 200))},
            "C6_bootstrap_lower_bound_positive",
        ),
        (
            {
                "high": _cell(
                    protocol.CANDIDATE, protocol.HIGH_COST_SCENARIO, sharpe=0.29
                )
            },
            "C3_high_cost_and_delay_sharpe",
        ),
        (
            {"delay": _cell(protocol.CANDIDATE, protocol.DELAY_SCENARIO, sharpe=0.39)},
            "C3_high_cost_and_delay_sharpe",
        ),
    ],
)
def test_each_condition_fails_alone_and_gives_not_supported(
    override: dict[str, OosCell], failed: str
) -> None:
    outcome = decision.decide_disjoint(_grid(**override))
    assert outcome["label"] == protocol.LABEL_NOT_SUPPORTED
    assert outcome["failed_conditions"] == [failed]


def test_the_thresholds_are_inclusive_where_the_spec_says_at_least() -> None:
    cells = _grid(
        base=_base(sharpe=0.6),
        high=_cell(protocol.CANDIDATE, protocol.HIGH_COST_SCENARIO, sharpe=0.3),
        delay=_cell(protocol.CANDIDATE, protocol.DELAY_SCENARIO, sharpe=0.4),
    )
    assert decision.decide_disjoint(cells)["label"] == protocol.LABEL_SUPPORTED


def test_any_invalid_cell_in_any_arm_makes_the_label_invalid() -> None:
    cells = _grid()
    cells[(protocol.ARM_EW_LONG, protocol.DELAY_SCENARIO)] = _cell(
        protocol.ARM_EW_LONG, protocol.DELAY_SCENARIO, valid=False
    )
    outcome = decision.decide_disjoint(cells)
    assert outcome["label"] == protocol.LABEL_INVALID
    assert outcome["invalid_cells"] == ["EW_LONG__DELAY"]
    assert not outcome["conditions"]["C1_no_invalid_cell"]


def test_the_grid_must_be_exactly_the_frozen_one() -> None:
    cells = _grid()
    del cells[(protocol.ARM_LOWVOL, protocol.BASE_SCENARIO)]
    with pytest.raises(ValueError):
        decision.decide_disjoint(cells)


def test_the_bootstrap_is_the_frozen_stationary_bootstrap_at_90_percent() -> None:
    boot = decision.bootstrap_lower_bound(_grid())
    assert boot["available"] and boot["gates"]
    assert boot["block_mean_bars"] == 42 and boot["resamples"] == 10_000
    assert boot["seed"] == 20261121 and boot["one_sided_confidence"] == 0.90
    assert boot["lower_bound"] < boot["mean"]
    cells = _grid(base=_base(returns=tuple([-0.001, 0.0005] * 200)))
    assert decision.bootstrap_lower_bound(cells)["lower_bound"] < 0.0


def test_forward_label_needs_a_supported_disjoint_label_and_its_own_conditions() -> (
    None
):
    cells = _grid()
    good = decision.decide_forward(cells, disjoint_label=protocol.LABEL_SUPPORTED)
    assert good["label"] == protocol.LABEL_FORWARD_SUPPORTED and good["supported"]
    not_supported = decision.decide_forward(
        cells, disjoint_label=protocol.LABEL_NOT_SUPPORTED
    )
    assert not_supported["label"] == protocol.LABEL_FORWARD_NOT_SUPPORTED
    weak = _grid(base=_base(sharpe=-0.1))
    assert (
        decision.decide_forward(weak, disjoint_label=protocol.LABEL_SUPPORTED)["label"]
        == protocol.LABEL_FORWARD_NOT_SUPPORTED
    )
    losing = _grid(
        high=_cell(protocol.CANDIDATE, protocol.HIGH_COST_SCENARIO, ret=-0.01)
    )
    assert (
        decision.decide_forward(losing, disjoint_label=protocol.LABEL_SUPPORTED)[
            "label"
        ]
        == protocol.LABEL_FORWARD_NOT_SUPPORTED
    )
    deep = _grid(base=_base(dd=0.25))
    assert (
        decision.decide_forward(deep, disjoint_label=protocol.LABEL_SUPPORTED)["label"]
        == protocol.LABEL_FORWARD_NOT_SUPPORTED
    )
    invalid = _grid()
    invalid[(protocol.CONTROL_CASH, protocol.BASE_SCENARIO)] = _cell(
        protocol.CONTROL_CASH, protocol.BASE_SCENARIO, valid=False
    )
    assert (
        decision.decide_forward(invalid, disjoint_label=protocol.LABEL_SUPPORTED)[
            "label"
        ]
        == protocol.LABEL_INVALID
    )
    with pytest.raises(ValueError):
        decision.decide_forward(cells, disjoint_label="NOPE")


def test_dev_and_interim_reports_carry_no_label_and_never_gate() -> None:
    dev = decision.dev_report(_grid())
    assert dev["label"] is None and dev["gates"] is False
    assert dev["status"] == decision.DEV_STATUS and dev["invalid_cells"] == []
    interim = decision.interim_report(_grid())
    assert interim["label"] is None and interim["status"] == protocol.LABEL_INTERIM
    for report in (dev, interim):
        assert "beta_to_ew_long" in report and "deflated_sharpe" in report
        assert report["deflated_sharpe"]["n_trials"] == 40
        assert report["deflated_sharpe"]["gates"] is False
        assert set(report["forced_close"]) == {
            f"{a}__{s}" for a in protocol.ARMS for s in protocol.SCENARIOS
        }


# ===================================================================== protocol


def test_costs_and_scenarios_are_the_spec_table() -> None:
    base = protocol.scenario_cost("BASE")
    assert (base.fee_rate, base.spread_rate, base.impact_rate) == (
        0.0005,
        0.0003,
        0.0002,
    )
    high = protocol.scenario_cost("HIGH_COST")
    assert (high.fee_rate, high.spread_rate, high.impact_rate) == (
        0.0005,
        0.0008,
        0.0006,
    )
    delay = protocol.scenario_cost("DELAY")
    assert (delay.fee_rate, delay.spread_rate, delay.impact_rate) == (
        0.0005,
        0.0003,
        0.0002,
    )
    assert [protocol.scenario_signal_lag_bars(s) for s in protocol.SCENARIOS] == [
        0,
        0,
        1,
    ]
    for name in protocol.SCENARIOS:
        cost = protocol.scenario_cost(name)
        assert cost.max_participation_rate == 0.01 and cost.minimum_notional == 10.0
        assert cost.processing_bar_volume_capacity is False
        assert cost.maintenance_margin_rate == 0.05 and cost.allow_short
    with pytest.raises(ValueError):
        protocol.scenario_cost("LOW_COST")


def test_the_risk_config_is_the_spec_one() -> None:
    risk = protocol.risk_config()
    assert (risk.max_gross, risk.max_abs_weight, risk.max_turnover) == (2.0, 0.25, 2.5)
    assert (risk.drawdown_start, risk.drawdown_stop) == (0.125, 0.25)
    assert risk.emergency_turnover_override is True
    protocol.verify_margin_binding()


def test_stage_windows_are_the_spec_table_and_decisions_are_mondays() -> None:
    expected = {
        "DEV": ("2022-01-01", "2022-04-04", "2026-07-27", "2026-08-01"),
        "DISJOINT": ("2021-01-01", "2021-04-05", "2026-07-27", "2026-08-01"),
        "FORWARD": ("2026-04-01", "2026-08-03", "2027-04-26", "2027-05-01"),
    }
    for stage, dates in expected.items():
        window = data.frozen_window(stage)
        assert (
            tuple(
                moment.date().isoformat()
                for moment in (
                    window.load_start,
                    window.first_decision,
                    window.last_decision_bar,
                    window.settles,
                )
            )
            == dates
        )
        assert window.first_decision.weekday() == 0
        assert window.last_decision_bar.weekday() == 0
        assert window.first_decision_index >= protocol.COV_BARS
    assert protocol.FINAL_NOT_BEFORE == "2027-05-08T00:00:00Z"
    assert protocol.AUTHORIZED_STAGES == ("DISJOINT", "FORWARD")


def test_the_frozen_candidate_list_is_bound_by_its_sha256(tmp_path: Path) -> None:
    candidates = protocol.load_candidates()
    assert len(candidates) == 441 and len(set(candidates)) == 441
    touched = {
        "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT",
        "AVAXUSDT", "LINKUSDT", "LTCUSDT", "DOTUSDT", "BCHUSDT", "SUIUSDT", "APTUSDT",
        "ARBUSDT", "OPUSDT", "TRXUSDT",
    }  # fmt: skip
    assert not touched & set(candidates)
    assert protocol.CANDIDATES_SHA256.startswith("fc680208")
    edited = tmp_path / "candidates.json"
    edited.write_bytes(protocol.candidates_path().read_bytes() + b" ")
    with pytest.raises(protocol.ProtocolBindingError):
        protocol.verify_candidates_binding(edited)
    with pytest.raises(protocol.ProtocolBindingError):
        protocol.verify_candidates_binding(tmp_path / "missing.json")


def test_forward_listing_exclusions() -> None:
    assert protocol.forward_symbol_allowed("BTCUSDT")
    assert protocol.forward_symbol_allowed(
        "1000PEPEUSDT"
    )  # perp-only: price scale is moot
    assert protocol.forward_symbol_allowed("1MBABYDOGEUSDT")
    assert not protocol.forward_symbol_allowed("币安人生USDT")
    assert not protocol.forward_symbol_allowed("btcUSDT")
    assert not protocol.forward_symbol_allowed("USDCUSDT")
    assert not protocol.forward_symbol_allowed("PAXGUSDT")
    assert not protocol.forward_symbol_allowed("BTCBUSD")
    assert not protocol.forward_symbol_allowed("USDT")


def test_amendment_1_declares_no_halt_and_pins_its_coverage_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert protocol.DECLARED_HALTS == ()
    assert protocol.COVERAGE_EVIDENCE_SHA256 == (
        "64149a6ff2e3770449481e048896d832d6e5eb558c17b950b76de0a06c25d594"
    )
    assert protocol.declared_halt_intervals() == ()
    monkeypatch.setattr(protocol, "COVERAGE_EVIDENCE_SHA256", None)
    monkeypatch.setattr(
        protocol, "DECLARED_HALTS", (("2021-05-19T10:00:00Z", "2021-05-19T12:00:00Z"),)
    )
    with pytest.raises(protocol.ProtocolBindingError):
        protocol.declared_halt_intervals()
    monkeypatch.setattr(protocol, "COVERAGE_EVIDENCE_SHA256", "a" * 64)
    ((lo, hi),) = protocol.declared_halt_intervals()
    assert hi - lo == timedelta(hours=2)


# The pinned protocol digest (a change to the constants or to any harness source is a new
# protocol and must be re-pinned here and in the spec).
PINNED_PROTOCOL_DIGEST = (
    "e9193110264427104e63bbe17f40dca35255ab25240394834a8916a9b5057487"
)


def test_the_protocol_digest_is_pinned_and_binds_the_harness_source() -> None:
    assert protocol.protocol_digest() == PINNED_PROTOCOL_DIGEST


def test_the_harness_digest_changes_with_any_harness_file(tmp_path: Path) -> None:
    root = tmp_path
    package = root / "pkg"
    package.mkdir()
    source = Path(protocol.__file__).resolve().parent
    for name in protocol.HARNESS_PACKAGE_FILES:
        shutil.copyfile(source / name, package / name)
    tool = root / "tool.py"
    tool.write_bytes((ROOT / "tools" / "xs_lowvol_funding.py").read_bytes())
    reference = protocol.harness_source_digest(package_dir=package, tool_path=tool)
    assert reference == protocol.harness_source_digest()
    # CRLF checkouts hash like LF ones
    crlf = root / "tool_crlf.py"
    crlf.write_bytes(tool.read_bytes().replace(b"\n", b"\r\n"))
    assert (
        protocol.harness_source_digest(package_dir=package, tool_path=crlf) == reference
    )
    for name in protocol.HARNESS_PACKAGE_FILES:
        path = package / name
        original = path.read_bytes()
        path.write_bytes(original + b"# edit\n")
        assert (
            protocol.harness_source_digest(package_dir=package, tool_path=tool)
            != reference
        )
        path.write_bytes(original)
    tool.write_bytes(tool.read_bytes() + b"# edit\n")
    assert (
        protocol.harness_source_digest(package_dir=package, tool_path=tool) != reference
    )
    tool.unlink()
    with pytest.raises(protocol.ProtocolBindingError):
        protocol.harness_source_digest(package_dir=package, tool_path=tool)


def test_the_payload_binds_candidates_costs_and_the_decision_rule() -> None:
    payload = protocol.protocol_payload()
    assert payload["protocol_id"] == "xs_lowvol_funding_v1"
    assert payload["candidates"]["sha256"] == protocol.CANDIDATES_SHA256  # type: ignore[index]
    text = str(payload)
    for fragment in ("C6", "0.0008", "0.0006", "20261121", "forced_close"):
        assert fragment in text
