"""Synthetic tests of the low-turnover family screen (``lowturn_family_screen_v1``).

Nothing here reads market data, the network or an economic result: every dataset is a
seeded synthetic fixture and every fixture value is hand-computed.
"""

from __future__ import annotations

import io
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tests.data.test_mtf6_signals import _make_alternating_dataset
from tests.evaluation import test_core7_mtf_tierd as tierd_tests
from tools import core7_g4_comment as g4
from tools import lowturn_screen as driver
from trade_rl.data.features import mtf6_signals
from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.core7_mtf_study import protocol as core7_protocol
from trade_rl.evaluation.core7_mtf_study import runner as core7_runner
from trade_rl.evaluation.lowturn_screen import policies, protocol, runner
from trade_rl.evaluation.lowturn_screen.decision import (
    ScreenCell,
    decide,
    deflated_sharpe_ratio,
)
from trade_rl.evaluation.lowturn_screen.policies import (
    LowturnPolicy,
    apply_caps,
    family_targets,
    risk_scale_of,
)
from trade_rl.evaluation.lowturn_screen.signals import (
    ROW_FIELDS,
    LowturnSignals,
    build_signals,
    signal_row,
)
from trade_rl.evaluation.replay import run_shared_portfolio_replay
from trade_rl.risk.pretrade import PreTradeRisk
from trade_rl.strategies.portfolio import (
    PortfolioAccountSnapshot,
    PortfolioDecisionSnapshot,
    PortfolioRiskLimits,
)
from trade_rl.strategies.rl.shadow_anchor_mtf import ShadowAnchorMtfPolicy

N_LEGS = 22
ALL_PERP = np.arange(11) * 2 + 1
ALL_SPOT = np.arange(11) * 2
# 1200 bars of 4h before 2024-07-01: bar index 1200 is the first with timestamp >= split
FIXTURE_START = "2023-12-13T00:00:00"
FIXTURE_BARS = 1400


def _fixture_dataset(seed: int = 5, **kwargs: Any) -> MarketDataset:
    return _make_alternating_dataset(
        n_bars=FIXTURE_BARS,
        pairs=protocol.ROSTER,
        seed=seed,
        start_iso=FIXTURE_START,
        **kwargs,
    )


# ----------------------------------------------------------------- hand-built signals


def _signals(rows: int = 2, **over: np.ndarray) -> LowturnSignals:
    shape = (rows, N_LEGS)
    arrays: dict[str, np.ndarray] = {
        name: np.full(shape, np.nan) for name in ROW_FIELDS if name != "usable"
    }
    arrays["usable"] = np.ones(shape, dtype=bool)
    arrays.update(over)
    return LowturnSignals(
        pair_names=protocol.ROSTER, symbols=protocol.LEG_ROSTER, **arrays
    )


def _grid(rows: int = 2, value: float = np.nan) -> np.ndarray:
    return np.full((rows, N_LEGS), value)


def _perp(pair: int) -> int:
    return 2 * pair + 1


def _spot(pair: int) -> int:
    return 2 * pair


def _expected(entries: dict[int, float]) -> np.ndarray:
    out = np.zeros(N_LEGS)
    for leg, weight in entries.items():
        out[leg] = weight
    return out


def test_leg_layout_is_alternating_spot_then_perp_in_roster_order() -> None:
    assert protocol.LEG_ROSTER[0] == "BTCUSDT:spot"
    assert protocol.LEG_ROSTER[1] == "BTCUSDT:perp"
    assert protocol.LEG_ROSTER[2] == "ETHUSDT:spot"
    assert len(protocol.LEG_ROSTER) == N_LEGS
    for pair, name in enumerate(protocol.ROSTER):
        assert protocol.LEG_ROSTER[_spot(pair)] == f"{name}:spot"
        assert protocol.LEG_ROSTER[_perp(pair)] == f"{name}:perp"


def test_ts_trend_ls_weights_are_hand_computed() -> None:
    vol = _grid()
    vol[0, :] = 0.5  # risk weight 0.50 / 0.5 = 1.0
    r30, r90, r180 = _grid(), _grid(), _grid()
    # pair0: (+,+,-) -> 1/3; pair1: (+,+,+) -> 1; pair2: (-,-,-) -> -1
    # pair3: (0,+,+) -> 2/3 (sign(0)=0); pair4: r30 unavailable -> weight 0
    for pair, signs in {
        0: (1, 1, -1),
        1: (1, 1, 1),
        2: (-1, -1, -1),
        3: (0, 1, 1),
    }.items():
        for arr, sign in zip((r30, r90, r180), signs, strict=True):
            arr[0, _perp(pair)] = 0.1 * sign
    r90[0, _perp(4)] = 0.1
    r180[0, _perp(4)] = 0.1
    sig = _signals(vol60=vol, r30=r30, r90=r90, r180=r180)

    got = family_targets("TS_TREND_LS", sig, 0)

    expected = _expected(
        {
            _perp(0): (1 / 3) / 11,
            _perp(1): 1 / 11,
            _perp(2): -1 / 11,
            _perp(3): (2 / 3) / 11,
        }
    )
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-15)
    assert np.all(got[ALL_SPOT] == 0.0)  # the unused leg of every pair is 0


def test_ts_trend_long_spot_clips_the_strength_to_zero_one_on_spot_legs() -> None:
    vol = _grid()
    vol[0, :] = 0.5
    r30, r90, r180 = _grid(), _grid(), _grid()
    for pair, signs in {0: (1, 1, 1), 1: (-1, -1, -1), 2: (1, 1, -1)}.items():
        for arr, sign in zip((r30, r90, r180), signs, strict=True):
            arr[0, _spot(pair)] = 0.1 * sign
            arr[0, _perp(pair)] = 0.1 * sign  # perp inputs must be ignored
    sig = _signals(vol60=vol, r30=r30, r90=r90, r180=r180)

    got = family_targets("TS_TREND_LONG_SPOT", sig, 0)

    np.testing.assert_allclose(
        got, _expected({_spot(0): 1 / 11, _spot(2): (1 / 3) / 11}), atol=1e-15
    )
    assert np.all(got[ALL_PERP] == 0.0)
    assert np.all(got >= 0.0)  # long-only: the bearish pair1 is 0, not a short


def test_ts_trend_btceth_trades_only_btc_and_eth_perps_with_divide_by_two() -> None:
    vol = _grid()
    vol[0, :] = 1.0  # risk weight 0.5
    r30, r90, r180 = _grid(), _grid(), _grid()
    for pair, signs in {0: (1, 1, 1), 1: (1, -1, -1), 2: (1, 1, 1)}.items():
        for arr, sign in zip((r30, r90, r180), signs, strict=True):
            arr[0, _perp(pair)] = 0.1 * sign
    sig = _signals(vol60=vol, r30=r30, r90=r90, r180=r180)

    got = family_targets("TS_TREND_BTCETH", sig, 0)

    # BTC: s=1 -> 0.5 / 2 = 0.25; ETH: s=-1/3 -> -(1/3) * 0.5 / 2; BNB is not traded
    np.testing.assert_allclose(
        got, _expected({_perp(0): 0.25, _perp(1): -(1 / 3) * 0.25}), atol=1e-15
    )


def test_xs_mom_goes_long_top_three_and_short_bottom_three() -> None:
    vol = _grid()
    vol[0, :] = 0.5
    score = _grid()
    # pair -> score; top 3: pairs 10, 2, 6 ; bottom 3: pairs 9, 3, 5
    values = [0.05, 0.03, 0.09, 0.01, 0.07, 0.02, 0.08, 0.04, 0.06, 0.0, 0.10]
    for pair, value in enumerate(values):
        score[0, _perp(pair)] = value
    sig = _signals(vol60=vol, xs_score=score)

    got = family_targets("XS_MOM", sig, 0)

    side = 1 / 6
    np.testing.assert_allclose(
        got,
        _expected(
            {_perp(10): side, _perp(2): side, _perp(6): side}
            | {_perp(9): -side, _perp(3): -side, _perp(5): -side}
        ),
        atol=1e-15,
    )
    assert float(np.abs(got).sum()) == pytest.approx(1.0)


def test_xs_mom_needs_six_available_names_and_breaks_ties_by_pair_index() -> None:
    vol = _grid()
    vol[0, :] = 0.5
    score = _grid()
    for pair in range(5):
        score[0, _perp(pair)] = 0.01 * pair
    assert not family_targets("XS_MOM", _signals(vol60=vol, xs_score=score), 0).any()
    # six names, all tied: top = the first three pairs, bottom = the next three
    score[0, _perp(5)] = 0.0
    tied = _grid()
    for pair in range(6):
        tied[0, _perp(pair)] = 0.02
    got = family_targets("XS_MOM", _signals(vol60=vol, xs_score=tied), 0)
    side = 1 / 6
    np.testing.assert_allclose(
        got,
        _expected(
            {_perp(0): side, _perp(1): side, _perp(2): side}
            | {_perp(3): -side, _perp(4): -side, _perp(5): -side}
        ),
        atol=1e-15,
    )


# the frozen 0.40 per-name cap applies to every family, so the 0.5 MA weight is clipped
MA_CAPPED = 0.40


def test_ma_btceth_spot_switches_on_sma180_above_sma900_per_asset() -> None:
    fast, slow = _grid(), _grid()
    fast[0, _spot(0)], slow[0, _spot(0)] = 2.0, 1.0  # BTC: up -> 0.5
    fast[0, _spot(1)], slow[0, _spot(1)] = 1.0, 2.0  # ETH: down -> 0
    fast[0, _spot(2)], slow[0, _spot(2)] = 9.0, 1.0  # BNB is not an MA asset
    sig = _signals(sma_fast=fast, sma_slow=slow)
    got = family_targets("MA_BTCETH_SPOT", sig, 0)
    np.testing.assert_allclose(got, _expected({_spot(0): MA_CAPPED}), atol=1e-15)
    fast[0, _spot(1)], slow[0, _spot(1)] = 3.0, 2.0
    got = family_targets("MA_BTCETH_SPOT", _signals(sma_fast=fast, sma_slow=slow), 0)
    np.testing.assert_allclose(
        got, _expected({_spot(0): MA_CAPPED, _spot(1): MA_CAPPED}), atol=1e-15
    )
    slow[0, _spot(1)] = np.nan  # an unavailable input gives 0
    got = family_targets("MA_BTCETH_SPOT", _signals(sma_fast=fast, sma_slow=slow), 0)
    assert got[_spot(1)] == 0.0


def test_controls_are_cash_equal_weight_spot_and_vol_targeted_perp() -> None:
    sig = _signals(vol60=np.full((2, N_LEGS), 1.0))
    assert not family_targets("CASH", sig, 0).any()
    ew = family_targets("EW_HOLD_SPOT", sig, 0)
    np.testing.assert_allclose(ew[ALL_SPOT], 1 / 11)
    assert not ew[ALL_PERP].any()
    vt = family_targets("VT_HOLD_PERP", sig, 0)
    np.testing.assert_allclose(vt[ALL_PERP], 0.5 / 11)  # (0.50 / 1.0) / 11
    assert not vt[ALL_SPOT].any()
    # an unavailable leg gets weight 0
    usable = np.ones((2, N_LEGS), dtype=bool)
    usable[0, _spot(3)] = False
    ew = family_targets("EW_HOLD_SPOT", _signals(usable=usable), 0)
    assert ew[_spot(3)] == 0.0 and ew[_spot(4)] == pytest.approx(1 / 11)
    vol = np.full((2, N_LEGS), 1.0)
    vol[0, _perp(2)] = np.nan
    assert family_targets("VT_HOLD_PERP", _signals(vol60=vol), 0)[_perp(2)] == 0.0


def test_caps_clip_each_name_first_then_scale_the_gross() -> None:
    # vol floor 0.10: 0.50 / 0.10 / 11 = 0.4545 > 0.40 per name; gross 4.4 -> 2.0
    sig = _signals(vol60=np.full((2, N_LEGS), 0.10))
    vt = family_targets("VT_HOLD_PERP", sig, 0)
    np.testing.assert_allclose(vt[ALL_PERP], 0.40 * 2.0 / 4.4)
    assert float(np.abs(vt).sum()) == pytest.approx(2.0)
    # name cap alone: three names at 0.4545 -> 0.40, gross 1.2 untouched
    vol = _grid()
    vol[0, [_perp(0), _perp(1), _perp(2)]] = 0.10
    vt = family_targets("VT_HOLD_PERP", _signals(vol60=vol), 0)
    np.testing.assert_allclose(vt[[_perp(0), _perp(1), _perp(2)]], 0.40)
    # gross alone: 0.50 / 0.20 / 11 = 0.2273 each, gross 2.5 -> 2.0
    vt = family_targets("VT_HOLD_PERP", _signals(vol60=np.full((2, N_LEGS), 0.20)), 0)
    np.testing.assert_allclose(vt[ALL_PERP], 2.0 / 11)
    # XS_MOM has the lower gross cap 1.5: six names clipped to 0.40 (gross 2.4) -> 0.25
    score = _grid()
    for pair in range(11):
        score[0, _perp(pair)] = 0.01 * pair
    vol = np.full((2, N_LEGS), 0.10)
    xs = family_targets("XS_MOM", _signals(vol60=vol, xs_score=score), 0)
    assert float(np.abs(xs).sum()) == pytest.approx(1.5)
    assert float(np.max(np.abs(xs))) == pytest.approx(0.40 * 1.5 / 2.4)


def test_apply_caps_is_the_frozen_clip_then_scale_order() -> None:
    capped = apply_caps(np.array([0.9, -0.9, 0.3, 0.0]), gross_limit=1.0)
    # clip -> (0.4, -0.4, 0.3, 0) gross 1.1 -> scaled by 1/1.1
    np.testing.assert_allclose(capped, np.array([0.4, -0.4, 0.3, 0.0]) / 1.1)
    # scaling first would give a different answer (0.9 -> 0.9*..., then no clip)
    unchanged = apply_caps(np.array([0.1, -0.2]), gross_limit=2.0)
    np.testing.assert_allclose(unchanged, [0.1, -0.2])


@pytest.mark.parametrize("family", protocol.FAMILIES)
def test_every_family_keeps_spot_long_only_and_the_unused_leg_flat(
    family: str,
) -> None:
    rng = np.random.default_rng(11)
    shape = (3, N_LEGS)
    sig = _signals(
        rows=3,
        r30=rng.normal(size=shape),
        r90=rng.normal(size=shape),
        r180=rng.normal(size=shape),
        vol60=rng.uniform(0.1, 1.5, size=shape),
        xs_score=rng.normal(size=shape),
        sma_fast=rng.uniform(1, 2, size=shape),
        sma_slow=rng.uniform(1, 2, size=shape),
    )
    for row in range(3):
        w = family_targets(family, sig, row)
        assert w.shape == (N_LEGS,) and np.all(np.isfinite(w))
        assert np.all(w[ALL_SPOT] >= 0.0)
        cap = protocol.XS_GROSS_CAP if family == "XS_MOM" else protocol.GROSS_CAP
        assert float(np.abs(w).sum()) <= cap + 1e-12
        assert float(np.max(np.abs(w))) <= protocol.PER_NAME_CAP + 1e-12
        if family in ("EW_HOLD_SPOT", "TS_TREND_LONG_SPOT", "MA_BTCETH_SPOT"):
            assert not w[ALL_PERP].any()
        if family in ("VT_HOLD_PERP", "TS_TREND_LS", "TS_TREND_BTCETH", "XS_MOM"):
            assert not w[ALL_SPOT].any()
        if family == "CASH":
            assert not w.any()


# ----------------------------------------------------------------- causal signals


def _manual_signals_at(dataset: MarketDataset, t: int, leg: int) -> dict[str, float]:
    """Independent re-computation of the signals of one leg at row ``t``."""
    c = np.asarray(dataset.close[:, leg], dtype=np.float64)
    log_returns = np.diff(np.log(c[t - 360 : t + 1]))
    assert log_returns.size == 360
    return {
        "r30": c[t] / c[t - 180] - 1.0,
        "r90": c[t] / c[t - 540] - 1.0,
        "r180": c[t] / c[t - 1080] - 1.0,
        "vol60": max(float(np.std(log_returns)) * math.sqrt(2190.0), 0.10),
        "xs_score": c[t - 42] / c[t - 402] - 1.0,
        "sma_fast": float(np.mean(c[t - 179 : t + 1])),
        "sma_slow": float(np.mean(c[t - 899 : t + 1])),
    }


def test_signals_match_an_independent_recomputation() -> None:
    ds = _fixture_dataset(seed=3)
    sig = build_signals(ds)
    for t, leg in ((1080, 0), (1200, 1), (1399, 5), (1250, 16)):
        for name, value in _manual_signals_at(ds, t, leg).items():
            assert getattr(sig, name)[t, leg] == pytest.approx(value, rel=1e-9), name


def test_signals_are_unavailable_until_their_history_exists() -> None:
    sig = build_signals(_fixture_dataset(seed=3))
    first_valid = {
        "r30": 180,
        "r90": 540,
        "r180": 1080,
        "vol60": 360,
        "xs_score": 402,
        "sma_fast": 179,
        "sma_slow": 899,
    }
    for name, first in first_valid.items():
        column = getattr(sig, name)[:, 0]
        assert np.isnan(column[first - 1]), name
        assert np.isfinite(column[first]), name


def test_vol_is_floored_and_annualised_with_2190_bars() -> None:
    ds = _fixture_dataset(seed=3)
    flat = replace(
        ds,
        open=np.ones_like(ds.close),
        high=np.ones_like(ds.close),
        low=np.ones_like(ds.close),
        close=np.ones_like(ds.close),
    )
    sig = build_signals(flat)
    assert sig.vol60[400, 0] == pytest.approx(protocol.VOL_FLOOR)
    # alternating +-1% log returns: population std 0.01 -> 0.01 * sqrt(2190)
    steps = np.where(np.arange(ds.n_bars) % 2 == 0, 0.01, -0.01)
    price = np.exp(np.cumsum(steps))[:, None] * np.ones((1, ds.n_symbols))
    wavy = replace(ds, open=price, high=price, low=price, close=price)
    assert build_signals(wavy).vol60[500, 0] == pytest.approx(
        0.01 * math.sqrt(2190.0), rel=1e-6
    )


def test_signals_never_see_future_bars() -> None:
    ds = _fixture_dataset(seed=3)
    base = build_signals(ds)
    cut = 1300
    perturbed = build_signals(
        core7_runner._perturb_future_market_rows(ds, cut_index=cut, seed=4)
    )
    for name in ROW_FIELDS:
        a, b = getattr(base, name), getattr(perturbed, name)
        assert np.array_equal(a[: cut + 1], b[: cut + 1], equal_nan=True), name
    # the perturbation is effective: later rows do differ
    assert not np.array_equal(
        base.r30[cut + 1 :], perturbed.r30[cut + 1 :], equal_nan=True
    )


def test_availability_matches_mtf6_and_spot_halts_mask_the_signals() -> None:
    ds = _fixture_dataset(seed=3, spot_halt_slice=(0, 1200, 1210))
    sig = build_signals(ds)
    np.testing.assert_array_equal(sig.usable, mtf6_signals._availability_mask(ds))
    assert not sig.usable[1205, 0] and sig.usable[1205, 1]
    for name in ("r30", "vol60", "sma_slow"):
        assert np.isnan(getattr(sig, name)[1205, 0])
        assert np.isfinite(getattr(sig, name)[1205, 1])
    assert np.isfinite(sig.r30[1210, 0])  # available again after the halt


def test_decision_row_follows_the_mtf6_snapshot_convention_and_delay_lags_one_bar() -> (
    None
):
    # mtf6 / ShadowAnchorMtfPolicy read row k = max(0, index - signal_lag_bars)
    assert signal_row(500, 0) == 500
    assert signal_row(500, 1) == 499
    assert signal_row(0, 1) == 0
    assert protocol.scenario_signal_lag_bars("BASE") == 0
    assert protocol.scenario_signal_lag_bars("HIGH_COST") == 0
    assert protocol.scenario_signal_lag_bars("DELAY_1BAR") == 1


# ----------------------------------------------------------------- policy mechanics


def _snapshot(
    index: int,
    *,
    weights: np.ndarray | None = None,
    value: float = 100_000.0,
    peak: float | None = None,
    max_dd: float = 0.0,
    stop: int = 100_000,
) -> PortfolioDecisionSnapshot:
    w = np.zeros(N_LEGS) if weights is None else np.asarray(weights, dtype=np.float64)
    rc = protocol.risk_config()
    account = PortfolioAccountSnapshot(
        cash=value,
        portfolio_value=value,
        peak_value=value if peak is None else peak,
        max_drawdown=max_dd,
        exact_quantities=("0",) * N_LEGS,
        quantities=(0.0,) * N_LEGS,
        weights=tuple(float(x) for x in w),
        mark_prices=(1.0,) * N_LEGS,
        contract_multipliers=(1.0,) * N_LEGS,
        turnover_total=0.0,
        total_cost=0.0,
        funding_pnl=0.0,
        borrow_cost=0.0,
        margin_used=0.0,
        maintenance_margin=0.0,
        maintenance_requirement=0.0,
        margin_deficit=0.0,
        insolvent=False,
        termination_reason=None,
    )
    return PortfolioDecisionSnapshot(
        index=index,
        timestamp_ns=index,
        agent_stop_index=stop,
        remaining_decisions=stop - index,
        symbols=protocol.LEG_ROSTER,
        feature_names=(),
        global_feature_names=(),
        features=((),) * N_LEGS,
        feature_available=((),) * N_LEGS,
        feature_staleness=((),) * N_LEGS,
        global_features=(),
        global_feature_available=(),
        account=account,
        active_orders=(),
        risk=PortfolioRiskLimits(
            max_gross=rc.max_gross,
            max_abs_weight=rc.max_abs_weight,
            max_turnover=rc.max_turnover,
            drawdown_start=rc.drawdown_start,
            drawdown_stop=rc.drawdown_stop,
            emergency_turnover_override=rc.emergency_turnover_override,
            fail_closed_tolerance=rc.fail_closed_tolerance,
        ),
    )


def _trend_signals(rows: int = 400) -> LowturnSignals:
    """Pair0 perp is +1 trend, pair1 perp is -1 trend, risk weight 1.0 -> +-1/11."""
    vol = _grid(rows, 0.5)
    r = _grid(rows)
    r[:, _perp(0)] = 0.1
    r[:, _perp(1)] = -0.1
    return _signals(rows, vol60=vol, r30=r.copy(), r90=r.copy(), r180=r.copy())


TARGET_PERP0 = 1 / 11
TARGET_PERP1 = -1 / 11


def test_rebalance_runs_on_the_weekly_cadence_and_holds_exact_quantities_between() -> (
    None
):
    policy = LowturnPolicy(
        family="TS_TREND_LS", signals=_trend_signals(), start_index=100
    )
    first = policy.decide_portfolio(_snapshot(100))
    assert first.target_weights[_perp(0)] == pytest.approx(TARGET_PERP0)
    assert first.target_weights[_perp(1)] == pytest.approx(TARGET_PERP1)
    traded = {_perp(0), _perp(1)}
    assert set(first.hold_current_quantity_symbols) == set(range(N_LEGS)) - traded

    current = _expected({_perp(0): 0.3, _perp(1): -0.2})  # drifted, away from target
    for index in range(101, 142):
        proposal = policy.decide_portfolio(_snapshot(index, weights=current))
        assert proposal.hold_current_quantity_symbols == tuple(range(N_LEGS)), index
        assert proposal.target_weights == tuple(float(x) for x in current)
    week2 = policy.decide_portfolio(_snapshot(142, weights=current))
    assert week2.target_weights[_perp(0)] == pytest.approx(TARGET_PERP0)
    assert _perp(0) not in week2.hold_current_quantity_symbols
    follow = policy.decide_portfolio(_snapshot(143, weights=current))
    assert follow.hold_current_quantity_symbols == tuple(range(N_LEGS))


def test_deadband_holds_a_symbol_below_0_02_and_trades_it_at_or_above() -> None:
    for offset, held in ((0.019, True), (0.021, False)):
        policy = LowturnPolicy(
            family="TS_TREND_LS", signals=_trend_signals(), start_index=0
        )
        current = _expected({_perp(0): TARGET_PERP0 + offset, _perp(1): TARGET_PERP1})
        proposal = policy.decide_portfolio(_snapshot(0, weights=current))
        assert (_perp(0) in proposal.hold_current_quantity_symbols) is held
        if held:
            assert proposal.target_weights[_perp(0)] == current[_perp(0)]
        else:
            assert proposal.target_weights[_perp(0)] == pytest.approx(TARGET_PERP0)
        assert _perp(1) in proposal.hold_current_quantity_symbols  # exactly on target


def test_delay_one_bar_reads_the_previous_signal_row() -> None:
    rows = 10
    vol = _grid(rows, 0.5)
    r = _grid(rows)
    r[:, _perp(0)] = 0.1
    r[6:, _perp(0)] = -0.1  # the sign flips at row 6
    sig = _signals(rows, vol60=vol, r30=r.copy(), r90=r.copy(), r180=r.copy())
    now = LowturnPolicy(family="TS_TREND_LS", signals=sig, start_index=6)
    lagged = LowturnPolicy(
        family="TS_TREND_LS", signals=sig, start_index=6, signal_lag_bars=1
    )
    assert now.decide_portfolio(_snapshot(6)).target_weights[_perp(0)] == pytest.approx(
        -1 / 11
    )
    assert lagged.decide_portfolio(_snapshot(6)).target_weights[
        _perp(0)
    ] == pytest.approx(1 / 11)


def test_risk_scale_matches_pretrade_risk_and_shadow_anchor() -> None:
    risk = PreTradeRisk(protocol.risk_config())
    for value, peak, max_dd in (
        (100.0, 100.0, 0.0),
        (95.0, 100.0, 0.05),
        (88.0, 100.0, 0.12),
        (88.0, 100.0, 0.15),
        (100.0, 100.0, 0.15),
        (80.0, 100.0, 0.20),
        (70.0, 100.0, 0.30),
    ):
        snap = _snapshot(0, value=value, peak=peak, max_dd=max_dd)
        drawdown = min(1.0, max(1.0 - value / peak, max_dd))
        assert risk_scale_of(snap) == pytest.approx(risk.risk_scale(drawdown))
        assert risk_scale_of(snap) == pytest.approx(
            ShadowAnchorMtfPolicy._risk_scale(snap)
        )


def _constrain(proposal: Any, current: np.ndarray, max_dd: float) -> np.ndarray:
    risk = PreTradeRisk(protocol.risk_config())
    return risk.constrain(
        np.asarray(proposal.target_weights), current=current, drawdown=max_dd
    ).weights


def test_risk_scale_decrease_retargets_once_and_never_compounds() -> None:
    policy = LowturnPolicy(
        family="TS_TREND_LS", signals=_trend_signals(), start_index=100
    )
    intended = _expected({_perp(0): TARGET_PERP0, _perp(1): TARGET_PERP1})
    policy.decide_portfolio(_snapshot(100, weights=intended))

    # drawdown 0.15 -> risk_scale 0.5, a decrease: re-target the last rebalance weights
    p1 = policy.decide_portfolio(_snapshot(101, weights=intended, max_dd=0.15))
    assert p1.hold_current_quantity_symbols == ()
    np.testing.assert_allclose(p1.target_weights, intended, atol=1e-15)
    scaled = _constrain(p1, intended, 0.15)
    np.testing.assert_allclose(scaled, intended * 0.5, atol=1e-15)

    # unchanged scale: no HOLD (it would be scaled again); the pre-image maps to current
    p2 = policy.decide_portfolio(_snapshot(102, weights=scaled, max_dd=0.15))
    assert p2.hold_current_quantity_symbols == ()
    np.testing.assert_allclose(_constrain(p2, scaled, 0.15), scaled, atol=1e-15)
    p3 = policy.decide_portfolio(_snapshot(103, weights=scaled, max_dd=0.15))
    np.testing.assert_allclose(_constrain(p3, scaled, 0.15), scaled, atol=1e-15)

    # a further (even small) decrease re-targets the intended book again, scaled once
    p4 = policy.decide_portfolio(_snapshot(104, weights=scaled, max_dd=0.18))
    np.testing.assert_allclose(p4.target_weights, intended, atol=1e-15)
    np.testing.assert_allclose(_constrain(p4, scaled, 0.18), intended * 0.2, atol=1e-15)

    # the hard stop: risk_scale 0 flattens whatever is proposed
    p5 = policy.decide_portfolio(_snapshot(105, weights=intended * 0.2, max_dd=0.25))
    assert not _constrain(p5, intended * 0.2, 0.25).any()


def test_rebalance_under_a_drawdown_uses_the_deadband_on_the_scaled_target() -> None:
    policy = LowturnPolicy(
        family="TS_TREND_LS", signals=_trend_signals(), start_index=0
    )
    policy.decide_portfolio(_snapshot(0))
    scaled = _expected({_perp(0): TARGET_PERP0 * 0.5, _perp(1): TARGET_PERP1 * 0.5})
    for index in range(1, 42):
        policy.decide_portfolio(_snapshot(index, weights=scaled, max_dd=0.15))
    # perp0 sits on its scaled target, perp1 is far from it
    current = scaled.copy()
    current[_perp(1)] = 0.3
    proposal = policy.decide_portfolio(_snapshot(42, weights=current, max_dd=0.15))
    assert proposal.hold_current_quantity_symbols == ()
    after = _constrain(proposal, current, 0.15)
    assert after[_perp(0)] == pytest.approx(current[_perp(0)])  # held, not rescaled
    assert after[_perp(1)] == pytest.approx(TARGET_PERP1 * 0.5)  # re-targeted once


def test_unscalable_pre_image_falls_back_to_the_intended_book() -> None:
    policy = LowturnPolicy(
        family="TS_TREND_LS", signals=_trend_signals(), start_index=0
    )
    policy.decide_portfolio(_snapshot(0))
    huge = _expected({_perp(0): 0.5})  # pre-image 0.5 / 0.5 = 1.0 > max_abs_weight
    proposal = policy.decide_portfolio(_snapshot(1, weights=huge, max_dd=0.15))
    intended = _expected({_perp(0): TARGET_PERP0, _perp(1): TARGET_PERP1})
    np.testing.assert_allclose(proposal.target_weights, intended, atol=1e-15)


def test_unscalable_pre_image_at_an_unchanged_scale_re_targets_the_intended_book() -> (
    None
):
    policy = LowturnPolicy(
        family="TS_TREND_LS", signals=_trend_signals(), start_index=0
    )
    policy.decide_portfolio(_snapshot(0))
    intended = _expected({_perp(0): TARGET_PERP0, _perp(1): TARGET_PERP1})
    # the scale decrease is consumed first, so the next bar has an unchanged 0.5 scale
    policy.decide_portfolio(_snapshot(1, weights=intended * 0.5, max_dd=0.15))
    within = intended * 0.5
    held = policy.decide_portfolio(_snapshot(2, weights=within, max_dd=0.15))
    np.testing.assert_allclose(held.target_weights, within / 0.5, atol=1e-15)
    drifted = _expected({_perp(0): 0.5})  # pre-image 0.5 / 0.5 = 1.0 > max_abs_weight
    proposal = policy.decide_portfolio(_snapshot(3, weights=drifted, max_dd=0.15))
    assert proposal.hold_current_quantity_symbols == ()
    np.testing.assert_allclose(proposal.target_weights, intended, atol=1e-15)


def test_policy_refuses_a_negative_spot_target_and_invalid_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sig = _trend_signals()
    policy = LowturnPolicy(family="XS_MOM", signals=sig, start_index=0)
    bad = np.zeros(N_LEGS)
    bad[_spot(0)] = -0.1
    monkeypatch.setattr(policies, "family_targets", lambda *a, **k: bad)
    with pytest.raises(ValueError, match="long-only"):
        policy.decide_portfolio(_snapshot(0))
    monkeypatch.undo()

    with pytest.raises(ValueError):
        LowturnPolicy(family="NOPE", signals=sig, start_index=0)
    with pytest.raises(ValueError):
        LowturnPolicy(family="CASH", signals=sig, start_index=0, signal_lag_bars=2)
    cash = LowturnPolicy(family="CASH", signals=sig, start_index=10)
    with pytest.raises(ValueError, match="precedes"):
        cash.decide_portfolio(_snapshot(5))
    cash.decide_portfolio(_snapshot(10))
    with pytest.raises(ValueError, match="monotonically"):
        cash.decide_portfolio(_snapshot(10))
    late = LowturnPolicy(family="CASH", signals=sig, start_index=10)
    with pytest.raises(ValueError, match="rebalance"):
        late.decide_portfolio(_snapshot(11))


# ----------------------------------------------------------------- decision rule


def _cell(
    family: str,
    segment: str,
    scenario: str,
    *,
    ret: float = 0.05,
    sharpe: float = 0.1,
    stopped: bool = False,
    ledger: bool = True,
    causality: bool = True,
    flat: bool = True,
    insolvent: bool = False,
) -> ScreenCell:
    return ScreenCell(
        family=family,
        segment=segment,
        scenario=scenario,
        total_return=ret,
        annual_sharpe=sharpe,
        max_drawdown=0.25 if stopped else 0.05,
        total_cost=10.0,
        funding_pnl=0.0,
        turnover_total=2.0,
        complete=True,
        terminal_flat=flat,
        execution_valid=(not insolvent) and ledger,
        causality_passed=causality,
        ledger_reconciled=ledger,
        stopped=stopped,
        stop_reason="max_drawdown_hard_stop" if stopped else None,
    )


def _grid_of_cells(
    **overrides: dict[str, Any],
) -> dict[tuple[str, str, str], ScreenCell]:
    """A grid in which only ``TS_TREND_LS`` passes DEV and HOLDOUT.

    ``overrides`` maps ``"FAMILY__SEGMENT__SCENARIO"`` to ``_cell`` keyword changes.
    """
    cells: dict[tuple[str, str, str], ScreenCell] = {}
    for family in protocol.FAMILIES:
        for segment in protocol.SEGMENTS:
            for scenario in protocol.SCENARIOS:
                kwargs: dict[str, Any] = {}
                if family == "EW_HOLD_SPOT":
                    kwargs["sharpe"] = 0.2
                elif family == "VT_HOLD_PERP":
                    kwargs["sharpe"] = 0.3
                elif family == "TS_TREND_LS":
                    kwargs["sharpe"] = 1.0 if segment == "DEV" else 0.4
                kwargs.update(overrides.get("__".join((family, segment, scenario)), {}))
                cells[(family, segment, scenario)] = _cell(
                    family, segment, scenario, **kwargs
                )
    return cells


def _label(**overrides: dict[str, Any]) -> str:
    return str(decide(_grid_of_cells(**overrides))["label"])


def test_a_stopped_cell_is_valid_but_not_a_pass() -> None:
    stopped = _cell("TS_TREND_LS", "DEV", "BASE", stopped=True)
    assert stopped.valid is True and stopped.stopped is True
    assert _cell("TS_TREND_LS", "DEV", "BASE", ledger=False).valid is False
    assert _cell("TS_TREND_LS", "DEV", "BASE", causality=False).valid is False
    assert _cell("TS_TREND_LS", "DEV", "BASE", flat=False).valid is False
    assert _cell("TS_TREND_LS", "DEV", "BASE", insolvent=True).valid is False


def test_the_baseline_grid_is_an_exploratory_candidate() -> None:
    decision = decide(_grid_of_cells())
    assert decision["label"] == "EXPLORATORY_CANDIDATE"
    assert decision["dev_pass_families"] == ["TS_TREND_LS"]
    assert decision["confirmed_families"] == ["TS_TREND_LS"]
    assert decision["claim_ceiling"] == "EXPLORATORY_CANDIDATE"
    assert decision["invalid_control_cells"] == []
    for family in protocol.CANDIDATE_FAMILIES[1:]:
        assert decision["families"][family]["dev_pass"] is False
        assert decision["families"][family]["holdout_confirmed"] is None


@pytest.mark.parametrize(
    ("override", "reason"),
    [
        ({"TS_TREND_LS__DEV__DELAY_1BAR": {"ledger": False}}, "dev cell invalid"),
        ({"TS_TREND_LS__DEV__BASE": {"causality": False}}, "dev causality"),
        ({"TS_TREND_LS__DEV__HIGH_COST": {"flat": False}}, "dev not flat"),
        ({"TS_TREND_LS__DEV__HIGH_COST": {"stopped": True}}, "dev stop"),
        ({"TS_TREND_LS__DEV__BASE": {"sharpe": 0.4999}}, "dev sharpe < 0.5"),
        (
            {
                "TS_TREND_LS__DEV__BASE": {"sharpe": 1.0},
                "EW_HOLD_SPOT__DEV__BASE": {"sharpe": 1.0},
            },
            "dev sharpe equal to EW is not above",
        ),
        (
            {"VT_HOLD_PERP__DEV__BASE": {"sharpe": 1.2}},
            "dev sharpe below VT_HOLD_PERP",
        ),
        ({"TS_TREND_LS__DEV__HIGH_COST": {"ret": 0.0}}, "dev high cost return = 0"),
        ({"TS_TREND_LS__DEV__HIGH_COST": {"ret": -0.01}}, "dev high cost negative"),
        ({"TS_TREND_LS__DEV__DELAY_1BAR": {"ret": 0.0}}, "dev delay return = 0"),
    ],
)
def test_each_dev_condition_is_load_bearing(
    override: dict[str, dict[str, Any]], reason: str
) -> None:
    decision = decide(_grid_of_cells(**override))
    assert decision["label"] == "NO_CANDIDATE", reason
    assert decision["families"]["TS_TREND_LS"]["dev_pass"] is False, reason
    assert decision["families"]["TS_TREND_LS"]["holdout_confirmed"] is None


def test_dev_sharpe_exactly_at_the_minimum_passes() -> None:
    assert _label(**{"TS_TREND_LS__DEV__BASE": {"sharpe": 0.5}}) == (
        "EXPLORATORY_CANDIDATE"
    )


@pytest.mark.parametrize(
    "override",
    [
        {"TS_TREND_LS__HOLDOUT__DELAY_1BAR": {"ledger": False}},
        {"TS_TREND_LS__HOLDOUT__BASE": {"flat": False}},
        {"TS_TREND_LS__HOLDOUT__HIGH_COST": {"stopped": True}},
        {"TS_TREND_LS__HOLDOUT__BASE": {"sharpe": 0.2999}},
        {"TS_TREND_LS__HOLDOUT__HIGH_COST": {"ret": 0.0}},
    ],
)
def test_each_holdout_condition_is_load_bearing(
    override: dict[str, dict[str, Any]],
) -> None:
    decision = decide(_grid_of_cells(**override))
    assert decision["label"] == "NO_CANDIDATE"
    assert decision["families"]["TS_TREND_LS"]["dev_pass"] is True
    assert decision["families"]["TS_TREND_LS"]["holdout_confirmed"] is False
    assert decision["dev_pass_families"] == ["TS_TREND_LS"]
    assert decision["confirmed_families"] == []


def test_holdout_sharpe_exactly_at_the_minimum_and_delay_return_are_not_gated() -> None:
    # HOLDOUT only needs BASE Sharpe >= 0.3 and HIGH_COST > 0; DELAY and the controls
    # are not read there (no condition may be invented)
    overrides = {
        "TS_TREND_LS__HOLDOUT__BASE": {"sharpe": 0.3},
        "TS_TREND_LS__HOLDOUT__DELAY_1BAR": {"ret": -0.5},
        "VT_HOLD_PERP__HOLDOUT__BASE": {"sharpe": 5.0},
    }
    assert _label(**overrides) == "EXPLORATORY_CANDIDATE"


@pytest.mark.parametrize("control", protocol.CONTROL_FAMILIES)
@pytest.mark.parametrize("segment", protocol.SEGMENTS)
def test_an_invalid_control_cell_makes_the_screen_invalid(
    control: str, segment: str
) -> None:
    decision = decide(
        _grid_of_cells(**{f"{control}__{segment}__HIGH_COST": {"ledger": False}})
    )
    assert decision["label"] == "INVALID"
    assert decision["invalid_control_cells"] == [f"{control}__{segment}__HIGH_COST"]
    # the candidate verdicts are still reported
    assert decision["families"]["TS_TREND_LS"]["dev_pass"] is True


def test_a_stopped_but_reconciled_control_is_a_stop_not_invalid() -> None:
    decision = decide(
        _grid_of_cells(**{"VT_HOLD_PERP__DEV__BASE": {"stopped": True, "sharpe": 0.3}})
    )
    assert decision["label"] == "EXPLORATORY_CANDIDATE"
    assert decision["invalid_control_cells"] == []


def test_an_invalid_candidate_cell_fails_that_family_only() -> None:
    decision = decide(
        _grid_of_cells(**{"XS_MOM__DEV__BASE": {"ledger": False, "sharpe": 3.0}})
    )
    assert decision["label"] == "EXPLORATORY_CANDIDATE"
    assert decision["families"]["XS_MOM"]["dev_pass"] is False


def test_the_decision_refuses_an_incomplete_or_mislabelled_grid() -> None:
    cells = _grid_of_cells()
    del cells[("XS_MOM", "DEV", "BASE")]
    with pytest.raises(ValueError, match="frozen grid"):
        decide(cells)
    cells = _grid_of_cells()
    cells[("XS_MOM", "DEV", "BASE")] = _cell("CASH", "DEV", "BASE")
    with pytest.raises(ValueError, match="does not match"):
        decide(cells)


def test_the_deflated_sharpe_ratio_matches_an_independent_value() -> None:
    # reference value computed independently with scipy.stats.norm (ppf / cdf):
    # SR0 = sqrt(V) * ((1 - g) z(1 - 1/N) + g z(1 - 1/(N e))) = 0.0238518800
    # DSR = Phi((SR - SR0) sqrt(T - 1) / sqrt(1 - skew SR + (kurt - 1)/4 SR^2))
    value = deflated_sharpe_ratio(
        sharpe=0.06,
        n_obs=1000,
        skewness=-0.5,
        kurtosis=5.0,
        sharpe_variance=0.0004,
        n_trials=5,
    )
    assert value == pytest.approx(0.8694528056867399, abs=1e-12)
    # with no trial dispersion the benchmark is 0
    flat = deflated_sharpe_ratio(
        sharpe=0.06,
        n_obs=1000,
        skewness=-0.5,
        kurtosis=5.0,
        sharpe_variance=0.0,
        n_trials=5,
    )
    assert flat == pytest.approx(0.9689330411924317, abs=1e-12)
    assert (
        deflated_sharpe_ratio(
            sharpe=0.06,
            n_obs=1000,
            skewness=0.0,
            kurtosis=3.0,
            sharpe_variance=0.0004,
            n_trials=1,
        )
        is None
    )
    assert (
        deflated_sharpe_ratio(
            sharpe=float("nan"),
            n_obs=1000,
            skewness=0.0,
            kurtosis=3.0,
            sharpe_variance=0.0004,
            n_trials=5,
        )
        is None
    )
    # more trials raise the benchmark and lower the DSR
    more = deflated_sharpe_ratio(
        sharpe=0.06,
        n_obs=1000,
        skewness=-0.5,
        kurtosis=5.0,
        sharpe_variance=0.0004,
        n_trials=50,
    )
    assert more is not None and value is not None and more < value


def test_the_deflated_sharpe_is_reported_with_five_trials_and_never_gates() -> None:
    cells = _grid_of_cells()
    for family in protocol.CANDIDATE_FAMILIES:
        key = (family, "DEV", "BASE")
        cells[key] = replace(
            cells[key],
            annual_sharpe=0.5 + 0.1 * protocol.CANDIDATE_FAMILIES.index(family),
            n_returns=1000,
            return_skewness=-0.2,
            return_kurtosis=4.0,
        )
    decision = decide(cells)
    dsr = decision["deflated_sharpe"]
    assert dsr["n_trials"] == 5 and dsr["gates"] is False
    assert set(dsr["dev_base"]) == set(protocol.CANDIDATE_FAMILIES)
    sharpes = np.array(
        [cells[(f, "DEV", "BASE")].annual_sharpe for f in protocol.CANDIDATE_FAMILIES]
    )
    variance = float(np.var(sharpes / math.sqrt(2190.0), ddof=1))
    assert dsr["trial_sharpe_variance_per_period"] == pytest.approx(variance)
    assert dsr["dev_base"]["XS_MOM"] == pytest.approx(
        deflated_sharpe_ratio(
            sharpe=cells[("XS_MOM", "DEV", "BASE")].annual_sharpe / math.sqrt(2190.0),
            n_obs=1000,
            skewness=-0.2,
            kurtosis=4.0,
            sharpe_variance=variance,
            n_trials=5,
        )
    )
    # a candidate that fails DEV_PASS keeps its reported DSR; the label ignores it
    assert decision["label"] == _label()


def test_calendar_year_returns_are_reported_per_cell() -> None:
    cells = _grid_of_cells()
    key = ("TS_TREND_LS", "DEV", "BASE")
    cells[key] = replace(cells[key], calendar_year_returns={"2022": 0.1, "2023": -0.05})
    decision = decide(cells)
    assert decision["calendar_year_returns"]["TS_TREND_LS__DEV__BASE"] == {
        "2022": 0.1,
        "2023": -0.05,
    }
    assert set(decision["cells"]) == {"__".join(k) for k in cells}


# ----------------------------------------------------------------- protocol


def test_the_protocol_constants_are_the_frozen_spec_values() -> None:
    assert protocol.PROTOCOL_ID == "lowturn_family_screen_v1"
    assert protocol.FROZEN_DATASET_ID == (
        "92336a938ac1229316a1e5ecce97f5d12a2f8f4864812b2b415e83389dbbaf8b"
    )
    assert protocol.LOOKBACK_BARS == {
        "7d": 42,
        "30d": 180,
        "60d": 360,
        "90d": 540,
        "150d": 900,
        "180d": 1080,
    }
    assert (protocol.R30_BARS, protocol.R90_BARS, protocol.R180_BARS) == (
        180,
        540,
        1080,
    )
    assert protocol.XS_PAST_BARS - protocol.XS_RECENT_BARS == 360
    assert (protocol.SMA_FAST_BARS, protocol.SMA_SLOW_BARS) == (180, 900)
    assert protocol.VOL_WINDOW_BARS == 360 and protocol.PERIODS_PER_YEAR == 2190.0
    assert (protocol.VOL_FLOOR, protocol.VOL_TARGET) == (0.10, 0.50)
    assert (protocol.PER_NAME_CAP, protocol.GROSS_CAP, protocol.XS_GROSS_CAP) == (
        0.40,
        2.0,
        1.5,
    )
    assert (protocol.REBALANCE_EVERY_BARS, protocol.DEADBAND) == (42, 0.02)
    assert protocol.WARMUP_BARS == 1080 and protocol.INITIAL_CAPITAL == 100_000.0
    assert (protocol.DEV_MIN_SHARPE, protocol.HOLDOUT_MIN_SHARPE) == (0.5, 0.3)
    assert protocol.DSR_TRIALS == 5
    assert protocol.SCENARIOS == ("BASE", "HIGH_COST", "DELAY_1BAR")
    assert protocol.CONTROL_FAMILIES == ("CASH", "EW_HOLD_SPOT", "VT_HOLD_PERP")
    assert protocol.CANDIDATE_FAMILIES == (
        "TS_TREND_LS",
        "TS_TREND_LONG_SPOT",
        "TS_TREND_BTCETH",
        "XS_MOM",
        "MA_BTCETH_SPOT",
    )


def test_risk_and_costs_are_the_core7_v4_values_unchanged() -> None:
    rc = protocol.risk_config()
    assert (
        rc.max_gross,
        rc.max_abs_weight,
        rc.max_turnover,
        rc.drawdown_start,
        rc.drawdown_stop,
    ) == (2.5, 0.55, 2.5, 0.10, 0.20)
    assert rc == core7_protocol.risk_config()
    for scenario in protocol.SCENARIOS:
        assert (
            protocol.scenario_cost(scenario).execution_policy_payload()
            == core7_protocol.scenario_cost(scenario).execution_policy_payload()
        )


def test_the_protocol_digest_is_canonical_stable_and_pinned() -> None:
    import hashlib

    from trade_rl.artifacts import canonical_json_bytes

    digest = protocol.protocol_digest()
    assert (
        digest
        == hashlib.sha256(canonical_json_bytes(protocol.protocol_payload())).hexdigest()
    )
    assert digest == protocol.protocol_digest()
    assert digest == PINNED_DIGEST


PINNED_DIGEST = "9de83d14ec756924cd9d5bc36da8a95fe7d3e3daf258c59a01584fd23eb2c433"


@pytest.mark.parametrize(
    "name", ["DEADBAND", "PER_NAME_CAP", "XS_PAST_BARS", "DEV_MIN_SHARPE", "VOL_FLOOR"]
)
def test_changing_any_frozen_constant_changes_the_digest(
    monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    before = protocol.protocol_digest()
    monkeypatch.setattr(protocol, name, getattr(protocol, name) + 1)
    assert protocol.protocol_digest() != before


def test_segment_bounds_split_at_the_first_stored_timestamp_at_or_after_the_date() -> (
    None
):
    ds = _fixture_dataset()
    bounds = protocol.segment_bounds(ds.timestamps)
    assert bounds == {"DEV": (1080, 1206), "HOLDOUT": (1206, FIXTURE_BARS - 1)}
    split = np.datetime64("2024-07-01T00:00:00", "ns")
    assert ds.timestamps[1206] >= split > ds.timestamps[1205]
    with pytest.raises(ValueError, match="warm-up"):
        protocol.segment_bounds(ds.timestamps[:1100])


# ----------------------------------------------------------------- end to end


@pytest.fixture(scope="module")
def synthetic_run(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[MarketDataset, Path, dict[str, Any]]:
    ds = _fixture_dataset()
    out = tmp_path_factory.mktemp("lowturn") / "run"
    decision = runner.run_screen(
        ds,
        output_dir=out,
        expected_protocol_digest=protocol.protocol_digest(),
        expected_dataset_id=ds.dataset_id,
        code_commit="a" * 40,
    )
    return ds, out, decision


def test_a_small_synthetic_replay_runs_every_cell_through_the_shared_replay(
    synthetic_run: tuple[MarketDataset, Path, dict[str, Any]],
) -> None:
    ds, out, decision = synthetic_run
    cells = decision["cells"]
    assert len(cells) == 8 * 2 * 3
    for name, cell in cells.items():
        assert cell["cell_valid"] is True, (name, cell["invalid_reasons"])
        assert cell["ledger_reconciled"] and cell["causality_passed"]
        assert cell["terminal_flat"]
        assert (out / "cells" / f"{name}.json").exists()
    assert decision["label"] in ("NO_CANDIDATE", "EXPLORATORY_CANDIDATE")
    assert cells["CASH__DEV__BASE"]["total_return"] == 0.0
    assert cells["CASH__DEV__BASE"]["turnover_total"] == 0.0
    # costs are the frozen scenarios: HIGH_COST costs more than BASE
    assert (
        cells["EW_HOLD_SPOT__DEV__HIGH_COST"]["total_cost"]
        > cells["EW_HOLD_SPOT__DEV__BASE"]["total_cost"]
    )
    # no signal input: DELAY_1BAR equals BASE for the controls, not for a signal family
    assert (
        cells["EW_HOLD_SPOT__DEV__DELAY_1BAR"]["total_return"]
        == cells["EW_HOLD_SPOT__DEV__BASE"]["total_return"]
    )
    assert (
        cells["TS_TREND_LS__DEV__DELAY_1BAR"]["total_return"]
        != cells["TS_TREND_LS__DEV__BASE"]["total_return"]
    ) or (
        cells["XS_MOM__DEV__DELAY_1BAR"]["total_return"]
        != cells["XS_MOM__DEV__BASE"]["total_return"]
    )
    # calendar years are reported and compound to the cell return
    years = cells["EW_HOLD_SPOT__DEV__BASE"]["calendar_year_returns"]
    assert set(years) == {"2024"}
    compounded = math.prod(1.0 + v for v in years.values()) - 1.0
    assert compounded == pytest.approx(
        cells["EW_HOLD_SPOT__DEV__BASE"]["total_return"], abs=1e-9
    )
    # a week of turnover: the cadence keeps turnover to rebalances (2 = buy + settle)
    assert cells["EW_HOLD_SPOT__DEV__BASE"]["turnover_total"] < 2.5


def test_the_outputs_and_manifest_are_written_before_the_decision_is_returned(
    synthetic_run: tuple[MarketDataset, Path, dict[str, Any]],
) -> None:
    ds, out, decision = synthetic_run
    manifest = json.loads((out / runner.MANIFEST_NAME).read_text(encoding="utf-8"))
    saved = json.loads((out / runner.DECISION_NAME).read_text(encoding="utf-8"))
    assert saved["label"] == decision["label"] == manifest["label"]
    assert manifest["protocol_digest"] == protocol.protocol_digest()
    assert manifest["protocol_id"] == protocol.PROTOCOL_ID
    assert manifest["dataset_id"] == ds.dataset_id
    assert manifest["code_commit"] == "a" * 40
    assert manifest["segments"] == {"DEV": [1080, 1206], "HOLDOUT": [1206, 1399]}
    assert len(manifest["cell_files"]) == 48
    assert all((out / path).exists() for path in manifest["cell_files"])
    assert len(manifest["causality_probes"]) == 8 * 2 * 2  # family x segment x lag
    assert all(p["passed"] for p in manifest["causality_probes"].values())


def test_run_screen_refuses_wrong_digest_wrong_dataset_and_an_existing_run(
    synthetic_run: tuple[MarketDataset, Path, dict[str, Any]], tmp_path: Path
) -> None:
    ds, out, _ = synthetic_run
    with pytest.raises(ValueError, match="protocol digest"):
        runner.run_screen(
            ds,
            output_dir=tmp_path / "a",
            expected_protocol_digest="0" * 64,
            expected_dataset_id=ds.dataset_id,
        )
    with pytest.raises(ValueError, match="dataset id"):
        runner.run_screen(
            ds,
            output_dir=tmp_path / "b",
            expected_protocol_digest=protocol.protocol_digest(),
            expected_dataset_id="f" * 64,
        )
    with pytest.raises(FileExistsError):
        runner.run_screen(
            ds,
            output_dir=out,
            expected_protocol_digest=protocol.protocol_digest(),
            expected_dataset_id=ds.dataset_id,
        )
    assert not (tmp_path / "a").exists() and not (tmp_path / "b").exists()


def test_run_screen_refuses_a_short_roster_and_the_embargo(tmp_path: Path) -> None:
    short = _make_alternating_dataset(
        n_bars=FIXTURE_BARS, pairs=("BTCUSDT", "ETHUSDT"), start_iso=FIXTURE_START
    )
    with pytest.raises(ValueError, match="ROSTER"):
        runner.run_screen(
            short,
            output_dir=tmp_path / "x",
            expected_protocol_digest=protocol.protocol_digest(),
            expected_dataset_id=short.dataset_id,
        )
    late = _make_alternating_dataset(
        n_bars=FIXTURE_BARS,
        pairs=protocol.ROSTER,
        start_iso="2026-06-01T00:00:00",
    )
    with pytest.raises(ValueError, match="embargo"):
        runner.run_screen(
            late,
            output_dir=tmp_path / "y",
            expected_protocol_digest=protocol.protocol_digest(),
            expected_dataset_id=late.dataset_id,
        )


def test_the_causality_probe_catches_a_policy_that_reads_the_next_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ds = _fixture_dataset()
    clean = runner.causality_probe_segment(ds, start_index=1080, stop_index=1206)
    assert all(p.passed for p in clean) and len(clean) == 16
    monkeypatch.setattr(policies, "signal_row", lambda index, lag: index + 1)
    leaky = runner.causality_probe_segment(ds, start_index=1080, stop_index=1206)
    failed = {(p.family, p.signal_lag_bars) for p in leaky if not p.passed}
    assert ("TS_TREND_LS", 0) in failed and ("XS_MOM", 0) in failed
    assert ("CASH", 0) not in failed  # CASH never reads a signal


def test_calendar_year_returns_compound_by_the_year_of_each_interval_end() -> None:
    from types import SimpleNamespace

    ts = np.array(
        [
            "2023-12-31T16:00:00",
            "2023-12-31T20:00:00",
            "2024-01-01T00:00:00",
            "2024-01-01T04:00:00",
        ],
        dtype="datetime64[ns]",
    )
    dataset = SimpleNamespace(timestamps=ts)
    intervals = [SimpleNamespace(next_index=i) for i in (1, 2, 3)]
    result = SimpleNamespace(
        ledger_evidence=SimpleNamespace(intervals=SimpleNamespace(intervals=intervals))
    )
    returns = np.array([0.10, 0.02, -0.01])
    got = runner._calendar_year_returns(dataset, result, returns)  # type: ignore[arg-type]
    assert got == {
        "2023": pytest.approx(0.10),
        "2024": pytest.approx(1.02 * 0.99 - 1.0),
    }
    assert runner._calendar_year_returns(dataset, result, returns[:2]) is None  # type: ignore[arg-type]


def _flat_then_drop(ds: MarketDataset, index: int, factor: float) -> MarketDataset:
    base = np.asarray(ds.close[0], dtype=np.float64)
    prices = np.tile(base, (ds.n_bars, 1))
    prices[index:] *= factor
    return replace(
        ds, open=prices, high=prices, low=prices, close=prices, mark_price=prices
    )


def test_a_hard_stop_is_a_valid_flat_reconciled_economic_stop() -> None:
    ds = _fixture_dataset()
    # a 40% gap in every leg at bar 1130 breaches the 20% stop before any policy can react
    crashed = _flat_then_drop(ds, 1130, 0.6)
    sig = build_signals(crashed)
    start, stop = protocol.segment_bounds(crashed.timestamps)["DEV"]
    policy = LowturnPolicy(family="EW_HOLD_SPOT", signals=sig, start_index=start)
    result = run_shared_portfolio_replay(
        crashed,
        policy,
        start_index=start,
        stop_index=stop,
        initial_capital=protocol.INITIAL_CAPITAL,
        execution_cost=protocol.scenario_cost("BASE"),
        risk=PreTradeRisk(protocol.risk_config()),
        settle_terminal_position=True,
        capture_ledger_evidence=True,
    )
    cell = runner._build_cell(
        family="EW_HOLD_SPOT",
        segment="DEV",
        scenario="BASE",
        result=result,
        dataset=crashed,
        causality_passed=True,
    )
    assert cell.max_drawdown >= 0.20
    assert cell.stopped and cell.stop_reason == "max_drawdown_hard_stop"
    # the engine does not terminate the replay at the stop: risk_scale hits 0, every later
    # proposal is flattened and settlement still runs, so the cell is an economic STOP
    assert result.diagnostics.termination_reasons == ()
    assert not result.book.insolvent
    assert cell.terminal_flat and cell.ledger_reconciled
    assert cell.valid is True and cell.invalid_reasons == ()
    assert float(np.abs(result.book.quantities).max()) == 0.0
    assert len(result.decisions) == stop - start - 1  # decided on every agent bar


def test_risk_scaling_does_not_compound_in_the_real_replay() -> None:
    ds = _fixture_dataset()
    # prices flat, then a 15% drop in every leg at bar 1100: drawdown 0.15 -> scale 0.5
    dropped = _flat_then_drop(ds, 1100, 0.85)
    sig = build_signals(dropped)
    start, stop = protocol.segment_bounds(dropped.timestamps)["DEV"]
    policy = LowturnPolicy(family="EW_HOLD_SPOT", signals=sig, start_index=start)
    result = run_shared_portfolio_replay(
        dropped,
        policy,
        start_index=start,
        stop_index=stop,
        initial_capital=protocol.INITIAL_CAPITAL,
        execution_cost=protocol.scenario_cost("BASE"),
        risk=PreTradeRisk(protocol.risk_config()),
        settle_terminal_position=True,
        capture_ledger_evidence=True,
    )
    gross = {d.index: float(np.abs(d.target_weights).sum()) for d in result.decisions}
    assert result.book.max_drawdown == pytest.approx(0.15, abs=0.01)
    before = gross[1099]
    assert before == pytest.approx(1.0, abs=0.01)
    # risk_scale ~0.5 is applied once, then the book is left alone until the next rebalance
    window = [gross[i] for i in range(1103, 1121)]
    assert max(window) - min(window) < 0.01
    assert window[0] == pytest.approx(0.5 * before, rel=0.05)
    # no trades between 1103 and the next rebalance (1122): nothing re-scales the book
    intervals = result.ledger_evidence.intervals.intervals  # type: ignore[union-attr]
    for iv in intervals:
        if 1103 <= iv.start_index < 1122:
            assert all(e.filled_quantity == 0.0 for e in iv.order_events), (
                iv.start_index
            )


# ----------------------------------------------------------------- the tool


AUTH = "https://github.com/shuntatsu/trade_rl/issues/810#issuecomment-1"


def _body(dataset_id: str, *, digest: str | None = None, pid: bool = True) -> str:
    lines = ["G4 authorization: low-turnover family screen"]
    if pid:
        lines.append(f"protocol_id: {protocol.PROTOCOL_ID}")
    lines.append(f"protocol_digest: {digest or protocol.protocol_digest()}")
    lines.append(f"dataset_id: {dataset_id}")
    return "\n".join(lines) + "\n"


def _argv(tmp_path: Path, url: str = AUTH, name: str = "out") -> list[str]:
    files = tierd_tests._cli_files(tmp_path)
    return [
        "run",
        *files,
        "--output-dir",
        str(tmp_path / name),
        "--authorization-url",
        url,
    ]


def _run_tool(argv: list[str], *, gh: Any, clean: bool = True) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    rows = tierd_tests.FakeRows()
    code = driver.main(
        argv,
        stdout=out,
        stderr=err,
        state=lambda: ("a" * 40, clean),
        rows_factory=lambda plan: rows,
        load_end=tierd_tests.END,
        gh=gh,
    )
    return code, out.getvalue(), err.getvalue()


def _fake_dataset_id(argv: list[str]) -> str:
    return tierd_tests._dataset_id_for(argv, None)


class _ScreenSpy:
    def __init__(self, label: str = "NO_CANDIDATE") -> None:
        self.calls: list[dict[str, Any]] = []
        self.label = label
        self.out: io.StringIO | None = None

    def __call__(self, dataset: Any, **kwargs: Any) -> dict[str, Any]:
        self.calls.append({"dataset": dataset, **kwargs})
        out_dir = kwargs["output_dir"]
        (out_dir / runner.DECISION_NAME).write_text("{}", encoding="utf-8")
        return {
            "label": self.label,
            "dev_pass_families": [],
            "confirmed_families": [],
        }


def test_plan_prints_the_protocol_digest_and_segment_indices() -> None:
    out, err = io.StringIO(), io.StringIO()
    assert driver.main(["plan"], stdout=out, stderr=err) == driver.EXIT_OK
    document = json.loads(out.getvalue())
    assert document["protocol_id"] == "lowturn_family_screen_v1"
    assert document["protocol_digest"] == protocol.protocol_digest()
    assert document["dataset_id"] == protocol.FROZEN_DATASET_ID
    assert document["clock"]["n_bars"] == 10589
    assert document["clock"]["first_bar_close"].startswith("2021-10-01T04:00")
    assert document["clock"]["last_bar_close"].startswith("2026-07-31T20:00")
    assert document["segments"]["DEV"]["start_index"] == 1080
    assert document["segments"]["DEV"]["stop_index"] == 6023
    assert document["segments"]["HOLDOUT"]["start_index"] == 6023
    assert document["segments"]["HOLDOUT"]["stop_index"] == 10588
    assert document["segments"]["DEV"]["stop_bar_close"].startswith("2024-07-01T00:00")
    assert document["protocol"] == json.loads(
        json.dumps(protocol.protocol_payload(), sort_keys=True)
    )


def test_run_needs_an_authorization_url_a_clean_tree_and_a_new_output_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        driver.runner, "run_screen", lambda *a, **k: pytest.fail("must not run")
    )
    argv = _argv(tmp_path)
    with pytest.raises(SystemExit):
        driver.main(["run", *tierd_tests._cli_files(tmp_path), "--output-dir", "x"])
    gh = tierd_tests.FakeGh(body=_body(_fake_dataset_id(argv)))
    code, _, err = _run_tool(_argv(tmp_path, "https://example.com/x"), gh=gh)
    assert code == driver.EXIT_REFUSED and "authorization_url" in err
    code, _, err = _run_tool(argv, gh=gh, clean=False)
    assert code == driver.EXIT_REFUSED and "clean" in err
    (tmp_path / "out").mkdir()
    code, _, err = _run_tool(argv, gh=gh)
    assert code == driver.EXIT_REFUSED and "write-once" in err


def test_run_refuses_any_dataset_that_is_not_the_frozen_tier_d_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spy = _ScreenSpy()
    monkeypatch.setattr(driver.runner, "run_screen", spy)
    argv = _argv(tmp_path)
    gh = tierd_tests.FakeGh(body=_body(_fake_dataset_id(argv)))
    code, _, err = _run_tool(argv, gh=gh)
    assert code == driver.EXIT_REFUSED and "frozen Tier D dataset id" in err
    assert not spy.calls and not (tmp_path / "out").exists()
    assert len(gh.calls) == 1  # refused before the second fetch


@pytest.mark.parametrize(
    "body_factory",
    [
        lambda ds: _body(ds, pid=False),
        lambda ds: _body(ds, digest=core7_protocol.protocol_digest()),
        lambda ds: _body("e" * 64),
        lambda ds: _body(ds).replace(protocol.PROTOCOL_ID, protocol.PROTOCOL_ID + "x"),
    ],
)
def test_run_requires_the_comment_to_quote_this_protocol_id_digest_and_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, body_factory: Any
) -> None:
    spy = _ScreenSpy()
    monkeypatch.setattr(driver.runner, "run_screen", spy)
    argv = _argv(tmp_path)
    dataset_id = _fake_dataset_id(argv)
    monkeypatch.setattr(protocol, "FROZEN_DATASET_ID", dataset_id)
    code, _, err = _run_tool(argv, gh=tierd_tests.FakeGh(body=body_factory(dataset_id)))
    assert code == driver.EXIT_UNAUTHENTICATED, err
    assert not spy.calls and not (tmp_path / "out").exists()


def test_a_non_owner_or_edited_comment_authorizes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spy = _ScreenSpy()
    monkeypatch.setattr(driver.runner, "run_screen", spy)
    argv = _argv(tmp_path)
    dataset_id = _fake_dataset_id(argv)
    monkeypatch.setattr(protocol, "FROZEN_DATASET_ID", dataset_id)
    body = _body(dataset_id)
    code, _, _ = _run_tool(
        argv, gh=tierd_tests.FakeGh(body=body, login="mallory", user_id=1)
    )
    assert code == driver.EXIT_UNAUTHENTICATED
    edited = tierd_tests.FakeGh(
        body=body, variants={2: {"updated_at": "2026-11-21T00:00:00Z"}}
    )
    code, _, err = _run_tool(argv, gh=edited)
    assert code == driver.EXIT_UNAUTHENTICATED and "edited" in err
    assert not spy.calls and not (tmp_path / "out").exists()


def test_an_authenticated_run_writes_receipts_then_prints_the_label(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spy = _ScreenSpy("EXPLORATORY_CANDIDATE")
    monkeypatch.setattr(driver.runner, "run_screen", spy)
    argv = _argv(tmp_path)
    dataset_id = _fake_dataset_id(argv)
    monkeypatch.setattr(protocol, "FROZEN_DATASET_ID", dataset_id)
    gh = tierd_tests.FakeGh(body=_body(dataset_id))
    code, out, err = _run_tool(argv, gh=gh)

    assert code == driver.EXIT_OK, err
    assert json.loads(out)["label"] == "EXPLORATORY_CANDIDATE"
    assert [c[0] for c in gh.calls] == ["api", "api"]
    (call,) = spy.calls
    assert call["expected_protocol_digest"] == protocol.protocol_digest()
    assert call["expected_dataset_id"] == dataset_id
    assert call["dataset"].dataset_id == dataset_id
    assert call["code_commit"] == "a" * 40
    assert call["authorization"]["author_login"] == "shuntatsu"
    assert call["authorization"]["quoted_protocol_digest"] == protocol.protocol_digest()
    out_dir = tmp_path / "out"
    data_receipt = json.loads((out_dir / driver.DATA_RECEIPT_NAME).read_text("utf-8"))
    assert data_receipt["dataset_id"] == dataset_id
    assert data_receipt["protocol_id"] == protocol.PROTOCOL_ID
    assert data_receipt["code_commit"] == "a" * 40
    run_receipt = json.loads((out_dir / driver.RUN_RECEIPT_NAME).read_text("utf-8"))
    assert run_receipt["label"] == "EXPLORATORY_CANDIDATE"
    assert run_receipt["claim_ceiling"] == "EXPLORATORY_CANDIDATE"
    assert run_receipt["authorization_receipt"]["comment_id"] == 1


def test_an_invalid_screen_is_not_reported_as_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(driver.runner, "run_screen", _ScreenSpy("INVALID"))
    argv = _argv(tmp_path)
    dataset_id = _fake_dataset_id(argv)
    monkeypatch.setattr(protocol, "FROZEN_DATASET_ID", dataset_id)
    code, out, err = _run_tool(argv, gh=tierd_tests.FakeGh(body=_body(dataset_id)))
    assert code == driver.EXIT_INVALID and "INVALID" in err
    assert json.loads(out)["label"] == "INVALID"


def test_the_g4_helpers_default_to_core7_and_accept_this_protocol_by_parameter() -> (
    None
):
    snap = g4.CommentSnapshot(
        comment_id=1,
        html_url=AUTH,
        login="shuntatsu",
        user_id=g4.OWNER_ID,
        body=_body("d" * 64),
        body_sha256="0" * 64,
        updated_at="t",
        fetched_at="t",
    )
    g4.require_protocol_digest(snap, protocol.protocol_digest())
    g4.require_protocol_id(snap, protocol.PROTOCOL_ID)
    with pytest.raises(g4.G4AuthError):
        g4.require_protocol_digest(snap)  # the Core7 digest is not quoted
    with pytest.raises(g4.G4AuthError):
        g4.require_protocol_id(snap, "lowturn_family_screen")  # not a whole token
    with pytest.raises(g4.G4AuthError):
        g4.require_protocol_digest(snap, "xyz")
