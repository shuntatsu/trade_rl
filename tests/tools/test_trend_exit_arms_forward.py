"""Synthetic-data tests for tools/trend_exit_arms_forward.py (no network, no market data).

Needs only numpy and the repository's own ``trade_rl`` package for the cost-constant pin. One
test uses an optional package and skips without it: the pandas frame parity oracle (``pandas``)."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Barrier
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np
import pytest

from tools import trend_exit_arms_forward as m

DIGEST_PIN = "d7b19ea2667ffcc18bc27f10188ed063360e7a3225b90ed0f1abd630c0443781"
COMMIT = "a" * 40
URL = "https://github.com/shuntatsu/trade_rl/issues/810#issuecomment-7040999999"
T0 = int(datetime(2026, 12, 1, tzinfo=UTC).timestamp())


# ----------------------------------------------------------------------------- digest


def test_digest_pin() -> None:
    assert m.protocol_digest() == DIGEST_PIN
    assert re.fullmatch(r"[0-9a-f]{64}", m.protocol_digest())


def test_constants_frozen() -> None:
    assert m.PROTOCOL_ID == "trend_exit_arms_v1"
    assert m.SOURCE_CLOSE_POLICY.value == "strict_source_close_v2"
    assert len(m.SYMBOLS) == 11 and m.SYMBOLS[0] == "BTCUSDT"
    assert m.REGIME_WINDOWS == (20, 50, 100, 200) and m.REGIME_THRESHOLD == 0.5
    assert (m.SIZE_VOL, m.VOL_DAYS) == (0.5, 30)
    assert (m.ATR_RANGE_BARS, m.ATR_MEAN_BARS) == (6, 84)
    assert (m.TRAIL_ATR, m.SHORT_HALF, m.TIME_EXIT_BARS) == (3.0, 0.5, 180)
    assert (m.EARLY_STOP_DD, m.DD_SLACK, m.GROSS_LIMIT) == (0.30, 0.05, 1.5)
    assert (m.BOOT_BLOCK, m.BOOT_DRAWS, m.BOOT_SEED, m.ALPHA) == (30, 5000, 0, 0.05)
    assert m.ARMS == ("A0_RULE", "A1_SHORT_TRAIL3", "A2_SHORT_HALF", "A3_SHORT_TIME30")
    assert m.VARIANTS == {"BASE": (1.0, 0), "HIGH_COST": (2.0, 0), "DELAY": (1.0, 1)}
    assert len(m.all_cell_names()) == 14 and m.all_cell_names()[-2:] == [
        "C_HOLD_BTC",
        "C_CASH",
    ]
    assert m.DATA_SCOPE_COMMENT_ID == 7040247247 and m.OWNER_LOGIN == "shuntatsu"
    fin = m.STAGES["FORWARD_FINAL"]
    assert fin.decision_start.isoformat() == "2026-12-01T00:00:00+00:00"
    assert fin.decision_end.isoformat() == "2027-12-01T00:00:00+00:00"
    assert m.STAGES["FORWARD_INTERIM"].not_before == datetime(2026, 12, 2, tzinfo=UTC)
    assert m.STAGES["FORWARD_FINAL"].not_before == datetime(2027, 12, 2, tzinfo=UTC)


def test_cost_constants_match_the_repository_execution_formula() -> None:
    """The cost constants and the formula mirror trade_rl.simulation.execution (FS1 constants):
    ``multiplier * (fee + spread + impact_rate * sqrt(participation))`` with a 5% cap."""
    from trade_rl.simulation.execution import ExecutionCostConfig

    cfg = ExecutionCostConfig()
    assert (m.FEE_RATE, m.SPREAD_RATE, m.IMPACT_RATE, m.MAX_PARTICIPATION) == (
        cfg.fee_rate,
        cfg.spread_rate,
        cfg.impact_rate,
        cfg.max_participation_rate,
    )
    for part in (1e-6, 0.01, 0.05):
        for mult in (1.0, 2.0):
            fee, spr, imp = m.unit_cost_rate(part, mult)
            repo = mult * (
                cfg.fee_rate + cfg.spread_rate + cfg.impact_rate * math.sqrt(part)
            )
            assert fee + spr + imp == pytest.approx(repo, rel=1e-15)


def test_rolling_primitives_propagate_nan() -> None:
    x = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0])
    assert np.isnan(m.r_max(x, 3)[:5]).all() and m.r_max(x, 3)[5] == 6.0
    assert m.r_mean(x, 2)[1] == 1.5 and np.isnan(m.r_mean(x, 2)[2])
    assert m.r_std(np.array([1.0, 3.0]), 2)[1] == pytest.approx(math.sqrt(2.0))
    assert np.array_equal(m.shift(x, 2)[2:4], [1.0, 2.0])


# ----------------------------------------------------------------------------- frame


def synth_hourly(
    n_hours: int, seed: int, holes: tuple[int, ...] = ()
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    r = rng.normal(0.0, 0.004, n_hours) + 0.0002 * np.sin(np.arange(n_hours) / 500)
    c = 100 * np.exp(np.cumsum(r))
    o = np.r_[100.0, c[:-1]]
    h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.002, n_hours)))
    lo = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.002, n_hours)))
    qv = rng.uniform(1e6, 5e6, n_hours)
    arrs = {
        "open": o,
        "high": h,
        "low": lo,
        "close": c,
        "qv": qv,
        "tbq": qv * rng.uniform(0.4, 0.6, n_hours),
    }
    for i in holes:
        for k in arrs:
            arrs[k][i] = np.nan
    return arrs


START_S = int(datetime(2025, 1, 1, tzinfo=UTC).timestamp())


def test_frame_shape_and_warmup() -> None:
    fr = m.build_frame(synth_hourly(24 * 300, 1), START_S)
    assert fr.close.shape == (24 * 300 // 4,)
    assert fr.open_s[1] - fr.open_s[0] == m.BAR_S and fr.open_s[0] == START_S
    # regime needs 200 completed daily bars: first known at the close of day 200 (bar 6*200 - 1)
    assert np.isnan(fr.regime[: 6 * 200 - 1]).all()
    assert np.isfinite(fr.regime[6 * 200 :]).all()
    fin = fr.regime[~np.isnan(fr.regime)]
    assert np.allclose(fin * 4, np.round(fin * 4)) and np.abs(fin).max() <= 1.0
    # vol needs 31 daily closes; the daily ATR needs 5 + 84 four-hour bars
    assert (
        np.isnan(fr.vol_ann[: 6 * 31 - 1]).all()
        and np.isfinite(fr.vol_ann[6 * 31 :]).all()
    )
    assert np.isnan(fr.datr[: 5 + 83]).all() and np.isfinite(fr.datr[5 + 83 :]).all()
    # the fill volume is the last completed hour of the 4h bar
    arrs = synth_hourly(24 * 300, 1)
    assert np.array_equal(fr.fill_qv, arrs["qv"][3::4])


@pytest.mark.parametrize("cut_day", [210.0, 240.25, 280.5])
def test_frame_is_causal_under_truncation(cut_day: float) -> None:
    """A truncation test: every column at every decision close <= the cut is identical whether or
    not the later bars exist, and unchanged when the later bars are rewritten."""
    full = synth_hourly(24 * 300, 2, holes=(5000, 5001))
    cut_h = int(cut_day * 24) // 4 * 4
    trunc = {k: v[:cut_h].copy() for k, v in full.items()}
    a = m.build_frame(full, START_S)
    b = m.build_frame(trunc, START_S)
    k = cut_h // 4
    future = {key: v.copy() for key, v in full.items()}
    rng = np.random.default_rng(9)
    for key in future:
        future[key][cut_h:] = rng.uniform(1.0, 500.0, len(future[key]) - cut_h)
    c = m.build_frame(future, START_S)
    for col in ("close", "regime", "vol_ann", "datr", "fill_qv"):
        assert np.array_equal(getattr(a, col)[:k], getattr(b, col), equal_nan=True), col
        assert np.array_equal(
            getattr(a, col)[:k], getattr(c, col)[:k], equal_nan=True
        ), col


def test_frame_uses_latest_completed_daily_bar() -> None:
    arrs = synth_hourly(24 * 260, 3)
    fr = m.build_frame(arrs, START_S)
    mod = {k: v.copy() for k, v in arrs.items()}
    day = 230
    for key in mod:
        mod[key][day * 24 + 23] *= 1.5
    fr2 = m.build_frame(mod, START_S)
    boundary = day * 6 + 5  # the 4h bar whose close is the day boundary
    assert np.array_equal(fr.vol_ann[:boundary], fr2.vol_ann[:boundary], equal_nan=True)
    assert fr.vol_ann[boundary] != fr2.vol_ann[boundary]


# G2 oracle: the exploratory pandas code (feat.agg, the q2 regime / vol_ann and the h_exit_lib daily
# ATR), copied verbatim, must give the same numbers as the pandas-free port.
def _pandas_reference(arrs: dict[str, Any], start: str) -> Any:  # pragma: no cover
    import pandas as pd

    grid = pd.date_range(start, periods=len(arrs["close"]), freq="h", tz="UTC")
    d1h = pd.DataFrame(
        {
            "open": arrs["open"],
            "high": arrs["high"],
            "low": arrs["low"],
            "close": arrs["close"],
            "quote_volume": arrs["qv"],
            "taker_buy_quote_volume": arrs["tbq"],
        },
        index=grid,
    )

    def agg(d: Any, rule: str, n: int) -> Any:
        r = d.resample(rule, label="left", closed="left")
        o = pd.DataFrame(
            {
                "open": r.open.first(),
                "high": r.high.max(),
                "low": r.low.min(),
                "close": r.close.last(),
                "qv": r.quote_volume.sum(min_count=n),
                "tbq": r.taker_buy_quote_volume.sum(min_count=n),
            }
        )
        cnt = d.close.resample(rule, label="left", closed="left").count()
        o[cnt < n] = np.nan
        return o

    b4, b1d = agg(d1h, "4h", 4), agg(d1h, "1D", 24)
    c4 = b4.index + pd.Timedelta("4h")
    cd = b1d.close
    fd = pd.DataFrame(index=b1d.index)
    fd["regime"] = (
        sum(np.sign(cd - cd.rolling(n).mean()) for n in (20, 50, 100, 200)) / 4
    )
    fd["vol_ann"] = np.log(cd).diff().rolling(30).std() * np.sqrt(365)
    fd.index = fd.index + pd.Timedelta("1D")
    f = pd.DataFrame(
        fd.reindex(c4, method="ffill").values, index=b4.index, columns=fd.columns
    )
    hh = b4.high.rolling(6).max()
    ll = b4.low.rolling(6).min()
    f["datr"] = (hh - ll).rolling(84, min_periods=84).mean().values
    f["close"] = b4.close.values
    return f


@pytest.mark.parametrize("holes", [(), (3000, 3001, 5000)])
def test_frame_matches_pandas_reference(holes: tuple[int, ...]) -> None:
    pytest.importorskip("pandas")
    arrs = synth_hourly(24 * 330, 1, holes)
    ref = _pandas_reference(arrs, "2025-01-01")
    fr = m.build_frame(arrs, START_S)
    for col in ("regime", "vol_ann", "datr", "close"):
        expect = ref[col].to_numpy(np.float64)
        got = getattr(fr, col)
        assert np.array_equal(np.isnan(expect), np.isnan(got)), col
        ok = ~np.isnan(expect)
        assert np.allclose(expect[ok], got[ok], rtol=1e-9, atol=1e-9), col


# ----------------------------------------------------------------------------- rule and exits


def _arr(*v: float) -> np.ndarray:
    return np.array(v, dtype=float)


def test_regime_direction_and_size_rule() -> None:
    r = m.regime_direction(_arr(1.0, 0.5, 0.25, 0.0, -0.25, -0.5, -1.0, np.nan))
    assert list(r) == [1, 1, 0, 0, 0, -1, -1, 0]
    assert m._size(0.25) == 1.0 and m._size(1.0) == 0.5 and m._size(0.0) == 1.0


def test_rule_positions_size_fixed_at_entry_and_held_on_nan() -> None:
    reg = _arr(0, 1, 1, 1, 1, 0, 1, -1, -1)
    vol = _arr(1.0, 1.0, 2.0, 0.1, np.nan, 1.0, 0.25, 0.5, 4.0)
    close = np.full(9, 100.0)
    pos = m.rule_positions(reg, vol, close)
    # entry size 0.5/1.0 kept while vol moves (2.0, 0.1) and on a NaN-vol bar; flat; new entry
    # sized 1.0 (vol 0.25); a flip re-sizes (-1 at vol 0.5), then kept (vol 4.0)
    assert pos == pytest.approx([0, 0.5, 0.5, 0.5, 0.5, 0, 1.0, -1.0, -1.0])
    close2 = close.copy()
    close2[2] = np.nan  # a non-finite close also holds the previous position
    assert m.rule_positions(reg, vol, close2)[2] == pos[1]


def _short_run(close: list[float], datr: float = 5.0) -> tuple[np.ndarray, ...]:
    n = len(close)
    pos = np.zeros(n)
    pos[1 : n - 1] = -1.0
    return pos, _arr(*close), np.full(n, datr)


def test_arm_exits_hand_cases() -> None:
    pos, close, datr = _short_run([100, 100, 98, 97, 101, 113, 90, 80, 100])
    # trailing: lowest close so far 97 at row 3; stop 97 + 3 * 5 = 112; row 5 (113) exits and the
    # short stays out although the price falls back; the rule exit (row 8 flat) ends the trade
    a1 = m.exit_multipliers("A1_SHORT_TRAIL3", pos, close, datr)
    assert list(a1) == [0, 1, 1, 1, 1, 0, 0, 0, 0]
    assert list(m.exit_multipliers("A0_RULE", pos, close, datr)) == [
        0,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        0,
    ]
    assert list(m.exit_multipliers("A2_SHORT_HALF", pos, close, datr)) == [
        0,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0,
    ]
    assert list(m.exit_multipliers("A3_SHORT_TIME30", pos, close, datr)) == list(
        m.exit_multipliers("A0_RULE", pos, close, datr)
    )  # a 7-bar short never reaches 180 bars


def test_trailing_stop_is_measured_from_the_best_close_including_the_entry_row() -> (
    None
):
    # exactly 3 ATR above the entry close exits on that row (>=); just below does not
    pos, close, datr = _short_run([100, 100, 114.999, 115.0, 100])
    assert list(m.exit_multipliers("A1_SHORT_TRAIL3", pos, close, datr)) == [
        0,
        1,
        1,
        0,
        0,
    ]


def test_time_exit_closes_the_short_at_180_bars() -> None:
    n = 300
    pos = np.zeros(n)
    pos[10:280] = -1.0
    close = np.linspace(100.0, 60.0, n)  # falling price: the trail never hits
    datr = np.full(n, 1.0)
    a3 = m.exit_multipliers("A3_SHORT_TIME30", pos, close, datr)
    assert (
        a3[10 + 179] == 1.0 and a3[10 + 180] == 0.0 and (a3[10 + 180 : 280] == 0).all()
    )
    assert (a3[10:190] == 1.0).all()
    # a trade that ends before 180 bars is untouched
    pos2 = np.zeros(n)
    pos2[10:100] = -1.0
    assert (
        m.exit_multipliers("A3_SHORT_TIME30", pos2, close, datr)[10:100] == 1.0
    ).all()


def test_longs_always_keep_the_rule_exit() -> None:
    pos = np.zeros(300)
    pos[5:290] = 1.0
    close = np.concatenate([np.linspace(100, 20, 150), np.linspace(20, 200, 150)])
    datr = np.full(300, 0.5)
    for arm in m.ARMS:
        assert (m.exit_multipliers(arm, pos, close, datr)[5:290] == 1.0).all(), arm


def test_no_reentry_and_new_trade_resets_the_exit() -> None:
    pos = _arr(0, -1, -1, -1, -1, -1, 1, 1, 0, -1, -1, -1, 0)
    close = _arr(100, 100, 100, 120, 80, 80, 80, 80, 80, 80, 80, 80, 80)
    datr = np.full(13, 5.0)
    a1 = m.exit_multipliers("A1_SHORT_TRAIL3", pos, close, datr)
    # row 3: close 120 >= best 100 + 15 exits; rows 4-5 stay out; the long at rows 6-7 is full; the
    # short that starts at row 9 is a new trade (best resets to its entry close 80, no exit)
    assert list(a1) == [0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0]
    # a zero row between two same-sign runs separates trades
    pos2 = _arr(0, -1, -1, 0, -1, -1, 0)
    close2 = _arr(100, 100, 100, 100, 100, 130, 100)
    a1b = m.exit_multipliers("A1_SHORT_TRAIL3", pos2, close2, np.full(7, 5.0))
    assert list(a1b) == [0, 1, 1, 0, 1, 0, 0]


def test_trade_without_entry_atr_keeps_the_rule_exit() -> None:
    pos, close, datr = _short_run([100, 100, 100, 200, 300])
    datr[1] = np.nan  # no entry ATR: the exploratory policy skips the trade
    assert (m.exit_multipliers("A1_SHORT_TRAIL3", pos, close, datr)[1:4] == 1.0).all()
    datr2 = np.full(5, 5.0)
    datr2[3] = np.nan  # a non-finite ATR on a later row cannot trigger the stop
    assert list(m.exit_multipliers("A1_SHORT_TRAIL3", pos, close, datr2))[1:4] == [
        1,
        1,
        1,
    ]
    with pytest.raises(ValueError, match="unknown arm"):
        m.exit_multipliers("A9", pos, close, datr)


def test_exit_multipliers_are_causal() -> None:
    rng = np.random.default_rng(5)
    n = 700
    pos = np.zeros(n)
    i = 0
    while i < n:
        run = int(rng.integers(20, 300))
        pos[i : i + run] = float(rng.choice([-1.0, 0.0, 1.0, -0.5]))
        i += run
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    datr = close * 0.03
    for arm in m.ARMS:
        full = m.exit_multipliers(arm, pos, close, datr)
        for cut in (150, 333, 500):
            part = m.exit_multipliers(arm, pos[:cut], close[:cut], datr[:cut])
            assert np.array_equal(part, full[:cut]), (arm, cut)


# ----------------------------------------------------------------------------- books


def make_sig(
    closes: dict[str, list[float]],
    fund: dict[str, list[float]] | None = None,
    qv: float = 1e12,
) -> m.Signals:
    syms = tuple(closes)
    n = len(closes[syms[0]]) - 1
    close = {s: _arr(*v) for s, v in closes.items()}
    z = {s: np.zeros(n + 1) for s in syms}
    return m.Signals(
        T0 + m.BAR_S * np.arange(n + 1),
        syms,
        close,
        z,
        {s: np.full(n + 1, 0.2) for s in syms},
        {s: np.full(n + 1, 1.0) for s in syms},
        {s: np.full(n + 1, qv) for s in syms},
        {s: _arr(*fund[s]) if fund and s in fund else np.zeros(n + 1) for s in syms},
    )


def _w(n: int, *cols: float) -> np.ndarray:
    return np.tile(np.array(cols, dtype=float), (n, 1))


def test_units_are_held_no_free_rebalancing() -> None:
    sig = make_sig({"BTCUSDT": [100, 100, 200, 100]})
    cell = m.simulate_weights(sig, _w(3, 0.5), 0.0, 0)
    # 500 contracts bought once: marks 1.0, 1.25 (cash 50k + 500 * 200 = 150k), back to exactly 1.0
    assert cell["equity"] == pytest.approx([1.0, 1.0, 1.0, 1.5, 1.0], abs=1e-12)
    assert cell["fills"] == 1 and cell["valid"]


def test_cost_is_charged_on_actual_fill_notional_with_the_repository_formula() -> None:
    sig = make_sig({"BTCUSDT": [100, 100, 100]}, qv=1e9)
    cell = m.simulate_weights(sig, _w(2, 1.0), 1.0, 0)
    notional = m.INITIAL_CAPITAL  # target 1.0 x pre-fee equity
    cost = notional * (0.0005 + 0.0002 + 0.0001 * math.sqrt(notional / 1e9))
    assert cell["costs_total"] == pytest.approx(cost / m.INITIAL_CAPITAL, rel=1e-12)
    assert cell["equity"][1] == pytest.approx(1.0 - cost / m.INITIAL_CAPITAL, rel=1e-12)
    assert (
        cell["cost_paid"]["long"] == cell["costs_total"]
        and cell["cost_paid"]["short"] == 0.0
    )
    doubled = m.simulate_weights(sig, _w(2, 1.0), 2.0, 0)
    assert doubled["costs_total"] == pytest.approx(2 * cell["costs_total"], rel=1e-12)


def test_fill_is_capped_at_five_percent_of_volume_and_the_residual_is_carried() -> None:
    sig = make_sig({"BTCUSDT": [100.0] * 7}, qv=1e5)  # capacity 5,000 notional per bar
    cell = m.simulate_weights(sig, _w(6, 1.0), 1.0, 0)
    # 6 fill events (bars 0..5) each fill 5,000 toward the SAME 1,000-contract target
    assert cell["fills"] == 6 and cell["capped_fills"] == 6
    per_fill = 5000.0 * (0.0005 + 0.0002 + 0.0001 * math.sqrt(0.05))
    assert cell["costs_total"] == pytest.approx(
        6 * per_fill / m.INITIAL_CAPITAL, rel=1e-12
    )
    assert cell["max_gross_at_fills"] == pytest.approx(0.30, abs=0.01)


def test_funding_is_paid_on_held_units_with_the_actual_sign_and_timing() -> None:
    f = [0.0, 0.0, 0.1, 0.0]  # price * rate summed over the bar's events (bar 2)
    sig = make_sig({"BTCUSDT": [100.0] * 4}, fund={"BTCUSDT": f})
    long_ = m.simulate_weights(sig, _w(3, 1.0), 0.0, 0)
    short = m.simulate_weights(sig, _w(3, -1.0), 0.0, 0)
    assert long_["funding_total"] == pytest.approx(-1000 * 0.1 / m.INITIAL_CAPITAL)
    assert short["funding_total"] == pytest.approx(+1000 * 0.1 / m.INITIAL_CAPITAL)
    assert long_["funding_pnl"]["long"] == long_["funding_total"]
    assert short["funding_pnl"]["short"] == short["funding_total"]
    # an event in bar 1 is paid only by units held BEFORE the fill of close c_1
    sig1 = make_sig({"BTCUSDT": [100.0] * 4}, fund={"BTCUSDT": [0.0, 0.1, 0.0, 0.0]})
    assert m.simulate_weights(sig1, _w(3, 1.0), 0.0, 0)["funding_total"] < 0.0
    assert m.simulate_weights(sig1, _w(3, 1.0), 0.0, 1)["funding_total"] == 0.0


def test_delay_fills_one_bar_later_at_that_close() -> None:
    sig = make_sig({"BTCUSDT": [100, 110, 121, 121]})
    base = m.simulate_weights(sig, _w(3, 1.0), 0.0, 0)
    late = m.simulate_weights(sig, _w(3, 1.0), 0.0, 1)
    assert base["equity"] == pytest.approx([1.0, 1.0, 1.1, 1.21, 1.21], abs=1e-12)
    assert late["equity"] == pytest.approx([1.0, 1.0, 1.0, 1.1, 1.1], abs=1e-12)


def test_gross_above_limit_at_a_fill_is_invalid() -> None:
    sig = make_sig({"BTCUSDT": [100.0] * 4, "ETHUSDT": [50.0] * 4})
    ok = m.simulate_weights(sig, _w(3, 0.7, 0.7), 0.0, 0)
    assert ok["valid"] and ok["max_gross_at_fills"] == pytest.approx(1.4)
    bad = m.simulate_weights(sig, _w(3, 0.8, 0.8), 0.0, 0)
    assert not bad["valid"] and any(
        "gross leverage" in r for r in bad["invalid_reasons"]
    )


def test_nonpositive_equity_is_invalid() -> None:
    sig = make_sig({"BTCUSDT": [100, 100, 260, 260]})
    cell = m.simulate_weights(sig, _w(3, -1.0), 0.0, 0)
    assert not cell["valid"] and any(
        "not positive" in r for r in cell["invalid_reasons"]
    )


def test_identity_break_is_detected(monkeypatch: pytest.MonkeyPatch) -> None:
    orig = m._Book._try_fill

    def leaky(self: Any, *a: Any) -> bool:
        out: bool = orig(self, *a)
        self.cash += 1.0
        return out

    monkeypatch.setattr(m._Book, "_try_fill", leaky)
    cell = m.simulate_weights(make_sig({"BTCUSDT": [100.0] * 3}), _w(2, 0.5), 1.0, 0)
    assert not cell["valid"] and any("identity" in r for r in cell["invalid_reasons"])


def test_prices_must_be_positive_and_present() -> None:
    with pytest.raises(ValueError, match="non-positive or missing"):
        m.simulate_weights(
            make_sig({"BTCUSDT": [100, float("nan"), 100]}), _w(2, 0.5), 0.0, 0
        )
    with pytest.raises(ValueError, match="non-positive or missing"):
        m.simulate_weights(make_sig({"BTCUSDT": [100, 0.0, 100]}), _w(2, 0.5), 0.0, 0)


def test_side_attribution_reconciles_with_equity() -> None:
    sig = make_sig(
        {"BTCUSDT": [100, 100, 120, 130, 125], "ETHUSDT": [50, 50, 45, 40, 44]},
        fund={"BTCUSDT": [0, 0, 0.05, 0.05, 0], "ETHUSDT": [0, 0, -0.02, 0.01, 0]},
    )
    cell = m.simulate_weights(sig, _w(4, 0.4, -0.4), 1.0, 0)
    assert cell["valid"]
    total = sum(cell["price_pnl"].values()) + sum(cell["funding_pnl"].values())
    total -= sum(cell["cost_paid"].values())
    assert total == pytest.approx(cell["total_return"], abs=1e-12)
    assert cell["price_pnl"]["long"] > 0.0 and cell["price_pnl"]["short"] > 0.0


def test_hold_and_cash_cells() -> None:
    sig = make_sig({"BTCUSDT": [100, 100, 110, 121], "ETHUSDT": [10, 10, 10, 10]})
    cells = m.evaluate_cells(sig)
    assert set(cells) == set(m.all_cell_names())
    hold = cells["C_HOLD_BTC"]
    assert hold["fills"] == 1 and hold["total_return"] > 0.2
    cash = cells["C_CASH"]
    assert cash["total_return"] == 0.0 and cash["sharpe"] == 0.0 and cash["valid"]
    assert len(cash["returns"]) == len(hold["returns"]) == sig.bars + 1


# ----------------------------------------------------------------------------- independent oracle


def random_signals(seed: int, bars: int = 700, symbols: int = 3) -> m.Signals:
    rng = np.random.default_rng(seed)
    syms = m.SYMBOLS[:symbols]
    close, regime, vol, datr, qv, fund = {}, {}, {}, {}, {}, {}
    for s in syms:
        close[s] = 100 * np.exp(np.cumsum(rng.normal(0, 0.012, bars + 1)))
        reg = np.zeros(bars + 1)
        i = 0
        while i <= bars:
            run = int(rng.integers(20, 300))
            reg[i : i + run] = float(rng.choice([-1.0, -0.5, 0.0, 0.5, 1.0]))
            i += run
        regime[s] = reg
        v = 0.3 + 0.9 * np.abs(np.sin(np.arange(bars + 1) / 40 + rng.uniform(0, 3)))
        v[rng.integers(0, bars + 1, 3)] = np.nan
        vol[s] = v
        a = close[s] * rng.uniform(0.01, 0.04, bars + 1)
        a[:2] = np.nan
        datr[s] = a
        qv[s] = rng.uniform(2e5, 6e6, bars + 1)
        f = np.zeros(bars + 1)
        f[1::2] = rng.normal(0, 0.05, len(f[1::2]))
        fund[s] = f
    return m.Signals(
        T0 + m.BAR_S * np.arange(bars + 1), syms, close, regime, vol, datr, qv, fund
    )


def _sg(x: float) -> int:
    return int(x > 0) - int(x < 0)


def oracle_positions(regime: Any, vol: Any, close: Any) -> list[float]:
    out, pos = [], 0.0
    for r, v, c in zip(regime, vol, close, strict=True):
        if math.isfinite(v) and math.isfinite(c):
            direction = 1 if r >= 0.5 else (-1 if r <= -0.5 else 0)
            new = direction * (1.0 if v <= 0 else min(1.0, 0.5 / v))
            same = pos != 0.0 and _sg(new) == _sg(pos)
            pos = pos if same else new
        out.append(pos)
    return out


def oracle_multipliers(
    arm: str, pos: list[float], close: Any, datr: Any
) -> list[float]:
    """Trade-first formulation (as the exploratory ``enumerate_trades`` + policy): find each
    maximal same-sign run, then apply the arm's exit to the whole run."""
    n = len(pos)
    mult = [0.0] * n
    sgn = [_sg(p) for p in pos]
    t = 0
    while t < n:
        if sgn[t] == 0:
            t += 1
            continue
        e = t
        while e + 1 < n and sgn[e + 1] == sgn[t]:
            e += 1
        short = sgn[t] < 0
        for b in range(t, e + 1):
            mult[b] = 0.5 if (short and arm == "A2_SHORT_HALF") else 1.0
        ok = math.isfinite(datr[t]) and datr[t] > 0 and math.isfinite(close[t])
        if short and ok and e > t:
            if arm == "A1_SHORT_TRAIL3":
                for b in range(t + 1, e + 1):
                    lowest = min(close[t : b + 1])
                    if close[b] >= lowest + 3.0 * datr[b]:
                        for k in range(b, e + 1):
                            mult[k] = 0.0
                        break
            if arm == "A3_SHORT_TIME30" and e >= t + 180:
                for k in range(t + 180, e + 1):
                    mult[k] = 0.0
        t = e + 1
    return mult


def oracle_book(sig: m.Signals, w: np.ndarray, mult: float, delay: int) -> list[float]:
    """Plain-python replay (floats and dicts, equity as cash plus units times price)."""
    syms, n = list(sig.symbols), sig.bars
    cap = m.INITIAL_CAPITAL
    cash = cap
    units = dict.fromkeys(syms, 0.0)
    prev_w = dict.fromkeys(syms, 0.0)
    pend: dict[str, float | None] = dict.fromkeys(syms)
    eq = [1.0]
    for j in range(n + 1):
        if j >= 1:
            for s in syms:
                cash -= units[s] * float(sig.fund[s][j])
        px = {s: float(sig.close[s][j]) for s in syms}
        d = j - delay
        if 0 <= d <= n - 1:
            e_pre = cash + sum(units[s] * px[s] for s in syms)
            for k, s in enumerate(syms):
                if abs(float(w[d][k]) - prev_w[s]) > 1e-12:
                    prev_w[s] = float(w[d][k])
                    pend[s] = prev_w[s] * e_pre / px[s]
                target = pend[s]
                if target is None:
                    continue
                delta = target - units[s]
                want = abs(delta) * px[s]
                if want <= 1e-15 * max(e_pre, 1.0):
                    pend[s] = None
                    continue
                vol = float(sig.fill_qv[s][j])
                limit = 0.05 * vol
                filled = min(want, limit)
                rate = mult * (0.0005 + 0.0002 + 0.0001 * math.sqrt(filled / vol))
                old = units[s]
                if want > limit:
                    units[s] = old + math.copysign(filled / px[s], delta)
                else:
                    units[s] = target
                    pend[s] = None
                cash -= (units[s] - old) * px[s] + filled * rate
        eq.append((cash + sum(units[s] * px[s] for s in syms)) / cap)
    return eq


def test_exit_multipliers_match_trade_first_oracle() -> None:
    sig = random_signals(0)
    for s in sig.symbols:
        pos = m.rule_positions(sig.regime[s][:-1], sig.vol[s][:-1], sig.close[s][:-1])
        assert list(pos) == pytest.approx(
            oracle_positions(sig.regime[s][:-1], sig.vol[s][:-1], sig.close[s][:-1])
        )
        for arm in m.ARMS:
            got = m.exit_multipliers(arm, pos, sig.close[s][:-1], sig.datr[s][:-1])
            want = oracle_multipliers(
                arm, list(pos), sig.close[s][:-1], sig.datr[s][:-1]
            )
            assert list(got) == want, (s, arm)


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("arm", m.ARMS)
@pytest.mark.parametrize("variant", ["BASE", "HIGH_COST", "DELAY"])
def test_engine_matches_independent_oracle(seed: int, arm: str, variant: str) -> None:
    sig = random_signals(seed)
    mult, delay = m.VARIANTS[variant]
    cell = m.simulate_weights(sig, m.arm_weights(sig, arm), mult, delay)
    syms = list(sig.symbols)
    w = np.zeros((sig.bars, len(syms)))
    for k, s in enumerate(syms):
        pos = oracle_positions(sig.regime[s][:-1], sig.vol[s][:-1], sig.close[s][:-1])
        mm = oracle_multipliers(arm, pos, sig.close[s][:-1], sig.datr[s][:-1])
        w[:, k] = np.array(pos) * np.array(mm) / len(syms)
    assert cell["valid"], cell["invalid_reasons"]
    assert np.allclose(
        cell["equity"], oracle_book(sig, w, mult, delay), rtol=0, atol=1e-9
    )


def test_oracle_scenario_is_nontrivial() -> None:
    sig = random_signals(0)
    cells = m.evaluate_cells(sig)
    assert all(c["valid"] for c in cells.values())
    base = cells["A0_RULE__BASE"]
    assert base["fills"] > 10 and base["capped_fills"] > 0
    assert base["price_pnl"]["short"] != 0.0 and base["price_pnl"]["long"] != 0.0
    assert base["funding_total"] != 0.0 and base["costs_total"] > 0.0
    # The oracle must distinguish each challenger from the BASELINE SHORT LEG, not compare
    # unrelated long and short totals.  Decisions on the long leg must remain identical.
    assert m.CHALLENGERS == ("A1_SHORT_TRAIL3", "A2_SHORT_HALF", "A3_SHORT_TIME30")
    base_short = base["price_pnl"]["short"]
    assert math.isfinite(base_short)
    w0 = m.arm_weights(sig, "A0_RULE")
    for arm in m.CHALLENGERS:
        other = cells[m.cell_name(arm, "BASE")]
        other_short = other["price_pnl"]["short"]
        assert math.isfinite(other_short), arm
        assert other_short != base_short, arm
        w = m.arm_weights(sig, arm)
        assert np.array_equal(np.maximum(w0, 0), np.maximum(w, 0)), arm
        assert (np.minimum(w, 0) >= np.minimum(w0, 0) - 1e-15).all(), arm


# ----------------------------------------------------------------------------- decision


def _cell(returns: np.ndarray, dd: float = 0.1, valid: bool = True) -> dict[str, Any]:
    return {
        "returns": returns,
        "max_dd": dd,
        "sharpe": float(np.mean(returns) / np.std(returns)) if np.std(returns) else 0.0,
        "valid": valid,
        "total_return": float(np.prod(1 + returns) - 1),
    }


def _cells(
    offsets: dict[str, tuple[float, float, float]], dd: dict[str, float] | None = None
) -> Any:
    """BASE / HIGH_COST / DELAY return offsets (per bar) of each challenger versus A0."""
    rng = np.random.default_rng(1)
    r0 = {v: rng.normal(0.0, 0.01, 600) for v in m.VARIANTS}
    noise = {a: rng.normal(0.0, 1e-4, 600) for a in m.CHALLENGERS}
    cells = {}
    for v in m.VARIANTS:
        cells[m.cell_name("A0_RULE", v)] = _cell(r0[v], 0.20)
        for a in m.CHALLENGERS:
            off = offsets[a][list(m.VARIANTS).index(v)]
            cells[m.cell_name(a, v)] = _cell(
                r0[v] + off + noise[a], (dd or {}).get(a, 0.20)
            )
    cells["C_HOLD_BTC"] = _cell(rng.normal(0.0, 0.02, 600), 0.5)
    cells["C_CASH"] = _cell(np.zeros(600), 0.0)
    return cells


GOOD = (2e-3, 1e-3, 1e-3)


def test_bootstrap_matches_reference_and_is_paired() -> None:
    rng = np.random.default_rng(3)
    x = rng.normal(0, 1, 500)
    y = rng.normal(0, 1, 500)
    out = m.block_bootstrap_means({"x": x, "y": y}, block=30, draws=400, seed=0)
    # reference: the exploratory block_bootstrap_ci scheme, row by row
    gen = np.random.default_rng(0)
    n_blocks = -(-500 // 30)
    starts = gen.integers(0, 500 - 30 + 1, size=(400, n_blocks))
    idx = (starts[:, :, None] + np.arange(30)[None, None, :]).reshape(400, -1)[:, :500]
    assert np.allclose(out["x"], [x[r].mean() for r in idx], atol=1e-12)
    assert np.allclose(out["y"], [y[r].mean() for r in idx], atol=1e-12)
    # paired: the bootstrap mean of a difference equals the difference of the bootstrap means
    d = m.block_bootstrap_means({"d": x - y}, block=30, draws=400, seed=0)["d"]
    assert np.allclose(d, out["x"] - out["y"], atol=1e-12)
    short = m.block_bootstrap_means({"a": np.arange(5.0)}, block=30, draws=10)["a"]
    assert short.shape == (10,)  # a block longer than the series is shortened
    with pytest.raises(ValueError):
        m.block_bootstrap_means({"a": np.arange(1.0)})
    with pytest.raises(ValueError):
        m.block_bootstrap_means({"a": np.arange(5.0), "b": np.arange(4.0)})


def test_holm_step_down() -> None:
    assert m.holm({"a": 0.01, "b": 0.02, "c": 0.03}) == dict.fromkeys("abc", True)
    assert m.holm({"a": 0.02, "b": 0.03, "c": 0.04}) == dict.fromkeys("abc", False)
    # a failure stops the step-down even when a later p-value is small
    assert m.holm({"a": 0.001, "b": 0.03, "c": 0.0301}) == {
        "a": True,
        "b": False,
        "c": False,
    }
    assert m.holm({"a": 0.049}) == {"a": True} and m.holm({"a": 0.051}) == {"a": False}


def test_analyze_each_condition_gates_the_arm() -> None:
    offsets = {
        "A1_SHORT_TRAIL3": GOOD,
        "A2_SHORT_HALF": (-2e-3, 1e-3, 1e-3),
        "A3_SHORT_TIME30": (0.0, 0.0, 0.0),
    }
    res = m.analyze(_cells(offsets))
    assert res["arms"]["A1_SHORT_TRAIL3"]["improves"]
    assert res["improving"] == ["A1_SHORT_TRAIL3"]
    assert not res["arms"]["A2_SHORT_HALF"]["checks"]["bootstrap_lower95_above_0"]
    assert not res["arms"]["A3_SHORT_TIME30"]["checks"]["holm_rejected"]
    # each other condition alone removes the arm
    for k, off in (
        ("high_cost_mean_diff_positive", (2e-3, -1e-3, 1e-3)),
        ("delay_mean_diff_positive", (2e-3, 1e-3, -1e-3)),
    ):
        r = m.analyze(_cells({**offsets, "A1_SHORT_TRAIL3": off}))["arms"][
            "A1_SHORT_TRAIL3"
        ]
        assert not r["checks"][k] and not r["improves"]
    r = m.analyze(_cells(offsets, dd={"A1_SHORT_TRAIL3": 0.2501}))["arms"][
        "A1_SHORT_TRAIL3"
    ]
    assert not r["checks"]["max_dd_within_slack"] and not r["improves"]
    r = m.analyze(_cells(offsets, dd={"A1_SHORT_TRAIL3": 0.25}))["arms"][
        "A1_SHORT_TRAIL3"
    ]
    assert r["checks"]["max_dd_within_slack"] and r["improves"]


def test_analyze_secondary_and_report_fields() -> None:
    res = m.analyze(_cells({a: GOOD for a in m.CHALLENGERS}))
    assert res["improving"] == list(m.CHALLENGERS)
    assert isinstance(res["secondary_a0_beats_hold_btc_sharpe"], bool)
    arm = res["arms"]["A2_SHORT_HALF"]
    assert (
        arm["base_mean_diff"] > 0
        and arm["base_lower95"] > 0
        and 0 < arm["p_one_sided"] < 0.01
    )


def _final_cells(**over: Any) -> dict[str, dict[str, Any]]:
    return {**_cells({a: GOOD for a in m.CHALLENGERS}), **over}


def test_decide_final_branches() -> None:
    out = m.decide("FORWARD_FINAL", _final_cells())
    assert out["label"] == "ARM_IMPROVES:A1_SHORT_TRAIL3,A2_SHORT_HALF,A3_SHORT_TIME30"
    cells = _cells(
        {
            "A1_SHORT_TRAIL3": GOOD,
            "A2_SHORT_HALF": (0.0,) * 3,
            "A3_SHORT_TIME30": (0.0,) * 3,
        }
    )
    assert m.decide("FORWARD_FINAL", cells)["label"] == "ARM_IMPROVES:A1_SHORT_TRAIL3"
    none = m.decide(
        "FORWARD_FINAL", _cells({a: (0.0, 0.0, 0.0) for a in m.CHALLENGERS})
    )
    assert none["label"] == "NO_ARM_IMPROVES"


def test_decide_invalid_has_priority() -> None:
    bad = _final_cells(**{"A3_SHORT_TIME30__DELAY": _cell(np.zeros(600), valid=False)})
    out = m.decide("FORWARD_FINAL", bad)
    assert out["label"] == "INVALID" and out["invalid_cells"] == [
        "A3_SHORT_TIME30__DELAY"
    ]
    assert "analysis" not in out
    assert (
        m.decide("FORWARD_FINAL", _final_cells(), ["data problem"])["label"]
        == "INVALID"
    )
    assert m.decide("FORWARD_FINAL", {}, [])["label"] == "INVALID"
    assert m.decide("FORWARD_INTERIM", bad)["label"] == "INVALID"


def test_decide_interim_branches() -> None:
    ok = m.decide("FORWARD_INTERIM", _final_cells())
    assert ok["label"] == "INTERIM_NOT_A_DECISION"
    for dd, label in ((0.2999, "INTERIM_NOT_A_DECISION"), (0.30, "STOPPED_EARLY")):
        cells = _final_cells(**{"A0_RULE__BASE": _cell(np.zeros(600), dd)})
        assert m.decide("FORWARD_INTERIM", cells)["label"] == label
    # the stop rule is on A0 BASE only: another cell's drawdown does not stop the line
    other = _final_cells(**{"A1_SHORT_TRAIL3__BASE": _cell(np.zeros(600), 0.9)})
    assert m.decide("FORWARD_INTERIM", other)["label"] == "INTERIM_NOT_A_DECISION"


def test_final_decides_even_when_a0_crosses_the_stop_and_records_it() -> None:
    """The 30% stop is an interim rule. A final whose A0 BASE drawdown crossed 30% after the last
    interim still decides the arm question and reports ``a0_stop_crossed``."""
    for dd, crossed in ((0.35, True), (0.30, True), (0.2999, False)):
        cells = _final_cells(
            **{"A0_RULE__BASE": _cell(_final_cells()["A0_RULE__BASE"]["returns"], dd)}
        )
        out = m.decide("FORWARD_FINAL", cells)
        assert (
            out["label"] == "ARM_IMPROVES:A1_SHORT_TRAIL3,A2_SHORT_HALF,A3_SHORT_TIME30"
        )
        assert out["a0_stop_crossed"] is crossed and out["a0_base_max_dd"] == dd
    none = _cells({a: (0.0, 0.0, 0.0) for a in m.CHALLENGERS})
    none["A0_RULE__BASE"] = _cell(none["A0_RULE__BASE"]["returns"], 0.9)
    out = m.decide("FORWARD_FINAL", none)
    assert out["label"] == "NO_ARM_IMPROVES" and out["a0_stop_crossed"]
    interim = m.decide("FORWARD_INTERIM", none)
    assert interim["label"] == "STOPPED_EARLY" and interim["a0_stop_crossed"]
    assert "a0_stop_crossed" not in m.decide("FORWARD_FINAL", {}, ["x"])


def test_summary_cell_is_json_safe() -> None:
    sig = random_signals(0, bars=120)
    cell = m.simulate_weights(sig, m.arm_weights(sig, "A0_RULE"), 1.0, 0)
    s = m.summarize_cell(cell)
    json.dumps(s)
    assert "returns" not in s and "equity" not in s and len(s["equity_sha256"]) == 64
    assert s["bars"] == sig.bars + 1


# ----------------------------------------------------------------------------- data parsing


def _kline(open_ms: int, c: float = 100.0) -> list[Any]:
    return [
        open_ms,
        "99",
        "101",
        "98",
        f"{c}",
        "5",
        open_ms + 3599999,
        "500",
        3,
        "2",
        "250",
        "0",
    ]


def test_parse_klines_and_funding() -> None:
    rows = m.parse_klines([_kline(3600000, 50.5)])
    assert rows == [(3600000, 99.0, 101.0, 98.0, 50.5, 500.0)]
    for bad in ({}, [[1]], [_kline(3600001)], [["x"] * 12]):
        with pytest.raises(SystemExit):
            m.parse_klines(bad)
    ev = m.parse_funding(
        [
            {"fundingTime": 1000, "fundingRate": "0.0001", "markPrice": "101.5"},
            {"fundingTime": 2000, "fundingRate": "-0.0002", "markPrice": ""},
            {"fundingTime": 3000, "fundingRate": "0", "markPrice": None},
        ]
    )
    assert ev[0] == (1000, 0.0001, 101.5)
    assert math.isnan(ev[1][2]) and math.isnan(ev[2][2]) and ev[1][1] == -0.0002
    for bad in ({}, [{"fundingTime": "x", "fundingRate": "1"}], [{"fundingRate": "1"}]):
        with pytest.raises(SystemExit):
            m.parse_funding(bad)


def test_hourly_arrays_leaves_holes_nan() -> None:
    rows = [(0, 1.0, 2.0, 0.5, 1.5, 10.0), (2 * m.HOUR_MS, 1, 1, 1, 1, 1)]
    a = m.hourly_arrays(rows, 0, 3 * m.HOUR_MS)
    assert len(a["close"]) == 4 and np.isnan(a["close"][1]) and a["close"][2] == 1.0
    assert a["qv"][0] == 10.0 and set(a) == set(m.HOURLY_KEYS)
    # rows outside the grid are ignored
    assert (
        np.isfinite(m.hourly_arrays(rows, 3600000, 3 * m.HOUR_MS)["close"]).sum() == 1
    )


def test_stage_problems_flags_missing_hours_and_bad_prices() -> None:
    good = synth_hourly(48, 0)
    assert m.stage_problems({"A": good}, 8) == []
    holey = synth_hourly(48, 0, holes=(20,))
    assert "missing hour" in m.stage_problems({"A": holey}, 8)[0]
    # a hole BEFORE the stage window only blanks feature bars: it is not a problem
    assert m.stage_problems({"A": synth_hourly(48, 0, holes=(3,))}, 8) == []
    neg = synth_hourly(48, 0)
    neg["low"][30] = 0.0
    assert any("non-positive low" in p for p in m.stage_problems({"A": neg}, 8))


# ----------------------------------------------------------------------------- fake exchange

_SYNTH: dict[str, dict[str, Any]] = {}
EXCHANGE_START_MS = int(datetime(2021, 1, 1, tzinfo=UTC).timestamp()) * 1000
EXCHANGE_HOURS = 24 * 365 * 7  # to the end of 2027


def exch(symbol: str) -> dict[str, Any]:
    if symbol not in _SYNTH:
        _SYNTH[symbol] = synth_hourly(EXCHANGE_HOURS, sum(map(ord, symbol)))
    return _SYNTH[symbol]


def fake_market(
    url: str, honour_start: bool = True, holes: dict[str, set[int]] | None = None
) -> bytes:
    u = urlparse(url)
    q = {k: v[0] for k, v in parse_qs(u.query).items()}
    sym = q["symbol"]
    a = exch(sym)
    lo_ms, hi_ms, lim = int(q["startTime"]), int(q["endTime"]), int(q["limit"])
    if u.path.endswith("klines"):
        first = (lo_ms - EXCHANGE_START_MS + m.HOUR_MS - 1) // m.HOUR_MS
        last = (hi_ms - EXCHANGE_START_MS) // m.HOUR_MS
        idx = list(range(max(first, 0), min(last, EXCHANGE_HOURS - 1) + 1))
        if (
            not honour_start
        ):  # adversarial: return the LAST ``limit`` bars at or before the end
            idx = list(range(max(0, last - lim + 1), min(last, EXCHANGE_HOURS - 1) + 1))
        idx = [i for i in idx if i not in (holes or {}).get(sym, set())][:lim]
        rows = [
            [
                EXCHANGE_START_MS + i * m.HOUR_MS,
                str(a["open"][i]),
                str(a["high"][i]),
                str(a["low"][i]),
                str(a["close"][i]),
                "1",
                EXCHANGE_START_MS + (i + 1) * m.HOUR_MS - 1,
                str(a["qv"][i]),
                1,
                "1",
                str(a["tbq"][i]),
                "0",
            ]
            for i in idx
        ]
        return json.dumps(rows).encode()
    # funding: every 8h at 00/08/16 UTC, a few ms of jitter, mark price = that hour's open
    t0 = (lo_ms + 8 * m.HOUR_MS - 1) // (8 * m.HOUR_MS) * 8 * m.HOUR_MS
    times = list(range(t0, hi_ms + 1, 8 * m.HOUR_MS))
    times = times[:lim] if honour_start else times[-lim:]
    out = []
    for t in times:
        i = (t - EXCHANGE_START_MS) // m.HOUR_MS
        out.append(
            {
                "symbol": sym,
                "fundingTime": t + 7,
                "fundingRate": f"{1e-4 * math.sin(i / 50.0):.8f}",
                "markPrice": f"{a['open'][i]:.6f}" if i % 11 else "",
            }
        )
    return json.dumps(out).encode()


def test_fetch_klines_pages_cache_hash_and_ignore_start(tmp_path: Path) -> None:
    first = EXCHANGE_START_MS
    last_open = first + (24 * 80 - 1) * m.HOUR_MS  # 80 days: two pages
    calls: list[str] = []

    def f(url: str) -> bytes:
        calls.append(url)
        return fake_market(url)

    rows, pages = m.fetch_klines("BTCUSDT", first, last_open, tmp_path / "raw", f)
    assert len(rows) == 24 * 80 and len(pages) == 2 and len(calls) == 2
    assert "interval=1h" in calls[0] and "limit=1500" in calls[0]
    for p in pages:
        assert m.sha256_file(tmp_path / "raw" / p["file"]) == p["sha256"]
    assert [r[0] for r in rows] == [first + i * m.HOUR_MS for i in range(24 * 80)]
    # an API that ignores startTime and returns the last 1500 bars before the end gives the same rows
    rows2, _ = m.fetch_klines(
        "BTCUSDT",
        first,
        last_open,
        tmp_path / "raw2",
        lambda u: fake_market(u, honour_start=False),
    )
    assert rows2 == rows


def test_fetch_klines_refuses_duplicates_bad_json_and_overwrite(tmp_path: Path) -> None:
    first = EXCHANGE_START_MS
    with pytest.raises(SystemExit, match="not JSON"):
        m.fetch_klines("BTCUSDT", first, first, tmp_path / "a", lambda u: b"<html>")
    dup = json.dumps([_kline(first), _kline(first)]).encode()
    with pytest.raises(SystemExit, match="duplicate"):
        m.fetch_klines("BTCUSDT", first, first, tmp_path / "b", lambda u: dup)
    # raw pages are write-once
    m.fetch_klines(
        "BTCUSDT",
        first,
        first,
        tmp_path / "c",
        lambda u: json.dumps([_kline(first)]).encode(),
    )
    with pytest.raises(FileExistsError):
        m.fetch_klines("BTCUSDT", first, first, tmp_path / "c", lambda u: b"[]")


def test_fetch_klines_missing_hour_is_a_hole_not_a_fill(tmp_path: Path) -> None:
    first = EXCHANGE_START_MS
    rows, _ = m.fetch_klines(
        "BTCUSDT",
        first,
        first + 47 * m.HOUR_MS,
        tmp_path / "raw",
        lambda u: fake_market(u, holes={"BTCUSDT": {5}}),
    )
    arrs = m.hourly_arrays(rows, first, first + 47 * m.HOUR_MS)
    assert np.isnan(arrs["close"][5]) and np.isfinite(arrs["close"][4])


def test_fetch_funding_paging_and_coverage(tmp_path: Path) -> None:
    start = EXCHANGE_START_MS
    end = start + 400 * 24 * m.HOUR_MS  # 1200 events: two pages
    ev, pages = m.fetch_funding("BTCUSDT", start, end, tmp_path / "raw", fake_market)
    assert len(pages) == 2 and len(ev) in (1200, 1201)
    times = [e[0] for e in ev]
    assert times == sorted(times) and len(set(times)) == len(times)
    assert m.funding_coverage_problems("BTCUSDT", ev, start // 1000, end // 1000) == []
    # an API that ignores startTime returns the newest 1000 events: coverage catches the hole
    try:
        ev2, _ = m.fetch_funding(
            "BTCUSDT",
            start,
            end,
            tmp_path / "raw2",
            lambda u: fake_market(u, honour_start=False),
        )
    except SystemExit as exc:  # the pager itself notices it cannot advance
        assert "no progress" in str(exc)
    else:
        problems = m.funding_coverage_problems(
            "BTCUSDT", ev2, start // 1000, end // 1000
        )
        assert problems and "starts late" in problems[0]
    assert m.funding_coverage_problems("X", [], 0, 10)[0].startswith("X: no funding")
    gap = [(0, 0.0, 1.0), (10 * m.HOUR_MS, 0.0, 1.0)]
    assert "gap" in m.funding_coverage_problems("X", gap, 0, 10 * 3600)[0]


def test_funding_coefficients_boundaries(tmp_path: Path) -> None:
    c0 = T0
    hour_open = np.full(24 * 3, 50.0)
    grid_start_ms = (c0 - 24 * 3600) * 1000  # grid starts one day before c0
    ev = [
        (c0 * 1000 - 1, 1.0, 100.0),  # before the first fill: ignored
        (c0 * 1000 + 5, 0.01, 100.0),  # just after c_0 -> bar 1
        (
            (c0 + m.BAR_S) * 1000 + 3,
            0.02,
            math.nan,
        ),  # exactly after c_1 -> bar 2, hour-open price
        ((c0 + 2 * m.BAR_S) * 1000 - 1, 0.03, 200.0),  # last ms of bar 2
        ((c0 + 3 * m.BAR_S) * 1000, 9.0, 100.0),  # at the final mark close: not paid
    ]
    coef, fallbacks = m.funding_coefficients(ev, hour_open, grid_start_ms, c0, 3)
    assert coef.tolist() == pytest.approx([0.0, 1.0, 0.02 * 50.0 + 0.03 * 200.0, 0.0])
    assert fallbacks == 1


def test_retrying_backs_off_and_gives_up() -> None:
    waits: list[float] = []
    state = {"n": 0}

    def once(url: str) -> bytes:
        state["n"] += 1
        if state["n"] < 3:
            raise urllib.error.HTTPError(url, 429, "slow down", {}, None)  # type: ignore[arg-type]
        return b"ok"

    assert m.retrying(once, waits.append)("https://x") == b"ok"
    assert waits == [5.0, 10.0]

    def dead(url: str) -> bytes:
        raise urllib.error.HTTPError(url, 404, "nope", {}, None)  # type: ignore[arg-type]

    with pytest.raises(SystemExit, match="HTTP 404"):
        m.retrying(dead, waits.append)("https://x")

    def flaky(url: str) -> bytes:
        raise TimeoutError("t")

    with pytest.raises(SystemExit, match="network failure"):
        m.retrying(flaky, lambda s: None, attempts=3)("https://x")


# ----------------------------------------------------------------------------- authorization


def _comment(
    body: str,
    login: str = "shuntatsu",
    issue_url: str = m.ISSUE_URL,
    updated: str = "t1",
) -> dict[str, Any]:
    return {
        "user": {"login": login},
        "issue_url": issue_url,
        "body": body,
        "updated_at": updated,
    }


def _gh(relay_body: str | None = None, **over: Any) -> Any:
    digest = "d" * 64
    relay_body = (
        relay_body
        if relay_body is not None
        else f"G4 identity: {m.PROTOCOL_ID} {digest} {COMMIT}\nG4 claim: FORWARD_FINAL"
    )

    def fetch(cid: int) -> dict[str, Any]:
        if cid == m.DATA_SCOPE_COMMENT_ID:
            return _comment("scope", **over.get("scope", {}))
        return _comment(relay_body, **over.get("relay", {}))

    return fetch


def test_g4_relay_requires_exact_stage_claim() -> None:
    digest = "d" * 64
    good = f"G4 identity: {m.PROTOCOL_ID} {digest} {COMMIT}\nG4 claim: FORWARD_FINAL"
    rec = m.authorization_receipt(
        URL, digest, COMMIT, "FORWARD_FINAL", _gh(relay_body=good)
    )
    assert rec["g4_relay"]["comment_id"] == 7040999999
    for invalid in (
        f"G4 identity: {m.PROTOCOL_ID} {digest} {COMMIT}",
        good.replace("FORWARD_FINAL", "FORWARD_INTERIM_2027-01"),
        good + "\nG4 claim: FORWARD_INTERIM_2027-01",
        good.replace("G4 claim:", "Not G4 claim:"),
    ):
        with pytest.raises(SystemExit, match="G4 claim"):
            m.authorization_receipt(
                URL, digest, COMMIT, "FORWARD_FINAL", _gh(relay_body=invalid)
            )


def test_g4_comment_id_is_exclusive_across_stages_and_digests(tmp_path: Path) -> None:
    root = tmp_path / "protocol"
    now = datetime(2027, 1, 20, tzinfo=UTC)
    first = m.reserve_g4_comment(
        root, 7040999999, "FORWARD_INTERIM_2027-01", "d" * 64, COMMIT, now
    )
    assert first.exists()
    with pytest.raises(SystemExit, match="G4 comment.*already"):
        m.reserve_g4_comment(root, 7040999999, "FORWARD_FINAL", "e" * 64, COMMIT, now)
    assert not (root / "FORWARD_FINAL.claim").exists()
    second = m.reserve_g4_comment(
        root, 7040999998, "FORWARD_FINAL", "e" * 64, COMMIT, now
    )
    assert second.exists()


def test_g4_reuse_detects_historical_claim_without_new_marker(tmp_path: Path) -> None:
    root = tmp_path / "protocol"
    now = datetime(2027, 1, 20, tzinfo=UTC)
    m.claim_stage(
        "FORWARD_INTERIM_2027-01", "d" * 64, COMMIT, 7040999999, now, root / ("d" * 64)
    )
    with pytest.raises(SystemExit, match="G4 comment.*already"):
        m.reserve_g4_comment(root, 7040999999, "FORWARD_FINAL", "e" * 64, COMMIT, now)


def test_authorization_ok() -> None:
    rec = m.authorization_receipt(URL, "d" * 64, COMMIT, "FORWARD_FINAL", _gh())
    assert rec["g4_relay"]["comment_id"] == 7040999999
    assert rec["data_scope"]["comment_id"] == m.DATA_SCOPE_COMMENT_ID
    assert len(rec["g4_relay"]["body_sha256"]) == 64


@pytest.mark.parametrize(
    "url",
    [
        "https://github.com/shuntatsu/trade_rl/issues/811#issuecomment-1",
        "https://github.com/shuntatsu/trade_rl/pull/810#issuecomment-1",
        "https://github.com/other/trade_rl/issues/810#issuecomment-1",
        URL + "x",
    ],
)
def test_authorization_wrong_url(url: str) -> None:
    with pytest.raises(SystemExit, match="must be an Issue #810 comment"):
        m.authorization_receipt(url, "d" * 64, COMMIT, "FORWARD_FINAL", _gh())


def test_authorization_owner_and_issue_binding() -> None:
    with pytest.raises(SystemExit, match="not owner-authored"):
        m.authorization_receipt(
            URL, "d" * 64, COMMIT, "FORWARD_FINAL", _gh(relay={"login": "mallory"})
        )
    with pytest.raises(SystemExit, match="not owner-authored"):
        m.authorization_receipt(
            URL, "d" * 64, COMMIT, "FORWARD_FINAL", _gh(scope={"login": "mallory"})
        )
    with pytest.raises(SystemExit, match="not on Issue #810"):
        m.authorization_receipt(
            URL, "d" * 64, COMMIT, "FORWARD_FINAL", _gh(relay={"issue_url": "x"})
        )


def test_authorization_edit_between_fetches() -> None:
    calls: dict[int, int] = {}

    def fetch(cid: int) -> dict[str, Any]:
        calls[cid] = calls.get(cid, 0) + 1
        body = f"G4 identity: {m.PROTOCOL_ID} {'d' * 64} {COMMIT}"
        return _comment(body, updated=f"t{calls[cid]}")

    with pytest.raises(SystemExit, match="changed between fetches"):
        m.authorization_receipt(URL, "d" * 64, COMMIT, "FORWARD_FINAL", fetch)


@pytest.mark.parametrize(
    "body",
    [
        "",
        "trend_exit_arms_v1",
        f"{'d' * 64} {COMMIT}",
        f"trend_exit_arms_v1 {'d' * 64}",
        f"trend_exit_arms_v1 {'e' * 64} {COMMIT}",
        f"trend_exit_arms_v1 {'d' * 64} {'b' * 40}",
    ],
)
def test_authorization_relay_must_quote_id_digest_commit(body: str) -> None:
    with pytest.raises(SystemExit, match="does not quote"):
        m.authorization_receipt(
            URL, "d" * 64, COMMIT, "FORWARD_FINAL", _gh(relay_body=body)
        )


def test_authorization_requires_full_commit() -> None:
    with pytest.raises(SystemExit, match="full 40-hex"):
        m.authorization_receipt(URL, "d" * 64, "abc123", "FORWARD_FINAL", _gh())


def test_harness_commit_refuses_untracked_files_but_not_ignored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import subprocess

    def git(*a: str) -> None:
        subprocess.run(
            ["git", "-C", str(tmp_path), *a], check=True, capture_output=True
        )

    git("init", "-q")
    git("config", "user.email", "t@example.com")
    git("config", "user.name", "t")
    (tmp_path / ".gitignore").write_text("ignored.txt\n")
    (tmp_path / "a.py").write_text("x = 1\n")
    git("add", ".")
    git("commit", "-q", "-m", "init")
    monkeypatch.setattr(m, "REPO_ROOT", tmp_path)
    assert m._COMMIT_RE.fullmatch(m.harness_commit())
    (tmp_path / "ignored.txt").write_text("ok\n")
    assert m._COMMIT_RE.fullmatch(m.harness_commit())
    (tmp_path / "stray_module.py").write_text("y = 2\n")
    with pytest.raises(SystemExit, match="not clean"):
        m.harness_commit()


def test_harness_commit_requires_clean_tree(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        m, "_git", lambda *a: " M tools/x.py\n" if a[0] == "status" else COMMIT
    )
    with pytest.raises(SystemExit, match="not clean"):
        m.harness_commit()
    monkeypatch.setattr(m, "_git", lambda *a: "" if a[0] == "status" else COMMIT + "\n")
    assert m.harness_commit() == COMMIT
    monkeypatch.setattr(m, "_git", lambda *a: "" if a[0] == "status" else "zz\n")
    with pytest.raises(SystemExit, match="unexpected HEAD"):
        m.harness_commit()


# ----------------------------------------------------------------------------- ledger & clock


def test_ledger_claim_is_exclusive(tmp_path: Path) -> None:
    now = datetime(2026, 12, 5, tzinfo=UTC)
    p = m.claim_stage("FORWARD_FINAL", "d" * 64, COMMIT, 7, now, tmp_path / "led")
    assert json.loads(p.read_text())["authorization_comment_id"] == 7
    with pytest.raises(SystemExit, match="already claimed"):
        m.claim_stage("FORWARD_FINAL", "d" * 64, COMMIT, 7, now, tmp_path / "led")


def test_interim_claim_is_once_per_month() -> None:
    st = m.STAGES["FORWARD_INTERIM"]
    assert (
        m.claim_name(st, datetime(2026, 12, 5, tzinfo=UTC)) == "FORWARD_INTERIM_2026-12"
    )
    assert (
        m.claim_name(st, datetime(2027, 1, 1, tzinfo=UTC)) == "FORWARD_INTERIM_2027-01"
    )
    assert (
        m.claim_name(m.STAGES["FORWARD_FINAL"], datetime(2027, 12, 5, tzinfo=UTC))
        == "FORWARD_FINAL"
    )


def test_clock_refusals() -> None:
    fi, ff = m.STAGES["FORWARD_INTERIM"], m.STAGES["FORWARD_FINAL"]
    with pytest.raises(SystemExit, match="may not run before"):
        m.check_clock(fi, datetime(2026, 12, 1, 23, 59, tzinfo=UTC))
    m.check_clock(fi, datetime(2026, 12, 2, tzinfo=UTC))
    with pytest.raises(SystemExit, match="may not run before"):
        m.check_clock(ff, datetime(2027, 12, 1, 23, 59, tzinfo=UTC))
    m.check_clock(ff, datetime(2027, 12, 2, tzinfo=UTC))
    with pytest.raises(SystemExit, match="may not run at or after"):
        m.check_clock(fi, datetime(2027, 12, 2, tzinfo=UTC))


def test_stage_window_interim_ends_at_last_settled_4h_close() -> None:
    fi = m.STAGES["FORWARD_INTERIM"]
    assert m.stage_window(fi, datetime(2026, 12, 5, 13, 30, tzinfo=UTC)) == datetime(
        2026, 12, 5, 12, tzinfo=UTC
    )
    # the newest 5 minutes are ignored: 12:04 still ends at 08:00
    assert m.stage_window(fi, datetime(2026, 12, 5, 12, 4, tzinfo=UTC)) == datetime(
        2026, 12, 5, 8, tzinfo=UTC
    )
    assert m.stage_window(fi, datetime(2026, 12, 5, 12, 5, tzinfo=UTC)) == datetime(
        2026, 12, 5, 12, tzinfo=UTC
    )
    assert m.stage_window(fi, datetime(2027, 11, 30, 22, tzinfo=UTC)) == datetime(
        2027, 11, 30, 20, tzinfo=UTC
    )
    # an interim never reaches past the stage end
    assert m.stage_window(fi, datetime(2027, 12, 1, 22, tzinfo=UTC)) == datetime(
        2027, 12, 1, tzinfo=UTC
    )
    assert m.stage_window(
        m.STAGES["FORWARD_FINAL"], datetime(2028, 1, 1, tzinfo=UTC)
    ) == datetime(2027, 12, 1, tzinfo=UTC)
    with pytest.raises(SystemExit, match="no complete 4h bar"):
        m.stage_window(fi, datetime(2026, 12, 1, 4, 1, tzinfo=UTC))


# ----------------------------------------------------------------------------- stage runs


# ----------------------------------------------------------------------------- stage runs


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    monkeypatch.setattr(m, "harness_commit", lambda: COMMIT)
    digest = m.protocol_digest()
    gh = _gh(relay_body=f"G4 identity: {m.PROTOCOL_ID} {digest} {COMMIT}")
    return {"ledger": tmp_path / "ledger", "tmp": tmp_path, "gh": gh, "digest": digest}


TWO = ("BTCUSDT", "ETHUSDT")


def _run(
    env: dict[str, Any],
    stage: Any,
    now: datetime,
    tag: str = "x",
    symbols: tuple[str, ...] = TWO,
    fetch: Any = fake_market,
) -> dict[str, Any]:
    concrete_stage = m.STAGES[stage] if isinstance(stage, str) else stage
    stage_claim = m.claim_name(concrete_stage, now)
    cid = 7000000000 + int.from_bytes(
        hashlib.sha256(stage_claim.encode()).digest()[:4], "big"
    )
    url = f"https://github.com/shuntatsu/trade_rl/issues/810#issuecomment-{cid}"
    gh = _gh(
        relay_body=f"G4 identity: {m.PROTOCOL_ID} {env['digest']} {COMMIT}\nG4 claim: {stage_claim}"
    )
    return m.run_stage(
        stage,
        url,
        env["tmp"] / f"work_{tag}",
        env["tmp"] / f"out_{tag}",
        now=now,
        fetch=fetch,
        gh_fetch=gh,
        ledger_dir=env["ledger"],
        symbols=symbols,
    )


def test_interim_end_to_end_write_once_and_ledger(env: dict[str, Any]) -> None:
    now = datetime(2027, 1, 20, 9, tzinfo=UTC)  # 50 days
    res = _run(env, "FORWARD_INTERIM", now)
    assert res["label"] in {"INTERIM_NOT_A_DECISION", "STOPPED_EARLY"}, (
        res["problems"] + res["invalid_cells"]
    )
    assert set(res["cells"]) == set(m.all_cell_names())
    assert res["window"]["bars"] == 50 * 6 + 2
    assert res["window"]["last_mark_close"].startswith("2027-01-20T08")
    assert res["claim_name"] == "FORWARD_INTERIM_2027-01"
    assert set(res["analysis"]["arms"]) == set(m.CHALLENGERS)
    assert res["a0_stop_crossed"] == (res["cells"]["A0_RULE__BASE"]["max_dd"] >= 0.30)
    out = env["tmp"] / "out_x" / "result.json"
    led = json.loads(
        (
            env["ledger"]
            / m.PROTOCOL_ID
            / env["digest"]
            / "FORWARD_INTERIM_2027-01.result.json"
        ).read_text()
    )
    assert led["result_sha256"] == m.sha256_file(out) and led["label"] == res["label"]
    on_disk = json.loads(out.read_text())
    assert on_disk["protocol_digest"] == env["digest"] and on_disk["commit"] == COMMIT
    raw = env["tmp"] / "work_x" / "raw"
    for sym in TWO:
        for kind in ("klines", "funding"):
            for p in res["data"]["pages"][sym][kind]:
                assert m.sha256_file(raw / p["file"]) == p["sha256"]
    with pytest.raises(SystemExit, match="already claimed"):
        _run(env, "FORWARD_INTERIM", now + timedelta(days=3), tag="again")
    with pytest.raises(SystemExit, match="output directory exists"):
        _run(env, "FORWARD_INTERIM", now + timedelta(days=40), tag="x")


def test_full_sequence_interims_and_final(env: dict[str, Any]) -> None:
    _run(env, "FORWARD_INTERIM", datetime(2026, 12, 20, 9, tzinfo=UTC), tag="dec")
    _run(env, "FORWARD_INTERIM", datetime(2027, 1, 20, 9, tzinfo=UTC), tag="jan")
    fin = _run(env, "FORWARD_FINAL", datetime(2027, 12, 2, 1, tzinfo=UTC), tag="fin")
    assert fin["label"] == "NO_ARM_IMPROVES" or fin["label"].startswith(
        "ARM_IMPROVES:"
    ), fin["problems"]
    assert fin["window"]["bars"] == 365 * 6
    assert fin["window"]["last_mark_close"].startswith("2027-12-01T00")
    assert fin["claim_name"] == "FORWARD_FINAL"
    assert fin["cells"]["A0_RULE__BASE"]["bars"] == 365 * 6 + 1
    with pytest.raises(SystemExit, match="already claimed"):
        _run(env, "FORWARD_FINAL", datetime(2027, 12, 3, tzinfo=UTC), tag="dup")
    # reported cells and equity digests are deterministic: a second environment agrees
    env2 = {**env, "ledger": env["tmp"] / "ledger2"}
    fin2 = _run(env2, "FORWARD_FINAL", datetime(2027, 12, 2, 1, tzinfo=UTC), tag="fin2")
    assert {k: v["equity_sha256"] for k, v in fin["cells"].items()} == {
        k: v["equity_sha256"] for k, v in fin2["cells"].items()
    }
    assert fin["label"] == fin2["label"] and fin["analysis"] == fin2["analysis"]


def _old_ns(env: dict[str, Any], digest: str = "9" * 64) -> Path:
    """A ledger namespace left by an EARLIER digest of the same protocol id."""
    return env["ledger"] / m.PROTOCOL_ID / digest


def _seed(
    ns: Path, name: str, label: str, claim: bool = True, digest: str = "9" * 64
) -> None:
    if claim:
        m.claim_stage(name, digest, COMMIT, 1, datetime(2026, 12, 20, tzinfo=UTC), ns)
    m.write_ledger_result(name, digest, label, False, "s", ns)


def test_old_stopped_early_blocks_interims_and_final_under_a_new_digest(
    env: dict[str, Any],
) -> None:
    ns = _old_ns(env)
    _seed(ns, "FORWARD_INTERIM_2026-12", "STOPPED_EARLY")
    assert env["digest"] != "9" * 64
    with pytest.raises(SystemExit, match="line stopped"):
        _run(env, "FORWARD_INTERIM", datetime(2027, 1, 20, 9, tzinfo=UTC))
    with pytest.raises(SystemExit, match="line stopped"):
        _run(env, "FORWARD_FINAL", datetime(2027, 12, 2, 1, tzinfo=UTC), tag="fin")
    assert not (env["ledger"] / m.PROTOCOL_ID / env["digest"]).exists()


def test_old_final_blocks_a_new_digest_from_reclaiming_the_final(
    env: dict[str, Any],
) -> None:
    _seed(_old_ns(env), "FORWARD_FINAL", "NO_ARM_IMPROVES")
    with pytest.raises(SystemExit, match="single use applies across digests"):
        _run(env, "FORWARD_FINAL", datetime(2027, 12, 2, 1, tzinfo=UTC), tag="fin")
    assert not (env["ledger"] / m.PROTOCOL_ID / env["digest"]).exists()


def test_old_claim_without_result_still_blocks_the_final(env: dict[str, Any]) -> None:
    m.claim_stage(
        "FORWARD_FINAL",
        "9" * 64,
        COMMIT,
        1,
        datetime(2027, 12, 2, tzinfo=UTC),
        _old_ns(env),
    )
    with pytest.raises(SystemExit, match="single use applies across digests"):
        _run(env, "FORWARD_FINAL", datetime(2027, 12, 2, 1, tzinfo=UTC), tag="fin")


def test_old_interim_of_the_same_month_blocks_but_another_month_runs(
    env: dict[str, Any],
) -> None:
    _seed(_old_ns(env), "FORWARD_INTERIM_2027-01", "INTERIM_NOT_A_DECISION")
    with pytest.raises(SystemExit, match="single use applies across digests"):
        _run(env, "FORWARD_INTERIM", datetime(2027, 1, 25, 9, tzinfo=UTC))
    res = _run(env, "FORWARD_INTERIM", datetime(2027, 2, 20, 9, tzinfo=UTC), tag="feb")
    assert res["claim_name"] == "FORWARD_INTERIM_2027-02"
    assert (
        env["ledger"] / m.PROTOCOL_ID / env["digest"] / "FORWARD_INTERIM_2027-02.claim"
    ).exists()


def test_protocol_root_claim_blocks_a_new_digest(env: dict[str, Any]) -> None:
    """The atomic single-use lock is the protocol-root claim: a root claim with no per-digest
    trace still refuses the same stage under any digest."""
    root = env["ledger"] / m.PROTOCOL_ID
    m.claim_stage(
        "FORWARD_INTERIM_2027-01",
        "1" * 64,
        COMMIT,
        1,
        datetime(2027, 1, 20, tzinfo=UTC),
        root,
    )
    with pytest.raises(SystemExit, match="protocol root"):
        _run(env, "FORWARD_INTERIM", datetime(2027, 1, 20, 9, tzinfo=UTC))


def test_missing_stage_hour_gives_invalid_result_and_burns_the_claim(
    env: dict[str, Any],
) -> None:
    bad = int(
        (datetime(2026, 12, 10, 3, tzinfo=UTC).timestamp() * 1000 - EXCHANGE_START_MS)
        // m.HOUR_MS
    )
    res = _run(
        env,
        "FORWARD_INTERIM",
        datetime(2027, 1, 20, 9, tzinfo=UTC),
        fetch=lambda u: fake_market(u, holes={"ETHUSDT": {bad}}),
    )
    assert res["label"] == "INVALID" and any(
        "ETHUSDT" in p and "missing hour" in p for p in res["problems"]
    )
    assert res["cells"] == {} and "analysis" not in res
    assert (
        env["ledger"]
        / m.PROTOCOL_ID
        / env["digest"]
        / "FORWARD_INTERIM_2027-01.result.json"
    ).exists()
    with pytest.raises(SystemExit, match="already claimed"):
        _run(env, "FORWARD_INTERIM", datetime(2027, 1, 21, 9, tzinfo=UTC), tag="retry")


def test_history_hole_blanks_the_frame_without_invalidating(
    env: dict[str, Any],
) -> None:
    bad = int(
        (datetime(2025, 6, 10, 3, tzinfo=UTC).timestamp() * 1000 - EXCHANGE_START_MS)
        // m.HOUR_MS
    )
    res = _run(
        env,
        "FORWARD_INTERIM",
        datetime(2027, 1, 20, 9, tzinfo=UTC),
        fetch=lambda u: fake_market(u, holes={"BTCUSDT": {bad}}),
    )
    assert res["label"] != "INVALID", res["problems"]
    assert res["data"]["history_missing_hours"]["BTCUSDT"] == 1


def test_gates_precede_claim(env: dict[str, Any]) -> None:
    with pytest.raises(SystemExit, match="may not run before"):
        _run(env, "FORWARD_FINAL", datetime(2027, 12, 1, 23, tzinfo=UTC))
    with pytest.raises(SystemExit, match="may not run before"):
        _run(env, "FORWARD_INTERIM", datetime(2026, 11, 30, tzinfo=UTC))
    with pytest.raises(SystemExit, match="does not quote"):
        m.run_stage(
            "FORWARD_FINAL",
            URL,
            env["tmp"] / "w",
            env["tmp"] / "o",
            now=datetime(2027, 12, 2, tzinfo=UTC),
            fetch=fake_market,
            gh_fetch=_gh(relay_body="no quote"),
            ledger_dir=env["ledger"],
            symbols=TWO,
        )
    assert not env["ledger"].exists()
    (env["tmp"] / "busy").mkdir()
    (env["tmp"] / "busy" / "f").write_text("x")
    with pytest.raises(SystemExit, match="work directory is not empty"):
        m.run_stage(
            "FORWARD_FINAL",
            URL,
            env["tmp"] / "busy",
            env["tmp"] / "o2",
            now=datetime(2027, 12, 2, tzinfo=UTC),
            fetch=fake_market,
            gh_fetch=env["gh"],
            ledger_dir=env["ledger"],
            symbols=TWO,
        )


def test_interim_equals_prefix_of_final() -> None:
    """Whole-pipeline causality: an interim replay is the exact prefix of the later, longer
    replay (no decision depends on a bar that did not exist yet), for every arm and variant."""
    symbols = TWO
    start_s = EXCHANGE_START_MS // 1000
    c0 = int(datetime(2026, 12, 1, tzinfo=UTC).timestamp())

    def cells_until(last_close: datetime) -> dict[str, np.ndarray]:
        last = int(last_close.timestamp())
        n_hours = (last - start_s) // m.HOUR_S
        frames = {
            s: m.build_frame({k: v[:n_hours] for k, v in exch(s).items()}, start_s)
            for s in symbols
        }
        bars = (last - c0) // m.BAR_S
        fund = {s: np.zeros(bars + 1) for s in symbols}
        fund["BTCUSDT"][5::7] = 3.0e-3
        sig = m.build_signals(frames, symbols, c0, bars, fund)
        return {k: v["equity"] for k, v in m.evaluate_cells(sig).items()}

    short = cells_until(datetime(2027, 2, 1, tzinfo=UTC))
    long = cells_until(datetime(2027, 4, 1, tzinfo=UTC))
    for name in m.all_cell_names():
        k = len(short[name])
        assert np.allclose(short[name][:-1], long[name][: k - 1], rtol=0, atol=1e-12), (
            name
        )


# ----------------------------------------------------------------------------- CLI


def test_cli_digest(capsys: pytest.CaptureFixture[str]) -> None:
    m.main(["digest"])
    assert capsys.readouterr().out.strip() == m.protocol_digest()
    with pytest.raises(SystemExit):
        m.main(
            [
                "run",
                "--stage",
                "BACKCAST",
                "--authorization-url",
                URL,
                "--work",
                "w",
                "--output-dir",
                "o",
            ]
        )


def _claims(env: dict[str, Any]) -> list[Path]:
    return sorted((env["ledger"]).rglob("*.claim")) if env["ledger"].exists() else []


def test_probe_reads_no_stage_window_data() -> None:
    assert (
        m.PROBE_END_MS + m.DAY_S * 1000
        < int(datetime(2026, 11, 1, tzinfo=UTC).timestamp()) * 1000
    )
    seen: list[str] = []

    def f(url: str) -> bytes:
        seen.append(url)
        return fake_market(url)

    m.connectivity_probe(f)
    assert len(seen) == 2 and "klines" in seen[0] and "fundingRate" in seen[1]
    assert all("symbol=BTCUSDT" in u for u in seen)


@pytest.mark.parametrize("failure", ["raise", "empty", "garbage"])
def test_connectivity_failure_refuses_without_claiming(
    env: dict[str, Any], failure: str
) -> None:
    def broken(url: str) -> bytes:
        if failure == "raise":
            raise SystemExit("network failure")
        return b"[]" if failure == "empty" else b"not json"

    with pytest.raises(SystemExit, match="connectivity probe failed before any claim"):
        _run(env, "FORWARD_FINAL", datetime(2027, 12, 2, 1, tzinfo=UTC), fetch=broken)
    assert _claims(env) == []
    # the stage is still available afterwards
    assert _run(env, "FORWARD_FINAL", datetime(2027, 12, 2, 1, tzinfo=UTC), tag="ok")[
        "label"
    ]


@pytest.mark.parametrize("which", ["relay_edit", "scope_owner", "relay_quote"])
def test_authorization_failure_leaves_no_claim(env: dict[str, Any], which: str) -> None:
    calls = {"n": 0}
    good = (
        f"G4 identity: {m.PROTOCOL_ID} {env['digest']} {COMMIT}\n"
        "G4 claim: FORWARD_FINAL"
    )
    expected = {
        "relay_edit": "g4_relay.*changed between fetches",
        "scope_owner": "data_scope.*not owner-authored",
        "relay_quote": "does not quote one exact protocol",
    }[which]

    def gh(cid: int) -> dict[str, Any]:
        calls["n"] += 1
        if which == "scope_owner" and cid == m.DATA_SCOPE_COMMENT_ID:
            return _comment("scope", login="someone")
        if which == "relay_quote" and cid != m.DATA_SCOPE_COMMENT_ID:
            return _comment("no quote")
        if which == "relay_edit" and cid != m.DATA_SCOPE_COMMENT_ID:
            return _comment(good, updated=f"t{calls['n']}")
        return _comment("scope") if cid == m.DATA_SCOPE_COMMENT_ID else _comment(good)

    fetched: list[str] = []

    def f(url: str) -> bytes:
        fetched.append(url)
        return fake_market(url)

    with pytest.raises(SystemExit, match=expected):
        m.run_stage(
            "FORWARD_FINAL",
            URL,
            env["tmp"] / "w",
            env["tmp"] / "o",
            now=datetime(2027, 12, 2, 1, tzinfo=UTC),
            fetch=f,
            gh_fetch=gh,
            ledger_dir=env["ledger"],
            symbols=TWO,
        )
    assert _claims(env) == [] and fetched == []
    assert calls["n"] == (4 if which == "scope_owner" else 2)


# Strict source-close admission is part of the prospective frozen data contract.
# This is synthetic only; no Binance request, funding read or economic stage runs.
@pytest.mark.parametrize(
    ("open_unit", "close_offset"),
    [
        (1, -m.HOUR_MS),  # source claims the bar ended before it opened
        (1, -1),  # one millisecond early
        (1, 1),  # one millisecond late
        (1, m.HOUR_MS),
        (1000, -1),  # one microsecond early
        (1000, 1),  # one microsecond late
    ],
)
def test_parse_klines_rejects_nonfinal_or_offgrid_native_source_close(
    open_unit: int,
    close_offset: int,
) -> None:
    open_ms = 1735689600000  # 2025-01-01 00:00 UTC, synthetic
    row = _kline(open_ms)
    row[0] = open_ms * open_unit
    row[6] = (open_ms + m.HOUR_MS) * open_unit - 1 + close_offset
    with pytest.raises(SystemExit, match="close_time"):
        m.parse_klines([row])


@pytest.mark.parametrize("open_unit,close_unit", [(1, 1000), (1000, 1)])
def test_parse_klines_rejects_mixed_native_timestamp_units(
    open_unit: int,
    close_unit: int,
) -> None:
    open_ms = 1735689600000
    row = _kline(open_ms)
    row[0] = open_ms * open_unit
    row[6] = (open_ms + m.HOUR_MS) * close_unit - 1
    with pytest.raises(SystemExit, match="close_time"):
        m.parse_klines([row])


@pytest.mark.parametrize("native_unit", [1, 1000])
def test_parse_klines_admits_exact_native_completed_hour_only(
    native_unit: int,
) -> None:
    open_ms = 1735689600000
    row = _kline(open_ms)
    row[0] = open_ms * native_unit
    row[6] = (open_ms + m.HOUR_MS) * native_unit - 1
    rows = m.parse_klines([row])
    assert rows[0][0] == open_ms
    arrs = m.hourly_arrays(rows, open_ms, open_ms)
    assert arrs["close"].tolist() == [100.0]
    assert arrs["qv"].tolist() == [500.0]


@pytest.mark.parametrize("bad_close", [None, 3.5, "1735693199999.0", -1])
def test_parse_klines_refuses_non_native_close_time(bad_close: object) -> None:
    row = _kline(1735689600000)
    row[6] = bad_close
    with pytest.raises(SystemExit, match="close_time"):
        m.parse_klines([row])


def test_page_parser_fail_closed_without_aggregating_corrupt_source_time(
    tmp_path: Path,
) -> None:
    open_ms = 1735689600000
    row = _kline(open_ms)
    row[6] = open_ms - 1
    raw = json.dumps([row]).encode()
    with pytest.raises(SystemExit, match="close_time"):
        m.fetch_klines(
            "BTCUSDT",
            open_ms,
            open_ms,
            tmp_path / "raw",
            fetch=lambda _url: raw,
        )
    # The source body is preserved exactly; it is not admitted as a tradable bar.
    assert (tmp_path / "raw" / "BTCUSDT_klines_000.json").read_bytes() == raw


def test_preclaim_probe_refuses_nonfinal_source_hour_without_funding_request() -> None:
    row = _kline(1735689600000)
    row[6] = row[0] - 1
    requests: list[str] = []

    def synthetic_fetch(url: str) -> bytes:
        requests.append(url)
        if "/klines" not in url:
            raise AssertionError(
                "Funding must not be requested after source clock refusal"
            )
        return json.dumps([row]).encode()

    with pytest.raises(SystemExit, match="connectivity probe failed"):
        m.connectivity_probe(synthetic_fetch)
    assert len(requests) == 1


@pytest.mark.parametrize("bad_open", [1735689600000001, "1735689600000.0", True])
def test_parse_klines_rejects_non_native_or_unaligned_open(bad_open: object) -> None:
    row = _kline(1735689600000)
    row[0] = bad_open
    with pytest.raises(SystemExit, match="close_time"):
        m.parse_klines([row])


@pytest.mark.parametrize("historical", ["marker", "legacy", "malformed"])
def test_preexisting_g4_refusal_does_not_consume_new_stage(
    env: dict[str, Any], historical: str
) -> None:
    root = env["ledger"] / m.PROTOCOL_ID
    before = datetime(2027, 1, 20, tzinfo=UTC)
    if historical == "marker":
        m.reserve_g4_comment(
            root, 7040999999, "FORWARD_INTERIM_2027-01", "9" * 64, COMMIT, before
        )
    elif historical == "legacy":
        m.claim_stage(
            "FORWARD_INTERIM_2027-01",
            "9" * 64,
            COMMIT,
            7040999999,
            before,
            root / ("9" * 64),
        )
    else:
        root.mkdir(parents=True)
        (root / "broken.claim").write_text("invalid JSON", encoding="utf-8")
    previous = {p: p.read_bytes() for p in root.rglob("*.claim")}
    fetched: list[str] = []

    def fetch(url: str) -> bytes:
        fetched.append(url)
        return fake_market(url)

    def forbid_github(_cid: int) -> dict[str, Any]:
        raise AssertionError("historical refusal must precede GitHub authorization")

    with pytest.raises(
        SystemExit, match="already consumed|unreadable historical G4 claim"
    ):
        m.run_stage(
            "FORWARD_FINAL",
            URL,
            env["tmp"] / "unused_work",
            env["tmp"] / "unused_output",
            now=datetime(2027, 12, 2, 1, tzinfo=UTC),
            fetch=fetch,
            gh_fetch=forbid_github,
            ledger_dir=env["ledger"],
            symbols=TWO,
        )
    assert {p: p.read_bytes() for p in root.rglob("*.claim")} == previous
    assert fetched == []


# Regression: do not allow a POSIX-relative Windows literal to fork the single-use ledger.
def test_stage_refuses_relative_ledger_before_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    when = datetime(2027, 1, 20, 9, tzinfo=UTC)
    for dirname in ("cwd1", "cwd2"):
        root = tmp_path / dirname
        root.mkdir()
        monkeypatch.chdir(root)
        # The argument must be rejected before any API call, claim or data read.
        with pytest.raises(SystemExit, match="absolute.*ledger|ledger.*absolute"):
            m.run_stage(
                "FORWARD_INTERIM",
                URL,
                root / "work",
                root / "out",
                now=when,
                ledger_dir=Path("relative_ledger"),
                fetch=lambda _url: (_ for _ in ()).throw(AssertionError("data read")),
            )
        assert not any(root.iterdir())


def test_claim_identity_is_utc_and_naive_datetimes_refuse() -> None:
    from datetime import timedelta, timezone

    jst = timezone(timedelta(hours=9))
    stage = m.STAGES["FORWARD_INTERIM"]
    assert (
        m.claim_name(stage, datetime(2026, 12, 1, 0, 30, tzinfo=jst))
        == "FORWARD_INTERIM_2026-11"
    )
    assert (
        m.claim_name(stage, datetime(2026, 12, 1, 9, 30, tzinfo=jst))
        == "FORWARD_INTERIM_2026-12"
    )
    with pytest.raises(SystemExit, match="timezone-aware"):
        m.claim_name(stage, datetime(2026, 12, 1))
    with pytest.raises(SystemExit, match="timezone-aware"):
        m.check_clock(stage, datetime(2026, 12, 1))


def test_posix_windows_legacy_default_must_not_be_accepted() -> None:
    import os

    if os.name != "nt":
        assert not m.LEDGER_DIR.is_absolute()


def test_g4_accepts_one_crlf_claim() -> None:
    digest = "d" * 64
    body = (
        f"G4 identity: {m.PROTOCOL_ID} {digest} {COMMIT}\r\nG4 claim: FORWARD_FINAL\r\n"
    )
    assert (
        m.authorization_receipt(
            URL, digest, COMMIT, "FORWARD_FINAL", _gh(relay_body=body)
        )["g4_relay"]["comment_id"]
        == 7040999999
    )


@pytest.mark.parametrize(
    "lines",
    [
        "G4 claim: FORWARD_INTERIM_2027-01, FORWARD_FINAL\nG4 claim: FORWARD_FINAL",
        "G4 claim: FORWARD_INTERIM_2027-01\r\nG4 claim: FORWARD_FINAL",
        "G4 claim: FORWARD_FINAL\nG4 claim: final",
        "G4 claim: FORWARD_FINAL\n G4 claim: FORWARD_INTERIM_2027-01",
        "G4 claim: FORWARD_INTERIM_2027-01",
    ],
)
def test_g4_invalid_claim_refuses_public_run_before_side_effects(
    env: dict[str, Any], lines: str
) -> None:
    def fetch(_url: str) -> bytes:
        raise AssertionError("invalid authorization must not fetch")

    with pytest.raises(SystemExit, match="G4 claim"):
        m.run_stage(
            "FORWARD_FINAL",
            URL,
            env["tmp"] / "w",
            env["tmp"] / "o",
            now=datetime(2027, 12, 2, 1, tzinfo=UTC),
            fetch=fetch,
            gh_fetch=_gh(
                relay_body=f"G4 identity: {m.PROTOCOL_ID} {env['digest']} {COMMIT}\n{lines}"
            ),
            ledger_dir=env["ledger"],
            symbols=TWO,
        )
    assert not env["ledger"].exists()
    assert not (env["tmp"] / "w").exists()
    assert not (env["tmp"] / "o").exists()


@pytest.mark.parametrize("kind", ["stage", "marker"])
def test_claim_is_complete_before_exclusive_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    root = tmp_path / "ledger"
    now = datetime(2027, 1, 20, tzinfo=UTC)
    real_link = os.link
    published: list[Path] = []

    def observed_link(src: Any, dst: Any) -> None:
        source, target = Path(src), Path(dst)
        assert source.parent == target.parent == root
        assert source.suffix != ".claim" and not target.exists()
        assert (
            json.loads(source.read_text(encoding="utf-8"))["authorization_comment_id"]
            == 42
        )
        real_link(src, dst)
        published.append(target)

    monkeypatch.setattr(os, "link", observed_link)
    if kind == "stage":
        target = m.claim_stage("FORWARD_FINAL", "d" * 64, COMMIT, 42, now, root)
    else:
        target = m.reserve_g4_comment(root, 42, "FORWARD_FINAL", "d" * 64, COMMIT, now)
    assert published == [target]
    assert list(root.iterdir()) == [target]
    assert (
        json.loads(target.read_text(encoding="utf-8"))["authorization_comment_id"] == 42
    )


@pytest.mark.parametrize("kind", ["stage", "marker"])
def test_failed_claim_flush_never_publishes_partial_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    root = tmp_path / "ledger"
    now = datetime(2027, 1, 20, tzinfo=UTC)

    def fail(_fd: int) -> None:
        raise OSError("synthetic fsync failure")

    monkeypatch.setattr(os, "fsync", fail)
    with pytest.raises(OSError, match="synthetic fsync failure"):
        if kind == "stage":
            m.claim_stage("FORWARD_FINAL", "d" * 64, COMMIT, 42, now, root)
        else:
            m.reserve_g4_comment(root, 42, "FORWARD_FINAL", "d" * 64, COMMIT, now)
    assert list(root.iterdir()) == []


def test_concurrent_g4_publication_has_one_winner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "ledger"
    now = datetime(2027, 1, 20, tzinfo=UTC)
    real_link = os.link
    gate = Barrier(2, timeout=10)

    def simultaneous_link(src: Any, dst: Any) -> None:
        gate.wait()
        real_link(src, dst)

    monkeypatch.setattr(os, "link", simultaneous_link)

    def reserve(stage: str) -> str:
        try:
            m.reserve_g4_comment(root, 42, stage, "d" * 64, COMMIT, now)
        except SystemExit as exc:
            assert "already consumed" in str(exc)
            return "refused"
        return stage

    stages = ["FORWARD_INTERIM_2027-01", "FORWARD_FINAL"]
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(reserve, stages))
    assert results.count("refused") == 1
    winner = next(x for x in results if x != "refused")
    marker = root / "g4-42.claim"
    assert list(root.iterdir()) == [marker]
    assert json.loads(marker.read_text(encoding="utf-8"))["stage"] == winner


@pytest.mark.parametrize("linked", ["root", "parent"])
def test_symlinked_ledger_refuses_before_authorization(
    env: dict[str, Any], linked: str
) -> None:
    real = env["tmp"] / "other_ledger"
    real.mkdir()
    try:
        if linked == "parent":
            env["ledger"].symlink_to(real, target_is_directory=True)
        else:
            env["ledger"].mkdir()
            (env["ledger"] / m.PROTOCOL_ID).symlink_to(real, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {exc}")

    def unexpected(_arg: Any) -> Any:
        raise AssertionError("unsafe ledger must be rejected before external access")

    with pytest.raises(SystemExit, match="G4 ledger root"):
        m.run_stage(
            "FORWARD_FINAL",
            URL,
            env["tmp"] / "w",
            env["tmp"] / "o",
            now=datetime(2027, 12, 2, 1, tzinfo=UTC),
            fetch=unexpected,
            gh_fetch=unexpected,
            ledger_dir=env["ledger"],
            symbols=TWO,
        )
    assert not list(real.iterdir())


def test_public_run_rechecks_g4_after_root_claim(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    root = env["ledger"] / m.PROTOCOL_ID
    now = datetime(2027, 12, 2, 1, tzinfo=UTC)
    real_claim = m.claim_stage

    def competing_claim(*args: Any, **kwargs: Any) -> Path:
        path = real_claim(*args, **kwargs)
        if path.parent == root:
            # A competing reservation arrives after read-only preflight, before the marker.
            m._publish_claim(
                root / "g4-7040999999.claim",
                {
                    "stage": "FORWARD_INTERIM_2027-01",
                    "protocol_digest": "9" * 64,
                    "commit": COMMIT,
                    "authorization_comment_id": 7040999999,
                    "claimed_at": now.isoformat(),
                },
            )
        return path

    monkeypatch.setattr(m, "claim_stage", competing_claim)
    # The trend probe is deliberately simulated; stage-window access is forbidden.
    if hasattr(m, "connectivity_probe"):
        monkeypatch.setattr(m, "connectivity_probe", lambda _fetch: None)

    def unexpected(_url: str) -> bytes:
        raise AssertionError("stage data must not be read after a G4 conflict")

    with pytest.raises(SystemExit, match="already consumed"):
        m.run_stage(
            "FORWARD_FINAL",
            URL,
            env["tmp"] / "w",
            env["tmp"] / "o",
            now=now,
            fetch=unexpected,
            gh_fetch=_gh(
                relay_body=f"G4 identity: {m.PROTOCOL_ID} {env['digest']} {COMMIT}\nG4 claim: FORWARD_FINAL"
            ),
            ledger_dir=env["ledger"],
            symbols=TWO,
        )
    assert (root / "FORWARD_FINAL.claim").exists()
    assert (root / "g4-7040999999.claim").exists()
    assert not (root / env["digest"] / "FORWARD_FINAL.claim").exists()
    assert not (env["tmp"] / "w").exists()


@pytest.mark.parametrize(
    "history",
    [
        "linked",
        "broken-link",
        "zero-id",
        "negative-id",
        "duplicate-id",
        "bool-id",
        "float-id",
        "string-id",
        "missing-id",
        "nested-duplicate",
        "nonobject",
        "corrupt",
        "directory",
    ],
)
def test_g4_invalid_history_refuses_before_claim_or_fetch(
    env: dict[str, Any], history: str
) -> None:
    root = env["ledger"] / m.PROTOCOL_ID
    root.mkdir(parents=True)
    ns = root / ("9" * 64)
    if history in {"linked", "broken-link"}:
        outside = env["tmp"] / "external-history"
        if history == "linked":
            outside.mkdir()
            (outside / "old.claim").write_text(
                '{"authorization_comment_id":7040999999}', encoding="utf-8"
            )
        try:
            ns.symlink_to(outside, target_is_directory=True)
        except OSError as exc:
            pytest.skip(f"symlinks unavailable: {exc}")
    else:
        ns.mkdir()
        record = {
            "zero-id": '{"authorization_comment_id":0}',
            "negative-id": '{"authorization_comment_id":-1}',
            "duplicate-id": '{"authorization_comment_id":7040999999,"authorization_comment_id":1}',
            "bool-id": '{"authorization_comment_id":true}',
            "float-id": '{"authorization_comment_id":1.0}',
            "string-id": '{"authorization_comment_id":"1"}',
            "missing-id": "{}",
            "nested-duplicate": '{"authorization_comment_id":1,"meta":{"x":1,"x":2}}',
            "nonobject": "[]",
            "corrupt": "{",
            "directory": None,
        }[history]
        if record is None:
            (ns / "old.claim").mkdir()
        else:
            (ns / "old.claim").write_text(record, encoding="utf-8")

    def unexpected(_arg: Any) -> Any:
        raise AssertionError("no external request after invalid ledger history")

    with pytest.raises(SystemExit, match="historical G4|G4 ledger"):
        m.run_stage(
            "FORWARD_FINAL",
            URL,
            env["tmp"] / "w",
            env["tmp"] / "o",
            now=datetime(2027, 12, 2, 1, tzinfo=UTC),
            fetch=unexpected,
            gh_fetch=unexpected,
            ledger_dir=env["ledger"],
            symbols=TWO,
        )
    assert not (root / "FORWARD_FINAL.claim").exists()
    assert not (root / "g4-7040999999.claim").exists()
    assert not (env["tmp"] / "w").exists()


def test_unreadable_history_refuses_before_external_access(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    root = env["ledger"] / m.PROTOCOL_ID
    ns = root / ("9" * 64)
    ns.mkdir(parents=True)
    real_iterdir = Path.iterdir

    def inaccessible(path: Path) -> Any:
        if path == ns:
            raise PermissionError("synthetic historical directory error")
        return real_iterdir(path)

    monkeypatch.setattr(Path, "iterdir", inaccessible)

    def unexpected(_arg: Any) -> Any:
        raise AssertionError("no external request after unsafe ledger")

    with pytest.raises(SystemExit, match="unreadable historical G4 ledger"):
        m.run_stage(
            "FORWARD_FINAL",
            URL,
            env["tmp"] / "w",
            env["tmp"] / "o",
            now=datetime(2027, 12, 2, 1, tzinfo=UTC),
            fetch=unexpected,
            gh_fetch=unexpected,
            ledger_dir=env["ledger"],
            symbols=TWO,
        )
    assert not (root / "FORWARD_FINAL.claim").exists()
    assert not (env["tmp"] / "w").exists()


@pytest.mark.parametrize(
    "mutation",
    (
        "long_id",
        "long_digest",
        "long_commit",
        "duplicate_identity",
        "conflicting_identity",
        "missing_identity",
    ),
)
def test_g4_identity_must_be_one_exact_tuple(mutation: str) -> None:
    digest = "d" * 64
    identity = f"G4 identity: {m.PROTOCOL_ID} {digest} {COMMIT}"
    if mutation == "long_id":
        identity = identity.replace(m.PROTOCOL_ID, m.PROTOCOL_ID + "_other")
    elif mutation == "long_digest":
        identity = identity.replace(digest, digest + "e")
    elif mutation == "long_commit":
        identity = identity.replace(COMMIT, "b" + COMMIT)
    elif mutation == "duplicate_identity":
        identity += "\n" + identity
    elif mutation == "conflicting_identity":
        identity += "\n" + identity.replace(digest, "e" * 64)
    else:
        identity = f"{m.PROTOCOL_ID} {digest} {COMMIT}"
    with pytest.raises(SystemExit, match="does not quote"):
        m.authorization_receipt(
            URL,
            digest,
            COMMIT,
            "FORWARD_FINAL",
            _gh(relay_body=identity + "\nG4 claim: FORWARD_FINAL"),
        )


@pytest.mark.parametrize("digest", ["d" * 63, "d" * 65, "D" * 64, "z" * 64])
def test_g4_invalid_digest_refuses_before_comment_fetch(digest: str) -> None:
    def unexpected_fetch(_cid: int) -> dict[str, Any]:
        pytest.fail("invalid digest must not reach comment fetch")

    with pytest.raises(SystemExit, match="full 64-hex"):
        m.authorization_receipt(URL, digest, COMMIT, "FORWARD_FINAL", unexpected_fetch)


@pytest.mark.parametrize(
    "separator", ["\r", "\v", "\f", "\x1c", "\x1d", "\x1e", "\x85", "\u2028", "\u2029"]
)
def test_g4_refuses_noncanonical_record_separators(separator: str) -> None:
    digest = "d" * 64
    body = f"G4 identity: {m.PROTOCOL_ID} {digest} {COMMIT}{separator}G4 claim: FORWARD_FINAL"
    with pytest.raises(SystemExit, match="unsupported line separator"):
        m.authorization_receipt(
            URL, digest, COMMIT, "FORWARD_FINAL", _gh(relay_body=body)
        )


@pytest.mark.parametrize("field", ["identity", "claim"])
def test_g4_refuses_malformed_second_record(field: str) -> None:
    digest = "d" * 64
    identity = f"G4 identity: {m.PROTOCOL_ID} {digest} {COMMIT}"
    claim = "G4 claim: FORWARD_FINAL"
    extra = (
        f"G4 identity {m.PROTOCOL_ID} {'e' * 64} {COMMIT}"
        if field == "identity"
        else "G4 claim FORWARD_INTERIM_2026-12"
    )
    body = "\n".join([identity, claim, extra])
    with pytest.raises(SystemExit, match="does not quote|must name exactly"):
        m.authorization_receipt(
            URL, digest, COMMIT, "FORWARD_FINAL", _gh(relay_body=body)
        )


@pytest.mark.parametrize("transient", [True, False])
def test_g4_reservation_disappearing_entry_only_tolerates_publisher_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, transient: bool
) -> None:
    root = tmp_path / "ledger"
    now = datetime(2027, 1, 20, tzinfo=UTC)
    prior = m.claim_stage("FORWARD_INTERIM_2027-01", "d" * 64, COMMIT, 41, now, root)
    before = prior.read_bytes()
    disappearing = root / (
        ".pending-claim-abc12345.tmp" if transient else "unknown.claim"
    )
    disappearing.write_text('{"authorization_comment_id": 99}', encoding="utf-8")
    real_lstat = Path.lstat
    vanished = False

    def vanish(path: Path, *args: Any, **kwargs: Any) -> Any:
        nonlocal vanished
        if path == disappearing and not vanished:
            vanished = True
            disappearing.unlink()
        return real_lstat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", vanish)
    if transient:
        marker = m.reserve_g4_comment(root, 42, "FORWARD_FINAL", "e" * 64, COMMIT, now)
        assert json.loads(marker.read_bytes())["authorization_comment_id"] == 42
    else:
        with pytest.raises(SystemExit, match="unreadable historical G4 ledger"):
            m.reserve_g4_comment(root, 42, "FORWARD_FINAL", "e" * 64, COMMIT, now)
        assert not (root / "g4-42.claim").exists()
    assert vanished and prior.read_bytes() == before


@pytest.mark.parametrize("malformed", [False, True])
def test_direct_g4_reservation_cannot_exclude_existing_same_stage_claim(
    tmp_path: Path, malformed: bool
) -> None:
    root = tmp_path / "ledger"
    now = datetime(2027, 1, 20, tzinfo=UTC)
    stage = m.claim_stage("FORWARD_FINAL", "d" * 64, COMMIT, 42, now, root)
    if malformed:
        stage.write_text('{"authorization_comment_id":true}', encoding="utf-8")
    before = stage.read_bytes()
    with pytest.raises(SystemExit, match="already consumed|invalid historical G4"):
        m.reserve_g4_comment(root, 42, "FORWARD_FINAL", "d" * 64, COMMIT, now)
    assert stage.read_bytes() == before
    assert not (root / "g4-42.claim").exists()


@pytest.mark.parametrize(
    "mutation",
    [
        "none",
        "missing",
        "wrong-digest",
        "wrong-commit",
        "wrong-id",
        "wrong-time",
        "wrong-stage",
    ],
)
def test_g4_current_root_claim_must_match_fresh_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    root = tmp_path / "ledger"
    now = datetime(2027, 1, 20, tzinfo=UTC)
    previous = m.claim_stage("FORWARD_INTERIM_2027-01", "e" * 64, COMMIT, 41, now, root)
    previous_bytes = previous.read_bytes()
    current = m.claim_stage("FORWARD_FINAL", "d" * 64, COMMIT, 42, now, root)
    if mutation == "missing":
        current.unlink()
    elif mutation != "none":
        rec = json.loads(current.read_bytes())
        field, value = {
            "wrong-digest": ("protocol_digest", "f" * 64),
            "wrong-commit": ("commit", "b" * 40),
            "wrong-id": ("authorization_comment_id", 43),
            "wrong-time": ("claimed_at", "2027-01-19T00:00:00+00:00"),
            "wrong-stage": ("stage", "FORWARD_INTERIM_2027-01"),
        }[mutation]
        rec[field] = value
        current.write_text(json.dumps(rec), encoding="utf-8")
    before = current.read_bytes() if current.exists() else None
    pending = root / ".pending-claim-other123.tmp"
    pending.write_text("temporary publisher bytes", encoding="utf-8")
    real_lstat = Path.lstat
    vanished = False

    def vanish(path: Path, *args: Any, **kwargs: Any) -> Any:
        nonlocal vanished
        if path == pending and not vanished:
            vanished = True
            pending.unlink()
        return real_lstat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", vanish)
    if mutation == "none":
        marker = m.reserve_g4_comment(
            root, 42, "FORWARD_FINAL", "d" * 64, COMMIT, now, current_claim=current
        )
        assert json.loads(marker.read_bytes()) == json.loads(current.read_bytes())
    else:
        with pytest.raises(SystemExit, match="current G4 stage claim"):
            m.reserve_g4_comment(
                root, 42, "FORWARD_FINAL", "d" * 64, COMMIT, now, current_claim=current
            )
        assert not (root / "g4-42.claim").exists()
    assert vanished and previous.read_bytes() == previous_bytes
    assert (current.read_bytes() if current.exists() else None) == before


@pytest.mark.parametrize(
    "mutated",
    [
        "G4 claim: FORWARD_FINAL\nG4 claim: FORWARD_FINAL",
        "g4 CLAIM: FORWARD_FINAL",
        " G4 claim: FORWARD_FINAL",
        "G4  claim: FORWARD_FINAL",
        "G4 claim: FORWARD_FINAL ",
        "G4 claim: FORWARD_FINAL\nG4 CLAIM FORWARD_INTERIM_2027-01",
    ],
)
def test_g4_exact_stage_refuses_duplicate_or_noncanonical_field(mutated: str) -> None:
    digest = "d" * 64
    body = f"G4 identity: {m.PROTOCOL_ID} {digest} {COMMIT}\n{mutated}"
    with pytest.raises(SystemExit, match="G4 claim"):
        m.authorization_receipt(
            URL, digest, COMMIT, "FORWARD_FINAL", _gh(relay_body=body)
        )


@pytest.mark.parametrize("commit", ["a" * 39, "a" * 41, "A" * 40, "z" * 40])
def test_g4_invalid_commit_refuses_before_comment_fetch(commit: str) -> None:
    def unexpected(_cid: int) -> dict[str, Any]:
        pytest.fail("invalid commit must not reach comment fetch")

    with pytest.raises(SystemExit, match="full 40-hex"):
        m.authorization_receipt(URL, "d" * 64, commit, "FORWARD_FINAL", unexpected)
