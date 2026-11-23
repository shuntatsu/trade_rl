"""Synthetic-data tests for tools/trend_breadth.py (no network, no market data).

Needs only numpy and the repository's own ``trade_rl`` package for the cost-constant pin. One test
uses an optional package and skips without it: the pandas frame parity oracle (``pandas``)."""

from __future__ import annotations

import io
import json
import math
import re
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tools import trend_breadth as m

DIGEST_PIN = "67e9c2fa75c95efee1a1ebb51f73df1b6a647dfe433929eaa8dd72fc4d3b3117"
COMMIT = "a" * 40
URL = "https://github.com/shuntatsu/trade_rl/issues/810#issuecomment-7040999999"
T0 = int(datetime(2021, 1, 1, tzinfo=UTC).timestamp())
STAGE = m.Stage(
    "BREADTH", datetime(2021, 1, 1, tzinfo=UTC), datetime(2022, 2, 1, tzinfo=UTC)
)
INIT = m.INITIAL_CAPITAL

# ----------------------------------------------------------------------------- digest pins


def test_digest_pin() -> None:
    assert m.protocol_digest() == DIGEST_PIN
    assert re.fullmatch(r"[0-9a-f]{64}", m.protocol_digest())


def test_constants_frozen() -> None:
    assert m.PROTOCOL_ID == "trend_breadth_v1"
    assert m.REGIME_WINDOWS == (20, 50, 100, 200) and m.REGIME_THRESHOLD == 0.5
    assert (m.SIZE_VOL, m.VOL_DAYS) == (0.5, 30)
    assert (m.SIDE_BUDGET, m.SIDE_CAP, m.BORROW_TOL, m.N_MIN) == (0.9, 1.0, 0.02, 20)
    assert (m.DEAD_RUN_HOURS, m.NOTICE_LOOKBACK_DAYS) == (24, 90)
    assert not hasattr(m, "FALLBACK_LEAD_BARS")  # no hindsight fallback
    assert (m.FORCED_EXTRA, m.FUNDING_UNCOVERED_MAX) == (0.05, 0.01)
    assert (m.SHARPE_MIN, m.MAX_DD_LIMIT) == (0.5, 0.40)
    assert (m.BOOT_BLOCK, m.BOOT_DRAWS, m.BOOT_SEED, m.ALPHA) == (30, 5000, 0, 0.05)
    assert m.INITIAL_CAPITAL == 100_000.0
    assert m.VARIANTS == {"BASE": (1.0, 0), "HIGH_COST": (2.0, 0), "DELAY": (1.0, 1)}
    assert m.CELL_NAMES == (
        "TREND_LS__BASE",
        "TREND_LONG__BASE",
        "TREND_LS__HIGH_COST",
        "TREND_LS__DELAY",
        "EW_LONG",
        "CASH",
    )
    assert m.DATA_SCOPE_COMMENT_ID == 7040247247 and m.OWNER_LOGIN == "shuntatsu"
    st = m.STAGES["BREADTH"]
    assert st.grid_start == datetime(2021, 1, 1, tzinfo=UTC)
    assert st.end == datetime(2026, 8, 1, tzinfo=UTC)
    assert st.first_close == datetime(2021, 1, 1, 4, tzinfo=UTC)
    assert st.n_hours == 2038 * 24 and st.n_hours % 24 == 0
    assert (st.n_hours // 4) == 12228


def test_cost_constants_match_the_repository_execution_formula() -> None:
    from trade_rl.simulation.execution import ExecutionCostConfig

    cfg = ExecutionCostConfig()
    assert (m.FEE_RATE, m.SPREAD_RATE, m.IMPACT_RATE, m.MAX_PARTICIPATION) == (
        cfg.fee_rate,
        cfg.spread_rate,
        cfg.impact_rate,
        cfg.max_participation_rate,
    )
    fee, spr, imp = m.unit_cost_rate(0.04, 2.0)
    assert (fee, spr) == (2 * cfg.fee_rate, 2 * cfg.spread_rate)
    assert imp == pytest.approx(2 * cfg.impact_rate * 0.2)


def test_universe_and_notices_are_pinned_and_complete() -> None:
    syms = m.load_universe()
    assert len(syms) == 441 and len(set(syms)) == 441
    assert m.sha256_file(m.UNIVERSE_PATH) == m.UNIVERSE_SHA256
    for touched in ("BTCUSDT", "ETHUSDT", "TRXUSDT", "SOLUSDT"):
        assert touched not in syms
    assert all(re.fullmatch(r"[A-Z0-9]+USDT", s) for s in syms)
    notices = m.load_notices()
    assert m.sha256_file(m.NOTICES_PATH) == m.NOTICES_SHA256
    assert sum(len(v) for v in notices.values()) == 234
    assert all(v == sorted(v) for v in notices.values())


def test_pinned_files_refuse_a_changed_copy(tmp_path: Path) -> None:
    bad = tmp_path / "u.json"
    bad.write_bytes(m.UNIVERSE_PATH.read_bytes() + b" ")
    with pytest.raises(SystemExit, match="pinned sha256"):
        m.load_universe(bad)
    badn = tmp_path / "n.json"
    badn.write_bytes(m.NOTICES_PATH.read_bytes() + b" ")
    with pytest.raises(SystemExit, match="pinned sha256"):
        m.load_notices(badn)


# ----------------------------------------------------------------------------- primitives and frame


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
        "vol": qv / c,
        "tbq": qv * rng.uniform(0.4, 0.6, n_hours),
    }
    for i in holes:
        for k in arrs:
            arrs[k][i] = np.nan
    return arrs


def test_rolling_primitives_propagate_nan() -> None:
    x = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0])
    assert np.isnan(m.r_mean(x, 3)[:4]).all()
    assert m.r_mean(x, 3)[5] == pytest.approx(5.0)
    assert m.r_std(np.array([1.0, 2.0, 4.0]), 3)[2] == pytest.approx(
        np.std([1, 2, 4], ddof=1)
    )
    assert np.isnan(m.shift(x, 2)[:2]).all() and m.shift(x, 2)[2] == 1.0


def test_frame_shape_and_warmup() -> None:
    arrs = synth_hourly(24 * 300, 1)
    fr = m.build_frame(arrs)
    assert fr.close.shape == (24 * 300 // 4,)
    assert np.isnan(fr.regime[: 6 * 200 - 1]).all()
    assert np.isfinite(fr.regime[6 * 200 :]).all()
    fin = fr.regime[~np.isnan(fr.regime)]
    assert np.allclose(fin * 4, np.round(fin * 4)) and np.abs(fin).max() <= 1.0
    assert (
        np.isnan(fr.vol_ann[: 6 * 31 - 1]).all()
        and np.isfinite(fr.vol_ann[6 * 31 :]).all()
    )
    assert np.array_equal(fr.fill_qv, arrs["qv"][3::4])
    with pytest.raises(ValueError, match="whole days"):
        m.build_frame({k: v[:-1] for k, v in arrs.items()})


@pytest.mark.parametrize("cut_day", [210.0, 240.25, 280.5])
def test_frame_is_causal_under_truncation(cut_day: float) -> None:
    full = synth_hourly(24 * 300, 2, holes=(5000, 5001))
    cut_h = int(cut_day * 24) // 4 * 4
    day_h = cut_h // 24 * 24
    trunc = {k: v[: day_h + 24].copy() for k, v in full.items()}
    a = m.build_frame(full)
    b = m.build_frame(trunc)
    k = cut_h // 4
    for col in ("close", "regime", "vol_ann", "fill_qv"):
        assert np.array_equal(
            getattr(a, col)[:k], getattr(b, col)[:k], equal_nan=True
        ), col
    future = {key: v.copy() for key, v in full.items()}
    rng = np.random.default_rng(9)
    for key in future:
        future[key][day_h + 24 :] = rng.uniform(
            1.0, 500.0, len(future[key]) - day_h - 24
        )
    c = m.build_frame(future)
    for col in ("close", "regime", "vol_ann", "fill_qv"):
        assert np.array_equal(
            getattr(a, col)[:k], getattr(c, col)[:k], equal_nan=True
        ), col


def test_frame_uses_latest_completed_daily_bar() -> None:
    arrs = synth_hourly(24 * 260, 3)
    fr = m.build_frame(arrs)
    mod = {k: v.copy() for k, v in arrs.items()}
    day = 230
    for key in mod:
        mod[key][day * 24 + 23] *= 1.5
    fr2 = m.build_frame(mod)
    boundary = day * 6 + 5  # the 4h bar whose close is the day boundary
    assert np.array_equal(fr.vol_ann[:boundary], fr2.vol_ann[:boundary], equal_nan=True)
    assert fr.vol_ann[boundary] != fr2.vol_ann[boundary]


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
    f["close"] = b4.close.values
    return f


@pytest.mark.parametrize("holes", [(), (3000, 3001, 5000)])
def test_frame_matches_pandas_reference(holes: tuple[int, ...]) -> None:
    pytest.importorskip("pandas")
    arrs = synth_hourly(24 * 330, 1, holes)
    ref = _pandas_reference(arrs, "2025-01-01")
    fr = m.build_frame(arrs)
    for col, got in (
        ("regime", fr.regime),
        ("vol_ann", fr.vol_ann),
        ("close", fr.close),
    ):
        expect = ref[col].to_numpy(np.float64)
        assert np.array_equal(np.isnan(expect), np.isnan(got)), col
        ok = ~np.isnan(expect)
        assert np.allclose(expect[ok], got[ok], rtol=1e-9, atol=1e-9), col


# ----------------------------------------------------------------------------- episodes and notices


def _live(n: int, dead: list[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray]:
    close = np.full(n, 100.0)
    traded = np.ones(n, dtype=bool)
    for a, b in dead:
        close[a:b] = np.nan
        traded[a:b] = False
    return close, traded


def test_dead_runs_and_episodes() -> None:
    assert m.dead_runs(np.array([0, 1, 1, 0, 1, 1, 1, 0], dtype=bool), 3) == [(4, 7)]
    # a hole of 23 hours stays inside the episode, 24 hours splits it
    c, t = _live(200, [(50, 73)])
    eps = m.find_episodes(c, t)
    assert eps == [m.Episode(0, 200, False)]
    c, t = _live(200, [(50, 74)])
    eps = m.find_episodes(c, t)
    assert eps == [m.Episode(0, 50, True), m.Episode(74, 200, False)]
    # leading dead hours are not part of an episode; a trailing long run means delisted
    c, t = _live(200, [(0, 30), (150, 200)])
    assert m.find_episodes(c, t) == [m.Episode(30, 150, True)]
    # a short trailing dead run is not a delisting
    c, t = _live(200, [(190, 200)])
    assert m.find_episodes(c, t) == [m.Episode(0, 190, False)]
    # zero volume with a finite price is dead (the Vision zombie tail)
    c, t = _live(200, [])
    t[120:] = False
    assert m.find_episodes(c, t) == [m.Episode(0, 120, True)]
    # no traded hour: no episode
    assert m.find_episodes(np.full(50, np.nan), np.zeros(50, dtype=bool)) == []


def test_forced_start_notice_pairing() -> None:
    times = T0 + m.BAR_S * (np.arange(2000) + 1)
    ev = 1500
    e_s = float(times[ev])
    day = m.DAY_S
    # earliest notice within 90 days wins; later postponement notices never delay the start
    rel1, rel2 = e_s - 20 * day, e_s - 5 * day
    start, status = m.forced_start(times, 0, ev, [rel1, rel2])
    assert status == "notice"
    first = start
    assert start is not None and times[start] - m.BAR_S >= rel1
    assert (
        times[start - 1] - m.BAR_S < rel1
    )  # the FIRST decision bar opening at/after release
    # a notice older than 90 days, after the event, or none at all: no forced window (the
    # position is held to the last bar and written off there; no hindsight)
    assert m.forced_start(times, 0, ev, [e_s - 91 * day]) == (None, "none")
    assert m.forced_start(times, 0, ev, [e_s + 5]) == (None, "none")
    assert m.forced_start(times, 1490, ev, []) == (None, "none")
    # released inside the last bar: too late to close, only the write-off applies
    assert m.forced_start(times, 0, ev, [e_s - 100]) == (None, "late")
    assert first is not None and first < ev
    # the earliest of two is used even when listed out of order after sorting
    assert m.notice_times({"X": [5.0, 3.0, 9.0e12]}, "X", 100) == [3.0, 5.0]


def test_zombie_tail_blank_and_relisting_start_a_new_episode_with_fresh_warmup() -> (
    None
):
    n = 24 * 700
    h = synth_hourly(n, 4)
    h["qv"][24 * 300 :] = 0.0
    h["vol"][24 * 300 :] = 0.0  # zombie: finite flat price, no volume
    stage = m.Stage(
        "T", datetime(2021, 1, 1, tzinfo=UTC), datetime(2022, 12, 2, tzinfo=UTC)
    )
    assert stage.n_hours == n
    # relisting: a second episode starts after a gap
    h2 = synth_hourly(n, 5)
    for k in ("open", "high", "low", "close"):
        h2[k][24 * 250 : 24 * 260] = np.nan
    h2["qv"][24 * 250 : 24 * 260] = 0.0
    h2["vol"][24 * 250 : 24 * 260] = 0.0
    panel, info = m.build_panel({"AAAUSDT": h, "BBBUSDT": h2}, stage, {}, None)
    assert panel.events == ((24 * 250 // 4 - 1, 1), (24 * 300 // 4 - 1, 0))
    # AAA: no price after its last traded bar
    assert np.isnan(panel.mark_px[24 * 300 // 4 :, 0]).all()
    assert np.isfinite(panel.mark_px[: 24 * 300 // 4, 0]).all()
    # BBB: the new episode has regime NaN until it has 200 fresh daily bars
    reg = panel.regime[:, 1]
    assert np.isfinite(reg[6 * 200 : 24 * 250 // 4 - 6]).all()
    relist = 24 * 260 // 4
    assert np.isnan(reg[relist : relist + 6 * 199]).all()
    assert np.isfinite(reg[relist + 6 * 201 :]).all()
    assert [r["delisted"] for r in info["episodes"]["BBBUSDT"]] == [True, False]
    # eligibility needs the regime: the symbol is out of the cross-section during the gap
    elig = m.eligible_matrix(panel)
    assert not elig[24 * 250 // 4 : relist + 6 * 200 - 1, 1].any()
    assert elig[relist + 6 * 200 + 5, 1]
    # no notice recorded: no forced window, both events are write-off-only
    assert info["notice_none"] == 2 and info["notice_matched"] == 0
    assert not panel.forced.any()
    assert info["no_notice_symbols"] == ["AAAUSDT", "BBBUSDT"]


def test_build_panel_marks_forced_window_ages_and_holes() -> None:
    n = 24 * 400
    h = synth_hourly(n, 6)
    stage = m.Stage(
        "T", datetime(2021, 1, 1, tzinfo=UTC), datetime(2022, 2, 5, tzinfo=UTC)
    )
    assert stage.n_hours == n
    end_hour = 24 * 330
    for k in ("open", "high", "low", "close"):
        h[k][end_hour:] = np.nan
    h["qv"][end_hour:] = 0.0
    h["vol"][end_hour:] = 0.0
    release = T0 + 24 * 3600 * 320 + 1800  # inside the 4h bar that opens at hour 24*320
    panel, info = m.build_panel({"DDDUSDT": h}, stage, {}, {"DDDUSDT": [release]})
    j_last = end_hour // 4 - 1
    assert panel.events == ((j_last, 0),)
    forced_rows = np.nonzero(panel.forced[:, 0])[0]
    first_open_after = math.ceil((release - panel.times[0] + m.BAR_S) / m.BAR_S)
    assert forced_rows[0] == first_open_after and forced_rows[-1] == j_last
    assert (
        panel.times[first_open_after] - m.BAR_S
        >= release
        > panel.times[first_open_after - 1] - m.BAR_S
    )
    assert info["notice_matched"] == 1
    # listing-age bucket: the episode starts at the first complete bar, not eligible before day 200
    ages = panel.age_bucket[:, 0]
    assert (
        ages[0] == 0
        and ages[6 * 201] == 1
        and ages[j_last] == 1
        and ages[j_last + 1] == 0
    )
    assert np.isnan(panel.close[j_last + 1 :, 0]).all()
    # a short hole inside the episode: price carried for marking, no tradable close
    h2 = synth_hourly(n, 7)
    for k in ("open", "high", "low", "close", "qv", "vol"):
        h2[k][24 * 100 + 2 : 24 * 100 + 6] = np.nan
    p2, _ = m.build_panel({"AAAUSDT": h2}, stage, {}, None)
    hole_bar = (24 * 100 + 2) // 4
    assert np.isnan(p2.close[hole_bar, 0]) and np.isfinite(p2.mark_px[hole_bar, 0])
    assert p2.mark_px[hole_bar, 0] == p2.close[hole_bar - 1, 0]


def test_a_trailing_hole_at_the_stage_end_is_marked_not_delisted() -> None:
    n = 24 * 260
    h = synth_hourly(n, 12)
    for k in ("open", "high", "low", "close", "qv", "vol"):
        h[k][n - 7 :] = np.nan  # 7 missing hours: shorter than a day, not a delisting
    stage = m.Stage(
        "T", datetime(2021, 1, 1, tzinfo=UTC), datetime(2021, 9, 18, tzinfo=UTC)
    )
    assert stage.n_hours == n
    panel, _ = m.build_panel({"AAAUSDT": h}, stage, {}, None)
    assert panel.events == ()
    nb = n // 4
    assert np.isnan(panel.close[nb - 1, 0]) and np.isfinite(panel.mark_px[nb - 1, 0])
    assert panel.mark_px[nb - 1, 0] == panel.close[nb - 3, 0]
    cell = m.simulate_weights(panel, m.trend_weights(panel, 1), 1.0, 0)
    assert cell["valid"], cell["invalid_reasons"]


def test_funding_events_map_to_the_bar_they_are_paid_over() -> None:
    n = 24 * 231
    h = synth_hourly(n, 8)
    stage = m.Stage(
        "T", datetime(2021, 1, 1, tzinfo=UTC), datetime(2021, 8, 20, tzinfo=UTC)
    )
    assert stage.n_hours == n
    c0 = (T0 + m.BAR_S) * 1000  # the first close c_0
    ev = [
        (c0 - 1, 0.001, 50.0),  # before c_0: ignored
        (c0, 0.001, 50.0),  # at c_0: paid over bar 1 (units held after the c_0 fill)
        (
            c0 + m.BAR_MS - 1,
            0.002,
            math.nan,
        ),  # last millisecond of bar 1: priced at the 1h open
        (c0 + m.BAR_MS, 0.003, 70.0),  # at c_1: paid over bar 2
        (n * m.HOUR_MS + T0 * 1000, 0.5, 1.0),  # at the stage end: never paid
    ]
    panel, info = m.build_panel({"AAAUSDT": h}, stage, {"AAAUSDT": ev}, None)
    assert panel.fund[0, 0] == 0.0
    assert panel.fund[1, 0] == pytest.approx(50.0 * 0.001 + float(h["open"][7]) * 0.002)
    assert panel.fund[2, 0] == pytest.approx(70.0 * 0.003)
    assert panel.fund[3:].sum() == 0.0
    assert info["funding_mark_fallbacks"] == 1


# ----------------------------------------------------------------------------- weights


def _arr(*v: float) -> np.ndarray:
    return np.array(v, dtype=float)


def test_regime_direction_and_size_rule() -> None:
    d = m.regime_direction(_arr(1.0, 0.5, 0.25, 0.0, -0.25, -0.5, -1.0, np.nan))
    assert d.tolist() == [1, 1, 0, 0, 0, -1, -1, 0]
    assert m._size(0.25) == 1.0 and m._size(1.0) == 0.5 and m._size(0.0) == 1.0


def test_symbol_weights_sleeve_is_fixed_at_entry_and_reset_on_a_flip() -> None:
    reg = _arr(1, 1, 1, 1, -1, -1, 1)
    vol = _arr(1.0, 0.8, 2.0, 1.0, 1.0, 1.0, 1.0)  # size 0.5 at entry; fixed while long
    cl = np.full(7, 100.0)
    el = np.ones(7, dtype=bool)
    nt = _arr(5, 5, 10, 10, 10, 20, 20)
    w = m.symbol_weights(reg, vol, cl, el, nt, n_min=3)
    assert w[0] == pytest.approx(0.5 * 0.9 / 5)
    assert (
        w[1] == w[2] == w[3] == w[0]
    )  # size and sleeve fixed although vol and N_t moved
    assert w[4] == pytest.approx(-0.5 * 0.9 / 10)  # flip: new size, new sleeve
    assert w[5] == w[4]
    assert w[6] == pytest.approx(0.5 * 0.9 / 20)


def test_symbol_weights_n_min_blocks_entries_only_and_hold_on_nan() -> None:
    reg = _arr(1, 1, 1, 1, 1, 1)
    vol = _arr(0.2, 0.2, 0.2, np.nan, 0.2, 0.2)
    cl = _arr(100, 100, 100, 100, np.nan, 100)
    el = np.ones(6, dtype=bool)
    nt = _arr(3, 3, 6, 2, 2, 2)
    w = m.symbol_weights(reg, vol, cl, el, nt, n_min=5)
    assert w[0] == w[1] == 0.0  # N_t below the minimum: no entry
    assert w[2] == pytest.approx(0.9 / 6)  # enters once N_t >= 5
    assert (
        w[3] == w[4] == w[2]
    )  # open positions continue below N_min and on non-finite inputs
    assert w[5] == w[2]
    # an ineligible row flattens the symbol; re-entry takes a new sleeve
    el2 = np.array([1, 1, 1, 0, 1, 1], dtype=bool)
    nt2 = _arr(3, 3, 6, 2, 8, 8)
    w2 = m.symbol_weights(reg, vol, cl, el2, nt2, n_min=5)
    assert w2[3] == 0.0 and w2[4] == 0.0 and w2[5] == pytest.approx(0.9 / 8)


def mk_panel(
    close: np.ndarray,
    qv: np.ndarray | float = 1e12,
    fund: np.ndarray | None = None,
    forced: np.ndarray | None = None,
    events: tuple[tuple[int, int], ...] = (),
    t0: int = T0 + m.BAR_S,
    mark_px: np.ndarray | None = None,
    regime: float = 1.0,
    vol: float = 0.2,
) -> m.Panel:
    close = np.asarray(close, dtype=float)
    if close.ndim == 1:
        close = close[:, None]
    rows, slots = close.shape
    return m.Panel(
        times=t0 + m.BAR_S * np.arange(rows),
        symbols=tuple(f"S{i}USDT" for i in range(slots)),
        close=close,
        mark_px=close.copy() if mark_px is None else mark_px,
        fill_qv=np.broadcast_to(np.asarray(qv, dtype=float), close.shape).copy(),
        regime=np.full(close.shape, regime),
        vol=np.full(close.shape, vol),
        forced=np.zeros(close.shape, dtype=bool) if forced is None else forced,
        age_bucket=np.zeros(close.shape, dtype=np.int8),
        fund=np.zeros(close.shape) if fund is None else fund,
        events=events,
    )


def test_rebalance_rows_and_ew_weights() -> None:
    n = 6 * 70
    p = mk_panel(np.full((n, 4), 100.0))
    reb = m.rebalance_rows(p.times)
    stamps = [datetime.fromtimestamp(int(t), UTC) for t in p.times[reb]]
    assert [s.day for s in stamps] == [1, 1] and {s.month for s in stamps} == {2, 3}
    w, force = m.ew_weights(p, n_min=3)
    # the first row with N_t >= n_min is a rebalance too, then month starts only
    assert np.allclose(w[0], 0.9 / 4) and force[0].all() and not force[1].any()
    j1 = int(np.nonzero(reb)[0][0])
    assert np.allclose(w[j1], 0.9 / 4) and force[j1].all() and not force[j1 + 1].any()
    assert force.all(axis=1).sum() == 3
    # eligibility only from row 100: the first rebalance is row 100, before the month start
    late = m.Panel(
        **{
            **p.__dict__,
            "regime": np.where(np.arange(n)[:, None] >= 100, 1.0, np.nan)
            * np.ones((1, 4)),
        }
    )
    wl, fl = m.ew_weights(late, n_min=3)
    assert (wl[:100] == 0).all() and np.allclose(wl[100], 0.9 / 4) and fl[100].all()
    assert 100 < j1 and fl.all(axis=1).sum() == 3
    # a symbol that leaves the eligible set is closed; nobody is added between rebalances
    p2 = mk_panel(np.full((n, 4), 100.0))
    reg = p2.regime.copy()
    reg[j1 + 5 :, 3] = np.nan
    p2 = m.Panel(**{**p2.__dict__, "regime": reg})
    w2, f2 = m.ew_weights(p2, n_min=3)
    assert w2[j1 + 4, 3] == pytest.approx(0.9 / 4) and w2[j1 + 5, 3] == 0.0
    assert w2[j1 + 5, 0] == pytest.approx(
        0.9 / 4
    )  # others are not re-sized between rebalances
    j2 = int(np.nonzero(reb)[0][1])
    assert np.allclose(w2[j2, :3], 0.9 / 3) and w2[j2, 3] == 0.0 and f2[j2].all()
    # fewer than n_min eligible: flat
    w3, _ = m.ew_weights(p, n_min=5)
    assert (w3 == 0).all()


def test_trend_weights_long_only_is_the_clipped_book() -> None:
    p = mk_panel(np.full((100, 3), 100.0))
    reg = p.regime.copy()
    reg[:, 1] = -1.0
    p = m.Panel(**{**p.__dict__, "regime": reg})
    ls = m.trend_weights(p, n_min=2)
    lo = m.trend_weights(p, n_min=2, long_only=True)
    assert (ls[:, 1] < 0).any() and (lo[:, 1] == 0).all()
    assert np.array_equal(lo, np.maximum(ls, 0.0))


# ----------------------------------------------------------------------------- the actual-units book


def _w(n: int, *cols: float) -> np.ndarray:
    return np.tile(np.array(cols, dtype=float), (n, 1))


def test_units_are_held_no_free_rebalancing() -> None:
    p = mk_panel([100.0, 200.0, 100.0])
    w = np.array([[0.5], [0.5]])
    cell = m.simulate_weights(p, w, 0.0, 0)
    assert cell["valid"]
    assert cell["equity"].tolist() == pytest.approx([1.0, 1.0, 1.5, 1.0])
    assert cell["fills"] == 1


def test_cost_is_charged_on_actual_fill_notional_with_the_repository_formula() -> None:
    p = mk_panel([100.0, 100.0], qv=1e6)
    cell = m.simulate_weights(p, np.array([[0.5]]), 1.0, 0)
    notional = 0.5 * INIT
    part = notional / 1e6
    expect = notional * (0.0005 + 0.0002 + 0.0001 * math.sqrt(part))
    assert cell["equity"][1] == pytest.approx(1.0 - expect / INIT, rel=1e-12)
    assert cell["costs_total"] == pytest.approx(expect / INIT)
    assert cell["cost_paid"]["long"] == pytest.approx(expect / INIT)
    hi = m.simulate_weights(p, np.array([[0.5]]), 2.0, 0)
    assert hi["costs_total"] == pytest.approx(2 * expect / INIT)


def test_fill_is_capped_at_five_percent_of_volume_and_the_residual_is_carried() -> None:
    p = mk_panel(np.full(14, 100.0), qv=1e5)  # cap 5,000 per bar; want 50,000
    cell = m.simulate_weights(p, _w(13, 0.5), 0.0, 0)
    assert cell["capped_fills"] == 9 and cell["fills"] == 10
    assert cell["turnover_x"] > 0 and cell["valid"]
    # the same 500 units are reached after ten bars, not more
    assert cell["price_pnl"]["long"] == 0.0


def test_increasing_fill_is_clamped_to_the_side_cap_and_never_borrows() -> None:
    p = mk_panel([100.0, 100.0, 100.0])
    cell = m.simulate_weights(p, _w(2, 1.5), 1.0, 0)
    assert cell["valid"] and cell["clamped_fills"] >= 1
    assert cell["min_cash_frac"] >= -1e-12
    # long notional <= equity after costs: cash stays non-negative
    short = m.simulate_weights(p, _w(2, -1.5), 1.0, 0)
    assert short["valid"] and short["clamped_fills"] >= 1
    assert short["max_side_at_marks"] <= 1.0 + 1e-9
    # reducing fills are never blocked, even with the side full
    w = np.array([[1.5], [0.0]])
    out = m.simulate_weights(p, w, 1.0, 0)
    assert out["valid"] and out["fills"] == 2


def test_funding_is_paid_on_held_units_with_the_actual_sign_and_timing() -> None:
    fund = np.zeros((4, 1))
    fund[1, 0] = 100.0 * 0.001  # price * rate, paid over bar 1
    p = mk_panel(np.full(4, 100.0), fund=fund)
    long_ = m.simulate_weights(p, _w(3, 0.5), 0.0, 0)
    assert long_["funding_pnl"]["long"] == pytest.approx(-500 * 0.1 / INIT)
    short = m.simulate_weights(p, _w(3, -0.5), 0.0, 0)
    assert short["funding_pnl"]["short"] == pytest.approx(500 * 0.1 / INIT)
    # a decision at c_1 fills after the event of bar 1 and pays nothing for it
    late = np.zeros((3, 1))
    late[1, 0] = 0.5
    lt = m.simulate_weights(p, late, 0.0, 0)
    assert lt["funding_pnl"]["long"] == 0.0


def test_delay_fills_one_bar_later_at_that_close() -> None:
    p = mk_panel([100.0, 110.0, 120.0, 120.0])
    base = m.simulate_weights(p, _w(3, 0.5), 0.0, 0)
    dly = m.simulate_weights(p, _w(3, 0.5), 0.0, 1)
    assert base["price_pnl"]["long"] == pytest.approx(500 * 20 / INIT)
    assert dly["price_pnl"]["long"] == pytest.approx((0.5 * INIT / 110) * 10 / INIT)


def test_forced_close_fill_pays_the_extra_five_percent() -> None:
    forced = np.zeros((4, 1), dtype=bool)
    forced[1:3, 0] = True
    p = mk_panel(np.full(4, 100.0), forced=forced)
    w = np.array([[0.5], [0.0], [0.0]])
    cell = m.simulate_weights(p, w, 0.0, 0)
    assert cell["forced_fills"] == 1
    assert cell["forced_charge"] == pytest.approx(0.05 * 0.5)
    assert cell["equity"][-1] == pytest.approx(1.0 - 0.05 * 0.5)
    # an ordinary (not forced) close pays no extra charge
    plain = m.simulate_weights(mk_panel(np.full(4, 100.0)), w, 0.0, 0)
    assert plain["forced_fills"] == 0 and plain["equity"][-1] == pytest.approx(1.0)


def test_write_off_settles_a_position_still_held_at_the_last_available_bar() -> None:
    close = np.full((5, 1), 100.0)
    close[3:, 0] = np.nan
    mark = close.copy()
    qv = np.full((5, 1), 1e12)
    p = mk_panel(close, qv=qv, mark_px=mark, events=((2, 0),))
    # the target stays 0.5 through the event bar: held at j=2 and settled there
    cell = m.simulate_weights(p, _w(4, 0.5), 0.0, 0)
    assert cell["write_off_count"] == 1 and cell["valid"]
    assert cell["write_offs"][0]["notional"] == pytest.approx(0.5 * INIT)
    assert cell["equity"][3] == pytest.approx(1.0 - 0.05 * 0.5)
    # no position after the settlement: the equity is flat afterwards
    assert cell["equity"][3] == cell["equity"][4] == pytest.approx(cell["equity"][-1])
    assert cell["forced_charge"] == pytest.approx(0.05 * 0.5)


def test_nonpositive_equity_is_invalid() -> None:
    p = mk_panel([100.0, 100.0, 400.0, 400.0])
    cell = m.simulate_weights(
        p, _w(3, -0.9), 0.0, 0
    )  # short 900 units, price x4: loss 2.7e5
    assert not cell["valid"] and "not positive" in cell["invalid_reasons"][0]


def test_cash_below_the_borrowing_limit_is_invalid_and_a_fill_never_creates_it() -> (
    None
):
    fund = np.zeros((4, 1))
    fund[2, 0] = 100.0 * 0.05  # a huge funding payment drains cash below -2% of equity
    p = mk_panel(np.full(4, 100.0), fund=fund)
    cell = m.simulate_weights(p, _w(3, 1.0), 0.0, 0)
    assert not cell["valid"]
    assert any(r.startswith("negative_cash_borrowing") for r in cell["invalid_reasons"])
    assert cell["min_cash_frac"] < -m.BORROW_TOL


def test_identity_break_is_detected(monkeypatch: pytest.MonkeyPatch) -> None:
    orig = m._Book.mark

    def leaky(self: Any, j: int) -> None:
        orig(self, j)
        self.cash -= 1.0

    monkeypatch.setattr(m._Book, "mark", leaky)
    p = mk_panel(np.full(5, 100.0))
    cell = m.simulate_weights(p, _w(4, 0.4), 0.0, 0)
    assert not cell["valid"]
    assert any("identity" in r for r in cell["invalid_reasons"])


def test_prices_of_held_symbols_must_be_positive() -> None:
    p = mk_panel([100.0, 0.0, 100.0])
    cell = m.simulate_weights(p, _w(2, 0.5), 0.0, 0)
    assert not cell["valid"]
    bad = m.panel_problems(p, {"_ev_times": {}})
    assert "non-positive price in an episode" in bad


def test_side_attribution_reconciles_with_equity() -> None:
    rng = np.random.default_rng(3)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, (60, 2)), axis=0))
    fund = rng.normal(0, 0.05, (60, 2))
    fund[0] = 0.0
    p = mk_panel(close, qv=5e6, fund=fund)
    w = np.tile([0.3, -0.3], (59, 1))
    cell = m.simulate_weights(p, w, 1.0, 0)
    assert cell["valid"]
    total = (
        sum(cell["price_pnl"].values())
        + sum(cell["funding_pnl"].values())
        - sum(cell["cost_paid"].values())
    )
    assert cell["total_return"] == pytest.approx(total, abs=1e-9)
    assert cell["price_pnl"]["short"] != 0 and cell["price_pnl"]["long"] != 0
    assert sum(cell["age_bucket_pnl"].values()) == pytest.approx(
        cell["total_return"], abs=1e-9
    )


def test_cash_cell() -> None:
    c = m.cash_cell(10)
    assert c["valid"] and c["total_return"] == 0.0 and len(c["returns"]) == 11
    assert c["max_dd"] == 0.0 and c["sharpe"] == 0.0


# ----------------------------------------------------------------------------- no-borrowing proofs


def bull_scenario(seed: int, n_sym: int = 8, rows: int = 6 * 300) -> m.Panel:
    """A strong bull market with thin and thick symbols: positions fill their budgets and the
    participation cap binds, the setting in which the xs EW_LONG control borrowed."""
    rng = np.random.default_rng(seed)
    r = rng.normal(0.004, 0.015, (rows, n_sym))
    close = 100 * np.exp(np.cumsum(r, axis=0))
    qv = np.tile(np.array([2e4, 5e4, 3e5, 2e6, 2e6, 5e6, 1e7, 1e9][:n_sym]), (rows, 1))
    return mk_panel(close, qv=qv, t0=T0 + m.BAR_S, regime=1.0, vol=0.2)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_long_only_books_never_borrow_cash(seed: int) -> None:
    p = bull_scenario(seed)
    w_ew, force = m.ew_weights(p, n_min=3)
    w_long = m.trend_weights(p, n_min=3, long_only=True)
    ew = m.simulate_weights(p, w_ew, 1.0, 0, force)
    tl = m.simulate_weights(p, w_long, 1.0, 0)
    for cell in (ew, tl):
        assert cell["valid"], cell["invalid_reasons"]
        assert cell["min_cash_frac"] >= -1e-9, cell["min_cash_frac"]
        assert cell["max_side_at_marks"] <= 1.0 + 1e-9
        assert cell["fills"] > 10 and cell["capped_fills"] > 0
    assert ew["mean_long_gross"] > 0.3 and tl["mean_long_gross"] > 0.3


def test_the_xs_defect_a_fully_invested_book_cannot_occur_with_the_clamp() -> None:
    """A full-budget long book (weights summing to 1.0) pays costs on its fills and would end a
    fill below zero cash without the clamp; with it the last fill is clamped and cash stays >= 0."""
    p = mk_panel(np.full((40, 4), 100.0))
    w = np.full((39, 4), 0.25)  # nominal gross exactly 1.0
    cell = m.simulate_weights(p, w, 1.0, 0)
    assert cell["valid"] and cell["min_cash_frac"] >= -1e-12
    assert cell["clamped_fills"] >= 1


# ----------------------------------------------------------------------------- the independent oracle


def oracle_weights(
    p: m.Panel, n_min: int, long_only: bool = False
) -> tuple[list[list[float]], list[list[bool]]]:
    S, T = len(p.symbols), len(p.times)
    fin = math.isfinite
    elig = [
        [
            fin(p.regime[j][k])
            and fin(p.vol[j][k])
            and fin(p.mark_px[j][k])
            and not p.forced[j][k]
            for k in range(S)
        ]
        for j in range(T)
    ]
    n_t = [sum(row) for row in elig]
    w = [[0.0] * S for _ in range(T)]
    for k in range(S):
        pos = 0.0
        sleeve = 0.0
        for j in range(T):
            if not elig[j][k]:
                pos = 0.0
                continue
            va, c = p.vol[j][k], p.close[j][k]
            if fin(va) and fin(c):
                r = p.regime[j][k]
                d = 1 if r >= 0.5 else (-1 if r <= -0.5 else 0)
                size = 1.0 if va <= 0 else min(1.0, 0.5 / va)
                new = d * size
                if pos != 0 and new != 0 and (new > 0) == (pos > 0):
                    new = pos
                elif new != 0:
                    if n_t[j] < n_min:
                        new = 0.0
                    else:
                        sleeve = 0.9 / n_t[j]
                pos = new
            w[j][k] = pos * sleeve
    if long_only:
        w = [[max(x, 0.0) for x in row] for row in w]
    return w, [[False] * S for _ in range(T)]


def oracle_ew(p: m.Panel, n_min: int) -> tuple[list[list[float]], list[list[bool]]]:
    S, T = len(p.symbols), len(p.times)
    fin = math.isfinite
    cur = [0.0] * S
    w, force = [], []
    started = False
    for j in range(T):
        el = [
            fin(p.regime[j][k])
            and fin(p.vol[j][k])
            and fin(p.mark_px[j][k])
            and not p.forced[j][k]
            for k in range(S)
        ]
        n = sum(el)
        dt = datetime.fromtimestamp(int(p.times[j]), UTC)
        reb = dt.day == 1 and dt.hour == 0 and dt.minute == 0
        first = (not started) and n >= n_min
        started = started or n >= n_min
        reb = reb or first
        if reb:
            cur = [0.9 / n if (el[k] and n >= n_min) else 0.0 for k in range(S)]
        else:
            cur = [cur[k] if el[k] else 0.0 for k in range(S)]
        w.append(list(cur))
        force.append([reb] * S)
    return w, force


def oracle_book(
    p: m.Panel,
    w: list[list[float]],
    force: list[list[bool]],
    mult: float,
    delay: int,
) -> list[float]:
    S, n = len(p.symbols), p.bars
    u = [0.0] * S
    wprev = [0.0] * S
    pend: list[float | None] = [None] * S
    pf = [False] * S
    cash = INIT
    settle: dict[int, list[int]] = {}
    for je, k in p.events:
        settle.setdefault(je, []).append(k)
    out = []
    for j in range(n + 1):
        px = [float(p.mark_px[j][k]) for k in range(S)]
        if j >= 1:
            for k in range(S):
                if u[k] != 0:
                    cash += -u[k] * float(p.fund[j][k])
        d = j - delay
        if 0 <= d <= n - 1:
            e_pre = cash + sum(u[k] * px[k] for k in range(S) if u[k] != 0)
            cost_ev = 0.0
            for k in range(S):
                if abs(w[d][k] - wprev[k]) > 1e-12 or force[d][k]:
                    wprev[k] = w[d][k]
                    if not (math.isfinite(px[k]) and px[k] > 0):
                        pend[k] = None
                        continue
                    pend[k] = w[d][k] * e_pre / px[k]
                    pf[k] = bool(p.forced[d][k])
            for k in range(S):
                if pend[k] is not None:
                    pf[k] = pf[k] or bool(p.forced[d][k])
            ks = [k for k in range(S) if pend[k] is not None]

            def reducing(k: int) -> bool:
                t = pend[k]
                assert t is not None
                return abs(t) <= abs(u[k]) and t * u[k] >= 0

            order = [k for k in ks if reducing(k)] + [k for k in ks if not reducing(k)]
            held_l = sum(max(u[k], 0) * px[k] for k in range(S) if u[k] != 0)
            held_s = sum(max(-u[k], 0) * px[k] for k in range(S) if u[k] != 0)
            for k in order:
                tgt = pend[k]
                assert tgt is not None
                pre = u[k]
                delta = tgt - pre
                want = abs(delta) * px[k]
                if want <= 1e-15 * max(e_pre, 1.0):
                    pend[k] = None
                    continue
                vol = float(p.fill_qv[j][k])
                if not (
                    math.isfinite(vol)
                    and vol > 0
                    and math.isfinite(float(p.close[j][k]))
                ):
                    continue
                x = min(want, 0.05 * vol)
                if delta > 0:
                    rc = mult * (0.0005 + 0.0002 + 0.0001 * math.sqrt(x / vol)) + (
                        0.05 if pf[k] else 0.0
                    )
                    x = min(x, max(0.0, cash) / (1.0 + rc))
                red = min(x, abs(pre) * px[k]) if delta * pre < 0 else 0.0
                opn = x - red
                side = 1 if delta > 0 else -1
                if opn > 0:
                    held = held_l if side > 0 else held_s
                    r = mult * (0.0005 + 0.0002 + 0.0001 * math.sqrt(x / vol)) + (
                        0.05 if pf[k] else 0.0
                    )
                    room = max(0.0, 1.0 * (e_pre - cost_ev - r * red) - held) / (
                        1.0 + r
                    )
                    opn = min(opn, room)
                x = red + opn
                if x <= 1e-15 * max(e_pre, 1.0):
                    continue
                full = x >= want * (1 - 1e-15)
                total = x * mult * (0.0005 + 0.0002 + 0.0001 * math.sqrt(x / vol))
                extra = 0.05 * x if pf[k] else 0.0
                if full:
                    qty = tgt - pre
                    u[k] = tgt
                    pend[k] = None
                else:
                    qty = math.copysign(x / px[k], delta)
                    u[k] = pre + qty
                cash -= qty * px[k] + total + extra
                held_l += max(u[k], 0) * px[k] - max(pre, 0) * px[k]
                held_s += max(-u[k], 0) * px[k] - max(-pre, 0) * px[k]
                cost_ev += total + extra
        for k in settle.get(j, []):
            pend[k] = None
            wprev[k] = 0.0
            if u[k] != 0:
                cash += u[k] * px[k] - 0.05 * abs(u[k]) * px[k]
                u[k] = 0.0
        out.append((cash + sum(u[k] * px[k] for k in range(S) if u[k] != 0)) / INIT)
    return out


NO_NOTICE = "GGGUSDT"


def scenario(seed: int) -> tuple[m.Panel, dict[str, Any]]:
    """Seven symbols over 396 days: two live all window (up and down drift), a late listing, a
    thin delisted name with a notice (so a write-off), a relisting gap, a thin live name (the
    cap binds) and a delisting without a notice (hindsight fallback)."""
    stage = m.Stage(
        "BREADTH", datetime(2021, 1, 1, tzinfo=UTC), datetime(2022, 2, 1, tzinfo=UTC)
    )
    n = stage.n_hours
    rng = np.random.default_rng(seed)

    def mk(
        listed: int,
        drift: float,
        qv0: float,
        delist: int | None = None,
        gap: tuple[int, int] | None = None,
    ):
        r = rng.normal(drift, 0.006, n) + 0.0006 * np.sin(np.arange(n) / (24 * 40))
        c = 100 * np.exp(np.cumsum(r))
        o = np.r_[100.0, c[:-1]]
        d = {
            "open": o,
            "high": np.maximum(o, c) * 1.002,
            "low": np.minimum(o, c) * 0.998,
            "close": c.copy(),
            "qv": qv0 * rng.uniform(0.3, 1.7, n),
        }
        d["vol"] = d["qv"] / c

        def blank(a: int, b: int) -> None:
            for key in d:
                d[key][a:b] = np.nan
            d["qv"][a:b] = 0.0
            d["vol"][a:b] = 0.0

        blank(0, listed * 24)
        if delist:
            blank(delist * 24, n)
        if gap:
            blank(gap[0] * 24, gap[1] * 24)
        return d

    hourly = {
        "AAAUSDT": mk(0, 0.0002, 3e6),
        "BBBUSDT": mk(0, -0.0002, 3e6),
        "CCCUSDT": mk(30, 0.0001, 2e6),
        "DDDUSDT": mk(10, 0.0004, 1.5e3, delist=340),
        "EEEUSDT": mk(0, 0.0001, 2e6, gap=(215, 226)),
        "FFFUSDT": mk(0, 0.0002, 8e3),
        NO_NOTICE: mk(5, 0.0003, 2e6, delist=330),
    }
    g0 = T0 * 1000
    funding = {
        s: [
            (g0 + k * 8 * m.HOUR_MS + 1, float(rng.normal(0.0001, 0.0002)), math.nan)
            for k in range(n // 8)
        ]
        for s in hourly
    }
    notices = {"DDDUSDT": [T0 + 330 * 86400 - 3 * 3600 - 20 * 86400]}
    panel, info = m.build_panel(hourly, stage, funding, notices, tuple(hourly))
    info.update(hourly=hourly, stage=stage, funding=funding, notices=notices)
    return panel, info


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_weights_match_independent_oracle(seed: int) -> None:
    p, _ = scenario(seed)
    for lo in (False, True):
        ow, _f = oracle_weights(p, 3, lo)
        assert np.allclose(
            m.trend_weights(p, 3, long_only=lo), np.array(ow), atol=1e-15, rtol=0
        )
    ew, ef = oracle_ew(p, 3)
    w, force = m.ew_weights(p, 3)
    assert np.allclose(w, np.array(ew), atol=1e-15, rtol=0) and np.array_equal(
        force, np.array(ef)
    )


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("variant", ["BASE", "HIGH_COST", "DELAY"])
def test_engine_matches_independent_loop_oracle(seed: int, variant: str) -> None:
    p, _ = scenario(seed)
    mult, delay = m.VARIANTS[variant]
    w = m.trend_weights(p, 3)
    cell = m.simulate_weights(p, w, mult, delay)
    assert cell["valid"], cell["invalid_reasons"]
    ow, of = oracle_weights(p, 3)
    got = oracle_book(p, ow, of, mult, delay)
    assert np.allclose(cell["equity"][1:], got, rtol=0, atol=1e-9)
    # the long-only book and the control too
    lw = m.simulate_weights(p, m.trend_weights(p, 3, long_only=True), mult, delay)
    olw, olf = oracle_weights(p, 3, True)
    assert np.allclose(
        lw["equity"][1:], oracle_book(p, olw, olf, mult, delay), rtol=0, atol=1e-9
    )
    ew, force = m.ew_weights(p, 3)
    ewc = m.simulate_weights(p, ew, mult, delay, force)
    oew, oef = oracle_ew(p, 3)
    assert np.allclose(
        ewc["equity"][1:], oracle_book(p, oew, oef, mult, delay), rtol=0, atol=1e-9
    )


def test_oracle_scenario_is_nontrivial() -> None:
    p, info = scenario(0)
    cells = m.evaluate_cells(p, 3)
    ls = cells[m.PRIMARY]
    assert all(c["valid"] for c in cells.values()), {
        k: c["invalid_reasons"] for k, c in cells.items()
    }
    assert ls["fills"] > 20 and ls["capped_fills"] > 0
    assert (
        ls["forced_fills"] >= 1
        and ls["write_off_count"] + cells[m.EW_CELL]["write_off_count"] >= 1
    )
    assert ls["price_pnl"]["short"] != 0 and ls["price_pnl"]["long"] != 0
    assert cells[m.EW_CELL]["fills"] > 10
    assert (
        len(p.events) == 3
    )  # the EEE gap, the DDD delisting and the no-notice delisting
    assert info["notice_matched"] == 1 and info["notice_none"] == 2
    # no hindsight: a no-notice name has no forced window at all
    k = p.symbols.index(NO_NOTICE)
    assert not p.forced[:, k].any()
    assert sorted(set(info["no_notice_symbols"])) == ["EEEUSDT", NO_NOTICE]


# ----------------------------------------------------------------------------- causality


def test_panel_features_and_weights_are_causal_under_truncation() -> None:
    p, info = scenario(1)
    hourly, funding, notices = info["hourly"], info["funding"], info["notices"]
    cut_h = (
        300 * 24
    )  # before the forced-close windows of the DDD and no-notice delistings
    stage = m.Stage(
        "T",
        datetime(2021, 1, 1, tzinfo=UTC),
        datetime.fromtimestamp(T0 + cut_h * 3600, UTC),
    )
    trunc = {s: {k: v[:cut_h].copy() for k, v in h.items()} for s, h in hourly.items()}
    cut_ms = (T0 + cut_h * 3600) * 1000
    ft = {s: [e for e in ev if e[0] < cut_ms] for s, ev in funding.items()}
    q, _ = m.build_panel(trunc, stage, ft, notices, p.symbols)
    rows = q.times.size
    for col in (
        "close",
        "mark_px",
        "fill_qv",
        "regime",
        "vol",
        "fund",
        "forced",
        "age_bucket",
    ):
        assert np.array_equal(
            getattr(p, col)[:rows], getattr(q, col), equal_nan=True
        ), col
    late = [p.symbols.index("DDDUSDT"), p.symbols.index(NO_NOTICE)]
    assert p.forced[rows:, late].any() and not p.forced[:rows, late].any()
    # rewriting every price after the cut changes nothing before it
    pert = {s: {k: v.copy() for k, v in h.items()} for s, h in hourly.items()}
    rng = np.random.default_rng(11)
    for h in pert.values():
        for k in ("open", "high", "low", "close"):
            h[k][cut_h:] = rng.uniform(10, 500, len(h[k]) - cut_h)
    r_, _ = m.build_panel(pert, info["stage"], funding, notices, p.symbols)
    for col in ("close", "regime", "vol"):
        assert np.array_equal(
            getattr(p, col)[:rows], getattr(r_, col)[:rows], equal_nan=True
        ), col
    # weights and the cell equities are exact prefixes of the full-data ones
    wa, wb = m.trend_weights(p, 3), m.trend_weights(q, 3)
    assert np.array_equal(wa[:rows], wb)
    ea, _f = m.ew_weights(p, 3)
    eb, _g = m.ew_weights(q, 3)
    assert np.array_equal(ea[:rows], eb)
    for mult, delay in ((1.0, 0), (1.0, 1), (2.0, 0)):
        ca = m.simulate_weights(p, wa, mult, delay)["equity"][:rows]
        cb = m.simulate_weights(q, wb, mult, delay)["equity"][:rows]
        assert np.allclose(ca, cb, rtol=0, atol=1e-12)


# ----------------------------------------------------------------------------- bootstrap and decision


def _cell(
    returns: np.ndarray, dd: float = 0.1, valid: bool = True, total: float | None = None
) -> dict[str, Any]:
    eq = np.concatenate(([1.0], np.cumprod(1.0 + returns)))
    sd = float(np.std(returns))
    return {
        "returns": returns,
        "equity": eq,
        "max_dd": dd,
        "sharpe": float(np.mean(returns)) / sd * m.ANN if sd > 0 else 0.0,
        "total_return": float(eq[-1] - 1.0) if total is None else total,
        "valid": valid,
        "invalid_reasons": [] if valid else ["x"],
    }


def _cells(
    base_mu: float = 0.003,
    ew_mu: float = 0.0,
    hc: float = 0.05,
    dl: float = 0.05,
    dd: float = 0.2,
    valid: bool = True,
    n: int = 600,
) -> dict[str, dict[str, Any]]:
    """Deterministic return series: mean exactly ``mu`` and population std 0.01."""
    pat1 = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
    pat2 = np.tile([1.0, 1.0, -1.0, -1.0], n // 4 + 1)[:n]
    base = base_mu + 0.01 * pat1
    ew = ew_mu + 0.01 * pat2
    return {
        m.PRIMARY: _cell(base, dd),
        m.LONG_CELL: _cell(base * 0.5),
        m.HIGH_COST_CELL: _cell(base, total=hc),
        m.DELAY_CELL: _cell(base, total=dl),
        m.EW_CELL: _cell(ew, valid=valid),
        m.CASH_CELL: m.cash_cell(n - 1),
    }


def test_bootstrap_matches_reference_and_is_paired() -> None:
    rng = np.random.default_rng(0)
    a, b = rng.normal(0, 1, 200), rng.normal(0.1, 1, 200)
    out = m.block_bootstrap_means({"a": a, "b": b}, block=10, draws=300, seed=0)
    ref = np.random.default_rng(0)
    starts = ref.integers(0, 200 - 10 + 1, size=(300, 20))
    idx = (starts[:, :, None] + np.arange(10)[None, None, :]).reshape(300, -1)[:, :200]
    assert np.allclose(out["a"], a[idx].mean(axis=1)) and np.allclose(
        out["b"], b[idx].mean(axis=1)
    )
    with pytest.raises(ValueError, match="paired"):
        m.block_bootstrap_means({"a": a, "b": b[:10]})


def test_analyze_each_condition_gates_the_label() -> None:
    good = m.analyze(_cells())
    assert good["supported"] and all(good["checks"].values())
    assert good["bootstrap_lower95"] > 0 and good["base_sharpe"] >= 0.5
    low_sharpe = m.analyze(_cells(base_mu=0.00005))
    assert (
        not low_sharpe["checks"]["base_sharpe_at_least_0.5"]
        and not low_sharpe["supported"]
    )
    no_edge = m.analyze(_cells(base_mu=0.003, ew_mu=0.003))
    assert (
        not no_edge["checks"]["bootstrap_lower95_vs_ew_long_above_0"]
        and not no_edge["supported"]
    )
    assert not m.analyze(_cells(hc=-0.01))["checks"]["high_cost_total_return_positive"]
    assert not m.analyze(_cells(dl=0.0))["checks"]["delay_total_return_positive"]
    assert not m.analyze(_cells(dd=0.40))["checks"]["base_max_dd_below_40pct"]
    assert m.analyze(_cells(dd=0.3999))["checks"]["base_max_dd_below_40pct"]


def test_decide_branches() -> None:
    assert m.decide(_cells())["label"] == m.LABEL_SUPPORTED
    assert m.decide(_cells(base_mu=0.00005))["label"] == m.LABEL_NOT_SUPPORTED
    inv = m.decide(_cells(valid=False))
    assert inv["label"] == m.LABEL_INVALID and inv["invalid_cells"] == [m.EW_CELL]
    assert "analysis" not in inv
    pr = m.decide(_cells(), ["funding coverage"])
    assert pr["label"] == m.LABEL_INVALID and pr["problems"] == ["funding coverage"]
    assert m.decide({})["label"] == m.LABEL_INVALID


def test_summary_cell_is_json_safe_and_has_per_year() -> None:
    p, _ = scenario(0)
    cell = m.simulate_weights(p, m.trend_weights(p, 3), 1.0, 0)
    s = m.summarize_cell(cell, p.times)
    json.dumps(s)
    assert "returns" not in s and len(s["equity_sha256"]) == 64
    assert set(s["per_year"]) == {"2021", "2022"}
    # the per-year returns compound to the total return
    tot = (1 + s["per_year"]["2021"]["return"]) * (
        1 + s["per_year"]["2022"]["return"]
    ) - 1
    assert tot == pytest.approx(cell["total_return"], abs=1e-9)
    assert set(s["age_bucket_pnl"]) == set(m.AGE_LABELS)


def test_yearly_boundaries() -> None:
    t = np.array(
        [
            datetime(2021, 12, 31, 20, tzinfo=UTC).timestamp(),
            datetime(2022, 1, 1, tzinfo=UTC).timestamp(),
            datetime(2022, 1, 1, 4, tzinfo=UTC).timestamp(),
        ],
        dtype=np.int64,
    )
    y = m.yearly(np.array([0.1, 0.2, 0.3]), t)
    assert set(y) == {
        "2021",
        "2022",
    }  # the 4h return ending exactly at midnight belongs to 2021
    assert y["2021"]["bars"] == 2 and y["2022"]["bars"] == 1


# ----------------------------------------------------------------------------- archives


def _csv_rows(
    h: dict[str, Any], lo: int, hi: int, start_ms: int, header: bool = True
) -> str:
    lines = (
        [
            "open_time,open,high,low,close,volume,close_time,quote_volume,count,taker_buy_volume,taker_buy_quote_volume,ignore"
        ]
        if header
        else []
    )
    for i in range(lo, hi):
        if not np.isfinite(h["close"][i]):
            continue
        t = start_ms + i * m.HOUR_MS
        lines.append(
            f"{t},{h['open'][i]},{h['high'][i]},{h['low'][i]},{h['close'][i]},{h['vol'][i]},"
            f"{t + m.HOUR_MS - 1},{h['qv'][i]},10,1,1,0"
        )
    return "\n".join(lines) + "\n"


def _write_zip(path: Path, name: str, text: str, checksum: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr(name, text)
    path.write_bytes(buf.getvalue())
    if checksum:
        import hashlib

        digest = hashlib.sha256(buf.getvalue()).hexdigest()
        path.with_name(path.name + ".CHECKSUM").write_text(f"{digest}  {path.name}\n")


def month_ranges(stage: m.Stage) -> list[tuple[int, int, int, int]]:
    """(year, month, first hour index, end hour index) of every month of the stage."""
    out = []
    y, mo = stage.grid_start.year, stage.grid_start.month
    while True:
        a = datetime(y, mo, 1, tzinfo=UTC)
        if a >= stage.end:
            return out
        y2, m2 = (y + 1, 1) if mo == 12 else (y, mo + 1)
        b = min(datetime(y2, m2, 1, tzinfo=UTC), stage.end)
        s0 = int(stage.grid_start.timestamp())
        out.append(
            (
                y,
                mo,
                (int(a.timestamp()) - s0) // 3600,
                (int(b.timestamp()) - s0) // 3600,
            )
        )
        y, mo = y2, m2


def write_archives(
    root: Path,
    hourly: dict[str, dict[str, Any]],
    funding: dict[str, list[Any]],
    stage: m.Stage,
    rest_month: tuple[int, int] | None = None,
) -> None:
    start_ms = int(stage.grid_start.timestamp()) * 1000
    for sym, h in hourly.items():
        for y, mo, lo, hi in month_ranges(stage):
            text = _csv_rows(h, lo, hi, start_ms, header=(mo % 2 == 0))
            if text.count("\n") <= (1 if mo % 2 == 0 else 0):
                continue  # nothing published for a month without bars
            _write_zip(
                root
                / "futures/um/monthly/klines"
                / sym
                / "1h"
                / f"{sym}-1h-{y:04d}-{mo:02d}.zip",
                f"{sym}-1h-{y:04d}-{mo:02d}.csv",
                text,
            )
            # funding of the month
            a = datetime(y, mo, 1, tzinfo=UTC).timestamp() * 1000
            b = a + (hi - lo) * m.HOUR_MS
            evs = [e for e in funding[sym] if a <= e[0] < b]
            if rest_month == (y, mo):
                (root / "rest/fundingRate" / sym).mkdir(parents=True, exist_ok=True)

                def rest_mark(t: int, h: dict[str, Any] = h) -> str:
                    op = h["open"][(t - start_ms) // m.HOUR_MS]
                    return repr(float(op)) if np.isfinite(op) else ""

                (
                    root
                    / "rest/fundingRate"
                    / sym
                    / f"{sym}-fundingRate-{y:04d}-{mo:02d}.json"
                ).write_text(
                    json.dumps(
                        [
                            {
                                "fundingRate": repr(r),
                                "fundingTime": t,
                                "markPrice": rest_mark(t),
                                "symbol": sym,
                            }
                            for t, r, _ in evs
                        ]
                    )
                )
            elif evs:
                body = (
                    "calc_time,funding_interval_hours,last_funding_rate\n"
                    + "\n".join(f"{t},8,{r!r}" for t, r, _ in evs)
                )
                _write_zip(
                    root
                    / "futures/um/monthly/fundingRate"
                    / sym
                    / f"{sym}-fundingRate-{y:04d}-{mo:02d}.zip",
                    f"{sym}-fundingRate-{y:04d}-{mo:02d}.csv",
                    body,
                )


def test_parse_kline_csv() -> None:
    body = "1609459200000,1,2,0.5,1.5,10,1609462799999,15,3,1,1,0\n1609462800000,1,2,0.5,1.5,10,1609466399999,15,3,1,1,0\n"
    assert m.parse_kline_csv(body).shape == (2, 12)
    assert m.parse_kline_csv("open_time,x\n" + body).shape == (2, 12)
    assert m.parse_kline_csv("").shape == (0, 12)
    for bad in ("1,2,3\n", "1,2,x,4,5,6,7,8,9,10,11,12\n"):
        with pytest.raises(ValueError):
            m.parse_kline_csv(bad)


def small_universe(
    seed: int = 0,
) -> tuple[
    m.Stage, dict[str, dict[str, Any]], dict[str, list[Any]], dict[str, list[float]]
]:
    p, info = scenario(seed)
    return info["stage"], info["hourly"], info["funding"], info["notices"]


def test_read_klines_roundtrips_months_headers_and_gaps(tmp_path: Path) -> None:
    stage, hourly, funding, _ = small_universe()
    write_archives(tmp_path, hourly, funding, stage)
    sym = "EEEUSDT"
    files = m.scan_symbol(tmp_path, sym, stage)
    arrs, stats = m.read_klines(files, stage)
    h = hourly[sym]
    ok = np.isfinite(h["close"])
    assert np.array_equal(np.isfinite(arrs["close"]), ok)
    for key in ("open", "high", "low", "close", "qv", "vol"):
        assert np.array_equal(arrs[key][ok], h[key][ok]), key
    assert stats["monthly"] == len(files.monthly) and stats["malformed_months"] == 0
    # the delisted name's archive simply ends
    f2 = m.scan_symbol(tmp_path, "DDDUSDT", stage)
    a2, _ = m.read_klines(f2, stage)
    assert np.isnan(a2["close"][340 * 24 :]).all()


def test_daily_files_rebuild_a_missing_or_malformed_month(tmp_path: Path) -> None:
    stage, hourly, funding, _ = small_universe()
    write_archives(tmp_path, {"AAAUSDT": hourly["AAAUSDT"]}, funding, stage)
    sym, h = "AAAUSDT", hourly["AAAUSDT"]
    start_ms = int(stage.grid_start.timestamp()) * 1000
    base = tmp_path / "futures/um"
    # month 2021-03: monthly zip is corrupt -> rebuilt from daily zips; month 2021-04: missing
    for mo, corrupt in ((3, True), (4, False)):
        path = base / "monthly/klines" / sym / "1h" / f"{sym}-1h-2021-{mo:02d}.zip"
        if corrupt:
            path.write_bytes(b"not a zip")
            path.with_name(path.name + ".CHECKSUM").unlink()
            import hashlib

            path.with_name(path.name + ".CHECKSUM").write_text(
                hashlib.sha256(b"not a zip").hexdigest() + "  x\n"
            )
        else:
            path.unlink()
            path.with_name(path.name + ".CHECKSUM").unlink()
        lo, hi = [
            (a, b) for y, mm, a, b in month_ranges(stage) if y == 2021 and mm == mo
        ][0]
        for day in range((hi - lo) // 24):
            d0 = lo + day * 24
            _write_zip(
                base
                / "daily/klines"
                / sym
                / "1h"
                / f"{sym}-1h-2021-{mo:02d}-{day + 1:02d}.zip",
                "x.csv",
                _csv_rows(h, d0, d0 + 24, start_ms, header=False),
            )
    files = m.scan_symbol(tmp_path, sym, stage)
    arrs, stats = m.read_klines(files, stage)
    assert np.array_equal(arrs["close"], h["close"])
    assert stats["daily_months"] == 2 and stats["malformed_months"] == 1
    assert m.archive_preflight(tmp_path, (sym,), stage)["files"] > 20


def test_embargo_files_are_never_opened_and_rows_are_refused(tmp_path: Path) -> None:
    stage, hourly, funding, _ = small_universe()
    sym = "AAAUSDT"
    write_archives(tmp_path, {sym: hourly[sym]}, funding, stage)
    start_ms = int(stage.grid_start.timestamp()) * 1000
    base = tmp_path / "futures/um"
    # files dated at or after the stage end (2022-02-01) are ignored (never opened) and counted
    _write_zip(
        base / "monthly/klines" / sym / "1h" / f"{sym}-1h-2022-02.zip",
        "x.csv",
        "garbage that cannot parse",
    )
    _write_zip(
        base / "daily/klines" / sym / "1h" / f"{sym}-1h-2022-02-01.zip",
        "x.csv",
        "garbage",
    )
    files = m.scan_symbol(tmp_path, sym, stage)
    assert (
        files.embargoed == 2
        and (2022, 2) not in files.monthly
        and (2022, 2) not in files.daily
    )
    assert all("2022-02" not in p.name for p in m.used_files(files))
    m.read_klines(files, stage)  # still fine
    # a row at or after the end inside an opened file refuses the run
    h = hourly[sym]
    lo, hi = [(a, b) for y, mm, a, b in month_ranges(stage) if (y, mm) == (2022, 1)][0]
    text = _csv_rows(h, lo, hi, start_ms, header=False)
    t_bad = start_ms + stage.n_hours * m.HOUR_MS
    text += f"{t_bad},1,1,1,1,1,{t_bad + 1},1,1,1,1,0\n"
    _write_zip(
        base / "monthly/klines" / sym / "1h" / f"{sym}-1h-2022-01.zip", "x.csv", text
    )
    with pytest.raises(SystemExit, match="embargo"):
        m.read_klines(m.scan_symbol(tmp_path, sym, stage), stage)


def test_funding_vision_and_rest_months(tmp_path: Path) -> None:
    stage, hourly, funding, _ = small_universe()
    write_archives(
        tmp_path, {"AAAUSDT": hourly["AAAUSDT"]}, funding, stage, rest_month=(2021, 5)
    )
    files = m.scan_symbol(tmp_path, "AAAUSDT", stage)
    ev, stats = m.read_funding(files, stage)
    assert stats["rest_months"] == 1 and stats["vision_months"] == len(files.funding)
    assert [e[0] for e in ev] == sorted(e[0] for e in funding["AAAUSDT"])
    marks = [e for e in ev if math.isfinite(e[2])]
    start_ms = int(stage.grid_start.timestamp()) * 1000
    h = hourly["AAAUSDT"]
    assert marks
    assert all(
        e[2] == float(h["open"][(e[0] - start_ms) // m.HOUR_MS]) for e in marks
    )  # REST only
    assert all(e[0] < int(stage.end.timestamp()) * 1000 for e in ev)


def test_a_malformed_vision_funding_month_falls_back_to_the_cached_rest_month(
    tmp_path: Path,
) -> None:
    stage, hourly, funding, _ = small_universe()
    write_archives(
        tmp_path, {"AAAUSDT": hourly["AAAUSDT"]}, funding, stage, rest_month=(2021, 6)
    )
    base = tmp_path / "futures/um/monthly/fundingRate/AAAUSDT"
    files = m.scan_symbol(tmp_path, "AAAUSDT", stage)
    good, _ = m.read_funding(files, stage)
    # the Vision zip of 2021-07 is corrupt; the same month is cached as REST json instead
    victim = base / "AAAUSDT-fundingRate-2021-07.zip"
    t_lo = datetime(2021, 7, 1, tzinfo=UTC).timestamp() * 1000
    t_hi = datetime(2021, 8, 1, tzinfo=UTC).timestamp() * 1000
    month = [e for e in funding["AAAUSDT"] if t_lo <= e[0] < t_hi]
    victim.write_bytes(b"not a zip")
    broken, st = m.read_funding(m.scan_symbol(tmp_path, "AAAUSDT", stage), stage)
    assert (
        len(broken) == len(good) - len(month)
        and st["vision_months"] == len(files.funding) - 1
    )
    rest = tmp_path / "rest/fundingRate/AAAUSDT"
    (rest / "AAAUSDT-fundingRate-2021-07.json").write_text(
        json.dumps(
            [
                {"fundingRate": repr(r), "fundingTime": t, "markPrice": ""}
                for t, r, _ in month
            ]
        )
    )
    fixed, st2 = m.read_funding(m.scan_symbol(tmp_path, "AAAUSDT", stage), stage)
    assert [e[0] for e in fixed] == [e[0] for e in good] and st2["rest_months"] == 2


def test_preflight_checksum_and_presence(tmp_path: Path) -> None:
    stage, hourly, funding, _ = small_universe()
    write_archives(tmp_path, {"AAAUSDT": hourly["AAAUSDT"]}, funding, stage)
    pre = m.archive_preflight(tmp_path, ("AAAUSDT",), stage)
    assert pre["files"] > 20 and len(pre["manifest_sha256"]) == 64
    assert pre == m.archive_preflight(tmp_path, ("AAAUSDT",), stage)
    victim = next((tmp_path / "futures/um/monthly/klines/AAAUSDT/1h").glob("*.zip"))
    original = victim.read_bytes()
    victim.write_bytes(original + b"x")
    with pytest.raises(SystemExit, match="checksum mismatch"):
        m.archive_preflight(tmp_path, ("AAAUSDT",), stage)
    victim.write_bytes(original)
    victim.with_name(victim.name + ".CHECKSUM").unlink()
    with pytest.raises(SystemExit, match="missing .CHECKSUM"):
        m.archive_preflight(tmp_path, ("AAAUSDT",), stage)
    with pytest.raises(SystemExit, match="no kline archive"):
        m.archive_preflight(tmp_path, ("ZZZUSDT",), stage)


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
        relay_body if relay_body is not None else f"{m.PROTOCOL_ID} {digest} {COMMIT}"
    )

    def fetch(cid: int) -> dict[str, Any]:
        if cid == m.DATA_SCOPE_COMMENT_ID:
            return _comment("scope", **over.get("scope", {}))
        return _comment(relay_body, **over.get("relay", {}))

    return fetch


def test_authorization_ok() -> None:
    rec = m.authorization_receipt(URL, "d" * 64, COMMIT, _gh())
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
        m.authorization_receipt(url, "d" * 64, COMMIT, _gh())


def test_authorization_owner_and_issue_binding() -> None:
    with pytest.raises(SystemExit, match="not owner-authored"):
        m.authorization_receipt(URL, "d" * 64, COMMIT, _gh(relay={"login": "mallory"}))
    with pytest.raises(SystemExit, match="not owner-authored"):
        m.authorization_receipt(URL, "d" * 64, COMMIT, _gh(scope={"login": "mallory"}))
    with pytest.raises(SystemExit, match="not on Issue #810"):
        m.authorization_receipt(URL, "d" * 64, COMMIT, _gh(relay={"issue_url": "x"}))


def test_authorization_edit_between_fetches() -> None:
    calls: dict[int, int] = {}

    def fetch(cid: int) -> dict[str, Any]:
        calls[cid] = calls.get(cid, 0) + 1
        return _comment(
            f"{m.PROTOCOL_ID} {'d' * 64} {COMMIT}", updated=f"t{calls[cid]}"
        )

    with pytest.raises(SystemExit, match="changed between fetches"):
        m.authorization_receipt(URL, "d" * 64, COMMIT, fetch)


@pytest.mark.parametrize(
    "body",
    [
        "",
        "trend_breadth_v1",
        f"{'d' * 64} {COMMIT}",
        f"trend_breadth_v1 {'d' * 64}",
        f"trend_breadth_v1 {'e' * 64} {COMMIT}",
        f"trend_breadth_v1 {'d' * 64} {'b' * 40}",
    ],
)
def test_authorization_relay_must_quote_id_digest_commit(body: str) -> None:
    with pytest.raises(SystemExit, match="does not quote"):
        m.authorization_receipt(URL, "d" * 64, COMMIT, _gh(relay_body=body))


def test_authorization_requires_full_commit() -> None:
    with pytest.raises(SystemExit, match="full 40-hex"):
        m.authorization_receipt(URL, "d" * 64, "abc123", _gh())


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


# ----------------------------------------------------------------------------- ledger and end to end


def test_ledger_claim_is_exclusive(tmp_path: Path) -> None:
    now = datetime(2026, 12, 5, tzinfo=UTC)
    p = m.claim_stage("BREADTH", "d" * 64, COMMIT, 7, now, tmp_path / "led")
    assert json.loads(p.read_text())["authorization_comment_id"] == 7
    with pytest.raises(SystemExit, match="already claimed"):
        m.claim_stage("BREADTH", "d" * 64, COMMIT, 7, now, tmp_path / "led")
    m.write_ledger_result("BREADTH", "d" * 64, "x", False, "s", tmp_path / "led")
    with pytest.raises(SystemExit, match="already exists"):
        m.write_ledger_result("BREADTH", "d" * 64, "x", False, "s", tmp_path / "led")


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    monkeypatch.setattr(m, "harness_commit", lambda: COMMIT)
    digest = m.protocol_digest()
    stage, hourly, funding, notices = small_universe()
    root = tmp_path / "archives"
    write_archives(root, hourly, funding, stage, rest_month=(2021, 6))
    gh = _gh(relay_body=f"{m.PROTOCOL_ID} {digest} {COMMIT}")
    return {
        "ledger": tmp_path / "ledger",
        "tmp": tmp_path,
        "gh": gh,
        "digest": digest,
        "root": root,
        "stage": stage,
        "syms": tuple(hourly),
        "notices": notices,
    }


def _run(env: dict[str, Any], tag: str = "x", **over: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = dict(
        archive_root=env["root"],
        now=datetime(2026, 11, 24, tzinfo=UTC),
        gh_fetch=env["gh"],
        ledger_dir=env["ledger"],
        symbols=env["syms"],
        notices=env["notices"],
        n_min=3,
    )
    kwargs.update(over)
    return m.run_stage(env["stage"], URL, env["tmp"] / f"out_{tag}", **kwargs)


def _claims(env: dict[str, Any]) -> list[Path]:
    return sorted(env["ledger"].rglob("*.claim")) if env["ledger"].exists() else []


def test_end_to_end_fake_archives_write_once_and_ledger(env: dict[str, Any]) -> None:
    res = _run(env)
    assert res["label"] in {m.LABEL_SUPPORTED, m.LABEL_NOT_SUPPORTED}, (
        res["problems"] + res["invalid_cells"]
    )
    assert set(res["cells"]) == set(m.CELL_NAMES)
    assert all(c["valid"] for c in res["cells"].values())
    assert res["window"]["bars"] == env["stage"].n_hours // 4 - 1
    assert res["breadth"]["events"] == 3 and res["breadth"]["notice_matched"] == 1
    assert (
        res["data"]["preflight"]["files"] > 100
        and len(res["data"]["preflight"]["manifest_sha256"]) == 64
    )
    assert res["authorization"]["g4_relay"]["comment_id"] == 7040999999
    out = env["tmp"] / "out_x" / "result.json"
    led = json.loads(
        (
            env["ledger"] / m.PROTOCOL_ID / env["digest"] / "BREADTH.result.json"
        ).read_text()
    )
    assert led["result_sha256"] == m.sha256_file(out) and led["label"] == res["label"]
    on_disk = json.loads(out.read_text())
    assert on_disk["protocol_digest"] == env["digest"] and on_disk["commit"] == COMMIT
    assert (env["ledger"] / m.PROTOCOL_ID / "BREADTH.claim").exists()
    with pytest.raises(SystemExit, match="already claimed at the protocol root"):
        _run(env, tag="again")
    with pytest.raises(SystemExit, match="output directory exists"):
        _run(env, tag="x")


def test_end_to_end_matches_the_in_memory_pipeline(env: dict[str, Any]) -> None:
    res = _run(env)
    p, _ = scenario(0)
    cells = m.evaluate_cells(p, 3)
    for name, c in cells.items():
        assert (
            res["cells"][name]["equity_sha256"]
            == m.summarize_cell(c, p.times)["equity_sha256"]
        ), name


def _old_ns(env: dict[str, Any], digest: str = "9" * 64) -> Path:
    return env["ledger"] / m.PROTOCOL_ID / digest


def test_old_result_or_claim_blocks_a_new_digest(env: dict[str, Any]) -> None:
    ns = _old_ns(env)
    m.claim_stage("BREADTH", "9" * 64, COMMIT, 1, datetime(2026, 11, 1, tzinfo=UTC), ns)
    m.write_ledger_result("BREADTH", "9" * 64, m.LABEL_NOT_SUPPORTED, False, "s", ns)
    assert env["digest"] != "9" * 64
    with pytest.raises(SystemExit, match="single use applies across digests"):
        _run(env)
    assert not (env["ledger"] / m.PROTOCOL_ID / env["digest"]).exists()
    # a claim alone (no result: a crashed run) also blocks
    (ns / "BREADTH.result.json").unlink()
    with pytest.raises(SystemExit, match="single use applies across digests"):
        _run(env)


def test_old_result_without_claim_file_blocks(env: dict[str, Any]) -> None:
    ns = _old_ns(env)
    ns.mkdir(parents=True)
    m.write_ledger_result("BREADTH", "9" * 64, m.LABEL_INVALID, False, "s", ns)
    with pytest.raises(SystemExit, match="single use applies across digests"):
        _run(env)


def test_protocol_root_claim_blocks_a_new_digest(env: dict[str, Any]) -> None:
    root = env["ledger"] / m.PROTOCOL_ID
    m.claim_stage(
        "BREADTH", "1" * 64, COMMIT, 1, datetime(2026, 11, 1, tzinfo=UTC), root
    )
    with pytest.raises(SystemExit, match="protocol root"):
        _run(env)


def test_gates_precede_claim(env: dict[str, Any]) -> None:
    # relay does not quote the digest
    with pytest.raises(SystemExit, match="does not quote"):
        _run(env, gh_fetch=_gh(relay_body="no quote"))
    assert _claims(env) == []
    # a tampered archive refuses WITHOUT claiming
    victim = next((env["root"] / "futures/um/monthly/klines/AAAUSDT/1h").glob("*.zip"))
    original = victim.read_bytes()
    victim.write_bytes(original + b"x")
    with pytest.raises(SystemExit, match="checksum mismatch"):
        _run(env)
    assert _claims(env) == []
    victim.write_bytes(original)
    # the stage is still available afterwards
    assert _run(env, tag="ok")["label"]


def test_unclean_checkout_refuses_before_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, env: dict[str, Any]
) -> None:
    monkeypatch.undo()  # drop the harness_commit stub installed by the fixture
    monkeypatch.setattr(
        m, "_git", lambda *a: "?? tools/stray.py\n" if a[0] == "status" else COMMIT
    )
    with pytest.raises(SystemExit, match="not clean"):
        _run(env)
    assert _claims(env) == []


@pytest.mark.parametrize("which", ["relay_edit", "scope_owner", "relay_quote"])
def test_authorization_failure_leaves_no_claim(env: dict[str, Any], which: str) -> None:
    calls = {"n": 0}
    good = f"{m.PROTOCOL_ID} {env['digest']} {COMMIT}"

    def gh(cid: int) -> dict[str, Any]:
        calls["n"] += 1
        if which == "scope_owner" and cid == m.DATA_SCOPE_COMMENT_ID:
            return _comment("scope", login="someone")
        if which == "relay_quote" and cid != m.DATA_SCOPE_COMMENT_ID:
            return _comment("no quote")
        if which == "relay_edit" and cid != m.DATA_SCOPE_COMMENT_ID:
            return _comment(good, updated=f"t{calls['n']}")
        return _comment("scope") if cid == m.DATA_SCOPE_COMMENT_ID else _comment(good)

    with pytest.raises(SystemExit):
        _run(env, gh_fetch=gh)
    assert _claims(env) == []


def test_funding_gap_makes_the_stage_invalid_and_burns_the_claim(
    env: dict[str, Any],
) -> None:
    for sub in (env["root"] / "futures/um/monthly/fundingRate" / "AAAUSDT",):
        for p in list(sub.iterdir()):
            p.unlink()
    for p in (env["root"] / "rest/fundingRate" / "AAAUSDT").glob("*"):
        p.unlink()
    res = _run(env)
    assert res["label"] == m.LABEL_INVALID and "cells" in res and res["cells"] == {}
    assert any("funding coverage" in p for p in res["problems"])
    assert len(_claims(env)) == 2  # the claim is consumed


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
                "--output-dir",
                "o",
            ]
        )


# ----------------------------------------------------------------------------- review 6016979366


@pytest.mark.parametrize("gap", [1e-6, 1.0, 1000.0, 5000.0])
def test_a_flip_at_the_side_cap_cannot_break_the_cap_or_cash(gap: float) -> None:
    """The reducing part of a flip pays costs too: the clamp must bound the whole fill (the
    reviewer's probe: side notional 1.000035 x equity and cash -3.5 before the fix)."""
    p = mk_panel(np.full((3, 2), 100.0))
    book = m._Book(p, 1.0)
    E = INIT
    short = 0.05 * E
    book.u[1] = -short / 100.0  # short B, notional 5,000
    book.u[0] = (E - gap) / 100.0  # long A, notional E - gap
    book.cash = E - (E - gap) + short
    book.long_n, book.short_n = E - gap, short
    book.pending[1] = 0.05 * E / 100.0  # flip B to a 5,000 long
    book.fill_event(1, np.array([0.0, 0.05]), np.zeros(2, bool), np.zeros(2, bool))
    eq = book._equity(p.mark_px[1])
    assert not book.reasons, book.reasons
    assert book.cash >= -1e-9 * eq
    # an INCREASING fill may never push a side above the cap; a pure reduction of the other side
    # pays its costs out of equity and cannot add notional
    assert book.long_n <= max(m.SIDE_CAP * eq * (1 + 1e-9), E - gap + 1e-9)
    assert book.long_n <= E - gap + 1e-6 or book.long_n <= m.SIDE_CAP * eq * (1 + 1e-9)


def test_the_forced_extra_is_inside_the_clamp_bound() -> None:
    p = mk_panel(np.full((3, 2), 100.0))
    book = m._Book(p, 1.0)
    E = INIT
    book.u[1] = -5000.0 / 100.0
    book.u[0] = (E - 10.0) / 100.0
    book.cash = E - (E - 10.0) + 5000.0
    book.long_n, book.short_n = E - 10.0, 5000.0
    book.pending[1] = 50.0
    book.pend_forced[1] = True
    book.fill_event(1, np.array([0.0, 0.05]), np.zeros(2, bool), np.ones(2, bool))
    eq = book._equity(p.mark_px[1])
    assert not book.reasons and book.cash >= -1e-9 * eq
    assert book.long_n <= max(m.SIDE_CAP * eq * (1 + 1e-9), E - 10.0 + 1e-9)


def _causal_equal(a: m.Panel, b: m.Panel, rows: int, cols: tuple[str, ...]) -> None:
    for col in cols:
        got, want = getattr(a, col)[:rows], getattr(b, col)[:rows]
        assert np.array_equal(got, want, equal_nan=True), col


def _truncated(
    info: dict[str, Any],
    cut_day: int,
    symbols: tuple[str, ...],
    notices: dict[str, list[float]] | None = None,
) -> m.Panel:
    cut_h = cut_day * 24
    end = datetime.fromtimestamp(T0 + cut_h * 3600, UTC)
    stage = m.Stage("T", datetime(2021, 1, 1, tzinfo=UTC), end)
    trunc = {
        s: {k: v[:cut_h].copy() for k, v in h.items()}
        for s, h in info["hourly"].items()
    }
    cut_ms = (T0 + cut_h * 3600) * 1000
    ft = {s: [e for e in ev if e[0] < cut_ms] for s, ev in info["funding"].items()}
    q, _ = m.build_panel(
        trunc, stage, ft, info["notices"] if notices is None else notices, symbols
    )
    return q


def test_cut_inside_a_no_notice_delisting_window_changes_nothing_before_it() -> None:
    """The removed 42-bar hindsight fallback would have flagged rows 7 days before the
    no-notice symbol's last bar (day 330); a cut at day 325 sits inside that old window."""
    _, info = scenario(1)
    # no notice for anybody: every delisting is write-off only
    p, _ = m.build_panel(
        info["hourly"], info["stage"], info["funding"], {}, tuple(info["hourly"])
    )
    k = p.symbols.index(NO_NOTICE)
    q = _truncated(info, 325, p.symbols, {})
    rows = q.times.size
    assert rows < max(j for j, kk in p.events if kk == k)  # the cut is before the event
    assert rows > max(j for j, kk in p.events if kk == k) - 42  # inside the old window
    cols = (
        "close",
        "mark_px",
        "fill_qv",
        "regime",
        "vol",
        "fund",
        "forced",
        "age_bucket",
    )
    _causal_equal(p, q, rows, cols)
    assert not p.forced[:, k].any()
    wa, wb = m.trend_weights(p, 3), m.trend_weights(q, 3)
    assert np.array_equal(wa[:rows], wb)
    for mult, delay in ((1.0, 0), (1.0, 1)):
        ca = m.simulate_weights(p, wa, mult, delay)["equity"][:rows]
        cb = m.simulate_weights(q, wb, mult, delay)["equity"][:rows]
        assert np.allclose(ca, cb, rtol=0, atol=1e-12)


def test_cut_inside_a_notice_window_changes_nothing_before_the_release() -> None:
    p, info = scenario(1)
    d = p.symbols.index("DDDUSDT")
    release_row = int(np.nonzero(p.forced[:, d])[0][0])
    q = _truncated(info, 320, p.symbols)  # inside the notice-to-event window of DDD
    rows = q.times.size
    assert release_row < rows
    cols = ("close", "mark_px", "fill_qv", "regime", "vol", "fund", "age_bucket")
    _causal_equal(p, q, rows, cols)
    # the notice pairing is the one disclosed exception: forced flags may differ only from the
    # release row of the paired symbol onward
    assert np.array_equal(p.forced[:release_row], q.forced[:release_row])
    wa, wb = m.trend_weights(p, 3), m.trend_weights(q, 3)
    assert np.array_equal(wa[:release_row], wb[:release_row])


def hashlib_sha(b: bytes) -> str:
    import hashlib

    return hashlib.sha256(b).hexdigest()


def test_preflight_refuses_a_malformed_zip_that_cannot_be_rebuilt(
    tmp_path: Path,
) -> None:
    stage, hourly, funding, _ = small_universe()
    write_archives(tmp_path, {"AAAUSDT": hourly["AAAUSDT"]}, funding, stage)
    for sub, pat in (
        ("futures/um/monthly/klines/AAAUSDT/1h", "*-2021-03.zip"),
        ("futures/um/monthly/fundingRate/AAAUSDT", "*-2021-07.zip"),
    ):
        victim = next((tmp_path / sub).glob(pat))
        keep = victim.read_bytes()
        side = victim.with_name(victim.name + ".CHECKSUM")
        victim.write_bytes(b"not a zip")
        side.write_text(hashlib_sha(b"not a zip") + "  x\n")
        with pytest.raises(SystemExit, match="no replacement"):
            m.archive_preflight(tmp_path, ("AAAUSDT",), stage)
        victim.write_bytes(keep)
        side.write_text(hashlib_sha(keep) + "  x\n")
    assert m.archive_preflight(tmp_path, ("AAAUSDT",), stage)["files"] > 20


def test_a_readable_zip_with_a_malformed_csv_is_a_data_problem(
    env: dict[str, Any],
) -> None:
    sub = env["root"] / "futures/um/monthly/klines/AAAUSDT/1h"
    victim = next(sub.glob("*2021-03.zip"))
    _write_zip(
        victim, "x.csv", "1,2,3\n"
    )  # a valid zip, a malformed csv, no daily files
    res = _run(env)
    assert res["label"] == m.LABEL_INVALID
    assert any("malformed archive" in q for q in res["problems"])


def test_notices_with_overrides_are_refused(tmp_path: Path) -> None:
    doc = json.loads(m.NOTICES_PATH.read_text(encoding="utf-8"))
    assert not doc.get("overrides")  # the pinned file has none
    doc["overrides"] = {"X": 1}
    bad = tmp_path / "n.json"
    bad.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(SystemExit, match="overrides"):
        m.load_notices(bad, sha=m.sha256_file(bad))


def test_blanked_symbol_days_are_reported() -> None:
    n = 24 * 400
    stage = m.Stage(
        "T", datetime(2021, 1, 1, tzinfo=UTC), datetime(2022, 2, 5, tzinfo=UTC)
    )
    clean = synth_hourly(n, 21)
    holed = synth_hourly(n, 22)
    for k in ("open", "high", "low", "close", "qv", "vol"):
        holed[k][24 * 150 + 5] = np.nan  # one missing hour at day 150
    panel, info = m.build_panel({"AAAUSDT": clean, "BBBUSDT": holed}, stage, {}, None)
    rep = m.breadth_report(panel, info, 1)
    assert 140 <= rep["blanked_symbol_days"] <= 160
    only, only_info = m.build_panel({"AAAUSDT": clean}, stage, {}, None)
    assert m.breadth_report(only, only_info, 1)["blanked_symbol_days"] == 0
    assert rep["symbols_no_notice"] == 0
