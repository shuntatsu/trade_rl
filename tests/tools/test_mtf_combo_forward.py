"""Synthetic-data tests for tools/mtf_combo_forward.py (no network, no market data).

Needs only numpy. Two tests use optional packages and skip without them: the pandas feature
parity oracle (``pandas``) and the real-model determinism test (``scikit-learn``)."""

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

from tools import mtf_combo_forward as m

DIGEST_PIN = "2d5d0872dfa8e86498d0a56ee7644ccdb81538d70e35e37cdcb3f4d871537c37"
COMMIT = "a" * 40
URL = "https://github.com/shuntatsu/trade_rl/issues/810#issuecomment-7040999999"
T0 = int(datetime(2026, 12, 1, tzinfo=UTC).timestamp())


# ----------------------------------------------------------------------------- digest


def test_digest_pin() -> None:
    assert m.protocol_digest() == DIGEST_PIN
    assert re.fullmatch(r"[0-9a-f]{64}", m.protocol_digest())


def test_constants_frozen() -> None:
    assert m.PROTOCOL_ID == "mtf_combo_v1"
    assert m.SOURCE_CLOSE_POLICY.value == "strict_source_close_v2"
    assert m.SYMBOLS == tuple(
        "BTCUSDT ETHUSDT SOLUSDT BNBUSDT XRPUSDT ADAUSDT DOGEUSDT LINKUSDT "
        "DOTUSDT AVAXUSDT LTCUSDT".split()
    )
    assert len(m.SYMBOLS) == 11 and m.SYMBOLS[0] == "BTCUSDT"
    assert m.REGIME_WINDOWS == (20, 50, 100, 200) and m.REGIME_THRESHOLD == 0.5
    assert (m.LABEL_HORIZON, m.BAND, m.SIZE_VOL, m.SLEEVE_WEIGHT) == (30, 0.4, 0.5, 0.5)
    assert (m.TARGET_VOL, m.LEVERAGE_CAP, m.VOL_BARS, m.LEVERAGE_BAND) == (
        0.20,
        2.0,
        180,
        0.20,
    )
    assert m.HGB_PARAMS == {
        "max_depth": 3,
        "max_iter": 300,
        "learning_rate": 0.03,
        "l2_regularization": 2.0,
        "min_samples_leaf": 300,
        "random_state": 0,
    }
    assert (m.FEE_BASE, m.FEE_HIGH) == (0.0006, 0.0012)
    assert (m.EARLY_STOP_DD, m.MAX_DD, m.MIN_SHARPE) == (0.25, 0.25, 0.5)
    assert m.DATA_SCOPE_COMMENT_ID == 7040247247 and m.OWNER_LOGIN == "shuntatsu"
    assert str(m.LEDGER_DIR) == r"C:\trade_rl_runs\mtf_combo_forward_ledger"
    assert m.CELLS == {
        "COMBO__BASE": ("combo", 0.0006, 0, True),
        "COMBO__HIGH_COST": ("combo", 0.0012, 0, True),
        "COMBO__DELAY": ("combo", 0.0006, 1, True),
        "BTC_RULE__BASE": ("btc_rule", 0.0006, 0, False),
        "ALT_AGREE__BASE": ("alt_agree", 0.0006, 0, False),
        "HOLD_BTC": ("hold", 0.0006, 0, False),
        "CASH": ("cash", 0.0, 0, False),
    }
    fi, ff = m.STAGES["FORWARD_INTERIM"], m.STAGES["FORWARD_FINAL"]
    for st in (fi, ff):
        assert st.history_start == datetime(2021, 1, 1, tzinfo=UTC)
        assert st.decision_start == datetime(2026, 12, 1, tzinfo=UTC)
        assert st.decision_end == datetime(2027, 12, 1, tzinfo=UTC)
    assert ff.not_before == datetime(2027, 12, 2, tzinfo=UTC) and ff.final
    assert fi.interim and fi.not_after == datetime(2027, 12, 2, tzinfo=UTC)
    assert len(m.FEATURE_COLUMNS) == 42 and len(set(m.FEATURE_COLUMNS)) == 42
    assert m.FEATURE_COLUMNS[-3:] == ("regime", "dsma200_1d", "vol_ann")


# ----------------------------------------------------------------------------- primitives


def test_rolling_primitives_propagate_nan() -> None:
    x = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0])
    assert np.isnan(m.r_max(x, 3)[:5]).all() and m.r_max(x, 3)[5] == 6.0
    assert m.r_mean(x, 2)[1] == 1.5 and np.isnan(m.r_mean(x, 2)[2])
    assert m.r_std(np.array([1.0, 3.0]), 2)[1] == pytest.approx(
        math.sqrt(2.0)
    )  # ddof 1
    assert np.array_equal(m.shift(x, 2)[2:4], [1.0, 2.0])


def test_ewm_matches_hand_values_and_pandas_nan_rule() -> None:
    # adjust=False: y0 = x0; y_t = (1-a) y_{t-1} + a x_t
    a = 0.5
    out = m.ewm_mean(np.array([1.0, 2.0, 4.0]), a, 1)
    assert out == pytest.approx([1.0, 1.5, 2.75])
    # a NaN decays the old weight (pandas ignore_na=False): y2 = (0.25*1.5 + 0.5*4)/(0.25+0.5)
    gap = m.ewm_mean(np.array([1.0, 2.0, np.nan, 4.0]), a, 1)
    assert gap[2] == pytest.approx(1.5)
    assert gap[3] == pytest.approx((0.25 * 1.5 + 0.5 * 4.0) / 0.75)
    # min_periods counts observations; leading NaNs are skipped
    lead = m.ewm_mean(np.array([np.nan, 1.0, 2.0, 3.0]), a, 3)
    assert np.isnan(lead[:3]).all() and np.isfinite(lead[3])


# ----------------------------------------------------------------------------- features


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


def test_features_shape_and_warmup() -> None:
    fr = m.build_frame(synth_hourly(24 * 300, 1), START_S)
    assert fr.x.shape == (24 * 300 // 4, len(m.FEATURE_COLUMNS))
    assert fr.open_s[1] - fr.open_s[0] == m.BAR_S and fr.open_s[0] == START_S
    reg = fr.x[:, m.I_REGIME]
    # regime needs 200 completed daily bars: first known at the close of day 200 (bar 6*200 - 1)
    assert np.isnan(reg[: 6 * 200 - 1]).all() and np.isfinite(reg[6 * 200 :]).all()
    fin = reg[~np.isnan(reg)]
    assert np.allclose(fin * 4, np.round(fin * 4)) and np.abs(fin).max() <= 1.0


@pytest.mark.parametrize("cut_day", [210.0, 240.25, 280.5])
def test_features_are_causal_under_truncation(cut_day: float) -> None:
    """A truncation test: features at every decision close <= the cut are identical whether or
    not the later bars exist, and unchanged when the later bars are rewritten."""
    full = synth_hourly(24 * 300, 2, holes=(5000, 5001))
    cut_h = int(cut_day * 24) // 4 * 4
    trunc = {k: v[:cut_h].copy() for k, v in full.items()}
    a = m.build_frame(full, START_S)
    b = m.build_frame(trunc, START_S)
    k = cut_h // 4
    assert np.array_equal(a.x[:k], b.x, equal_nan=True)
    assert np.array_equal(a.close[:k], b.close, equal_nan=True)
    future = {key: v.copy() for key, v in full.items()}
    rng = np.random.default_rng(9)
    for key in future:
        future[key][cut_h:] = rng.uniform(1.0, 500.0, len(future[key]) - cut_h)
    c = m.build_frame(future, START_S)
    assert np.array_equal(a.x[:k], c.x[:k], equal_nan=True)


def test_features_use_latest_completed_daily_bar() -> None:
    arrs = synth_hourly(24 * 260, 3)
    fr = m.build_frame(arrs, START_S)
    base = fr.x.copy()
    # rewrite the last hour of day 230: only decision closes at or after its end may change
    # (the 4h bar containing that hour closes exactly at the day boundary)
    mod = {k: v.copy() for k, v in arrs.items()}
    day = 230
    for key in mod:
        mod[key][day * 24 + 23] *= 1.5
    fr2 = m.build_frame(mod, START_S)
    first_changed_close_bar = (
        day * 6 + 5
    )  # bar index whose 4h close is the day boundary
    col = m.FEATURE_COLUMNS.index("dsma20_1d")
    unchanged = slice(0, first_changed_close_bar)
    assert np.array_equal(base[unchanged, col], fr2.x[unchanged, col], equal_nan=True)
    assert base[first_changed_close_bar, col] != fr2.x[first_changed_close_bar, col]


# G2 oracle for the feature bank: the exploratory screens' pandas code, copied verbatim, must give
# the same numbers as the pandas-free port (NaN pattern equal, values to 1e-8).
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
    ).dropna(how="all")

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

    def ichimoku(b: Any) -> Any:
        h, lo, c = b.high, b.low, b.close
        tenkan = (h.rolling(9).max() + lo.rolling(9).min()) / 2
        kijun = (h.rolling(26).max() + lo.rolling(26).min()) / 2
        span_a = (tenkan + kijun) / 2
        span_b = (h.rolling(52).max() + lo.rolling(52).min()) / 2
        ca, cb = span_a.shift(26), span_b.shift(26)
        top, bot = np.maximum(ca, cb), np.minimum(ca, cb)
        pos = pd.Series(
            np.where(c > top, 1.0, np.where(c < bot, -1.0, 0.0)), index=c.index
        ).where(top.notna() & c.notna())
        tk = pd.Series(np.where(tenkan > kijun, 1.0, -1.0), index=c.index).where(
            tenkan.notna() & kijun.notna()
        )
        cs = c.shift(26)
        ck = pd.Series(np.where(c > cs, 1.0, -1.0), index=c.index).where(
            cs.notna() & c.notna()
        )
        return pos + tk + ck

    def wilder(s: Any, n: int) -> Any:
        return s.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()

    def indicators(b: Any) -> Any:
        h, lo, c, qv = b.high, b.low, b.close, b.qv
        d = c.diff()
        rs = wilder(d.clip(lower=0), 14) / wilder((-d).clip(lower=0), 14)
        ind = {"rsi": 100 - 100 / (1 + rs)}
        macd = (
            c.ewm(span=12, adjust=False, min_periods=26).mean()
            - c.ewm(span=26, adjust=False, min_periods=26).mean()
        )
        ind["macdh"] = macd - macd.ewm(span=9, adjust=False, min_periods=9).mean()
        ll, hh = lo.rolling(14).min(), h.rolling(14).max()
        ind["stoch"] = (100 * (c - ll) / (hh - ll)).rolling(3).mean()
        tp = (h + lo + c) / 3
        ind["cci"] = (tp - tp.rolling(20).mean()) / (
            0.015
            * tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
        )
        pos = qv.where(tp.diff() > 0, 0.0).where(tp.diff().notna())
        neg = qv.where(tp.diff() < 0, 0.0).where(tp.diff().notna())
        ind["mfi"] = (
            100
            * pos.rolling(14).sum()
            / (pos.rolling(14).sum() + neg.rolling(14).sum())
        )
        ind["obv"] = (np.sign(d).fillna(0) * qv.fillna(0)).cumsum().where(c.notna())
        ind["roc"] = c / c.shift(10) - 1
        ind["willr"] = -100 * (hh - c) / (hh - ll)
        ind["tbr"] = (b.tbq / qv).ewm(span=14, adjust=False, min_periods=14).mean()
        return pd.DataFrame(ind)

    def pivots(x: Any, left_w: int, right_w: int, kind: str) -> Any:
        s = pd.Series(x)
        if kind == "low":
            left = s.shift(1).rolling(left_w).min()
            right = s[::-1].shift(1).rolling(right_w).min()[::-1]
            ok = (s < left) & (s <= right)
        else:
            left = s.shift(1).rolling(left_w).max()
            right = s[::-1].shift(1).rolling(right_w).max()[::-1]
            ok = (s > left) & (s >= right)
        return np.flatnonzero(ok.values)

    def divergence(
        b: Any, left_w: int = 5, right_w: int = 3, maxgap: int = 60, life: int = 6
    ) -> Any:
        ind = indicators(b)
        lo, hi = b.low.values, b.high.values
        n = len(b)
        pl, ph = pivots(lo, left_w, right_w, "low"), pivots(hi, left_w, right_w, "high")
        bull, bear = np.zeros(n), np.zeros(n)
        for name in ind.columns:
            v = ind[name].values
            for kind, piv, price in (("bull", pl, lo), ("bear", ph, hi)):
                if len(piv) < 2:
                    continue
                p1, p2 = piv[:-1], piv[1:]
                gap = (p2 - p1) <= maxgap
                if kind == "bull":
                    cond = gap & (price[p2] < price[p1]) & (v[p2] > v[p1])
                else:
                    cond = gap & (price[p2] > price[p1]) & (v[p2] < v[p1])
                ev = np.zeros(n)
                conf = (p2 + right_w)[cond]
                conf = conf[conf < n]
                ev[conf] = 1
                act = pd.Series(ev).rolling(life, min_periods=1).max().values
                (bull if kind == "bull" else bear)[:] += act
        valid = b.close.notna().values
        return (
            pd.Series(np.where(valid, bull, np.nan), index=b.index),
            pd.Series(np.where(valid, bear, np.nan), index=b.index),
        )

    def ppo(c: Any, f: int = 12, s: int = 26) -> Any:
        return (
            c.ewm(span=f, adjust=False, min_periods=s).mean()
            / c.ewm(span=s, adjust=False, min_periods=s).mean()
            - 1
        ) * 100

    def tf_feats(b: Any, tag: str) -> Any:
        ind = indicators(b)
        c = b.close
        out = pd.DataFrame(index=b.index)
        out[f"rsi_{tag}"] = ind.rsi
        out[f"macdh_{tag}"] = ind.macdh / c
        out[f"ppo_{tag}"] = ppo(c)
        out[f"stoch_{tag}"] = ind.stoch
        out[f"ich_{tag}"] = ichimoku(b)
        bu, be = divergence(b)
        out[f"bull_{tag}"], out[f"bear_{tag}"] = bu, be
        out[f"net_{tag}"] = bu - be
        for n in (20, 50):
            out[f"dsma{n}_{tag}"] = np.log(c / c.rolling(n).mean())
        out[f"r1_{tag}"] = np.log(c / c.shift(1))
        out[f"r6_{tag}"] = np.log(c / c.shift(6))
        out[f"vol_{tag}"] = np.log(c).diff().rolling(30).std()
        return out

    h1 = d1h.reindex(grid).rename(
        columns={"quote_volume": "qv", "taker_buy_quote_volume": "tbq"}
    )
    b4, b1d = agg(d1h.reindex(grid), "4h", 4), agg(d1h.reindex(grid), "1D", 24)
    c4 = b4.index + pd.Timedelta("4h")
    frame = pd.DataFrame(index=b4.index)
    frame = frame.join(tf_feats(b4, "4h"))
    f1 = tf_feats(h1, "1h")
    f1.index = f1.index + pd.Timedelta("1h")
    frame = frame.join(
        pd.DataFrame(f1.reindex(c4).values, index=frame.index, columns=f1.columns)
    )
    fd = tf_feats(b1d, "1d")
    cd = b1d.close
    fd["regime"] = (
        sum(np.sign(cd - cd.rolling(n).mean()) for n in (20, 50, 100, 200)) / 4
    )
    fd["dsma200_1d"] = np.log(cd / cd.rolling(200).mean())
    fd["vol_ann"] = np.log(cd).diff().rolling(30).std() * np.sqrt(365)
    fd.index = fd.index + pd.Timedelta("1D")
    frame = frame.join(
        pd.DataFrame(
            fd.reindex(c4, method="ffill").values, index=frame.index, columns=fd.columns
        )
    )
    return frame, b4.close.to_numpy()


@pytest.mark.parametrize("holes", [(), (3000, 3001, 5000)])
def test_feature_bank_matches_pandas_reference(holes: tuple[int, ...]) -> None:
    pytest.importorskip("pandas")
    arrs = synth_hourly(24 * 330, 1, holes)
    ref, ref_close = _pandas_reference(arrs, "2025-01-01")
    fr = m.build_frame(arrs, START_S)
    assert tuple(ref.columns) == m.FEATURE_COLUMNS
    expect = ref.to_numpy(np.float64)
    assert expect.shape == fr.x.shape
    assert np.array_equal(np.isnan(expect), np.isnan(fr.x))
    ok = ~np.isnan(expect)
    assert np.allclose(expect[ok], fr.x[ok], rtol=1e-8, atol=1e-8)
    assert np.allclose(ref_close, fr.close, equal_nan=True)


# ----------------------------------------------------------------------------- model & training


def _frame(n: int, close: np.ndarray | None = None) -> m.Frame:
    c = close if close is not None else 100 + np.arange(n, dtype=float)
    x = np.full((n, len(m.FEATURE_COLUMNS)), 0.5)
    x[:, 0] = np.arange(n)
    return m.Frame(T0 - 4000 * m.BAR_S + m.BAR_S * np.arange(n), c, x)


def test_training_set_cutoff_is_exact() -> None:
    n = 4100
    cutoff = T0
    fr = _frame(n)
    x, y = m.training_set({"A": fr}, ("A",), cutoff)
    # row t is used iff the bar t+30 that closes its label window OPENS before the cutoff
    used = int(np.sum(fr.open_s + m.LABEL_HORIZON * m.BAR_S < cutoff))
    assert len(y) == used == len(x)
    assert x[-1, 0] == used - 1
    # a rising series: every label is positive
    assert set(y.tolist()) == {1}
    # mutating any price at or after the first excluded label bar cannot change the set
    c2 = fr.close.copy()
    first_bad = int(np.flatnonzero(fr.open_s >= cutoff)[0])
    c2[first_bad:] = 1.0
    x2, y2 = m.training_set({"A": _frame(n, c2)}, ("A",), cutoff)
    assert np.array_equal(y, y2) and np.array_equal(x, x2)


def test_training_label_is_strictly_positive_forward_return() -> None:
    n = 300
    c = np.full(n, 100.0)
    c[30:] = 100.0  # equal -> not > 0 -> label 0
    c[40] = 101.0  # bar 10's forward return is +1%
    fr = _frame(n, c)
    _, y = m.training_set({"A": fr}, ("A",), T0 + 10**9)
    lab = {i: v for i, v in enumerate(y.tolist())}
    assert lab[10] == 1 and lab[11] == 0 and lab[0] == 0


def test_training_requires_finite_regime_and_close() -> None:
    n = 200
    fr = _frame(n)
    fr.x[:50, m.I_REGIME] = np.nan
    fr.close[60] = np.nan
    x, y = m.training_set({"A": fr}, ("A",), T0 + 10**9)
    ok = np.arange(n - m.LABEL_HORIZON)
    ok = ok[(ok >= 50) & (ok != 60) & (ok + m.LABEL_HORIZON != 60)]
    assert len(y) == len(ok)


class FakeModel:
    """A deterministic stand-in for the gradient-boosting model."""

    def __init__(self, jitter: float = 0.0) -> None:
        self.jitter = jitter
        self.n_iter_ = 1

    def fit(self, x: np.ndarray, y: np.ndarray) -> FakeModel:
        self.shift = float(np.nanmedian(x[:, 8]))
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        z = np.nan_to_num(x[:, 8] - self.shift) * 20.0
        p = 1.0 / (1.0 + np.exp(-z)) + self.jitter
        return np.column_stack([1.0 - p, p])


def test_fit_direction_model_bands_and_receipt() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(500, len(m.FEATURE_COLUMNS)))
    y = (x[:, 8] > 0).astype(int)
    model, lo, hi, rec = m.fit_direction_model(x, y, FakeModel)
    p = model.predict_proba(x)[:, 1]
    assert lo == pytest.approx(np.quantile(p, 0.4)) and hi == pytest.approx(
        np.quantile(p, 0.6)
    )
    assert lo < hi and rec["deterministic"] is True and rec["train_rows"] == 500
    assert len(rec["model_prob_sha256"]) == 64
    _, _, _, rec2 = m.fit_direction_model(x, y, FakeModel)
    assert rec2 == rec  # same data, same receipt
    drifting = iter([0.0, 1e-9])
    _, _, _, bad = m.fit_direction_model(x, y, lambda: FakeModel(next(drifting)))
    assert bad["deterministic"] is False


def test_real_model_is_deterministic_and_frozen_params() -> None:
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(1)
    x = rng.normal(size=(1200, len(m.FEATURE_COLUMNS)))
    x[rng.random(x.shape) < 0.05] = np.nan
    y = (x[:, 3] + 0.5 * rng.normal(size=1200) > 0).astype(int)
    model, lo, hi, rec = m.fit_direction_model(x, y, m.default_model_factory)
    assert rec["deterministic"] is True and lo < hi
    assert (
        model.get_params()["max_depth"] == 3
        and model.get_params()["min_samples_leaf"] == 300
    )
    _, _, _, rec2 = m.fit_direction_model(x, y, m.default_model_factory)
    assert rec2["model_prob_sha256"] == rec["model_prob_sha256"]


def test_regime_direction_and_bands() -> None:
    d = m.regime_direction(np.array([1.0, 0.5, 0.25, 0.0, -0.25, -0.5, -1.0, np.nan]))
    assert d.tolist() == [1, 1, 0, 0, 0, -1, -1, 0]
    b = m.ml_band(np.array([0.7, 0.6, 0.5, 0.4, 0.3, np.nan]), 0.4, 0.6)
    assert b.tolist() == [1, 0, 0, 0, -1, 0]  # bands are strict


# ----------------------------------------------------------------------------- sleeves


def _arr(*v: float) -> np.ndarray:
    return np.asarray(v, dtype=float)


def test_sleeve_a_long_only_and_size_fixed_at_entry() -> None:
    rule = _arr(0, 1, 1, 1, -1, 0, 1)
    vol = _arr(
        1.0, 1.0, 0.25, 4.0, 1.0, 1.0, 2.0
    )  # sizes 0.5, 0.5, 1, 0.125, .5, .5, .25
    w = m.sleeve_a_weights(rule, vol)
    # enters at t=1 with size 0.5 and keeps it (no re-sizing) while the direction holds;
    # a -1 or 0 regime exits (long only); re-entry at t=6 uses the then-current size 0.25
    assert w.tolist() == [0.0, 0.5, 0.5, 0.5, 0.0, 0.0, 0.25]


def test_sleeve_a_skips_bars_without_volatility() -> None:
    w = m.sleeve_a_weights(_arr(1, 1, 1), _arr(np.nan, 1.0, np.nan))
    assert w.tolist() == [0.0, 0.5, 0.5]  # a NaN-vol bar holds, it never decides


def test_sleeve_b_agree_entry_hold_exit_and_btc_filter() -> None:
    rule = _arr(1, 1, 1, 1, 1, 0, 1, 1)
    mlb = _arr(0, 1, 1, 0, -1, 1, 1, 1)
    vol = np.ones(8)
    btc = np.zeros(8)
    w = m.sleeve_b_weights(rule, mlb, vol, btc, True)
    # t0 mlb 0: no entry; t1 agree: enter 0.5; t2 hold; t3 mlb neutral still holds;
    # t4 mlb flips against: exit; t5 rule 0: flat; t6 re-enter; t7 hold
    assert w.tolist() == [0.0, 0.5, 0.5, 0.5, 0.0, 0.0, 0.5, 0.5]
    # a rule flip exits even when mlb agrees with the old side
    w2 = m.sleeve_b_weights(
        _arr(1, 1, -1), _arr(1, 1, 1), np.ones(3), np.zeros(3), True
    )
    assert w2.tolist() == [0.5, 0.5, 0.0]
    # an opposite agreement flips only through flat: exit bar, then re-entry on the next bar
    w3 = m.sleeve_b_weights(
        _arr(1, -1, -1), _arr(1, -1, -1), np.ones(3), np.zeros(3), True
    )
    assert w3.tolist() == [0.5, 0.0, -0.5]


def test_sleeve_b_alt_short_needs_btc_regime_down() -> None:
    rule, mlb, vol = _arr(-1, -1, -1, -1), _arr(-1, -1, -1, -1), np.ones(4)
    btc = _arr(0, -1, -1, 0)
    alt = m.sleeve_b_weights(rule, mlb, vol, btc, False)
    assert alt.tolist() == [0.0, -0.5, -0.5, 0.0]  # blocked, enter, hold, forced out
    own = m.sleeve_b_weights(rule, mlb, vol, btc, True)  # BTC itself is exempt
    assert own.tolist() == [-0.5, -0.5, -0.5, -0.5]
    # longs are never filtered
    lng = m.sleeve_b_weights(_arr(1, 1), _arr(1, 1), np.ones(2), _arr(0, 0), False)
    assert lng.tolist() == [0.5, 0.5]


def test_size_rule() -> None:
    assert m._size(0.25) == 1.0 and m._size(0.5) == 1.0 and m._size(1.0) == 0.5
    assert m._size(0.0) == 1.0


# ----------------------------------------------------------------------------- the book


def make_sig(
    close: dict[str, list[float]],
    rule: dict[str, Any] | None = None,
    mlb: dict[str, Any] | None = None,
    vol: dict[str, Any] | None = None,
    fund: dict[str, Any] | None = None,
) -> m.Signals:
    syms = tuple(close)
    n1 = len(next(iter(close.values())))
    z = np.zeros(n1)
    return m.Signals(
        T0 + m.BAR_S * np.arange(n1),
        syms,
        {s: np.asarray(v, float) for s, v in close.items()},
        {s: np.asarray((rule or {}).get(s, z), np.int8) for s in syms},
        {s: np.asarray((mlb or {}).get(s, z), np.int8) for s in syms},
        {s: np.asarray((vol or {}).get(s, np.ones(n1)), float) for s in syms},
        {s: np.asarray((fund or {}).get(s, z), float) for s in syms},
    )


def test_hold_cell_keeps_units_no_free_rebalancing() -> None:
    # price 100 -> 200 -> 100 on a held 1x long: exactly flat, not a product of re-weighted returns
    sig = make_sig({"BTCUSDT": [100.0, 200.0, 100.0]})
    cell = m.simulate(sig, "hold", 0.0, 0)
    assert cell["equity"].tolist() == pytest.approx([1.0, 1.0, 2.0, 1.0])
    assert cell["total_return"] == pytest.approx(0.0) and cell["valid"]
    # half-weight held from entry: equity = 1 + 0.5 * (P/100 - 1) exactly
    sig2 = make_sig(
        {"BTCUSDT": [100.0, 200.0, 100.0]},
        rule={"BTCUSDT": [1, 1, 1]},
        vol={"BTCUSDT": [1.0] * 3},
    )
    cell2 = m.simulate(
        sig2, "btc_rule", 0.0, 0
    )  # sleeve A alone: size 0.5, L = 1 (< 180 bars)
    assert cell2["equity"].tolist() == pytest.approx([1.0, 1.0, 1.5, 1.0])


def test_fee_is_on_actual_fill_notional_and_entry_is_sized_pre_fee() -> None:
    sig = make_sig({"BTCUSDT": [100.0, 110.0, 110.0]})
    cell = m.simulate(sig, "hold", 0.001, 0)
    # entry: 1.0 of notional bought at 100 costs 0.001; no later trade
    assert cell["fees"] == pytest.approx(0.001) and cell["fills"] == 1
    assert cell["equity"].tolist() == pytest.approx(
        [1.0, 0.999, 0.999 + 0.01 * 10, 0.999 + 0.1]
    )
    assert cell["turnover"] == pytest.approx(1.0)


def test_funding_is_paid_on_held_notional() -> None:
    # 0.01 BTC held; funding stamped in bar 1 of mark*rate = 100 * 0.001 -> pays 0.01 * 0.1
    sig = make_sig(
        {"BTCUSDT": [100.0, 100.0, 100.0]}, fund={"BTCUSDT": [0.0, 0.1, -0.2]}
    )
    cell = m.simulate(sig, "hold", 0.0, 0)
    assert cell["funding"] == pytest.approx(-0.01 * 0.1 + 0.01 * 0.2)
    assert cell["equity"].tolist() == pytest.approx([1.0, 1.0, 0.999, 1.001])


def test_delay_fills_one_bar_later_at_that_close() -> None:
    sig = make_sig({"BTCUSDT": [100.0, 100.0, 120.0, 120.0]})
    base = m.simulate(sig, "hold", 0.0, 0)
    late = m.simulate(sig, "hold", 0.0, 1)
    # entry at c0 (100) earns 20%; entry at c1 (100) also earns the move 100 -> 120 ... use 3 prices
    assert base["total_return"] == pytest.approx(0.2)
    assert late["total_return"] == pytest.approx(0.2)
    sig2 = make_sig({"BTCUSDT": [100.0, 110.0, 121.0]})
    assert m.simulate(sig2, "hold", 0.0, 0)["total_return"] == pytest.approx(0.21)
    assert m.simulate(sig2, "hold", 0.0, 1)["total_return"] == pytest.approx(
        0.1
    )  # buys at 110


def test_leverage_schedule_before_and_after_180_bars() -> None:
    assert (
        m.next_leverage([0.01] * 180, 1.0) == 1.0
    )  # 179 marks -> not enough in-stage bars
    rng = np.random.default_rng(0)
    r = list(rng.normal(0.0, 0.01, 181))  # index 0 is the entry bar
    sigma = float(np.std(np.asarray(r[-180:]), ddof=0)) * math.sqrt(6 * 365)
    want = min(2.0, 0.20 / sigma)
    assert m.next_leverage(r, 1.0) == pytest.approx(want)
    # inside the +/-20% band the leverage stays put; outside it moves
    assert m.next_leverage(r, want * 1.1) == want * 1.1
    assert m.next_leverage(r, want * 1.5) == pytest.approx(want)
    assert m.next_leverage(r, want * 0.7) == pytest.approx(want)
    assert m.next_leverage([0.0] * 181, 1.0) == 2.0  # flat book: cap
    big = list(rng.normal(0.0, 0.2, 181))  # vol so high the target is far below 1
    assert m.next_leverage(big, 1.0) < 0.5


def test_hard_cap_clamps_fills_to_two_times_post_fee_equity() -> None:
    sig = make_sig({"BTCUSDT": [100.0, 100.0]})
    book = m._Book(0.001, 0, {"H": "BTCUSDT"})
    book._fill(0, {"H": 1.0}, 2.0, sig)
    # units 2s/100 with fee 0.001 * 2s: gross 2s = 2 * (1 - 0.002 s)  =>  s = 1 / 1.002
    s = 1.0 / 1.002
    assert book.units["H"] == pytest.approx(2.0 * s / 100.0, rel=1e-12)
    assert book.clamps == 1 and book.min_clamp_scale == pytest.approx(s, rel=1e-12)
    assert book.max_gross_fill == pytest.approx(2.0, abs=1e-9) and not book.reasons
    assert book.eq == pytest.approx(1.0 - 0.001 * 2.0 * s, rel=1e-12)
    # below the cap nothing is clamped
    ok = m._Book(0.001, 0, {"H": "BTCUSDT"})
    ok._fill(0, {"H": 1.0}, 1.5, sig)
    assert ok.clamps == 0 and ok.units["H"] == pytest.approx(1.5 / 100.0)


def test_gross_breach_is_invalid_if_the_clamp_were_bypassed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(m, "_clamp_scale", lambda *a, **k: 1.0)
    sig = make_sig({"BTCUSDT": [100.0, 100.0]})
    book = m._Book(0.0, 0, {"H": "BTCUSDT"})
    book._fill(0, {"H": 1.0}, 2.5, sig)
    assert any("gross leverage" in r for r in book.reasons)


def test_clamp_scale_accounts_for_existing_positions_and_fee() -> None:
    # hold 1.0 BTC-notional already; the fill wants 3x equity: must land exactly on 2x post-fee
    price = {"A": 100.0}
    old = {"A": 0.5}  # notional 50 on equity 100
    new = {"A": 3.0}  # notional 300
    s = m._clamp_scale(new, old, price, 0.001, 100.0)
    fee = 0.001 * abs(s * 3.0 - 0.5) * 100.0
    assert s * 300.0 == pytest.approx(2.0 * (100.0 - fee), rel=1e-12)
    assert m._clamp_scale({"A": 1.0}, old, price, 0.001, 100.0) == 1.0
    assert (
        m._clamp_scale(new, old, price, 0.001, 0.0) == 1.0
    )  # no equity: caught elsewhere


def test_prices_must_be_positive_and_present() -> None:
    for bad in ([100.0, 0.0], [100.0, -1.0], [100.0, float("nan")]):
        with pytest.raises(ValueError, match="non-positive or missing"):
            m.simulate(make_sig({"BTCUSDT": bad}), "hold", 0.0, 0)


def test_book_invalid_on_nonpositive_equity() -> None:
    sig = make_sig({"BTCUSDT": [100.0, 40.0, 40.0]}, rule={"BTCUSDT": [1, 1, 1]})
    book = m._Book(0.0, 0, {"H": "BTCUSDT"})
    book._fill(0, {"H": 1.0}, 2.0, sig)  # 2x long, then -60%: equity -0.2
    book.mark(1, sig)
    assert any("not positive" in r for r in book.reasons)


def test_identity_break_is_detected(monkeypatch: pytest.MonkeyPatch) -> None:
    sig = make_sig({"BTCUSDT": [100.0, 101.0, 102.0, 103.0]})
    real = m._Book.execute_due

    def corrupt(self: m._Book, j: int, s: m.Signals) -> None:
        real(self, j, s)
        if j == 1:
            self.cash += 1e-6  # money appears from nowhere

    monkeypatch.setattr(m._Book, "execute_due", corrupt)
    cell = m.simulate(sig, "hold", 0.0, 0)
    assert not cell["valid"] and any("identity" in r for r in cell["invalid_reasons"])


def test_cash_cell_is_flat() -> None:
    c = m.cash_cell(5)
    assert c["valid"] and c["total_return"] == 0.0 and c["max_dd"] == 0.0
    assert len(c["equity"]) == 7 and len(c["returns"]) == 6


# ----------------------------------------------------------------------------- G2 oracle
# An independent, deliberately plain reimplementation of the COMBO book. It shares no code with
# the engine: it runs the unscaled account first, derives the leverage schedule from its returns,
# then runs the scaled account, with its own state machines.


def oracle_book(
    sig: m.Signals,
    wa: float,
    wb: float,
    fee: float,
    delay: int,
    lev_sched: list[float] | None,
) -> list[float]:
    syms, n = sig.symbols, sig.bars
    px = {s: [float(v) for v in sig.close[s]] for s in syms}
    # held sizes of both sleeves, evaluated bar by bar
    a_size = 0.0
    b_size = dict.fromkeys(syms, 0.0)
    nominal_path: list[dict[str, float]] = []
    for j in range(n):
        va = float(sig.vol[m.BTC][j])
        if math.isfinite(va):
            want = 1 if int(sig.rule[m.BTC][j]) > 0 else 0
            new = want * min(1.0, 0.5 / va) if va > 0 else float(want)
            a_size = a_size if (a_size > 0 and new > 0) else new
        nom = {"A": wa * a_size}
        for s in syms:
            v, r, ml = float(sig.vol[s][j]), int(sig.rule[s][j]), int(sig.mlb[s][j])
            if math.isfinite(v):
                cur = (b_size[s] > 0) - (b_size[s] < 0)
                if cur == 0:
                    want = r if (r != 0 and ml == r) else 0
                else:
                    want = cur if (r == cur and ml != -cur) else 0
                if s != m.BTC and want < 0 and int(sig.rule[m.BTC][j]) != -1:
                    want = 0
                sz = want * (min(1.0, 0.5 / v) if v > 0 else 1.0)
                if cur != 0 and ((sz > 0) - (sz < 0)) == cur:
                    sz = b_size[s]
                b_size[s] = sz
            nom["B" + s] = wb * b_size[s] / len(syms)
        nominal_path.append(nom)
    keys = list(nominal_path[0])
    sym_of = {k: (m.BTC if k == "A" else k[1:]) for k in keys}
    orders: dict[int, tuple[dict[str, float], float]] = {}
    prev_n: dict[str, float] | None = None
    prev_l: float | None = None
    for j in range(n):
        lev = 1.0 if lev_sched is None else lev_sched[j]
        base = prev_n if prev_n is not None else dict.fromkeys(keys, 0.0)
        if nominal_path[j] != base or lev != (1.0 if prev_l is None else prev_l):
            orders[j + delay] = (nominal_path[j], lev)
        prev_n, prev_l = nominal_path[j], lev
    cash = 1.0
    units = dict.fromkeys(keys, 0.0)
    held_nom = dict.fromkeys(keys, 0.0)
    held_lev = 1.0
    post: list[float] = []
    for j in range(n + 1):
        if j >= 1:  # only funding moves cash; equity is valued as cash + units * price
            for s in syms:
                cash -= sum(units[k] for k in keys if sym_of[k] == s) * float(
                    sig.fund[s][j]
                )
        if j in orders:
            nom, lev = orders[j]
            eq = cash + sum(units[k] * px[sym_of[k]][j] for k in keys)
            old = {s: sum(units[k] for k in keys if sym_of[k] == s) for s in syms}
            for k in keys:
                if nom[k] != held_nom[k]:
                    units[k] = lev * nom[k] * eq / px[sym_of[k]][j]
                elif lev != held_lev:
                    units[k] = units[k] * lev / held_lev
            held_nom, held_lev = dict(nom), lev
            for s in syms:
                new_u = sum(units[k] for k in keys if sym_of[k] == s)
                d = new_u - old[s]
                cash -= d * px[s][j] + fee * abs(d) * px[s][j]
        post.append(cash + sum(units[k] * px[sym_of[k]][j] for k in keys))
    return [1.0, *post]


def oracle_cell(sig: m.Signals, kind: str, fee: float, delay: int) -> np.ndarray:
    wa, wb = {"combo": (0.5, 0.5), "btc_rule": (1.0, 0.0), "alt_agree": (0.0, 1.0)}[
        kind
    ]
    unscaled = oracle_book(sig, wa, wb, fee, delay, None)
    if kind == "alt_agree":
        return np.asarray(unscaled)
    rets = [unscaled[1] / 1.0 - 1.0] + [
        unscaled[i + 1] / unscaled[i] - 1.0 for i in range(1, len(unscaled) - 1)
    ]
    ann = math.sqrt(6 * 365)
    sched: list[float] = []
    lev = 1.0
    for j in range(sig.bars):
        if j >= 180:
            w = rets[j - 179 : j + 1]
            mean = sum(w) / 180
            sd = math.sqrt(sum((v - mean) ** 2 for v in w) / 180) * ann
            new = min(2.0, 0.2 / sd) if sd > 1e-6 else 2.0
            if lev == 0 or abs(new / lev - 1) > 0.2:
                lev = new
        sched.append(lev)
    return np.asarray(oracle_book(sig, wa, wb, fee, delay, sched))


def random_signals(seed: int, bars: int = 520, symbols: int = 3) -> m.Signals:
    rng = np.random.default_rng(seed)
    syms = m.SYMBOLS[:symbols]
    close, rule, mlb, vol, fund = {}, {}, {}, {}, {}
    for s in syms:
        close[s] = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.012, bars + 1)))

        def markov(k: int, stay: float) -> np.ndarray:
            out = np.zeros(k, np.int8)
            cur = 0
            for i in range(k):
                if rng.random() > stay:
                    cur = int(rng.integers(-1, 2))
                out[i] = cur
            return out

        rule[s], mlb[s] = markov(bars + 1, 0.93), markov(bars + 1, 0.85)
        vol[s] = rng.uniform(0.6, 1.3, bars + 1)
        f = np.zeros(bars + 1)
        idx = np.arange(2, bars + 1, 2)
        f[idx] = close[s][idx] * rng.normal(0.0, 1e-4, len(idx))
        fund[s] = f
    return make_sig({s: list(close[s]) for s in syms}, rule, mlb, vol, fund)


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize(
    "kind,fee,delay",
    [
        ("combo", 0.0006, 0),
        ("combo", 0.0012, 0),
        ("combo", 0.0006, 1),
        ("btc_rule", 0.0006, 0),
        ("alt_agree", 0.0006, 0),
    ],
)
def test_engine_matches_independent_oracle(
    seed: int, kind: str, fee: float, delay: int
) -> None:
    sig = random_signals(seed)
    cell = m.simulate(sig, kind, fee, delay)
    expect = oracle_cell(sig, kind, fee, delay)
    assert cell["valid"], cell["invalid_reasons"]
    assert len(expect) == len(cell["equity"])
    assert np.allclose(cell["equity"], expect, rtol=0, atol=1e-9)
    if kind in ("combo", "btc_rule"):
        assert cell["leverage_changes"] > 0  # the vol target actually engaged
    assert cell["fills"] > 0
    assert (
        cell["hard_cap_clamps"] == 0
    )  # the oracle has no clamp: the scenario stays below it


def test_oracle_scenario_is_nontrivial() -> None:
    sig = random_signals(0)
    cell = m.simulate(sig, "combo", 0.0006, 0)
    assert cell["fees"] > 0 and cell["funding"] != 0.0
    assert cell["max_gross_at_fills"] > 0.05
    assert cell["max_held_gross"] <= 2.0 + 1e-9


# ----------------------------------------------------------------------------- decision


def _cell(
    ret: float = 0.1, sharpe: float = 1.0, dd: float = 0.1, valid: bool = True
) -> dict[str, Any]:
    return {"total_return": ret, "sharpe": sharpe, "max_dd": dd, "valid": valid}


def _cells(**over: dict[str, Any]) -> dict[str, dict[str, Any]]:
    cells = {k: _cell() for k in m.CELLS}
    cells.update(over)
    return cells


def test_decide_final_branches() -> None:
    ok = m.decide("FORWARD_FINAL", _cells())
    assert ok["label"] == "MTF_COMBO_SUPPORTED" and ok["supported"]
    for over in (
        {"COMBO__BASE": _cell(ret=0.0)},
        {"COMBO__BASE": _cell(sharpe=0.4999)},
        {"COMBO__HIGH_COST": _cell(ret=-0.01)},
        {"COMBO__DELAY": _cell(ret=0.0)},
        {"COMBO__BASE": _cell(dd=0.25)},
        {"COMBO__HIGH_COST": _cell(dd=0.25)},
        {"COMBO__DELAY": _cell(dd=0.30)},
    ):
        d = m.decide("FORWARD_FINAL", _cells(**over))
        assert d["label"] == "MTF_COMBO_NOT_SUPPORTED" and d["failed_conditions"]
    # thresholds are inclusive only where the spec says so
    assert m.decide("FORWARD_FINAL", _cells(COMBO__BASE=_cell(sharpe=0.5)))["supported"]
    assert m.decide("FORWARD_FINAL", _cells(COMBO__BASE=_cell(dd=0.2499)))["supported"]
    # reported cells never gate (even a drawdown or a loss)
    assert m.decide("FORWARD_FINAL", _cells(HOLD_BTC=_cell(ret=-0.9, dd=0.9)))[
        "supported"
    ]


def test_decide_invalid_has_priority() -> None:
    d = m.decide("FORWARD_FINAL", _cells(HOLD_BTC=_cell(valid=False)))
    assert (
        d["label"] == "INVALID"
        and d["invalid_cells"] == ["HOLD_BTC"]
        and not d["supported"]
    )
    d = m.decide(
        "FORWARD_INTERIM", _cells(CASH=_cell(valid=False), COMBO__BASE=_cell(dd=0.9))
    )
    assert d["label"] == "INVALID"
    d = m.decide("FORWARD_FINAL", _cells(), problems=["BTCUSDT: 1 missing hour(s)"])
    assert d["label"] == "INVALID" and d["problems"] and not d["supported"]
    assert m.decide("FORWARD_FINAL", {}, ["data"])["label"] == "INVALID"


def test_decide_interim_branches() -> None:
    assert m.decide("FORWARD_INTERIM", _cells())["label"] == "INTERIM_NOT_A_DECISION"
    assert m.decide("FORWARD_INTERIM", _cells(COMBO__BASE=_cell(dd=0.2499)))[
        "label"
    ] == ("INTERIM_NOT_A_DECISION")
    assert (
        m.decide("FORWARD_INTERIM", _cells(COMBO__BASE=_cell(dd=0.25)))["label"]
        == "STOPPED_EARLY"
    )
    # only the primary cell stops the line; an interim never claims SUPPORTED
    assert m.decide("FORWARD_INTERIM", _cells(COMBO__DELAY=_cell(dd=0.5)))["label"] == (
        "INTERIM_NOT_A_DECISION"
    )
    assert m.decide("FORWARD_INTERIM", _cells(COMBO__BASE=_cell(ret=-0.5)))[
        "label"
    ] != ("MTF_COMBO_SUPPORTED")


def test_summary_cell_is_json_safe() -> None:
    sig = random_signals(0, bars=40)
    cell = m.simulate(sig, "combo", 0.0006, 0)
    s = m.summarize_cell(cell)
    json.dumps(s)
    assert s["bars"] == 41 and len(s["equity_sha256"]) == 64 and "equity" not in s


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
    assert rows == [(3600000, 99.0, 101.0, 98.0, 50.5, 500.0, 250.0)]
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
    rows = [(0, 1.0, 2.0, 0.5, 1.5, 10.0, 5.0), (2 * m.HOUR_MS, 1, 1, 1, 1, 1, 1)]
    a = m.hourly_arrays(rows, 0, 3 * m.HOUR_MS)
    assert len(a["close"]) == 4 and np.isnan(a["close"][1]) and a["close"][2] == 1.0
    assert a["qv"][0] == 10.0 and a["tbq"][0] == 5.0
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
        "mtf_combo_v1",
        f"{'d' * 64} {COMMIT}",
        f"mtf_combo_v1 {'d' * 64}",
        f"mtf_combo_v1 {'e' * 64} {COMMIT}",
        f"mtf_combo_v1 {'d' * 64} {'b' * 40}",
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
    factory: Any = FakeModel,
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
        model_factory=factory,
        symbols=symbols,
    )


def test_interim_end_to_end_write_once_and_ledger(env: dict[str, Any]) -> None:
    now = datetime(2027, 1, 20, 9, tzinfo=UTC)  # 50 days: the vol target is engaged
    res = _run(env, "FORWARD_INTERIM", now)
    assert res["label"] in {"INTERIM_NOT_A_DECISION", "STOPPED_EARLY", "INVALID"}
    assert res["label"] != "INVALID", res["problems"] + res["invalid_cells"]
    assert set(res["cells"]) == set(m.CELLS)
    assert res["window"]["bars"] == 50 * 6 + 2 and res["window"][
        "last_mark_close"
    ].startswith("2027-01-20T08")
    assert res["claim_name"] == "FORWARD_INTERIM_2027-01"
    assert res["model"]["deterministic"] and res["model"]["train_rows"] > 1000
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
    assert led["model_prob_sha256"] == res["model"]["model_prob_sha256"]
    on_disk = json.loads(out.read_text())
    assert on_disk["protocol_digest"] == env["digest"] and on_disk["commit"] == COMMIT
    raw = env["tmp"] / "work_x" / "raw"
    assert len(list(raw.glob("BTCUSDT_klines_*.json"))) == len(
        res["data"]["pages"]["BTCUSDT"]["klines"]
    )
    for sym in TWO:
        for kind in ("klines", "funding"):
            for p in res["data"]["pages"][sym][kind]:
                assert m.sha256_file(raw / p["file"]) == p["sha256"]
    with pytest.raises(SystemExit, match="already claimed"):
        _run(env, "FORWARD_INTERIM", now + timedelta(days=3), tag="again")
    with pytest.raises(SystemExit, match="output directory exists"):
        _run(env, "FORWARD_INTERIM", now + timedelta(days=40), tag="x")


def test_full_sequence_interims_and_final_share_one_model(env: dict[str, Any]) -> None:
    r1 = _run(env, "FORWARD_INTERIM", datetime(2026, 12, 20, 9, tzinfo=UTC), tag="dec")
    r2 = _run(env, "FORWARD_INTERIM", datetime(2027, 1, 20, 9, tzinfo=UTC), tag="jan")
    fin = _run(env, "FORWARD_FINAL", datetime(2027, 12, 2, 1, tzinfo=UTC), tag="fin")
    assert (
        r1["model"]["model_prob_sha256"]
        == r2["model"]["model_prob_sha256"]
        == fin["model"]["model_prob_sha256"]
    )
    assert fin["label"] in {"MTF_COMBO_SUPPORTED", "MTF_COMBO_NOT_SUPPORTED"}, fin[
        "problems"
    ]
    assert fin["window"]["bars"] == 365 * 6 and fin["window"][
        "last_mark_close"
    ].startswith("2027-12-01T00")
    assert fin["claim_name"] == "FORWARD_FINAL"
    with pytest.raises(SystemExit, match="already claimed"):
        _run(env, "FORWARD_FINAL", datetime(2027, 12, 3, tzinfo=UTC), tag="dup")
    # reported cells and equity digests are deterministic: a second environment agrees
    env2 = {**env, "ledger": env["tmp"] / "ledger2"}
    fin2 = _run(env2, "FORWARD_FINAL", datetime(2027, 12, 2, 1, tzinfo=UTC), tag="fin2")
    assert {k: v["equity_sha256"] for k, v in fin["cells"].items()} == {
        k: v["equity_sha256"] for k, v in fin2["cells"].items()
    }


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
    # refused before claiming: the current digest namespace was never created
    assert not (env["ledger"] / m.PROTOCOL_ID / env["digest"]).exists()


def test_old_final_blocks_a_new_digest_from_reclaiming_the_final(
    env: dict[str, Any],
) -> None:
    _seed(_old_ns(env), "FORWARD_FINAL", "MTF_COMBO_NOT_SUPPORTED")
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


def test_model_drift_between_stages_is_invalid(env: dict[str, Any]) -> None:
    led = env["ledger"] / m.PROTOCOL_ID / env["digest"]
    m.claim_stage(
        "FORWARD_INTERIM_2026-12",
        env["digest"],
        COMMIT,
        1,
        datetime(2026, 12, 20, tzinfo=UTC),
        led,
    )
    m.write_ledger_result(
        "FORWARD_INTERIM_2026-12",
        env["digest"],
        "INTERIM_NOT_A_DECISION",
        False,
        "s",
        led,
        "f" * 64,
    )
    res = _run(env, "FORWARD_INTERIM", datetime(2027, 1, 20, 9, tzinfo=UTC))
    assert res["label"] == "INVALID" and any(
        "model drift" in p for p in res["problems"]
    )


def test_model_drift_is_detected_across_digests(env: dict[str, Any]) -> None:
    """An interim ledgered under an OLD digest with a different model hash makes a run under the
    current digest INVALID: drift detection is keyed by protocol id, not by digest."""
    old = env["ledger"] / m.PROTOCOL_ID / ("0" * 64)
    old.mkdir(parents=True, exist_ok=True)
    m.write_ledger_result(
        "FORWARD_INTERIM_2026-12",
        "0" * 64,
        "INTERIM_NOT_A_DECISION",
        False,
        "s",
        old,
        "e" * 64,
    )
    res = _run(env, "FORWARD_INTERIM", datetime(2027, 1, 20, 9, tzinfo=UTC))
    assert res["label"] == "INVALID" and any(
        "model drift" in p and ("0" * 64) in p for p in res["problems"]
    )


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
    assert res["cells"] == {} and "model" not in res
    assert (
        env["ledger"]
        / m.PROTOCOL_ID
        / env["digest"]
        / "FORWARD_INTERIM_2027-01.result.json"
    ).exists()


def test_history_hole_blanks_features_without_invalidating(env: dict[str, Any]) -> None:
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
            model_factory=FakeModel,
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
            model_factory=FakeModel,
            symbols=TWO,
        )


def test_interim_equals_prefix_of_final() -> None:
    """Whole-pipeline causality: an interim replay is the exact prefix of the later, longer replay
    (no decision depends on a bar that did not exist yet), including the vol-target leverage."""
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
        x, y = m.training_set(frames, symbols, c0)
        model, lo, hi, _ = m.fit_direction_model(x, y, FakeModel)
        bars = (last - c0) // m.BAR_S
        fund = {s: np.zeros(bars + 1) for s in symbols}
        fund["BTCUSDT"][5::7] = 3.0e-3
        sig = m.build_signals(frames, symbols, c0, bars, model, lo, hi, fund)
        return {k: v["equity"] for k, v in m.evaluate_cells(sig).items()}

    short = cells_until(datetime(2027, 2, 1, tzinfo=UTC))  # 62 days, vol target engaged
    long = cells_until(datetime(2027, 4, 1, tzinfo=UTC))
    for name in m.CELLS:
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


# Synthetic regression for source candle finality. These tests do not fetch
# market data, train the MTF classifier or authorize any research stage.
@pytest.mark.parametrize(
    ("native_unit", "close_offset"),
    [
        (1, -m.HOUR_MS),
        (1, -1),
        (1, 1),
        (1, m.HOUR_MS),
        (1000, -1),
        (1000, 1),
    ],
)
def test_raw_klines_refuses_incomplete_or_shifted_close(
    native_unit: int, close_offset: int
) -> None:
    open_ms = 1735689600000  # synthetic UTC 2025-01-01 00:00
    row = _kline(open_ms)
    row[0] = open_ms * native_unit
    row[6] = (open_ms + m.HOUR_MS) * native_unit - 1 + close_offset
    with pytest.raises(SystemExit, match="close_time"):
        m.parse_klines([row])


@pytest.mark.parametrize(("open_unit", "close_unit"), [(1, 1000), (1000, 1)])
def test_raw_klines_refuses_native_timestamp_unit_mismatch(
    open_unit: int, close_unit: int
) -> None:
    open_ms = 1735689600000
    row = _kline(open_ms)
    row[0] = open_ms * open_unit
    row[6] = (open_ms + m.HOUR_MS) * close_unit - 1
    with pytest.raises(SystemExit, match="close_time"):
        m.parse_klines([row])


@pytest.mark.parametrize("native_unit", [1, 1000])
def test_raw_klines_accepts_only_full_native_completed_hour(native_unit: int) -> None:
    open_ms = 1735689600000
    row = _kline(open_ms)
    row[0] = open_ms * native_unit
    row[6] = (open_ms + m.HOUR_MS) * native_unit - 1
    rows = m.parse_klines([row])
    assert rows[0][0] == open_ms
    arrs = m.hourly_arrays(rows, open_ms, open_ms)
    assert arrs["close"].tolist() == [100.0]
    assert arrs["qv"].tolist() == [500.0]
    assert arrs["tbq"].tolist() == [250.0]


@pytest.mark.parametrize("bad_close", [None, 3.5, "1735693199999.0", -1])
def test_raw_klines_rejects_non_native_close_clock(bad_close: object) -> None:
    row = _kline(1735689600000)
    row[6] = bad_close
    with pytest.raises(SystemExit, match="close_time"):
        m.parse_klines([row])


def test_page_reader_preserves_raw_rejected_source_bytes(tmp_path: Path) -> None:
    open_ms = 1735689600000
    row = _kline(open_ms)
    row[6] = open_ms - 1
    body = json.dumps([row]).encode()
    with pytest.raises(SystemExit, match="close_time"):
        m.fetch_klines("BTCUSDT", open_ms, open_ms, tmp_path / "raw", lambda _: body)
    assert (tmp_path / "raw" / "BTCUSDT_klines_000.json").read_bytes() == body


@pytest.mark.parametrize("bad_open", [1735689600000001, "1735689600000.0", True])
def test_raw_klines_rejects_non_native_or_unaligned_open(bad_open: object) -> None:
    row = _kline(1735689600000)
    row[0] = bad_open
    with pytest.raises(SystemExit, match="close_time"):
        m.parse_klines([row])


def test_claim_name_uses_the_utc_month_and_refuses_naive_datetimes() -> None:
    from datetime import timedelta, timezone

    st = m.STAGES["FORWARD_INTERIM"]
    jst = timezone(timedelta(hours=9))
    assert (
        m.claim_name(st, datetime(2026, 12, 1, 0, 30, tzinfo=jst))
        == "FORWARD_INTERIM_2026-11"
    )
    assert (
        m.claim_name(st, datetime(2026, 12, 1, 9, 30, tzinfo=jst))
        == "FORWARD_INTERIM_2026-12"
    )
    with pytest.raises(SystemExit, match="timezone-aware"):
        m.claim_name(st, datetime(2026, 12, 20))


def test_run_stage_refuses_a_relative_ledger_directory(tmp_path: Any) -> None:
    from pathlib import Path

    with pytest.raises(SystemExit, match="absolute"):
        m.run_stage(
            "FORWARD_INTERIM",
            "https://github.com/shuntatsu/trade_rl/issues/810#issuecomment-1",
            tmp_path / "work",
            tmp_path / "out",
            now=datetime(2027, 1, 20, 9, tzinfo=UTC),
            ledger_dir=Path("relative_ledger"),
        )


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
            gh_fetch=_gh(
                relay_body=f"G4 identity: {m.PROTOCOL_ID} {env['digest']} {COMMIT}\nG4 claim: FORWARD_FINAL"
            ),
            ledger_dir=env["ledger"],
            symbols=TWO,
        )
    assert {p: p.read_bytes() for p in root.rglob("*.claim")} == previous
    assert fetched == []


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
