"""Hierarchical MTF long/short perp system, protocol ``mtf_combo_v1``.

Protocol: docs/specs/mtf-combo-forward.md (the spec is the authority for every constant).

A prospective FORWARD test of a regime + ML-direction long/short system on 11 Binance USDT-M
perpetuals. Stages (``STAGES``) fetch public 1h klines and funding history, build causal
multi-timeframe features, train one direction model on pre-stage data only, replay seven frozen
cells on an actual-units book and write a write-once result. A local ledger claim is created
exclusively before any data is fetched. Importing this module needs only the standard library
and numpy; scikit-learn is imported only when the real model is fitted, and the network code
runs only in real ``run`` invocations.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import re
import stat
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from trade_rl.integrations.binance.source_clock import (
    BinanceSourceClosePolicy,
    _check_source_close_time,
    _native_timestamp,
)

PROTOCOL_ID = "mtf_combo_v1"
SOURCE_CLOSE_POLICY = BinanceSourceClosePolicy.STRICT_SOURCE_CLOSE_V2

# ----------------------------------------------------------------------------- constants

SYMBOLS = (
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "BNBUSDT",
    "XRPUSDT",
    "ADAUSDT",
    "DOGEUSDT",
    "LINKUSDT",
    "DOTUSDT",
    "AVAXUSDT",
    "LTCUSDT",
)
BTC = "BTCUSDT"

HOUR_S = 3600
BAR_S = 4 * HOUR_S  # 4h decision bars
DAY_S = 86400
HOUR_MS = 1000 * HOUR_S
BAR_MS = 1000 * BAR_S
BAR_PER_YEAR = 6 * 365
ANN = math.sqrt(BAR_PER_YEAR)

REGIME_WINDOWS = (20, 50, 100, 200)  # daily SMA lookbacks of the regime vote
REGIME_THRESHOLD = 0.5
LABEL_HORIZON = 30  # 4h bars
BAND = 0.4  # model bands: quantiles 0.4 and 0.6 of the training probabilities
SIZE_VOL = 0.5  # size = min(1, 0.5 / vol_ann)
SLEEVE_WEIGHT = 0.5  # COMBO = 0.5 * sleeve A + 0.5 * sleeve B
TARGET_VOL = 0.20
LEVERAGE_CAP = 2.0
VOL_BARS = 180
LEVERAGE_BAND = 0.20
VOL_EPS = 1e-6
HGB_PARAMS: dict[str, Any] = {
    "max_depth": 3,
    "max_iter": 300,
    "learning_rate": 0.03,
    "l2_regularization": 2.0,
    "min_samples_leaf": 300,
    "random_state": 0,
}

FEE_BASE = 0.0006
FEE_HIGH = 0.0012
IDENTITY_TOL = 1e-9
GROSS_CAP = 2.0
EARLY_STOP_DD = 0.25
MAX_DD = 0.25
MIN_SHARPE = 0.5
TRADE_EPS = 1e-12

# name -> (kind, one-way fee on fill notional, fill delay in 4h bars, gates the decision)
CELLS: dict[str, tuple[str, float, int, bool]] = {
    "COMBO__BASE": ("combo", FEE_BASE, 0, True),
    "COMBO__HIGH_COST": ("combo", FEE_HIGH, 0, True),
    "COMBO__DELAY": ("combo", FEE_BASE, 1, True),
    "BTC_RULE__BASE": ("btc_rule", FEE_BASE, 0, False),
    "ALT_AGREE__BASE": ("alt_agree", FEE_BASE, 0, False),
    "HOLD_BTC": ("hold", FEE_BASE, 0, False),
    "CASH": ("cash", 0.0, 0, False),
}
GATING_PRIMARY = "COMBO__BASE"
GATING_POSITIVE = ("COMBO__HIGH_COST", "COMBO__DELAY")

TF_FIELDS = (
    "rsi",
    "macdh",
    "ppo",
    "stoch",
    "ich",
    "bull",
    "bear",
    "net",
    "dsma20",
    "dsma50",
    "r1",
    "r6",
    "vol",
)
FEATURE_COLUMNS = (
    tuple(f"{f}_4h" for f in TF_FIELDS)
    + tuple(f"{f}_1h" for f in TF_FIELDS)
    + tuple(f"{f}_1d" for f in TF_FIELDS)
    + ("regime", "dsma200_1d", "vol_ann")
)
I_REGIME = FEATURE_COLUMNS.index("regime")
I_VOL_ANN = FEATURE_COLUMNS.index("vol_ann")


@dataclass(frozen=True)
class Stage:
    name: str
    history_start: datetime  # first hourly bar loaded (features, training)
    decision_start: datetime  # first decision close c_0, also the training cutoff
    decision_end: datetime  # last mark close; decisions are made at closes before it
    interim: bool
    final: bool
    not_before: datetime | None
    not_after: datetime | None


_HISTORY = datetime(2021, 1, 1, tzinfo=UTC)
_START = datetime(2026, 12, 1, tzinfo=UTC)
_END = datetime(2027, 12, 1, tzinfo=UTC)

STAGES: dict[str, Stage] = {
    "FORWARD_INTERIM": Stage(
        "FORWARD_INTERIM",
        _HISTORY,
        _START,
        _END,
        interim=True,
        final=False,
        not_before=datetime(2026, 12, 2, tzinfo=UTC),
        not_after=datetime(2027, 12, 2, tzinfo=UTC),
    ),
    "FORWARD_FINAL": Stage(
        "FORWARD_FINAL",
        _HISTORY,
        _START,
        _END,
        interim=False,
        final=True,
        not_before=datetime(2027, 12, 2, tzinfo=UTC),
        not_after=None,
    ),
}

LABEL_SUPPORTED = "MTF_COMBO_SUPPORTED"
LABEL_NOT_SUPPORTED = "MTF_COMBO_NOT_SUPPORTED"
LABEL_INVALID = "INVALID"
LABEL_INTERIM = "INTERIM_NOT_A_DECISION"
LABEL_STOPPED = "STOPPED_EARLY"

DATA_SCOPE_COMMENT_ID = 7040247247  # owner scope relay on Issue #810
OWNER_LOGIN = "shuntatsu"
REPO = "shuntatsu/trade_rl"
ISSUE_URL = f"https://api.github.com/repos/{REPO}/issues/810"
LEDGER_DIR = Path(r"C:\trade_rl_runs\mtf_combo_forward_ledger")
REPO_ROOT = Path(__file__).resolve().parent.parent
SPEC_RELPATH = "docs/specs/mtf-combo-forward.md"

USER_AGENT = "trade_rl-research/1.0"
HTTP_TIMEOUT_S = 30
KLINES_URL = "https://fapi.binance.com/fapi/v1/klines"
FUNDING_URL = "https://fapi.binance.com/fapi/v1/fundingRate"
KLINES_LIMIT = 1500
FUNDING_LIMIT = 1000
MAX_PAGES = 120
MAX_FUNDING_GAP_S = 8 * HOUR_S + 60
SETTLE_S = 300  # an interim ignores the newest 5 minutes
HTTP_ATTEMPTS = 6
HTTP_PACE_S = 0.5


# ----------------------------------------------------------------------------- digests


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _iso(d: datetime | None) -> str | None:
    return d.isoformat() if d else None


def _constants() -> dict[str, Any]:
    return {
        "protocol_id": PROTOCOL_ID,
        "source_close_policy": SOURCE_CLOSE_POLICY.value,
        "symbols": list(SYMBOLS),
        "regime_windows": list(REGIME_WINDOWS),
        "regime_threshold": REGIME_THRESHOLD,
        "label_horizon": LABEL_HORIZON,
        "band": BAND,
        "size_vol": SIZE_VOL,
        "sleeve_weight": SLEEVE_WEIGHT,
        "target_vol": TARGET_VOL,
        "leverage_cap": LEVERAGE_CAP,
        "vol_bars": VOL_BARS,
        "leverage_band": LEVERAGE_BAND,
        "vol_eps": VOL_EPS,
        "hgb_params": HGB_PARAMS,
        "bar_per_year": BAR_PER_YEAR,
        "fee_base": FEE_BASE,
        "fee_high": FEE_HIGH,
        "identity_tol": IDENTITY_TOL,
        "gross_cap": GROSS_CAP,
        "early_stop_dd": EARLY_STOP_DD,
        "max_dd": MAX_DD,
        "min_sharpe": MIN_SHARPE,
        "cells": {k: list(v) for k, v in CELLS.items()},
        "feature_columns": list(FEATURE_COLUMNS),
        "stages": {
            k: [
                _iso(v.history_start),
                _iso(v.decision_start),
                _iso(v.decision_end),
                v.interim,
                v.final,
                _iso(v.not_before),
                _iso(v.not_after),
            ]
            for k, v in STAGES.items()
        },
        "data_scope_comment": DATA_SCOPE_COMMENT_ID,
        "owner_login": OWNER_LOGIN,
        "issue_url": ISSUE_URL,
        "ledger_dir": str(LEDGER_DIR),
        "klines_url": KLINES_URL,
        "funding_url": FUNDING_URL,
        "klines_limit": KLINES_LIMIT,
        "funding_limit": FUNDING_LIMIT,
        "max_funding_gap_s": MAX_FUNDING_GAP_S,
        "settle_s": SETTLE_S,
    }


def spec_sha256() -> str:
    """sha256 of the spec with CRLF normalized and the ``Protocol digest:`` line removed."""
    text = (REPO_ROOT / SPEC_RELPATH).read_bytes().replace(b"\r\n", b"\n")
    kept = [ln for ln in text.split(b"\n") if not ln.startswith(b"Protocol digest:")]
    return hashlib.sha256(b"\n".join(kept)).hexdigest()


def protocol_digest() -> str:
    source = Path(__file__).read_bytes().replace(b"\r\n", b"\n")
    payload = json.dumps(
        {
            "constants": _constants(),
            "harness_sha256": hashlib.sha256(source).hexdigest(),
            "spec_path": SPEC_RELPATH,
            "spec_sha256": spec_sha256(),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ----------------------------------------------------------------------------- array primitives
# Pandas-free ports of the exploratory screens' rolling/ewm primitives (NaN-propagating, with
# ``min_periods`` equal to the window, exactly as the screens used them). Tests compare them with
# the pandas reference when pandas is installed.


def _roll(x: np.ndarray, w: int, fn: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    out = np.full(len(x), np.nan)
    if len(x) >= w:
        out[w - 1 :] = fn(np.lib.stride_tricks.sliding_window_view(x, w))
    return out


def r_max(x: np.ndarray, w: int) -> np.ndarray:
    return _roll(x, w, lambda v: v.max(axis=1))


def r_min(x: np.ndarray, w: int) -> np.ndarray:
    return _roll(x, w, lambda v: v.min(axis=1))


def r_mean(x: np.ndarray, w: int) -> np.ndarray:
    return _roll(x, w, lambda v: v.mean(axis=1))


def r_sum(x: np.ndarray, w: int) -> np.ndarray:
    return _roll(x, w, lambda v: v.sum(axis=1))


def r_std(x: np.ndarray, w: int) -> np.ndarray:
    """Sample standard deviation (ddof 1, the pandas default) over the window."""
    return _roll(x, w, lambda v: v.std(axis=1, ddof=1))


def shift(x: np.ndarray, k: int) -> np.ndarray:
    out = np.full(len(x), np.nan)
    if k < len(x):
        out[k:] = x[: len(x) - k]
    return out


def diff(x: np.ndarray) -> np.ndarray:
    return x - shift(x, 1)


def clip0(x: np.ndarray) -> np.ndarray:
    return np.where(np.isnan(x), np.nan, np.maximum(x, 0.0))


def ewm_mean(x: np.ndarray, alpha: float, min_periods: int) -> np.ndarray:
    """``Series.ewm(alpha=..., adjust=False, min_periods=...).mean()`` including pandas' NaN rule
    (a missing observation decays the old weight and carries the last average)."""
    n = len(x)
    out = np.full(n, np.nan)
    old_factor = 1.0 - alpha
    minp = max(min_periods, 1)
    avg = float(x[0]) if n else math.nan
    nobs = 1 if avg == avg else 0
    if nobs >= minp:
        out[0] = avg
    old_wt = 1.0
    for i in range(1, n):
        cur = float(x[i])
        obs = cur == cur
        nobs += obs
        if avg == avg:
            old_wt *= old_factor
            if obs:
                if avg != cur:
                    avg = (old_wt * avg + alpha * cur) / (old_wt + alpha)
                old_wt = 1.0
        elif obs:
            avg = cur
        if nobs >= minp:
            out[i] = avg
    return out


def ewm_span(x: np.ndarray, span: int, min_periods: int) -> np.ndarray:
    return ewm_mean(x, 2.0 / (span + 1.0), min_periods)


def wilder(x: np.ndarray, n: int) -> np.ndarray:
    return ewm_mean(x, 1.0 / n, n)


# ----------------------------------------------------------------------------- features


def aggregate(
    hourly: dict[str, np.ndarray], hours: int, n_hours: int
) -> dict[str, np.ndarray]:
    """Aggregate the hourly grid (starting at UTC midnight) to bars of ``hours`` hours labelled by
    OPEN time. A bar is kept only if all its hourly bars exist; only complete bars are returned."""
    nb = n_hours // hours

    def cut(key: str) -> np.ndarray:
        return hourly[key][: nb * hours].reshape(nb, hours)

    close = cut("close")
    full = np.isfinite(close).all(axis=1)
    out = {
        "open": cut("open")[:, 0],
        "high": cut("high").max(axis=1),
        "low": cut("low").min(axis=1),
        "close": close[:, -1],
        "qv": cut("qv").sum(axis=1),
        "tbq": cut("tbq").sum(axis=1),
    }
    return {k: np.where(full, v, np.nan) for k, v in out.items()}


def ichimoku_score(h: np.ndarray, lo: np.ndarray, c: np.ndarray) -> np.ndarray:
    tenkan = (r_max(h, 9) + r_min(lo, 9)) / 2
    kijun = (r_max(h, 26) + r_min(lo, 26)) / 2
    span_a = (tenkan + kijun) / 2
    span_b = (r_max(h, 52) + r_min(lo, 52)) / 2
    # The cloud visible at bar t was computed at t-26 (spans shifted FORWARD; never backward).
    ca, cb = shift(span_a, 26), shift(span_b, 26)
    top, bot = np.maximum(ca, cb), np.minimum(ca, cb)
    ok = np.isfinite(top) & np.isfinite(c)
    pos = np.where(ok, np.where(c > top, 1.0, np.where(c < bot, -1.0, 0.0)), np.nan)
    tk = np.where(
        np.isfinite(tenkan) & np.isfinite(kijun),
        np.where(tenkan > kijun, 1.0, -1.0),
        np.nan,
    )
    cs = shift(c, 26)
    ck = np.where(np.isfinite(cs) & np.isfinite(c), np.where(c > cs, 1.0, -1.0), np.nan)
    return pos + tk + ck


def indicators(
    h: np.ndarray, lo: np.ndarray, c: np.ndarray, qv: np.ndarray, tbq: np.ndarray
) -> dict[str, np.ndarray]:
    with np.errstate(divide="ignore", invalid="ignore"):
        d = diff(c)
        rs = wilder(clip0(d), 14) / wilder(clip0(-d), 14)
        ind: dict[str, np.ndarray] = {"rsi": 100 - 100 / (1 + rs)}
        macd = ewm_span(c, 12, 26) - ewm_span(c, 26, 26)
        ind["macdh"] = macd - ewm_span(macd, 9, 9)
        ll, hh = r_min(lo, 14), r_max(h, 14)
        ind["stoch"] = r_mean(100 * (c - ll) / (hh - ll), 3)
        tp = (h + lo + c) / 3
        tp_mean = r_mean(tp, 20)
        mad = _roll(
            tp,
            20,
            lambda v: np.abs(v - v.mean(axis=1, keepdims=True)).mean(axis=1),
        )
        ind["cci"] = (tp - tp_mean) / (0.015 * mad)
        dtp = diff(tp)
        pos = np.where(np.isfinite(dtp), np.where(dtp > 0, qv, 0.0), np.nan)
        neg = np.where(np.isfinite(dtp), np.where(dtp < 0, qv, 0.0), np.nan)
        ind["mfi"] = 100 * r_sum(pos, 14) / (r_sum(pos, 14) + r_sum(neg, 14))
        obv = np.cumsum(np.nan_to_num(np.sign(d)) * np.nan_to_num(qv))
        ind["obv"] = np.where(np.isfinite(c), obv, np.nan)
        ind["roc"] = c / shift(c, 10) - 1
        ind["willr"] = -100 * (hh - c) / (hh - ll)
        ind["tbr"] = ewm_span(tbq / qv, 14, 14)
    return ind


def _pivots(x: np.ndarray, left_w: int, right_w: int, kind: str) -> np.ndarray:
    rev = x[::-1]
    with np.errstate(invalid="ignore"):
        if kind == "low":
            left = r_min(shift(x, 1), left_w)
            right = r_min(shift(rev, 1), right_w)[::-1]
            ok = (x < left) & (x <= right)
        else:
            left = r_max(shift(x, 1), left_w)
            right = r_max(shift(rev, 1), right_w)[::-1]
            ok = (x > left) & (x >= right)
    return np.flatnonzero(ok)


def divergence(
    h: np.ndarray,
    lo: np.ndarray,
    c: np.ndarray,
    qv: np.ndarray,
    tbq: np.ndarray,
    left_w: int = 5,
    right_w: int = 3,
    maxgap: int = 60,
    life: int = 6,
) -> tuple[np.ndarray, np.ndarray]:
    """(bull_count, bear_count). A pivot at p is confirmed (known) only at bar p + right_w."""
    ind = indicators(h, lo, c, qv, tbq)
    n = len(c)
    pl, ph = _pivots(lo, left_w, right_w, "low"), _pivots(h, left_w, right_w, "high")
    bull, bear = np.zeros(n), np.zeros(n)
    for name in ind:
        v = ind[name]
        for kind, piv, price in (("bull", pl, lo), ("bear", ph, h)):
            if len(piv) < 2:
                continue
            p1, p2 = piv[:-1], piv[1:]
            gap = (p2 - p1) <= maxgap
            with np.errstate(invalid="ignore"):
                if kind == "bull":
                    cond = gap & (price[p2] < price[p1]) & (v[p2] > v[p1])
                else:
                    cond = gap & (price[p2] > price[p1]) & (v[p2] < v[p1])
            ev = np.zeros(n)
            conf = (p2 + right_w)[cond]
            conf = conf[conf < n]
            ev[conf] = 1.0
            act = ev.copy()  # trailing window of ``life`` bars (causal)
            for k in range(1, life):
                act[k:] = np.maximum(act[k:], ev[:-k])
            (bull if kind == "bull" else bear)[:] += act
    valid = np.isfinite(c)
    return np.where(valid, bull, np.nan), np.where(valid, bear, np.nan)


def ppo(c: np.ndarray, fast: int = 12, slow: int = 26) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return (ewm_span(c, fast, slow) / ewm_span(c, slow, slow) - 1) * 100


def tf_feats(bars: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """The per-timeframe feature bank (field names in ``TF_FIELDS``)."""
    h, lo, c, qv, tbq = (bars[k] for k in ("high", "low", "close", "qv", "tbq"))
    ind = indicators(h, lo, c, qv, tbq)
    bu, be = divergence(h, lo, c, qv, tbq)
    with np.errstate(divide="ignore", invalid="ignore"):
        logc = np.log(c)
        out = {
            "rsi": ind["rsi"],
            "macdh": ind["macdh"] / c,
            "ppo": ppo(c),
            "stoch": ind["stoch"],
            "ich": ichimoku_score(h, lo, c),
            "bull": bu,
            "bear": be,
            "net": bu - be,
            "dsma20": np.log(c / r_mean(c, 20)),
            "dsma50": np.log(c / r_mean(c, 50)),
            "r1": logc - shift(logc, 1),
            "r6": logc - shift(logc, 6),
            "vol": r_std(diff(logc), 30),
        }
    return out


@dataclass(frozen=True)
class Frame:
    """Per-symbol 4h table: ``open_s[i]`` is the open time of bar i (decision at open + 4h)."""

    open_s: np.ndarray
    close: np.ndarray
    x: np.ndarray  # (bars, len(FEATURE_COLUMNS)) float64


def build_frame(hourly: dict[str, np.ndarray], start_s: int) -> Frame:
    """Causal features for one symbol from its full hourly grid (UTC midnight start; NaN holes
    for missing hours; length a multiple of 4)."""
    n_hours = len(hourly["close"])
    if n_hours % 4:
        raise ValueError("hourly grid must hold whole 4h bars")
    b4 = aggregate(hourly, 4, n_hours)
    b1d = aggregate(hourly, 24, n_hours)
    n4 = len(b4["close"])
    cols: dict[str, np.ndarray] = {}
    for name, arr in tf_feats(b4).items():
        cols[f"{name}_4h"] = arr
    h1 = {k: hourly[k] for k in ("high", "low", "close", "qv", "tbq")}
    for name, arr in tf_feats(h1).items():
        # the 1h value of the bar opening at c - 1h is known at the 4h decision close c
        cols[f"{name}_1h"] = arr[3::4][:n4]
    fd = tf_feats(b1d)
    cd = b1d["close"]
    with np.errstate(divide="ignore", invalid="ignore"):
        fd["regime"] = sum(np.sign(cd - r_mean(cd, w)) for w in REGIME_WINDOWS) / 4
        fd["dsma200_1d"] = np.log(cd / r_mean(cd, 200))
        fd["vol_ann"] = r_std(diff(np.log(cd)), 30) * math.sqrt(365)
    # latest COMPLETED daily bar at each 4h close: day index floor(hours since start / 24) - 1
    day_of = (np.arange(n4) + 1) * 4 // 24 - 1
    ok = (day_of >= 0) & (day_of < len(cd))
    for name, arr in fd.items():
        key = name if name in ("regime", "dsma200_1d", "vol_ann") else f"{name}_1d"
        cols[key] = np.where(ok, arr[np.clip(day_of, 0, max(len(cd) - 1, 0))], np.nan)
    x = np.column_stack([cols[k] for k in FEATURE_COLUMNS]).astype(np.float64)
    return Frame(start_s + BAR_S * np.arange(n4), b4["close"], x)


# ----------------------------------------------------------------------------- direction model


def default_model_factory() -> Any:  # pragma: no cover - needs scikit-learn
    ensemble = importlib.import_module("sklearn.ensemble")
    return ensemble.HistGradientBoostingClassifier(**HGB_PARAMS)


def training_set(
    frames: dict[str, Frame], symbols: tuple[str, ...], cutoff_s: int
) -> tuple[np.ndarray, np.ndarray]:
    """Pooled raw (unaligned) features and labels ``close[t+30]/close[t] - 1 > 0``.

    A bar is used only if the bar t+30 that closes its label window OPENS before the cutoff, so
    every hourly bar behind a training row is dated before the cutoff."""
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for sym in symbols:
        fr = frames[sym]
        c = fr.close
        n = len(c)
        fwd = np.full(n, np.nan)
        if n > LABEL_HORIZON:
            with np.errstate(invalid="ignore", divide="ignore"):
                fwd[: n - LABEL_HORIZON] = c[LABEL_HORIZON:] / c[:-LABEL_HORIZON] - 1
        label_open = fr.open_s + LABEL_HORIZON * BAR_S
        sel = (
            np.isfinite(c)
            & np.isfinite(fr.x[:, I_REGIME])
            & np.isfinite(fwd)
            & (label_open < cutoff_s)
        )
        xs.append(fr.x[sel])
        ys.append((fwd[sel] > 0).astype(np.int64))
    return np.vstack(xs), np.concatenate(ys)


def fit_direction_model(
    x: np.ndarray, y: np.ndarray, factory: Callable[[], Any]
) -> tuple[Any, float, float, dict[str, Any]]:
    """Fit once (twice for the determinism check). Returns (model, lo, hi, receipt)."""
    model = factory()
    model.fit(x, y)
    p_train = np.asarray(model.predict_proba(x)[:, 1], dtype="<f8")
    twin = factory()
    twin.fit(x, y)
    p_twin = np.asarray(twin.predict_proba(x)[:, 1], dtype="<f8")
    lo, hi = float(np.quantile(p_train, BAND)), float(np.quantile(p_train, 1 - BAND))
    receipt = {
        "train_rows": int(len(y)),
        "train_positive_rate": float(np.mean(y)),
        "x_sha256": hashlib.sha256(
            np.ascontiguousarray(x, "<f8").tobytes()
        ).hexdigest(),
        "y_sha256": hashlib.sha256(
            np.ascontiguousarray(y, "<i8").tobytes()
        ).hexdigest(),
        "model_prob_sha256": hashlib.sha256(p_train.tobytes()).hexdigest(),
        "deterministic": bool(np.array_equal(p_train, p_twin)),
        "band_lo": lo,
        "band_hi": hi,
        "n_iter": getattr(model, "n_iter_", None),
    }
    return model, lo, hi, receipt


def ml_band(p: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """mlb in {-1, 0, +1}: +1 if p > hi, -1 if p < lo (NaN -> 0)."""
    return np.where(p > hi, 1, np.where(p < lo, -1, 0)).astype(np.int8)


def regime_direction(regime: np.ndarray) -> np.ndarray:
    """Rule direction: +1 if regime >= 0.5, -1 if <= -0.5, else 0 (NaN -> 0)."""
    return np.where(
        regime >= REGIME_THRESHOLD, 1, np.where(regime <= -REGIME_THRESHOLD, -1, 0)
    ).astype(np.int8)


# ----------------------------------------------------------------------------- sleeves


@dataclass(frozen=True)
class Signals:
    """Decision-close inputs. Index j is the 4h close c_j (j = 0..n); ``fund[sym][j]`` is the sum
    of ``mark price * funding rate`` of the funding events paid over bar j (see
    ``funding_coefficients``; 0 at j = 0)."""

    times: np.ndarray
    symbols: tuple[str, ...]
    close: dict[str, np.ndarray]
    rule: dict[str, np.ndarray]
    mlb: dict[str, np.ndarray]
    vol: dict[str, np.ndarray]
    fund: dict[str, np.ndarray]

    @property
    def bars(self) -> int:
        return len(self.times) - 1


def _size(vol_ann: float) -> float:
    return 1.0 if vol_ann <= 0.0 else min(1.0, SIZE_VOL / vol_ann)


def sleeve_a_weights(rule: np.ndarray, vol: np.ndarray) -> np.ndarray:
    """Sleeve A, BTC rule long: direction max(rule, 0), size fixed at entry (no re-sizing).
    Returns the held signed size after each decision."""
    out = np.zeros(len(rule))
    pos = 0.0
    for t in range(len(rule)):
        va = float(vol[t])
        if math.isfinite(va):
            newpos = float(max(int(rule[t]), 0)) * _size(va)
            if pos != 0.0 and np.sign(newpos) == np.sign(pos):
                newpos = pos
            pos = newpos
        out[t] = pos
    return out


def sleeve_b_weights(
    rule: np.ndarray,
    mlb: np.ndarray,
    vol: np.ndarray,
    btc_rule: np.ndarray,
    is_btc: bool,
) -> np.ndarray:
    """Sleeve B for one coin: AGREE long/short with the BTC filter. Enter ``rule`` only when
    rule != 0 and mlb == rule; hold while rule == current and mlb != -current; a non-BTC short
    only while the BTC regime is <= -0.5. Size fixed at entry."""
    out = np.zeros(len(rule))
    pos = 0.0
    for t in range(len(rule)):
        va = float(vol[t])
        if math.isfinite(va):
            cur = int(np.sign(pos))
            r, m = int(rule[t]), int(mlb[t])
            if cur == 0:
                tgt = r if (r != 0 and m == r) else 0
            else:
                tgt = cur if (r == cur and m != -cur) else 0
            if not is_btc and tgt < 0 and int(btc_rule[t]) != -1:
                tgt = 0
            newpos = float(tgt) * _size(va)
            if pos != 0.0 and np.sign(newpos) == np.sign(pos):
                newpos = pos
            pos = newpos
        out[t] = pos
    return out


def components(sig: Signals, kind: str) -> dict[str, tuple[str, np.ndarray]]:
    """Nominal (pre-leverage) signed weights per component, length ``bars`` (decisions 0..n-1).
    Component name -> (symbol, equity-fraction path)."""
    n = sig.bars
    slots = len(sig.symbols)
    out: dict[str, tuple[str, np.ndarray]] = {}
    if kind == "hold":
        return {"H": (BTC, np.ones(n))}
    if kind in ("combo", "btc_rule"):
        w = SLEEVE_WEIGHT if kind == "combo" else 1.0
        out["A"] = (BTC, w * sleeve_a_weights(sig.rule[BTC][:n], sig.vol[BTC][:n]))
    if kind in ("combo", "alt_agree"):
        w = SLEEVE_WEIGHT if kind == "combo" else 1.0
        for s in sig.symbols:
            wb = sleeve_b_weights(
                sig.rule[s][:n],
                sig.mlb[s][:n],
                sig.vol[s][:n],
                sig.rule[BTC][:n],
                s == BTC,
            )
            out[f"B:{s}"] = (s, w * wb / slots)
    return out


# ----------------------------------------------------------------------------- actual-units book


def _clamp_scale(
    new_net: dict[str, float],
    old_net: dict[str, float],
    price: dict[str, float],
    fee: float,
    e_pre: float,
) -> float:
    """Hard gross-leverage cap at a fill. Returns the largest ``s <= 1`` such that the book that
    holds ``s * new_net`` has gross notional at most GROSS_CAP times the post-fee equity (the fee
    itself depends on ``s``). 1.0 when the fill already respects the cap."""
    if not e_pre > 0.0:
        return 1.0

    def excess(s: float) -> float:
        gross = sum(abs(s * u * price[k]) for k, u in new_net.items())
        fees = fee * sum(
            abs(s * u - old_net.get(k, 0.0)) * price[k] for k, u in new_net.items()
        )
        return gross + GROSS_CAP * fees - GROSS_CAP * e_pre

    if excess(1.0) <= 0.0:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(80):
        mid = (lo + hi) / 2.0
        if excess(mid) <= 0.0:
            lo = mid
        else:
            hi = mid
    return lo


class _Book:
    """Actual contracts and cash. Units change only at an accepted change (a component's nominal
    weight changed, or the portfolio leverage changed); between those they are held exactly and
    the marked weights drift. Fees are charged on the actual net fill notional per symbol;
    funding is paid or received on the held notional. A fill whose gross leverage would exceed
    GROSS_CAP after fees is scaled down to the cap (``_clamp_scale``); the post-fill check then
    only asserts that the clamp worked."""

    def __init__(self, fee: float, delay: int, comp_sym: dict[str, str]) -> None:
        self.fee, self.delay, self.comp_sym = fee, delay, comp_sym
        self.units = dict.fromkeys(comp_sym, 0.0)
        self.nominal = dict.fromkeys(comp_sym, 0.0)  # nominal weights currently held
        self.lev = 1.0  # leverage currently applied
        self.dec_nominal = dict.fromkeys(comp_sym, 0.0)  # last DECIDED state
        self.dec_lev = 1.0
        self.pending: list[tuple[int, dict[str, float], float]] = []
        self.cash = 1.0
        self.eq = 1.0
        self.post: list[float] = []
        self.reasons: list[str] = []
        self.err = 0.0
        self.fills = 0
        self.turnover = 0.0
        self.fees = 0.0
        self.funding = 0.0
        self.max_gross_fill = 0.0
        self.max_gross_mark = 0.0
        self.lev_changes = 0
        self.clamps = 0
        self.min_clamp_scale = 1.0

    def net(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for comp, sym in self.comp_sym.items():
            out[sym] = out.get(sym, 0.0) + self.units[comp]
        return out

    def mark(self, j: int, sig: Signals) -> None:
        """Funding of bar j on the units held over it, then mark to the close c_j."""
        net = self.net()
        fund, flow, notional = 0.0, 0.0, 0.0
        for sym, u in net.items():
            p0, p1 = float(sig.close[sym][j - 1]), float(sig.close[sym][j])
            fund -= u * float(sig.fund[sym][j])
            flow += u * (p1 - p0)
            notional += abs(u * p1)
        self.cash += fund
        self.funding += fund
        eq = self.cash + sum(u * float(sig.close[s][j]) for s, u in net.items())
        self.err = max(self.err, abs(eq - (self.eq + flow + fund)))
        self.eq = eq
        if not math.isfinite(eq) or eq <= 0.0:
            self.reasons.append(f"equity {eq!r} not positive at bar {j}")
            return
        self.max_gross_mark = max(self.max_gross_mark, notional / eq)

    def decide(
        self, j: int, nominal: dict[str, float], lev: float, sig: Signals
    ) -> None:
        if nominal == self.dec_nominal and lev == self.dec_lev:
            return
        self.dec_nominal, self.dec_lev = dict(nominal), lev
        if self.delay == 0:
            self._fill(j, nominal, lev, sig)
        else:
            self.pending.append((j + self.delay, dict(nominal), lev))

    def execute_due(self, j: int, sig: Signals) -> None:
        due = [o for o in self.pending if o[0] == j]
        self.pending = [o for o in self.pending if o[0] != j]
        for _, nominal, lev in due:
            self._fill(j, nominal, lev, sig)

    def _fill(
        self, j: int, nominal: dict[str, float], lev: float, sig: Signals
    ) -> None:
        old_net = self.net()
        px = {s: float(sig.close[s][j]) for s in old_net}
        e_pre = self.cash + sum(u * px[s] for s, u in old_net.items())
        cand = dict(self.units)
        for comp, sym in self.comp_sym.items():
            if nominal[comp] != self.nominal[comp]:
                cand[comp] = lev * nominal[comp] * e_pre / px[sym]
            elif lev != self.lev:
                cand[comp] = self.units[comp] * lev / self.lev
        cand_net: dict[str, float] = {}
        for comp, sym in self.comp_sym.items():
            cand_net[sym] = cand_net.get(sym, 0.0) + cand[comp]
        scale = _clamp_scale(cand_net, old_net, px, self.fee, e_pre)
        if scale < 1.0:
            self.clamps += 1
            self.min_clamp_scale = min(self.min_clamp_scale, scale)
        self.units = {k: v * scale for k, v in cand.items()}
        if lev != self.lev:
            self.lev_changes += 1
        self.lev = lev
        self.nominal = dict(nominal)
        new_net = self.net()
        traded, gross = 0.0, 0.0
        cash_delta = 0.0
        for sym, u in new_net.items():
            price = float(sig.close[sym][j])
            d = u - old_net.get(sym, 0.0)
            cash_delta -= d * price
            traded += abs(d) * price
            gross += abs(u * price)
            if abs(d) * price > TRADE_EPS:
                self.fills += 1
        fee = self.fee * traded
        self.cash += cash_delta - fee
        self.fees += fee
        self.turnover += traded / e_pre if e_pre > 0.0 else 0.0
        e_post = self.cash + sum(u * float(sig.close[s][j]) for s, u in new_net.items())
        self.err = max(self.err, abs(e_post - (e_pre - fee)))
        self.eq = e_post
        if not math.isfinite(e_post) or e_post <= 0.0:
            self.reasons.append(f"equity {e_post!r} not positive after fill at bar {j}")
            return
        lev_post = gross / e_post
        self.max_gross_fill = max(self.max_gross_fill, lev_post)
        if lev_post > GROSS_CAP + IDENTITY_TOL:
            self.reasons.append(
                f"gross leverage {lev_post!r} > {GROSS_CAP!r} after fill at bar {j}"
            )


def _stats(equity: np.ndarray) -> dict[str, Any]:
    with np.errstate(divide="ignore", invalid="ignore"):
        rets = equity[1:] / equity[:-1] - 1.0
        peak = np.maximum.accumulate(equity)
        max_dd = float(np.max((peak - equity) / peak))
    sd = float(np.std(rets, ddof=0)) if len(rets) else 0.0
    sharpe = float(np.mean(rets)) / sd * ANN if sd > 0.0 else 0.0
    return {
        "returns": rets,
        "equity": equity,
        "max_dd": max_dd,
        "sharpe": sharpe,
        "total_return": float(equity[-1] - 1.0),
    }


def next_leverage(raw_returns: list[float], current: float) -> float:
    """Portfolio vol targeting on the UNSCALED book (screens' ``q5.scale``): after VOL_BARS
    in-stage bars, L = min(cap, target / sigma) from the trailing VOL_BARS returns, changed only
    when it moves by more than LEVERAGE_BAND."""
    if len(raw_returns) - 1 < VOL_BARS:  # raw_returns[0] is the entry bar, not a mark
        return current
    w = np.asarray(raw_returns[-VOL_BARS:], dtype=float)
    sigma = float(np.std(w, ddof=0)) * ANN
    new = min(LEVERAGE_CAP, TARGET_VOL / sigma) if sigma > VOL_EPS else LEVERAGE_CAP
    if current == 0.0 or abs(new / current - 1.0) > LEVERAGE_BAND:
        return new
    return current


def simulate(sig: Signals, kind: str, fee: float, delay: int) -> dict[str, Any]:
    """Replay one cell. Decisions at closes c_0..c_{n-1}; marks at c_1..c_n; a decision fills at
    its own close (delay 0) or one 4h close later (delay 1)."""
    n = sig.bars
    for s in sig.symbols:
        px = sig.close[s]
        if not np.all(np.isfinite(px)) or np.any(px <= 0.0):
            raise ValueError(f"non-positive or missing close for {s}")
    comps = components(sig, kind)
    comp_sym = {k: v[0] for k, v in comps.items()}
    scaled = kind in ("combo", "btc_rule")
    shadow = _Book(fee, delay, comp_sym)  # unscaled book (leverage 1)
    scaled_book = _Book(fee, delay, comp_sym) if scaled else None
    books = [shadow] + ([scaled_book] if scaled_book else [])
    raw: list[float] = []
    lev = 1.0
    prev_post = 1.0
    for j in range(n + 1):
        if j >= 1:
            for b in books:
                b.mark(j, sig)
        for b in books:
            b.execute_due(j, sig)
        if j <= n - 1:
            nominal = {k: float(v[1][j]) for k, v in comps.items()}
            shadow.decide(j, nominal, 1.0, sig)
            if scaled_book is not None:
                raw.append(shadow.eq / prev_post - 1.0)
                lev = next_leverage(raw, lev)
                scaled_book.decide(j, nominal, lev, sig)
        for b in books:
            b.post.append(b.eq)
        prev_post = shadow.eq
        if any(b.reasons for b in books):
            break
    book = scaled_book if scaled_book is not None else shadow
    equity = np.asarray([1.0] + book.post)
    reasons = list(book.reasons)
    if scaled_book is not None:
        reasons += [f"unscaled book: {r}" for r in shadow.reasons]
    err = max(b.err for b in books)
    if err > IDENTITY_TOL:
        reasons.append(f"accounting identity gap {err!r} > {IDENTITY_TOL!r}")
    if len(book.post) != n + 1 and not reasons:
        reasons.append("incomplete replay")
    out = _stats(equity)
    out.update(
        fills=book.fills,
        turnover=book.turnover,
        fees=book.fees,
        funding=book.funding,
        leverage_changes=book.lev_changes,
        hard_cap_clamps=book.clamps,
        min_clamp_scale=book.min_clamp_scale,
        final_leverage=book.lev,
        max_gross_at_fills=book.max_gross_fill,
        max_held_gross=book.max_gross_mark,
        max_identity_err=err,
        invalid_reasons=reasons,
        valid=not reasons,
    )
    return out


def cash_cell(bars: int) -> dict[str, Any]:
    out = _stats(np.ones(bars + 2))
    out.update(
        fills=0,
        turnover=0.0,
        fees=0.0,
        funding=0.0,
        leverage_changes=0,
        hard_cap_clamps=0,
        min_clamp_scale=1.0,
        final_leverage=0.0,
        max_gross_at_fills=0.0,
        max_held_gross=0.0,
        max_identity_err=0.0,
        invalid_reasons=[],
        valid=True,
    )
    return out


def summarize_cell(cell: dict[str, Any]) -> dict[str, Any]:
    """JSON-safe cell record: stats plus digests of the equity series."""
    out = {k: v for k, v in cell.items() if k not in ("returns", "equity")}
    out["bars"] = int(len(cell["returns"]))
    out["equity_sha256"] = hashlib.sha256(
        np.asarray(cell["equity"], dtype="<f8").tobytes()
    ).hexdigest()
    return out


def evaluate_cells(sig: Signals) -> dict[str, dict[str, Any]]:
    cells: dict[str, dict[str, Any]] = {}
    for name, (kind, fee, delay, _gate) in CELLS.items():
        if kind == "cash":
            cells[name] = cash_cell(sig.bars)
        else:
            cells[name] = simulate(sig, kind, fee, delay)
    return cells


# ----------------------------------------------------------------------------- decision


def supported(cells: dict[str, dict[str, Any]]) -> tuple[bool, list[str]]:
    """Every condition of the decision rule. Returns (ok, failed conditions)."""
    failed: list[str] = []
    if any(not c["valid"] for c in cells.values()):
        failed.append("all cells valid")
    if not cells[GATING_PRIMARY]["total_return"] > 0.0:
        failed.append(f"{GATING_PRIMARY} total return > 0")
    if not cells[GATING_PRIMARY]["sharpe"] >= MIN_SHARPE:
        failed.append(f"{GATING_PRIMARY} sharpe >= {MIN_SHARPE}")
    for name in GATING_POSITIVE:
        if not cells[name]["total_return"] > 0.0:
            failed.append(f"{name} total return > 0")
    for name, spec in CELLS.items():
        if spec[3] and not cells[name]["max_dd"] < MAX_DD:
            failed.append(f"{name} max DD < {MAX_DD}")
    return not failed, failed


def decide(
    stage: Stage | str,
    cells: dict[str, dict[str, Any]],
    problems: list[str] | None = None,
) -> dict[str, Any]:
    """Label. Priority: INVALID (any problem or invalid cell), then interim STOPPED_EARLY /
    INTERIM_NOT_A_DECISION, then the final rule."""
    st = STAGES[stage] if isinstance(stage, str) else stage
    problems = list(problems or [])
    invalid = sorted(k for k, c in cells.items() if not c["valid"])
    ok, failed = supported(cells) if cells else (False, ["no cells"])
    out: dict[str, Any] = {
        "invalid_cells": invalid,
        "problems": problems,
        "failed_conditions": failed,
        "supported": ok and not problems,
    }
    if invalid or problems or not cells:
        out["label"] = LABEL_INVALID
        out["supported"] = False
    elif st.interim:
        stopped = cells[GATING_PRIMARY]["max_dd"] >= EARLY_STOP_DD
        out["label"] = LABEL_STOPPED if stopped else LABEL_INTERIM
    else:
        out["label"] = LABEL_SUPPORTED if ok else LABEL_NOT_SUPPORTED
    return out


# ----------------------------------------------------------------------------- data parsing


def parse_klines(
    payload: Any,
) -> list[tuple[int, float, float, float, float, float, float]]:
    """Binance klines page -> [(open_ms, open, high, low, close, quote_volume, taker_buy_quote)]."""
    if not isinstance(payload, list):
        raise SystemExit("malformed Binance klines payload")
    rows = []
    for it in payload:
        try:
            source_open = _native_timestamp(it[0])
            # Preserve exact sub-ms source finality while normalizing the grid.
            if source_open >= 10**15:
                if source_open % 1000:
                    raise ValueError("microsecond open time is not ms aligned")
                open_ms = source_open // 1000
            else:
                open_ms = source_open
            if open_ms % HOUR_MS:
                raise ValueError("source open time is not on a UTC hour")
            _check_source_close_time(
                it,
                open_ms=open_ms,
                interval_ms=HOUR_MS,
                source_close_policy=SOURCE_CLOSE_POLICY,
            )
        except (IndexError, KeyError, TypeError, ValueError) as error:
            raise SystemExit(f"malformed Binance source close_time: {error}") from None
        try:
            o, h, lo, c = (float(it[i]) for i in (1, 2, 3, 4))
            qv, tbq = float(it[7]), float(it[10])
        except (IndexError, KeyError, TypeError, ValueError):
            raise SystemExit("malformed Binance klines row") from None
        rows.append((open_ms, o, h, lo, c, qv, tbq))
    return rows


def parse_funding(payload: Any) -> list[tuple[int, float, float]]:
    """Binance fundingRate page -> [(funding time ms, rate, mark price or NaN)]."""
    if not isinstance(payload, list):
        raise SystemExit("malformed Binance funding payload")
    out = []
    for it in payload:
        try:
            t, rate = int(it["fundingTime"]), float(it["fundingRate"])
        except (KeyError, TypeError, ValueError):
            raise SystemExit("malformed Binance funding row") from None
        raw_mark = it.get("markPrice") if isinstance(it, dict) else None
        try:
            mark = float(raw_mark) if raw_mark not in (None, "") else math.nan
        except (TypeError, ValueError):
            mark = math.nan
        out.append((t, rate, mark))
    return out


def hourly_arrays(
    rows: list[tuple[int, float, float, float, float, float, float]],
    start_ms: int,
    last_open_ms: int,
) -> dict[str, np.ndarray]:
    """Full hourly grid start_ms..last_open_ms with NaN holes for missing hours."""
    n = (last_open_ms - start_ms) // HOUR_MS + 1
    keys = ("open", "high", "low", "close", "qv", "tbq")
    arrs = {k: np.full(n, np.nan) for k in keys}
    for row in rows:
        i = (row[0] - start_ms) // HOUR_MS
        if 0 <= i < n:
            for k, v in zip(keys, row[1:], strict=True):
                arrs[k][i] = v
    return arrs


# ----------------------------------------------------------------------------- data fetching


def _http_get_once(url: str) -> bytes:  # pragma: no cover - real network
    if not url.startswith("https://"):
        raise SystemExit(f"refusing non-https URL {url}")
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_S) as resp:  # noqa: S310
        data: bytes = resp.read()
    return data


_TRANSIENT_HTTP = {418, 429, 500, 502, 503, 504}


def retrying(
    once: Callable[[str], bytes],
    sleep: Callable[[float], None] = time.sleep,
    attempts: int = HTTP_ATTEMPTS,
) -> Callable[[str], bytes]:
    """Retry transient transport failures with exponential backoff. A retry re-requests the same
    window, so it cannot change which bars are read."""

    def fetch(url: str) -> bytes:
        for k in range(attempts):
            try:
                return once(url)
            except urllib.error.HTTPError as exc:
                if exc.code not in _TRANSIENT_HTTP or k == attempts - 1:
                    raise SystemExit(f"HTTP {exc.code} for {url}") from None
            except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                if k == attempts - 1:
                    raise SystemExit(f"network failure for {url}: {exc}") from None
            sleep(min(120.0, 5.0 * 2**k))
        raise SystemExit(f"unreachable retry state for {url}")  # pragma: no cover

    return fetch


_last_request = [0.0]


def _http_get(url: str) -> bytes:  # pragma: no cover - real network
    wait = HTTP_PACE_S - (time.monotonic() - _last_request[0])
    if wait > 0:
        time.sleep(wait)
    try:
        return retrying(_http_get_once)(url)
    finally:
        _last_request[0] = time.monotonic()


def _save_page(raw_dir: Path, name: str, body: bytes) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / name
    with path.open("xb") as f:
        f.write(body)
    return path


def fetch_klines(
    symbol: str,
    start_ms: int,
    last_open_ms: int,
    raw_dir: Path,
    fetch: Callable[[str], bytes] = _http_get,
) -> tuple[
    list[tuple[int, float, float, float, float, float, float]], list[dict[str, Any]]
]:
    """Hourly klines for the hourly open times start_ms..last_open_ms. Every request spans at most
    KLINES_LIMIT hours and rows outside its window are dropped, so the paging is correct whether
    or not the API honours ``startTime`` (precedent: btc_trend_ensemble Amendment 1)."""
    rows: list[tuple[int, float, float, float, float, float, float]] = []
    pages: list[dict[str, Any]] = []
    seen: set[int] = set()
    cursor = start_ms
    for page_no in range(MAX_PAGES + 1):
        if cursor > last_open_ms:
            break
        if page_no == MAX_PAGES:
            raise SystemExit(f"{symbol} klines paging exceeded {MAX_PAGES} pages")
        page_end = min(last_open_ms + HOUR_MS - 1, cursor + KLINES_LIMIT * HOUR_MS - 1)
        url = (
            f"{KLINES_URL}?symbol={symbol}&interval=1h&startTime={cursor}"
            f"&endTime={page_end}&limit={KLINES_LIMIT}"
        )
        body = fetch(url)
        path = _save_page(raw_dir, f"{symbol}_klines_{page_no:03d}.json", body)
        try:
            payload = json.loads(body)
        except ValueError:
            raise SystemExit(f"{symbol} klines page {page_no} is not JSON") from None
        page = [r for r in parse_klines(payload) if cursor <= r[0] <= last_open_ms]
        for r in page:
            if r[0] in seen:
                raise SystemExit(f"duplicate {symbol} hourly bar at {r[0]}")
            seen.add(r[0])
        rows.extend(page)
        pages.append(
            {
                "file": path.name,
                "url": url,
                "sha256": hashlib.sha256(body).hexdigest(),
                "rows": len(page),
            }
        )
        cursor = page_end + 1
    return rows, pages


def fetch_funding(
    symbol: str,
    start_ms: int,
    end_ms: int,
    raw_dir: Path,
    fetch: Callable[[str], bytes] = _http_get,
) -> tuple[list[tuple[int, float, float]], list[dict[str, Any]]]:
    """Funding events with start_ms <= fundingTime <= end_ms (rows outside are dropped)."""
    events: list[tuple[int, float, float]] = []
    pages: list[dict[str, Any]] = []
    seen: set[int] = set()
    cursor = start_ms
    for page_no in range(MAX_PAGES):
        url = (
            f"{FUNDING_URL}?symbol={symbol}&startTime={cursor}&endTime={end_ms}"
            f"&limit={FUNDING_LIMIT}"
        )
        body = fetch(url)
        path = _save_page(raw_dir, f"{symbol}_funding_{page_no:03d}.json", body)
        try:
            payload = json.loads(body)
        except ValueError:
            raise SystemExit(f"{symbol} funding page {page_no} is not JSON") from None
        parsed = parse_funding(payload)
        page = [e for e in parsed if cursor <= e[0] <= end_ms]
        for e in page:
            if e[0] in seen:
                raise SystemExit(f"duplicate {symbol} funding event at {e[0]}")
            seen.add(e[0])
        events.extend(page)
        pages.append(
            {
                "file": path.name,
                "url": url,
                "sha256": hashlib.sha256(body).hexdigest(),
                "rows": len(page),
            }
        )
        if len(parsed) < FUNDING_LIMIT:
            break
        newest = max(e[0] for e in parsed)
        if newest >= end_ms:
            break
        if newest + 1 <= cursor:
            raise SystemExit(f"{symbol} funding paging made no progress at {cursor}")
        cursor = newest + 1
    else:
        raise SystemExit(f"{symbol} funding paging exceeded {MAX_PAGES} pages")
    events.sort()
    return events, pages


def funding_coverage_problems(
    symbol: str, events: list[tuple[int, float, float]], first_s: int, end_s: int
) -> list[str]:
    """The funding history must cover [first_s, end_s] without a gap above 8h (+1 minute)."""
    times = [e[0] / 1000.0 for e in events]
    if not times:
        return [f"{symbol}: no funding events in the stage window"]
    problems = []
    if times[0] - first_s > MAX_FUNDING_GAP_S:
        problems.append(f"{symbol}: funding starts late ({times[0] - first_s:.0f} s)")
    if end_s - times[-1] > MAX_FUNDING_GAP_S:
        problems.append(f"{symbol}: funding ends early ({end_s - times[-1]:.0f} s)")
    gaps = np.diff(times)
    if len(gaps) and float(gaps.max()) > MAX_FUNDING_GAP_S:
        problems.append(f"{symbol}: funding gap {float(gaps.max()):.0f} s")
    return problems


def funding_coefficients(
    events: list[tuple[int, float, float]],
    hour_open: np.ndarray,
    grid_start_ms: int,
    first_close_s: int,
    bars: int,
) -> tuple[np.ndarray, int]:
    """coef[j] = sum of (mark price * funding rate) over events with c_{j-1} <= t < c_j.

    A fill at c_{j-1} happens before an event stamped at or just after c_{j-1}, so the units held
    after that fill pay it. The price is the record's mark price, else the 1h open of the funding
    hour (count of fallbacks returned)."""
    coef = np.zeros(bars + 1)
    fallbacks = 0
    c0_ms = first_close_s * 1000
    for t_ms, rate, mark in events:
        if t_ms < c0_ms or t_ms >= c0_ms + bars * BAR_MS:
            continue
        j = int((t_ms - c0_ms) // BAR_MS) + 1
        price = mark
        if not (math.isfinite(price) and price > 0.0):
            fallbacks += 1
            price = float(hour_open[(t_ms - grid_start_ms) // HOUR_MS])
        coef[j] += price * rate
    return coef, fallbacks


# ----------------------------------------------------------------------------- authorization


def _gh_comment(comment_id: int) -> dict[str, Any]:  # pragma: no cover - real network
    out = subprocess.run(
        ["gh", "api", f"repos/{REPO}/issues/comments/{comment_id}"],
        capture_output=True,
        text=True,
        check=True,
    )
    parsed: dict[str, Any] = json.loads(out.stdout)
    return parsed


_COMMENT_URL = re.compile(
    r"https://github\.com/shuntatsu/trade_rl/issues/810#issuecomment-(\d+)"
)
_COMMIT_RE = re.compile(r"[0-9a-f]{40}")


def comment_id_from_url(url: str) -> int:
    m = _COMMENT_URL.fullmatch(url)
    if m is None:
        raise SystemExit("authorization URL must be an Issue #810 comment")
    return int(m.group(1))


def _git(*args: str) -> str:  # pragma: no cover - real git
    out = subprocess.run(
        ["git", "-C", str(REPO_ROOT), *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout


def harness_commit() -> str:
    """Exact HEAD of the repository holding this file; refuses on modified or untracked files."""
    # untracked files count (a stray module could change behaviour); .gitignore is respected
    if _git("status", "--porcelain", "--untracked-files=all").strip():
        raise SystemExit(
            "harness repository is not clean (modified or untracked files); commit first"
        )
    commit = _git("rev-parse", "HEAD").strip()
    if not _COMMIT_RE.fullmatch(commit):
        raise SystemExit(f"unexpected HEAD from git: {commit!r}")
    return commit


def authorization_receipt(
    url: str,
    digest: str,
    commit: str,
    claim: str,
    fetch: Callable[[int], dict[str, Any]] = _gh_comment,
) -> dict[str, Any]:
    if not _COMMIT_RE.fullmatch(commit):
        raise SystemExit("commit must be a full 40-hex SHA")
    out: dict[str, Any] = {}
    for role, cid in (
        ("g4_relay", comment_id_from_url(url)),
        ("data_scope", DATA_SCOPE_COMMENT_ID),
    ):
        first, second = fetch(cid), fetch(cid)
        if first["user"]["login"] != OWNER_LOGIN:
            raise SystemExit(f"{role} comment {cid} is not owner-authored")
        if first.get("issue_url") != ISSUE_URL:
            raise SystemExit(f"{role} comment {cid} is not on Issue #810")
        if (
            first["body"] != second["body"]
            or first["updated_at"] != second["updated_at"]
        ):
            raise SystemExit(f"{role} comment {cid} changed between fetches")
        if role == "g4_relay":
            if not all(
                token in first["body"] for token in (PROTOCOL_ID, digest, commit)
            ):
                raise SystemExit(
                    "G4 relay does not quote this protocol id, digest and exact commit"
                )
            # One explicit, unambiguous claim line; citing another stage in prose
            # or listing several stages must never admit the current stage.
            claims = [
                line
                for line in first["body"].splitlines()
                if re.match(r"^\s*G4\s+claim\s*:", line, re.IGNORECASE)
            ]
            if claims != [f"G4 claim: {claim}"]:
                raise SystemExit(f"G4 claim must name exactly this stage: {claim}")
        out[role] = {
            "comment_id": cid,
            "author": first["user"]["login"],
            "updated_at": first["updated_at"],
            "body_sha256": hashlib.sha256(first["body"].encode("utf-8")).hexdigest(),
        }
    return out


# ----------------------------------------------------------------------------- ledger & clocks


def claim_name(stage: Stage, now: datetime) -> str:
    """Ledger name: interims are once per UTC month (the month of ``now`` in UTC)."""
    now = _utc(now)
    return f"{stage.name}_{now:%Y-%m}" if stage.interim else stage.name


def _utc(now: datetime) -> datetime:
    """Refuse naive datetimes; normalise aware ones to UTC."""
    if now.tzinfo is None or now.utcoffset() is None:
        raise SystemExit("now must be a timezone-aware datetime")
    return now.astimezone(UTC)


def check_clock(stage: Stage, now: datetime) -> None:
    if stage.not_before is not None and now < stage.not_before:
        raise SystemExit(
            f"{stage.name} may not run before {stage.not_before.isoformat()}"
        )
    if stage.not_after is not None and now >= stage.not_after:
        raise SystemExit(
            f"{stage.name} may not run at or after {stage.not_after.isoformat()}"
        )


def _publish_claim(path: Path, record: dict[str, Any]) -> None:
    """Publish complete JSON without replacement; never expose a partial *.claim."""
    pending: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=".pending-claim-",
            suffix=".tmp",
            delete=False,
        ) as f:
            pending = Path(f.name)
            f.write(json.dumps(record, indent=1, sort_keys=True))
            f.flush()
            os.fsync(f.fileno())
        # A same-directory hard link is exclusive on supporting filesystems.
        # Unlike replace, it can never overwrite an earlier claim.
        os.link(pending, path)
    finally:
        if pending is not None:
            pending.unlink(missing_ok=True)


def claim_stage(
    name: str,
    digest: str,
    commit: str,
    auth_comment_id: int,
    now: datetime,
    ledger_dir: Path = LEDGER_DIR,
) -> Path:
    """Local single-use enforcement: exclusive-create ``<ledger>/<name>.claim``."""
    ledger_dir.mkdir(parents=True, exist_ok=True)
    path = ledger_dir / f"{name}.claim"
    try:
        _publish_claim(
            path,
            {
                "stage": name,
                "protocol_digest": digest,
                "commit": commit,
                "authorization_comment_id": auth_comment_id,
                "claimed_at": now.isoformat(),
            },
        )
    except FileExistsError:
        raise SystemExit(f"stage {name} was already claimed: {path}") from None
    return path


def _validate_g4_ledger_root(root: Path) -> None:
    if not root.is_absolute():
        raise SystemExit("G4 ledger root must be a real absolute directory")
    for part in reversed((root, *root.parents)):
        try:
            mode = part.lstat().st_mode
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise SystemExit(f"unreadable G4 ledger root: {part}") from exc
        if not stat.S_ISDIR(mode) or part.is_junction():
            raise SystemExit("G4 ledger root must be a real absolute directory")


def _historical_claim_paths(root: Path) -> list[Path]:
    """Enumerate the trusted local ledger without silently skipping linked subtrees."""
    _validate_g4_ledger_root(root)
    try:
        root.lstat()
    except FileNotFoundError:
        return []
    except OSError as exc:
        raise SystemExit(f"unreadable historical G4 ledger: {root}") from exc
    pending = [root]
    claims: list[Path] = []
    while pending:
        directory = pending.pop()
        try:
            for path in directory.iterdir():
                mode = path.lstat().st_mode
                if stat.S_ISDIR(mode) and not path.is_junction():
                    if path.name.endswith(".claim"):
                        raise SystemExit(f"unsafe historical G4 claim path: {path}")
                    pending.append(path)
                elif not stat.S_ISREG(mode):
                    raise SystemExit(f"unsafe historical G4 ledger entry: {path}")
                elif path.name.endswith(".claim"):
                    claims.append(path)
        except OSError as exc:
            raise SystemExit(f"unreadable historical G4 ledger: {directory}") from exc
    return sorted(claims)


def _unique_claim_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    record: dict[str, Any] = {}
    for key, value in pairs:
        if key in record:
            raise ValueError("duplicate historical G4 claim key")
        record[key] = value
    return record


def _check_g4_comment_available(
    root: Path, comment_id: int, *, current_claim: Path | None = None
) -> None:
    """Read-only preflight; reject known invalid or consumed authorization."""
    if type(comment_id) is not int or comment_id <= 0:
        raise SystemExit("invalid G4 comment ID")
    for path in _historical_claim_paths(root):
        if path == current_claim:
            continue  # only the root claim created by this attempt
        if path.is_symlink() or not path.is_file():
            raise SystemExit(f"unsafe historical G4 claim path: {path}")
        try:
            record = json.loads(
                path.read_text(encoding="utf-8"), object_pairs_hook=_unique_claim_keys
            )
        except (OSError, ValueError) as exc:
            raise SystemExit(f"unreadable historical G4 claim: {path}") from exc
        if (
            not isinstance(record, dict)
            or type(record.get("authorization_comment_id")) is not int
            or record["authorization_comment_id"] <= 0
        ):
            raise SystemExit(f"invalid historical G4 claim: {path}")
        if record["authorization_comment_id"] == comment_id:
            raise SystemExit(f"G4 comment {comment_id} already consumed: {path}")


def reserve_g4_comment(
    root: Path,
    comment_id: int,
    name: str,
    digest: str,
    commit: str,
    now: datetime,
) -> Path:
    """Consume one owner G4 relay ID per protocol, including historical digests.

    Recheck after the protocol-root stage claim and before stage data access.
    A new conflict does not undo the stage claim.
    This deliberate fail-closed behavior avoids double use on a retry.
    """
    _check_g4_comment_available(root, comment_id, current_claim=root / f"{name}.claim")
    root.mkdir(parents=True, exist_ok=True)
    marker = root / f"g4-{comment_id}.claim"
    try:
        _publish_claim(
            marker,
            {
                "authorization_comment_id": comment_id,
                "stage": name,
                "protocol_digest": digest,
                "commit": commit,
                "claimed_at": now.isoformat(),
            },
        )
    except FileExistsError:
        raise SystemExit(
            f"G4 comment {comment_id} already consumed: {marker}"
        ) from None
    return marker


def write_ledger_result(
    name: str,
    digest: str,
    label: str,
    supported_flag: bool,
    result_sha256: str,
    ledger_dir: Path = LEDGER_DIR,
    model_prob_sha256: str | None = None,
) -> Path:
    path = ledger_dir / f"{name}.result.json"
    try:
        f = path.open("x", encoding="utf-8")
    except FileExistsError:
        raise SystemExit(f"ledger result already exists: {path}") from None
    with f:
        f.write(
            json.dumps(
                {
                    "stage": name,
                    "protocol_digest": digest,
                    "label": label,
                    "supported": supported_flag,
                    "result_sha256": result_sha256,
                    "model_prob_sha256": model_prob_sha256,
                },
                indent=1,
                sort_keys=True,
            )
        )
    return path


def ledger_results(ledger_dir: Path) -> list[dict[str, Any]]:
    if not ledger_dir.exists():
        return []
    out: list[dict[str, Any]] = [
        json.loads(p.read_text(encoding="utf-8"))
        for p in sorted(ledger_dir.glob("*.result.json"))
    ]
    return out


def _namespaces(root: Path) -> list[Path]:
    """Every per-digest provenance folder under the protocol-id ledger root."""
    if not root.exists():
        return []
    return sorted(p for p in root.iterdir() if p.is_dir())


def check_cross_digest(root: Path, stage: Stage, name: str) -> None:
    """Single-use and stop rules are keyed by protocol id + stage (+ interim month), across ALL
    digest namespaces under ``root``: a harness commit changes the digest but never reopens a
    consumed stage or a stopped line. Refuses before any claim, fetch or fit."""
    if (root / f"{name}.claim").exists():
        raise SystemExit(
            f"stage {name} was already claimed at the protocol root: single use applies across digests"
        )
    for ns in _namespaces(root):
        results = ledger_results(ns)
        for led in results:
            if led.get("label") == LABEL_STOPPED:
                raise SystemExit(
                    f"line stopped by {led.get('stage')} (digest {ns.name}): "
                    "no further interim or final runs"
                )
        used = (
            (ns / f"{name}.claim").exists()
            or (ns / f"{name}.result.json").exists()
            or any(led.get("stage") == name for led in results)
        )
        if used:
            raise SystemExit(
                f"stage {name} was already claimed under digest {ns.name}: "
                "single use applies across digests"
            )


# ----------------------------------------------------------------------------- stage run


def _write_once(path: Path, obj: dict[str, Any]) -> None:
    try:
        f = path.open("x", encoding="utf-8")
    except FileExistsError:
        raise SystemExit(f"refusing to overwrite {path}") from None
    with f:
        f.write(json.dumps(obj, indent=1, sort_keys=True))


def _epoch(d: datetime) -> int:
    return int(d.timestamp())


def stage_window(stage: Stage, now: datetime) -> datetime:
    """Last mark close c_n. Interims end at the last complete 4h close older than SETTLE_S."""
    last = stage.decision_end
    if stage.interim:
        cut = _epoch(now) - SETTLE_S
        last = min(last, datetime.fromtimestamp(cut // BAR_S * BAR_S, UTC))
    if (last - stage.decision_start).total_seconds() < BAR_S:
        raise SystemExit(f"no complete 4h bar after {stage.decision_start.isoformat()}")
    return last


def stage_problems(
    hourly: dict[str, dict[str, np.ndarray]], first_hour_idx: int
) -> list[str]:
    """In-stage data validity: no missing hour and no non-positive price from the open of the
    first decision bar onward. (Earlier holes only blank the affected feature bars.)"""
    problems = []
    for sym, arrs in hourly.items():
        close = arrs["close"][first_hour_idx:]
        missing = int(np.sum(~np.isfinite(close)))
        if missing:
            problems.append(f"{sym}: {missing} missing hour(s) in the stage window")
        for key in ("open", "high", "low", "close"):
            v = arrs[key][first_hour_idx:]
            if np.any(np.isfinite(v) & (v <= 0.0)):
                problems.append(f"{sym}: non-positive {key} in the stage window")
    return problems


def build_signals(
    frames: dict[str, Frame],
    symbols: tuple[str, ...],
    first_close_s: int,
    bars: int,
    model: Any,
    lo: float,
    hi: float,
    fund: dict[str, np.ndarray],
) -> Signals:
    times = first_close_s + BAR_S * np.arange(bars + 1)
    close: dict[str, np.ndarray] = {}
    rule: dict[str, np.ndarray] = {}
    mlb: dict[str, np.ndarray] = {}
    vol: dict[str, np.ndarray] = {}
    for sym in symbols:
        fr = frames[sym]
        i0 = int((first_close_s - BAR_S - int(fr.open_s[0])) // BAR_S)
        sl = slice(i0, i0 + bars + 1)
        close[sym] = fr.close[sl]
        x = fr.x[sl]
        rule[sym] = regime_direction(x[:, I_REGIME])
        vol[sym] = x[:, I_VOL_ANN]
        p = np.asarray(model.predict_proba(x)[:, 1], dtype=float)
        mlb[sym] = ml_band(p, lo, hi)
    return Signals(times, symbols, close, rule, mlb, vol, fund)


def library_versions() -> dict[str, str | None]:  # pragma: no cover - environment
    out: dict[str, str | None] = {}
    for pkg in ("numpy", "scikit-learn"):
        try:
            out[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            out[pkg] = None
    return out


def run_stage(
    stage_name: Stage | str,
    authorization_url: str,
    work: Path,
    output_dir: Path,
    now: datetime | None = None,
    fetch: Callable[[str], bytes] = _http_get,
    gh_fetch: Callable[[int], dict[str, Any]] = _gh_comment,
    ledger_dir: Path = LEDGER_DIR,
    model_factory: Callable[[], Any] | None = None,
    symbols: tuple[str, ...] = SYMBOLS,
) -> dict[str, Any]:
    stage = STAGES[stage_name] if isinstance(stage_name, str) else stage_name
    now = _utc(now or datetime.now(UTC))
    if not ledger_dir.is_absolute():
        raise SystemExit(f"ledger directory must be an absolute path: {ledger_dir}")
    factory = model_factory or default_model_factory
    check_clock(stage, now)
    if output_dir.exists():
        raise SystemExit(f"output directory exists: {output_dir}")
    if work.exists() and any(work.iterdir()):
        raise SystemExit(f"work directory is not empty: {work}")
    last_close = stage_window(stage, now)
    digest = protocol_digest()
    # Claims and the stop state are keyed by protocol id + stage (interims once per UTC month)
    # across every digest; the per-digest subfolder only keeps provenance.
    name = claim_name(stage, now)
    ledger_root = ledger_dir / PROTOCOL_ID
    _historical_claim_paths(ledger_root)
    check_cross_digest(ledger_root, stage, name)
    ledger_dir = ledger_root / digest
    commit = harness_commit()
    auth = authorization_receipt(authorization_url, digest, commit, name, gh_fetch)
    _check_g4_comment_available(ledger_root, auth["g4_relay"]["comment_id"])
    # The protocol-root claim is the atomic single-use lock across digests (exclusive create);
    # the per-digest copy only records provenance.
    claim_stage(name, digest, commit, auth["g4_relay"]["comment_id"], now, ledger_root)
    reserve_g4_comment(
        ledger_root, auth["g4_relay"]["comment_id"], name, digest, commit, now
    )
    claim = claim_stage(
        name, digest, commit, auth["g4_relay"]["comment_id"], now, ledger_dir
    )

    first_close_s = _epoch(stage.decision_start)
    last_close_s = _epoch(last_close)
    bars = (last_close_s - first_close_s) // BAR_S
    grid_start_ms = _epoch(stage.history_start) * 1000
    last_open_ms = (last_close_s - HOUR_S) * 1000
    raw_dir = work / "raw"

    hourly: dict[str, dict[str, np.ndarray]] = {}
    funding_events: dict[str, list[tuple[int, float, float]]] = {}
    data_pages: dict[str, dict[str, list[dict[str, Any]]]] = {}
    history_missing: dict[str, int] = {}
    for sym in symbols:
        rows, kpages = fetch_klines(sym, grid_start_ms, last_open_ms, raw_dir, fetch)
        hourly[sym] = hourly_arrays(rows, grid_start_ms, last_open_ms)
        events, fpages = fetch_funding(
            sym, first_close_s * 1000, last_close_s * 1000 - 1, raw_dir, fetch
        )
        funding_events[sym] = events
        data_pages[sym] = {"klines": kpages, "funding": fpages}
        history_missing[sym] = int(np.sum(~np.isfinite(hourly[sym]["close"])))

    first_hour_idx = (first_close_s * 1000 - grid_start_ms) // HOUR_MS - 4
    problems = stage_problems(hourly, first_hour_idx)
    for sym in symbols:
        problems += funding_coverage_problems(
            sym, funding_events[sym], first_close_s, last_close_s
        )

    result: dict[str, Any] = {
        "protocol_id": PROTOCOL_ID,
        "protocol_digest": digest,
        "commit": commit,
        "stage": stage.name,
        "claim_name": name,
        "claim": str(claim),
        "authorization": auth,
        "libraries": library_versions(),
        "data": {
            "symbols": list(symbols),
            "pages": data_pages,
            "history_missing_hours": history_missing,
            "closes_sha256": {
                s: hashlib.sha256(
                    hourly[s]["close"].astype("<f8").tobytes()
                ).hexdigest()
                for s in symbols
            },
        },
        "window": {
            "history_start": stage.history_start.isoformat(),
            "first_decision_close": stage.decision_start.isoformat(),
            "last_mark_close": last_close.isoformat(),
            "bars": int(bars),
        },
        "run_at": now.isoformat(),
    }
    cells: dict[str, dict[str, Any]] = {}
    model_hash: str | None = None
    if not problems:
        start_s = grid_start_ms // 1000
        frames = {s: build_frame(hourly[s], start_s) for s in symbols}
        x, y = training_set(frames, symbols, first_close_s)
        model, lo, hi, model_receipt = fit_direction_model(x, y, factory)
        model_hash = model_receipt["model_prob_sha256"]
        result["model"] = model_receipt
        if not model_receipt["deterministic"]:
            problems.append("direction model is not deterministic on identical data")
        for ns in _namespaces(
            ledger_root
        ):  # model drift is checked across every digest
            for led in ledger_results(ns):
                prior = led.get("model_prob_sha256")
                if prior and prior != model_hash:
                    problems.append(
                        f"model probabilities differ from {led.get('stage')} "
                        f"(digest {ns.name}): model drift"
                    )
        fund: dict[str, np.ndarray] = {}
        fallbacks: dict[str, int] = {}
        for sym in symbols:
            fund[sym], fallbacks[sym] = funding_coefficients(
                funding_events[sym],
                hourly[sym]["open"],
                grid_start_ms,
                first_close_s,
                int(bars),
            )
        result["data"]["funding_mark_fallbacks"] = fallbacks
        sig = build_signals(
            frames, symbols, first_close_s, int(bars), model, lo, hi, fund
        )
        cells = evaluate_cells(sig)
    decision = decide(stage, cells, problems)
    result.update(
        {"cells": {k: summarize_cell(v) for k, v in cells.items()}, **decision}
    )
    output_dir.mkdir(parents=True)
    _write_once(output_dir / "result.json", result)
    write_ledger_result(
        name,
        digest,
        result["label"],
        result["supported"],
        sha256_file(output_dir / "result.json"),
        ledger_dir,
        model_hash,
    )
    return result


# ----------------------------------------------------------------------------- CLI


def cmd_digest(_args: argparse.Namespace) -> None:
    print(protocol_digest())


def cmd_run(args: argparse.Namespace) -> None:  # pragma: no cover - real network
    result = run_stage(
        args.stage, args.authorization_url, Path(args.work), Path(args.output_dir)
    )
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "stage",
                    "label",
                    "invalid_cells",
                    "problems",
                    "failed_conditions",
                )
            },
            indent=1,
        )
    )
    for name, st in result["cells"].items():
        print(
            f"{name:18s} ret {st['total_return'] * 100:+8.2f}% dd {st['max_dd'] * 100:5.2f}% "
            f"sr {st['sharpe']:+.2f} valid {st['valid']}"
        )


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="mtf_combo_forward")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("digest").set_defaults(fn=cmd_digest)
    p = sub.add_parser("run")
    p.add_argument("--stage", required=True, choices=sorted(STAGES))
    p.add_argument("--authorization-url", required=True)
    p.add_argument("--work", required=True)
    p.add_argument("--output-dir", required=True)
    p.set_defaults(fn=cmd_run)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
