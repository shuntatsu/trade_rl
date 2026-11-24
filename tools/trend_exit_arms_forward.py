"""Trend short-exit arms, protocol ``trend_exit_arms_v1``.

Protocol: docs/specs/trend-exit-arms-forward.md (the spec is the authority for every constant).

A prospective FORWARD test of four frozen variants of a daily-regime trend book on 11 Binance
USDT-M perpetuals. The variants share the entries, the sizing, the long leg and the account; they
differ only in the exit rule of the short leg. Stages (``STAGES``) fetch public 1h klines and
funding history, build the causal daily regime, replay twelve frozen cells (four arms x three
variants) and two reference cells on an actual-units book with the repository's execution-cost
formula, and write a write-once result. A local ledger claim is created exclusively before any
data is fetched. Importing this module needs only the standard library and numpy; the network code
runs only in real ``run`` invocations.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import re
import subprocess
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

PROTOCOL_ID = "trend_exit_arms_v1"
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
SIZE_VOL = 0.5  # size = min(1, 0.5 / vol_ann), fixed at entry
VOL_DAYS = 30  # daily log-return std window
ATR_RANGE_BARS = 6  # 24h high-low range from 4h bars
ATR_MEAN_BARS = 84  # 14 days of 4h bars
TRAIL_ATR = 3.0  # A1: stop at 3 daily ATR from the best close since entry
SHORT_HALF = 0.5  # A2: short size multiplier
TIME_EXIT_BARS = 180  # A3: short closed 180 4h bars (30 days) after entry

# Execution cost: repository ExecutionCostConfig formula, FS1 constants.
FEE_RATE = 0.0005
SPREAD_RATE = 0.0002
IMPACT_RATE = 0.0001
MAX_PARTICIPATION = 0.05
COST_BASE = 1.0
COST_HIGH = 2.0
INITIAL_CAPITAL = (
    100_000.0  # USDT reference account; sizes the participation of every fill
)
IDENTITY_TOL = 1e-9
GROSS_LIMIT = 1.5
TRADE_EPS = 1e-12
EARLY_STOP_DD = 0.30
DD_SLACK = 0.05
BOOT_BLOCK = 30
BOOT_DRAWS = 5000
BOOT_SEED = 0
ALPHA = 0.05

ARMS = ("A0_RULE", "A1_SHORT_TRAIL3", "A2_SHORT_HALF", "A3_SHORT_TIME30")
CHALLENGERS = ARMS[1:]
VARIANTS: dict[
    str, tuple[float, int]
] = {  # name -> (cost multiplier, fill delay in 4h bars)
    "BASE": (COST_BASE, 0),
    "HIGH_COST": (COST_HIGH, 0),
    "DELAY": (COST_BASE, 1),
}
HOLD_CELL = "C_HOLD_BTC"
CASH_CELL = "C_CASH"
PRIMARY = "A0_RULE__BASE"


def cell_name(arm: str, variant: str) -> str:
    return f"{arm}__{variant}"


def all_cell_names() -> list[str]:
    return [cell_name(a, v) for a in ARMS for v in VARIANTS] + [HOLD_CELL, CASH_CELL]


@dataclass(frozen=True)
class Stage:
    name: str
    history_start: datetime  # first hourly bar loaded (regime warm-up)
    decision_start: datetime  # first decision close c_0
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

LABEL_IMPROVES = "ARM_IMPROVES"
LABEL_NONE = "NO_ARM_IMPROVES"
LABEL_INVALID = "INVALID"
LABEL_INTERIM = "INTERIM_NOT_A_DECISION"
LABEL_STOPPED = "STOPPED_EARLY"

DATA_SCOPE_COMMENT_ID = 7040247247  # owner scope relay on Issue #810
OWNER_LOGIN = "shuntatsu"
REPO = "shuntatsu/trade_rl"
ISSUE_URL = f"https://api.github.com/repos/{REPO}/issues/810"
LEDGER_DIR = Path(r"C:\trade_rl_runs\trend_exit_arms_forward_ledger")
REPO_ROOT = Path(__file__).resolve().parent.parent
SPEC_RELPATH = "docs/specs/trend-exit-arms-forward.md"

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

HOURLY_KEYS = ("open", "high", "low", "close", "qv")
# Pre-claim connectivity probe: one BTCUSDT klines page and one funding page, both dated before
# 2026-11-01, so no stage-window data is read before the claim.
PROBE_START_MS = int(datetime(2026, 10, 1, tzinfo=UTC).timestamp()) * 1000
PROBE_END_MS = PROBE_START_MS + DAY_S * 1000 - 1


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
        "size_vol": SIZE_VOL,
        "vol_days": VOL_DAYS,
        "atr_range_bars": ATR_RANGE_BARS,
        "atr_mean_bars": ATR_MEAN_BARS,
        "trail_atr": TRAIL_ATR,
        "short_half": SHORT_HALF,
        "time_exit_bars": TIME_EXIT_BARS,
        "bar_per_year": BAR_PER_YEAR,
        "fee_rate": FEE_RATE,
        "spread_rate": SPREAD_RATE,
        "impact_rate": IMPACT_RATE,
        "max_participation": MAX_PARTICIPATION,
        "initial_capital": INITIAL_CAPITAL,
        "identity_tol": IDENTITY_TOL,
        "gross_limit": GROSS_LIMIT,
        "early_stop_dd": EARLY_STOP_DD,
        "dd_slack": DD_SLACK,
        "boot": [BOOT_BLOCK, BOOT_DRAWS, BOOT_SEED, ALPHA],
        "arms": list(ARMS),
        "variants": {k: list(v) for k, v in VARIANTS.items()},
        "reference_cells": [HOLD_CELL, CASH_CELL],
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
        "probe": [PROBE_START_MS, PROBE_END_MS],
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
    }
    return {k: np.where(full, v, np.nan) for k, v in out.items()}


# ----------------------------------------------------------------------------- frame


@dataclass(frozen=True)
class Frame:
    """Per-symbol 4h table: ``open_s[i]`` is the open time of bar i (decision at open + 4h)."""

    open_s: np.ndarray
    close: np.ndarray
    regime: np.ndarray  # latest completed daily regime vote at the 4h close
    vol_ann: np.ndarray  # latest completed 30-day daily log-return std x sqrt(365)
    datr: np.ndarray  # causal daily ATR: 84-bar mean of the 6-bar 4h high-low range
    fill_qv: np.ndarray  # quote volume of the last completed 1h bar of the 4h bar


def build_frame(hourly: dict[str, np.ndarray], start_s: int) -> Frame:
    """Causal frame for one symbol from its full hourly grid (UTC midnight start; NaN holes for
    missing hours; length a multiple of 4). 4h and 1d bars are labelled by OPEN time and exist
    only if every hourly bar exists (the exploratory ``feat.agg`` rule)."""
    n_hours = len(hourly["close"])
    if n_hours % 4:
        raise ValueError("hourly grid must hold whole 4h bars")
    b4 = aggregate(hourly, 4, n_hours)
    b1d = aggregate(hourly, 24, n_hours)
    n4 = len(b4["close"])
    cd = b1d["close"]
    with np.errstate(divide="ignore", invalid="ignore"):
        regime = sum(np.sign(cd - r_mean(cd, w)) for w in REGIME_WINDOWS) / len(
            REGIME_WINDOWS
        )
        vol_ann = r_std(diff(np.log(cd)), VOL_DAYS) * math.sqrt(365)
    # latest COMPLETED daily bar at each 4h close: day index floor(hours since start / 24) - 1
    day_of = (np.arange(n4) + 1) * 4 // 24 - 1
    ok = (day_of >= 0) & (day_of < len(cd))
    pick = np.clip(day_of, 0, max(len(cd) - 1, 0))
    rng = r_max(b4["high"], ATR_RANGE_BARS) - r_min(b4["low"], ATR_RANGE_BARS)
    return Frame(
        start_s + BAR_S * np.arange(n4),
        b4["close"],
        np.where(ok, regime[pick], np.nan),
        np.where(ok, vol_ann[pick], np.nan),
        r_mean(rng, ATR_MEAN_BARS),
        hourly["qv"][3::4][:n4],
    )


# ----------------------------------------------------------------------------- signals and arms


@dataclass(frozen=True)
class Signals:
    """Decision-close inputs. Index j is the 4h close c_j (j = 0..n); ``fund[sym][j]`` is the sum
    of ``mark price * funding rate`` of the funding events paid over bar j (0 at j = 0)."""

    times: np.ndarray
    symbols: tuple[str, ...]
    close: dict[str, np.ndarray]
    regime: dict[str, np.ndarray]
    vol: dict[str, np.ndarray]
    datr: dict[str, np.ndarray]
    fill_qv: dict[str, np.ndarray]
    fund: dict[str, np.ndarray]

    @property
    def bars(self) -> int:
        return len(self.times) - 1


def regime_direction(regime: np.ndarray) -> np.ndarray:
    """Rule direction: +1 if regime >= 0.5, -1 if <= -0.5, else 0 (NaN -> 0)."""
    return np.where(
        regime >= REGIME_THRESHOLD, 1, np.where(regime <= -REGIME_THRESHOLD, -1, 0)
    ).astype(np.int8)


def _size(vol_ann: float) -> float:
    return 1.0 if vol_ann <= 0.0 else min(1.0, SIZE_VOL / vol_ann)


def rule_positions(
    regime: np.ndarray, vol: np.ndarray, close: np.ndarray
) -> np.ndarray:
    """The rule book's signed size per decision row (fraction of the coin's sleeve), ported from
    the exploratory ``positions``: direction from the regime vote, size ``min(1, 0.5/vol_ann)``
    fixed while the direction is unchanged, the previous position held on a row whose close or
    volatility is not finite."""
    rule = regime_direction(regime)
    out = np.zeros(len(rule))
    pos = 0.0
    for t in range(len(rule)):
        va, c = float(vol[t]), float(close[t])
        if math.isfinite(va) and math.isfinite(c):
            newpos = float(rule[t]) * _size(va)
            if pos != 0.0 and np.sign(newpos) == np.sign(pos):
                newpos = pos
            pos = newpos
        out[t] = pos
    return out


def exit_multipliers(
    arm: str, pos: np.ndarray, close: np.ndarray, datr: np.ndarray
) -> np.ndarray:
    """Per-row multiplier on the rule position. Longs always keep the rule exit (multiplier 1).
    A trade is a maximal run of one position sign; the multiplier of a SHORT trade starts at 1
    (``SHORT_HALF`` for A2) and an exit only lowers it to 0 for the rest of the trade (no
    re-entry until a new trade starts). Exits look only backwards from the decision row:

    * A1: the close is at least ``TRAIL_ATR`` daily ATR above the lowest close since entry
      (entry row included);
    * A3: the row is ``TIME_EXIT_BARS`` or more rows after the entry row.

    A trade whose entry row has no finite positive ATR keeps the rule exit (the exploratory
    ``apply_policy`` skips a trade without entry features)."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    n = len(pos)
    mult = np.zeros(n)
    prev = 0
    m = 1.0
    entry = 0
    best = math.nan
    enabled = False
    for b in range(n):
        s = int(np.sign(pos[b]))
        if s == 0:
            prev = 0
            continue
        if s != prev:
            entry, best, m = b, float(close[b]), 1.0
            a0 = float(datr[b])
            enabled = math.isfinite(a0) and a0 > 0.0 and math.isfinite(best)
            if s < 0 and arm == "A2_SHORT_HALF":
                m = SHORT_HALF
        elif s < 0 and enabled and m > 0.0:
            best = min(best, float(close[b]))
            if arm == "A1_SHORT_TRAIL3" and float(close[b]) >= best + TRAIL_ATR * float(
                datr[b]
            ):
                m = 0.0
            if arm == "A3_SHORT_TIME30" and b - entry >= TIME_EXIT_BARS:
                m = 0.0
        mult[b] = m
        prev = s
    return mult


def arm_weights(sig: Signals, arm: str) -> np.ndarray:
    """(decisions, symbols) target weights of equity: position x multiplier / number of coins."""
    n = sig.bars
    slots = len(sig.symbols)
    w = np.zeros((n, slots))
    for k, s in enumerate(sig.symbols):
        pos = rule_positions(sig.regime[s][:n], sig.vol[s][:n], sig.close[s][:n])
        w[:, k] = (
            pos * exit_multipliers(arm, pos, sig.close[s][:n], sig.datr[s][:n]) / slots
        )
    return w


def hold_weights(sig: Signals) -> np.ndarray:
    w = np.zeros((sig.bars, len(sig.symbols)))
    w[:, list(sig.symbols).index(BTC)] = 1.0
    return w


def build_signals(
    frames: dict[str, Frame],
    symbols: tuple[str, ...],
    first_close_s: int,
    bars: int,
    fund: dict[str, np.ndarray],
) -> Signals:
    times = first_close_s + BAR_S * np.arange(bars + 1)
    out: dict[str, dict[str, np.ndarray]] = {
        k: {} for k in ("close", "regime", "vol", "datr", "fill_qv")
    }
    for sym in symbols:
        fr = frames[sym]
        i0 = int((first_close_s - BAR_S - int(fr.open_s[0])) // BAR_S)
        sl = slice(i0, i0 + bars + 1)
        out["close"][sym] = fr.close[sl]
        out["regime"][sym] = fr.regime[sl]
        out["vol"][sym] = fr.vol_ann[sl]
        out["datr"][sym] = fr.datr[sl]
        out["fill_qv"][sym] = fr.fill_qv[sl]
    return Signals(
        times,
        symbols,
        out["close"],
        out["regime"],
        out["vol"],
        out["datr"],
        out["fill_qv"],
        fund,
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


# ----------------------------------------------------------------------------- actual-units book


def unit_cost_rate(
    participation: float, multiplier: float
) -> tuple[float, float, float]:
    """(fee, spread, impact) per unit of filled notional, the repository ExecutionCostConfig
    formula: ``multiplier * (fee + spread + impact_rate * sqrt(participation))``."""
    return (
        multiplier * FEE_RATE,
        multiplier * SPREAD_RATE,
        multiplier * IMPACT_RATE * math.sqrt(participation),
    )


class _Book:
    """Actual contracts and cash. A symbol whose target weight is unchanged keeps its units (no
    free re-weighting). A changed target is converted to units ONCE, at the fill, from the equity
    marked at the fill prices before that event's fills. A fill is at the close (no price
    slippage), is capped at ``MAX_PARTICIPATION`` of the last completed 1h quote volume, pays
    ``unit_cost_rate`` on the filled notional and carries an unfilled residual while the target
    is unchanged. Funding is ``-units * mark * rate`` on the units held over the bar."""

    def __init__(self, sig: Signals, multiplier: float) -> None:
        self.sig, self.mult = sig, multiplier
        self.syms = sig.symbols
        n = len(self.syms)
        self.u = np.zeros(n)
        self.w_prev = np.zeros(n)
        self.pending = np.full(n, np.nan)
        self.cash = INITIAL_CAPITAL
        self.eq = INITIAL_CAPITAL
        self.post: list[float] = []
        self.reasons: list[str] = []
        self.err = 0.0
        self.fills = 0
        self.capped = 0
        self.turnover = 0.0
        self.max_gross_fill = 0.0
        self.max_gross_mark = 0.0
        self._event_cost = 0.0
        self.cum_price = {1: 0.0, -1: 0.0}
        self.cum_fund = {1: 0.0, -1: 0.0}
        self.cum_cost = {1: 0.0, -1: 0.0}

    def _px(self, j: int) -> np.ndarray:
        return np.array([float(self.sig.close[s][j]) for s in self.syms])

    def _equity(self, px: np.ndarray) -> float:
        return self.cash + float(np.dot(self.u, px))

    def mark(self, j: int) -> None:
        p0, p1 = self._px(j - 1), self._px(j)
        for k, s in enumerate(self.syms):
            u = self.u[k]
            if u == 0.0:
                continue
            sg = 1 if u > 0 else -1
            self.cum_price[sg] += u * (p1[k] - p0[k])
            fund = -u * float(self.sig.fund[s][j])
            self.cash += fund
            self.cum_fund[sg] += fund
        eq = self._equity(p1)
        self.eq = eq
        if not math.isfinite(eq) or eq <= 0.0:
            self.reasons.append(f"equity {eq!r} not positive at bar {j}")
            return
        attr = (
            INITIAL_CAPITAL
            + sum(self.cum_price.values())
            + sum(self.cum_fund.values())
            - sum(self.cum_cost.values())
        )
        self.err = max(self.err, abs(eq - attr) / max(abs(eq), 1.0))
        self.max_gross_mark = max(
            self.max_gross_mark, float(np.abs(self.u * p1).sum()) / eq
        )

    def _try_fill(self, j: int, k: int, px: float, e_pre: float) -> bool:
        tgt = self.pending[k]
        delta = tgt - self.u[k]
        want = abs(delta) * px
        if want <= 1e-15 * max(e_pre, 1.0):
            self.pending[k] = np.nan
            return False
        vol = float(self.sig.fill_qv[self.syms[k]][j])
        if not (math.isfinite(vol) and vol > 0.0):
            return False  # not tradable this bar: the target stays pending
        cap = MAX_PARTICIPATION * vol
        filled = min(want, cap)
        qty = math.copysign(filled / px, delta)
        fee_r, spr_r, imp_r = unit_cost_rate(filled / vol, self.mult)
        total = filled * (fee_r + spr_r + imp_r)
        pre = self.u[k]
        if want > cap:
            self.capped += 1
            self.u[k] = pre + qty
        else:
            self.u[k] = tgt
            self.pending[k] = np.nan
            qty = tgt - pre
        self.cash -= qty * px + total
        sg = 1 if self.u[k] > 0 else (-1 if self.u[k] < 0 else (1 if pre > 0 else -1))
        self.cum_cost[sg] += total
        self.turnover += filled
        self.fills += 1
        self._event_cost += total
        return True

    def fill_event(self, j: int, w: np.ndarray) -> None:
        px = self._px(j)
        e_pre = self._equity(px)
        if not math.isfinite(e_pre) or e_pre <= 0.0:
            self.reasons.append(f"equity {e_pre!r} not positive before fill at bar {j}")
            return
        self._event_cost = 0.0
        filled_any = False
        for k in range(len(self.syms)):
            if abs(float(w[k]) - self.w_prev[k]) > TRADE_EPS:
                self.w_prev[k] = float(w[k])
                self.pending[k] = float(w[k]) * e_pre / px[k]
            if not np.isnan(self.pending[k]):
                filled_any |= self._try_fill(j, k, px[k], e_pre)
        e_post = self._equity(px)
        self.eq = e_post
        self.err = max(
            self.err, abs(e_post - (e_pre - self._event_cost)) / max(abs(e_pre), 1.0)
        )
        if not math.isfinite(e_post) or e_post <= 0.0:
            self.reasons.append(f"equity {e_post!r} not positive after fill at bar {j}")
            return
        if filled_any:
            gross = float(np.abs(self.u * px).sum()) / e_post
            self.max_gross_fill = max(self.max_gross_fill, gross)
            if gross > GROSS_LIMIT + IDENTITY_TOL:
                self.reasons.append(
                    f"gross leverage {gross!r} > {GROSS_LIMIT!r} at fill, bar {j}"
                )


def _side(d: dict[int, float]) -> dict[str, float]:
    return {"long": d[1] / INITIAL_CAPITAL, "short": d[-1] / INITIAL_CAPITAL}


def simulate_weights(
    sig: Signals, w: np.ndarray, multiplier: float, delay: int
) -> dict[str, Any]:
    """Replay one cell. Decisions at closes c_0..c_{n-1}; marks at c_1..c_n; the decision of
    close c_j fills at c_{j+delay}. The book starts flat with ``INITIAL_CAPITAL``."""
    n = sig.bars
    for s in sig.symbols:
        px = sig.close[s]
        if not np.all(np.isfinite(px)) or np.any(px <= 0.0):
            raise ValueError(f"non-positive or missing close for {s}")
    book = _Book(sig, multiplier)
    for j in range(n + 1):
        if j >= 1:
            book.mark(j)
        d = j - delay
        if not book.reasons and 0 <= d <= n - 1:
            book.fill_event(j, w[d])
        book.post.append(book.eq)
        if book.reasons:
            break
    equity = np.asarray([INITIAL_CAPITAL] + book.post) / INITIAL_CAPITAL
    reasons = list(book.reasons)
    if book.err > IDENTITY_TOL:
        reasons.append(f"accounting identity gap {book.err!r} > {IDENTITY_TOL!r}")
    if len(book.post) != n + 1 and not reasons:
        reasons.append("incomplete replay")
    out = _stats(equity)
    mean_eq = float(np.mean(equity)) * INITIAL_CAPITAL
    out.update(
        fills=book.fills,
        capped_fills=book.capped,
        turnover_x=book.turnover / mean_eq if mean_eq > 0.0 else 0.0,
        price_pnl=_side(book.cum_price),
        funding_pnl=_side(book.cum_fund),
        cost_paid=_side(book.cum_cost),
        costs_total=sum(book.cum_cost.values()) / INITIAL_CAPITAL,
        funding_total=sum(book.cum_fund.values()) / INITIAL_CAPITAL,
        max_gross_at_fills=book.max_gross_fill,
        max_held_gross=book.max_gross_mark,
        max_identity_err=book.err,
        invalid_reasons=reasons,
        valid=not reasons,
    )
    return out


def cash_cell(bars: int) -> dict[str, Any]:
    zero = {"long": 0.0, "short": 0.0}
    out = _stats(np.ones(bars + 2))
    out.update(
        fills=0,
        capped_fills=0,
        turnover_x=0.0,
        price_pnl=dict(zero),
        funding_pnl=dict(zero),
        cost_paid=dict(zero),
        costs_total=0.0,
        funding_total=0.0,
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
    for arm in ARMS:
        w = arm_weights(sig, arm)
        for variant, (mult, delay) in VARIANTS.items():
            cells[cell_name(arm, variant)] = simulate_weights(sig, w, mult, delay)
    cells[HOLD_CELL] = simulate_weights(sig, hold_weights(sig), COST_BASE, 0)
    cells[CASH_CELL] = cash_cell(sig.bars)
    return cells


# ----------------------------------------------------------------------------- decision


def block_bootstrap_means(
    series: dict[str, np.ndarray],
    block: int = BOOT_BLOCK,
    draws: int = BOOT_DRAWS,
    seed: int = BOOT_SEED,
) -> dict[str, np.ndarray]:
    """Paired moving-block bootstrap of the mean: overlapping blocks of ``block`` rows (shortened
    to the series length), one shared set of block starts for every series, so the series stay
    paired. Same scheme as the exploratory ``block_bootstrap_ci``."""
    names = list(series)
    n = len(series[names[0]])
    if n < 2 or any(len(series[k]) != n for k in names):
        raise ValueError("need paired series of at least two returns")
    block = min(block, n)
    rng = np.random.default_rng(seed)
    n_blocks = -(-n // block)
    starts = rng.integers(0, n - block + 1, size=(draws, n_blocks))
    out = {k: np.empty(draws) for k in names}
    step = 250
    for lo in range(0, draws, step):
        idx = (
            starts[lo : lo + step, :, None] + np.arange(block)[None, None, :]
        ).reshape(len(starts[lo : lo + step]), -1)[:, :n]
        for k in names:
            out[k][lo : lo + step] = np.asarray(series[k], dtype=float)[idx].mean(
                axis=1
            )
    return out


def holm(p_values: dict[str, float], alpha: float = ALPHA) -> dict[str, bool]:
    """Holm step-down: sorted ascending, reject while ``p_(r) <= alpha / (m - r)``."""
    order = sorted(p_values, key=lambda k: (p_values[k], k))
    m = len(order)
    out = dict.fromkeys(order, False)
    for r, name in enumerate(order):
        if p_values[name] <= alpha / (m - r):
            out[name] = True
        else:
            break
    return out


def analyze(cells: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """The decision rule's statistics for the three challenger arms against ``A0_RULE``. Needs
    valid cells. ``IMPROVES(Ai)`` requires all of: paired BASE bootstrap one-sided lower 95%
    bound above 0 and Holm rejection across the arms; positive mean differences in HIGH_COST and
    DELAY; Ai BASE max drawdown not above A0's plus ``DD_SLACK``."""
    base = {
        a: np.asarray(cells[cell_name(a, "BASE")]["returns"], dtype=float) for a in ARMS
    }
    diffs = {a: base[a] - base["A0_RULE"] for a in CHALLENGERS}
    means = block_bootstrap_means(diffs)
    p = {
        a: float((1 + np.sum(means[a] <= 0.0)) / (len(means[a]) + 1))
        for a in CHALLENGERS
    }
    rejected = holm(p)
    a0_dd = float(cells[PRIMARY]["max_dd"])
    arms: dict[str, Any] = {}
    for a in CHALLENGERS:
        lower = float(np.quantile(means[a], ALPHA))
        point = float(np.mean(diffs[a]))
        mean_diff = {
            v: float(
                np.mean(
                    np.asarray(cells[cell_name(a, v)]["returns"], dtype=float)
                    - np.asarray(cells[cell_name("A0_RULE", v)]["returns"], dtype=float)
                )
            )
            for v in ("HIGH_COST", "DELAY")
        }
        dd = float(cells[cell_name(a, "BASE")]["max_dd"])
        checks = {
            "bootstrap_lower95_above_0": lower > 0.0,
            "holm_rejected": rejected[a],
            "high_cost_mean_diff_positive": mean_diff["HIGH_COST"] > 0.0,
            "delay_mean_diff_positive": mean_diff["DELAY"] > 0.0,
            "max_dd_within_slack": dd <= a0_dd + DD_SLACK,
        }
        arms[a] = {
            "base_mean_diff": point,
            "base_lower95": lower,
            "p_one_sided": p[a],
            "high_cost_mean_diff": mean_diff["HIGH_COST"],
            "delay_mean_diff": mean_diff["DELAY"],
            "base_max_dd": dd,
            "a0_base_max_dd": a0_dd,
            "checks": checks,
            "improves": all(checks.values()),
        }
    hold = cells[HOLD_CELL]
    return {
        "arms": arms,
        "improving": sorted(a for a, r in arms.items() if r["improves"]),
        "secondary_a0_sharpe": float(cells[PRIMARY]["sharpe"]),
        "secondary_hold_btc_sharpe": float(hold["sharpe"]),
        "secondary_a0_beats_hold_btc_sharpe": bool(
            cells[PRIMARY]["sharpe"] > hold["sharpe"]
        ),
    }


def decide(
    stage: Stage | str,
    cells: dict[str, dict[str, Any]],
    problems: list[str] | None = None,
) -> dict[str, Any]:
    """Label. Priority: INVALID (any problem or invalid cell), then interim STOPPED_EARLY /
    INTERIM_NOT_A_DECISION, then the final rule. The A0 drawdown stop applies to interims only;
    the final records ``a0_stop_crossed`` and ``a0_base_max_dd`` and still decides."""
    st = STAGES[stage] if isinstance(stage, str) else stage
    problems = list(problems or [])
    invalid = sorted(k for k, c in cells.items() if not c["valid"])
    out: dict[str, Any] = {"invalid_cells": invalid, "problems": problems}
    if invalid or problems or not cells:
        out["label"] = LABEL_INVALID
        return out
    out["analysis"] = analyze(cells)
    a0_dd = float(cells[PRIMARY]["max_dd"])
    # Recorded for every valid stage; only an interim acts on it. The final always decides the
    # arm question (a relative comparison), whatever A0's drawdown.
    out["a0_base_max_dd"] = a0_dd
    out["a0_stop_crossed"] = bool(a0_dd >= EARLY_STOP_DD)
    if st.interim:
        out["label"] = LABEL_STOPPED if out["a0_stop_crossed"] else LABEL_INTERIM
    else:
        ids = out["analysis"]["improving"]
        out["label"] = f"{LABEL_IMPROVES}:{','.join(ids)}" if ids else LABEL_NONE
    return out


# ----------------------------------------------------------------------------- data parsing


def parse_klines(
    payload: Any,
) -> list[tuple[int, float, float, float, float, float]]:
    """Binance klines page -> [(open_ms, open, high, low, close, quote_volume)]."""
    if not isinstance(payload, list):
        raise SystemExit("malformed Binance klines payload")
    rows = []
    for it in payload:
        try:
            source_open = _native_timestamp(it[0])
            # Normalize exact native microseconds without rounding a bad source close.
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
            qv = float(it[7])
        except (IndexError, KeyError, TypeError, ValueError):
            raise SystemExit("malformed Binance klines row") from None
        rows.append((open_ms, o, h, lo, c, qv))
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
    rows: list[tuple[int, float, float, float, float, float]],
    start_ms: int,
    last_open_ms: int,
) -> dict[str, np.ndarray]:
    """Full hourly grid start_ms..last_open_ms with NaN holes for missing hours."""
    n = (last_open_ms - start_ms) // HOUR_MS + 1
    keys = HOURLY_KEYS
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
) -> tuple[list[tuple[int, float, float, float, float, float]], list[dict[str, Any]]]:
    """Hourly klines for the hourly open times start_ms..last_open_ms. Every request spans at most
    KLINES_LIMIT hours and rows outside its window are dropped, so the paging is correct whether
    or not the API honours ``startTime`` (precedent: btc_trend_ensemble Amendment 1)."""
    rows: list[tuple[int, float, float, float, float, float]] = []
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
        if role == "g4_relay" and not all(
            token in first["body"] for token in (PROTOCOL_ID, digest, commit)
        ):
            raise SystemExit(
                "G4 relay does not quote this protocol id, digest and exact commit"
            )
        out[role] = {
            "comment_id": cid,
            "author": first["user"]["login"],
            "updated_at": first["updated_at"],
            "body_sha256": hashlib.sha256(first["body"].encode("utf-8")).hexdigest(),
        }
    return out


# ----------------------------------------------------------------------------- ledger & clocks
def _utc(now: datetime) -> datetime:
    """Refuse naive datetimes; normalize an aware instant before claim identity."""
    if now.tzinfo is None or now.utcoffset() is None:
        raise SystemExit("now must be a timezone-aware datetime")
    return now.astimezone(UTC)


def claim_name(stage: Stage, now: datetime) -> str:
    """Ledger name: interims are once per UTC month."""
    now = _utc(now)
    return f"{stage.name}_{now:%Y-%m}" if stage.interim else stage.name


def check_clock(stage: Stage, now: datetime) -> None:
    now = _utc(now)
    if stage.not_before is not None and now < stage.not_before:
        raise SystemExit(
            f"{stage.name} may not run before {stage.not_before.isoformat()}"
        )
    if stage.not_after is not None and now >= stage.not_after:
        raise SystemExit(
            f"{stage.name} may not run at or after {stage.not_after.isoformat()}"
        )


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
        f = path.open("x", encoding="utf-8")
    except FileExistsError:
        raise SystemExit(f"stage {name} was already claimed: {path}") from None
    with f:
        f.write(
            json.dumps(
                {
                    "stage": name,
                    "protocol_digest": digest,
                    "commit": commit,
                    "authorization_comment_id": auth_comment_id,
                    "claimed_at": now.isoformat(),
                },
                indent=1,
                sort_keys=True,
            )
        )
    return path


def write_ledger_result(
    name: str,
    digest: str,
    label: str,
    supported_flag: bool,
    result_sha256: str,
    ledger_dir: Path = LEDGER_DIR,
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


def connectivity_probe(fetch: Callable[[str], bytes]) -> None:
    """Read one klines page and one funding page of BTCUSDT dated before the stage window. Any
    failure refuses the run before the single-use claim, so an outage cannot burn a stage."""
    kurl = (
        f"{KLINES_URL}?symbol={BTC}&interval=1h&startTime={PROBE_START_MS}"
        f"&endTime={PROBE_END_MS}&limit=24"
    )
    furl = (
        f"{FUNDING_URL}?symbol={BTC}&startTime={PROBE_START_MS}"
        f"&endTime={PROBE_END_MS + DAY_S * 1000}&limit=3"
    )
    try:
        krows = parse_klines(json.loads(fetch(kurl)))
        frows = parse_funding(json.loads(fetch(furl)))
    except (ValueError, SystemExit) as exc:
        raise SystemExit(f"connectivity probe failed before any claim: {exc}") from None
    if not krows or not frows:
        raise SystemExit("connectivity probe failed before any claim: empty page")


def library_versions() -> dict[str, str | None]:  # pragma: no cover - environment
    out: dict[str, str | None] = {}
    for pkg in ("numpy",):
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
    symbols: tuple[str, ...] = SYMBOLS,
) -> dict[str, Any]:
    stage = STAGES[stage_name] if isinstance(stage_name, str) else stage_name
    now = _utc(now or datetime.now(UTC))
    if not ledger_dir.is_absolute():
        raise SystemExit(f"ledger directory must be an absolute path: {ledger_dir}")
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
    check_cross_digest(ledger_root, stage, name)
    ledger_dir = ledger_root / digest
    commit = harness_commit()
    auth = authorization_receipt(authorization_url, digest, commit, gh_fetch)
    connectivity_probe(fetch)  # refuses WITHOUT claiming if the exchange is unreachable
    # The protocol-root claim is the atomic single-use lock across digests (exclusive create);
    # the per-digest copy only records provenance.
    claim_stage(name, digest, commit, auth["g4_relay"]["comment_id"], now, ledger_root)
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
    if not problems:
        start_s = grid_start_ms // 1000
        frames = {s: build_frame(hourly[s], start_s) for s in symbols}
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
        sig = build_signals(frames, symbols, first_close_s, int(bars), fund)
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
        result["label"].startswith(LABEL_IMPROVES),
        sha256_file(output_dir / "result.json"),
        ledger_dir,
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
            {k: result[k] for k in ("stage", "label", "invalid_cells", "problems")},
            indent=1,
        )
    )
    for name, st in result["cells"].items():
        print(
            f"{name:26s} ret {st['total_return'] * 100:+8.2f}% dd {st['max_dd'] * 100:5.2f}% "
            f"sr {st['sharpe']:+.2f} valid {st['valid']}"
        )


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="trend_exit_arms_forward")
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
