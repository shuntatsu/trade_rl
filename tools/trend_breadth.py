"""Daily-trend breadth screen, protocol ``trend_breadth_v1``.

Protocol: docs/specs/trend-breadth.md (the spec is the authority for every constant).

One single-use stage, ``BREADTH``: the frozen daily-regime trend rule is replayed on the 441 USDT-M
perpetuals of the xs DISJOINT candidate list (frozen by sha256) from 2021-01-01 to 2026-08-01, read
from local Binance Vision archives. Six cells (long/short, long only, long/short with doubled
costs, long/short with delayed fills, an equal-weight long control, cash) are replayed on an
actual-units book with the repository's execution-cost formula. A local ledger claim is created
exclusively before any archive value is parsed. Importing this module needs only the standard
library and numpy; there is no network access except the GitHub authorization comments.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import re
import subprocess
import warnings
import zipfile
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

PROTOCOL_ID = "trend_breadth_v1"

# ----------------------------------------------------------------------------- constants

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

SIDE_BUDGET = (
    0.9  # a sleeve is SIDE_BUDGET / N_t of equity (10% reserve for costs and funding)
)
SIDE_CAP = (
    1.0  # an increasing fill may not leave a side's notional above SIDE_CAP x equity
)
BORROW_TOL = (
    0.02  # cash below -2% of equity at a mark is borrowing (funding drift allowance)
)
N_MIN = 20  # new entries need at least this many eligible symbols
DEAD_RUN_HOURS = 24  # a run of dead hours this long ends a listing episode
NOTICE_LOOKBACK_DAYS = 90
FALLBACK_LEAD_BARS = (
    42  # forced-close lead of an event with no recorded notice (7 days)
)
FORCED_EXTRA = 0.05  # extra adverse charge on forced-close fills and write-offs
FUNDING_UNCOVERED_MAX = (
    0.01  # share of eligible symbol-bars without a recent funding event
)
MAX_FUNDING_GAP_S = 8 * HOUR_S + 60

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
TRADE_EPS = 1e-12
SHARPE_MIN = 0.5
MAX_DD_LIMIT = 0.40
BOOT_BLOCK = 30
BOOT_DRAWS = 5000
BOOT_SEED = 0
ALPHA = 0.05

AGE_EDGES_DAYS = (200, 365, 730, 1095)
AGE_LABELS = ("lt200d", "200-365d", "1-2y", "2-3y", "3y+")

VARIANTS: dict[
    str, tuple[float, int]
] = {  # name -> (cost multiplier, fill delay in 4h bars)
    "BASE": (COST_BASE, 0),
    "HIGH_COST": (COST_HIGH, 0),
    "DELAY": (COST_BASE, 1),
}
PRIMARY = "TREND_LS__BASE"
LONG_CELL = "TREND_LONG__BASE"
HIGH_COST_CELL = "TREND_LS__HIGH_COST"
DELAY_CELL = "TREND_LS__DELAY"
EW_CELL = "EW_LONG"
CASH_CELL = "CASH"
CELL_NAMES = (PRIMARY, LONG_CELL, HIGH_COST_CELL, DELAY_CELL, EW_CELL, CASH_CELL)


@dataclass(frozen=True)
class Stage:
    name: str
    grid_start: datetime  # first hourly bar read; the first 4h bar opens here
    end: datetime  # exclusive: no bar opening at or after it is read

    @property
    def first_close(self) -> datetime:
        return datetime.fromtimestamp(int(self.grid_start.timestamp()) + BAR_S, UTC)

    @property
    def n_hours(self) -> int:
        return (int(self.end.timestamp()) - int(self.grid_start.timestamp())) // HOUR_S


STAGES: dict[str, Stage] = {
    "BREADTH": Stage(
        "BREADTH", datetime(2021, 1, 1, tzinfo=UTC), datetime(2026, 8, 1, tzinfo=UTC)
    )
}

LABEL_SUPPORTED = "TREND_BREADTH_SUPPORTED"
LABEL_NOT_SUPPORTED = "TREND_BREADTH_NOT_SUPPORTED"
LABEL_INVALID = "INVALID"

DATA_SCOPE_COMMENT_ID = 7040247247  # owner scope relay on Issue #810
OWNER_LOGIN = "shuntatsu"
REPO = "shuntatsu/trade_rl"
ISSUE_URL = f"https://api.github.com/repos/{REPO}/issues/810"
LEDGER_DIR = Path(r"C:\trade_rl_runs\trend_breadth_ledger")
ARCHIVE_ROOT = Path(r"C:\trade_rl_runs\xs-archives")
REPO_ROOT = Path(__file__).resolve().parent.parent
SPEC_RELPATH = "docs/specs/trend-breadth.md"
DATA_DIR = Path(__file__).resolve().parent / "trend_breadth_data"
UNIVERSE_PATH = DATA_DIR / "universe_candidates.json"
UNIVERSE_SHA256 = "fc680208899cd4166c9923f9cf9e8f8ebe829010aae073a935de10d6e6fefdf6"
NOTICES_PATH = DATA_DIR / "delisting_notices.json"
NOTICES_SHA256 = "52b0245a754bf381861d7fe892f34d52c129c4cd911c2113dd098287f89bbcc0"
N_UNIVERSE = 441


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
        "regime_windows": list(REGIME_WINDOWS),
        "regime_threshold": REGIME_THRESHOLD,
        "size_vol": SIZE_VOL,
        "vol_days": VOL_DAYS,
        "side_budget": SIDE_BUDGET,
        "side_cap": SIDE_CAP,
        "borrow_tol": BORROW_TOL,
        "n_min": N_MIN,
        "dead_run_hours": DEAD_RUN_HOURS,
        "notice_lookback_days": NOTICE_LOOKBACK_DAYS,
        "fallback_lead_bars": FALLBACK_LEAD_BARS,
        "forced_extra": FORCED_EXTRA,
        "funding_uncovered_max": FUNDING_UNCOVERED_MAX,
        "max_funding_gap_s": MAX_FUNDING_GAP_S,
        "bar_per_year": BAR_PER_YEAR,
        "fee_rate": FEE_RATE,
        "spread_rate": SPREAD_RATE,
        "impact_rate": IMPACT_RATE,
        "max_participation": MAX_PARTICIPATION,
        "initial_capital": INITIAL_CAPITAL,
        "identity_tol": IDENTITY_TOL,
        "sharpe_min": SHARPE_MIN,
        "max_dd_limit": MAX_DD_LIMIT,
        "boot": [BOOT_BLOCK, BOOT_DRAWS, BOOT_SEED, ALPHA],
        "age_edges_days": list(AGE_EDGES_DAYS),
        "variants": {k: list(v) for k, v in VARIANTS.items()},
        "cells": list(CELL_NAMES),
        "stages": {
            k: [v.name, _iso(v.grid_start), _iso(v.end)] for k, v in STAGES.items()
        },
        "universe_sha256": UNIVERSE_SHA256,
        "n_universe": N_UNIVERSE,
        "notices_sha256": NOTICES_SHA256,
        "data_scope_comment": DATA_SCOPE_COMMENT_ID,
        "owner_login": OWNER_LOGIN,
        "issue_url": ISSUE_URL,
        "ledger_dir": str(LEDGER_DIR),
        "archive_root": str(ARCHIVE_ROOT),
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
# Pandas-free ports of the exploratory screens' rolling primitives (NaN-propagating, with
# ``min_periods`` equal to the window).


def _roll(x: np.ndarray, w: int, fn: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    out = np.full(len(x), np.nan)
    if len(x) >= w:
        out[w - 1 :] = fn(np.lib.stride_tricks.sliding_window_view(x, w))
    return out


def r_mean(x: np.ndarray, w: int) -> np.ndarray:
    return _roll(x, w, lambda v: v.mean(axis=1))


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
    """Per-symbol 4h table: ``close[i]`` is the close of the bar opening at ``start + 4h * i``,
    i.e. the decision close ``c_i`` (the first 4h close is ``c_0``)."""

    close: np.ndarray
    regime: np.ndarray  # latest completed daily regime vote at the 4h close
    vol_ann: np.ndarray  # latest completed 30-day daily log-return std x sqrt(365)
    fill_qv: np.ndarray  # quote volume of the last 1h bar of the 4h bar


def build_frame(hourly: dict[str, np.ndarray]) -> Frame:
    """Causal frame for one symbol from its full hourly grid (UTC midnight start; NaN holes for
    missing hours; length a multiple of 24). 4h and 1d bars are labelled by OPEN time and exist
    only if every hourly bar exists (the exploratory ``feat.agg`` rule)."""
    n_hours = len(hourly["close"])
    if n_hours % 24:
        raise ValueError("hourly grid must hold whole days")
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
    return Frame(
        b4["close"],
        np.where(ok, regime[pick], np.nan),
        np.where(ok, vol_ann[pick], np.nan),
        hourly["qv"][3::4][:n4],
    )


# ----------------------------------------------------------------------------- listing episodes


def dead_runs(dead: np.ndarray, min_len: int) -> list[tuple[int, int]]:
    """``[start, end)`` of every run of True values with at least ``min_len`` entries."""
    pad = np.concatenate(([0], np.asarray(dead, dtype=np.int8), [0]))
    edges = np.diff(pad)
    starts = np.nonzero(edges == 1)[0]
    ends = np.nonzero(edges == -1)[0]
    return [
        (int(a), int(b)) for a, b in zip(starts, ends, strict=True) if b - a >= min_len
    ]


@dataclass(frozen=True)
class Episode:
    """One contiguous listing of a perpetual on the hourly grid: hours ``[start, end)``.
    ``delisted`` is True when a run of at least ``DEAD_RUN_HOURS`` dead hours follows it."""

    start: int
    end: int
    delisted: bool


def find_episodes(
    close: np.ndarray, traded: np.ndarray, min_run: int = DEAD_RUN_HOURS
) -> list[Episode]:
    """Episodes of one symbol. A *dead* hour is missing, non-finite or without volume. A run of at
    least ``min_run`` dead hours (a delisting, a relisting gap or the archive's zombie tail) ends
    an episode; shorter holes stay inside it. Leading and trailing dead hours are not part of an
    episode; an episode followed by a long dead run (interior or trailing) is ``delisted``."""
    live = np.isfinite(close) & traded
    runs = dead_runs(~live, min_run)
    n = len(close)
    segments: list[tuple[int, int, bool]] = []
    pos = 0
    for a, b in runs:
        if a > pos:
            segments.append((pos, a, True))
        pos = b
    if pos < n:
        segments.append((pos, n, False))
    out: list[Episode] = []
    for s, e, delisted in segments:
        idx = np.nonzero(live[s:e])[0]
        if idx.size:
            out.append(Episode(s + int(idx[0]), s + int(idx[-1]) + 1, delisted))
    return out


# ----------------------------------------------------------------------------- panel


@dataclass(frozen=True)
class Panel:
    """Decision-close inputs on one 4h grid. Row ``j`` is the close ``c_j`` (j = 0..n) and column
    ``k`` a symbol. ``fund[j, k]`` is the sum of ``mark price * funding rate`` of the funding
    events paid over bar j (0 at j = 0)."""

    times: np.ndarray  # epoch seconds of c_j
    symbols: tuple[str, ...]
    close: np.ndarray  # 4h close, NaN where the bar is incomplete
    mark_px: np.ndarray  # close carried over short holes inside an episode, else NaN
    fill_qv: np.ndarray
    regime: np.ndarray
    vol: np.ndarray
    forced: (
        np.ndarray
    )  # bool: forced-close window (notice released, event not yet passed)
    age_bucket: np.ndarray  # int8 listing-age bucket, 0 outside episodes
    fund: np.ndarray
    events: tuple[tuple[int, int], ...]  # (row j_E of the last available bar, symbol k)

    @property
    def bars(self) -> int:
        return len(self.times) - 1


def _ffill(x: np.ndarray) -> np.ndarray:
    idx = np.where(np.isfinite(x), np.arange(len(x)), 0)
    np.maximum.accumulate(idx, out=idx)
    out = x[idx]
    out[: int(np.argmax(np.isfinite(x)))] = np.nan
    return out


def notice_times(
    notices: dict[str, list[float]] | None, symbol: str, end_s: int
) -> list[float]:
    """Release times (epoch seconds) of the recorded notices of a symbol (sorted, before the
    stage end)."""
    return sorted(t for t in (notices or {}).get(symbol, []) if t < end_s)


def forced_start(
    times: np.ndarray, j_first: int, j_event: int, releases: list[float]
) -> tuple[int | None, str]:
    """First decision row of the forced-close window of an event and how it was paired.

    The notice is the EARLIEST of the symbol with ``release <= event`` and ``release >=
    event - 90 days``; the window starts at the first decision close whose 4h bar opens at or
    after the release. Without a notice the hindsight fallback starts ``FALLBACK_LEAD_BARS``
    rows before the event. Returns ``(None, status)`` when the notice came too late."""
    event_s = float(times[j_event])
    lo = event_s - NOTICE_LOOKBACK_DAYS * DAY_S
    paired = [r for r in releases if lo <= r <= event_s]
    if not paired:
        return max(j_first, j_event - FALLBACK_LEAD_BARS), "fallback"
    release = paired[0]
    start = math.ceil((release + BAR_S - float(times[0])) / BAR_S)
    start = max(j_first, start)
    if start > j_event:
        return None, "late"
    return start, "notice"


def build_panel(
    hourly: Mapping[str, dict[str, np.ndarray]],
    stage: Stage,
    funding: dict[str, list[tuple[int, float, float]]],
    notices: dict[str, list[float]] | None,
    symbols: tuple[str, ...] | None = None,
) -> tuple[Panel, dict[str, Any]]:
    """Cleaned hourly grids -> Panel (listing episodes, frames, forced-close windows, funding).

    ``hourly[sym]`` holds ``open high low close qv vol`` on the stage grid. Dead hours of a long
    run (see ``find_episodes``) are blanked before any bar is aggregated, so a feature never
    spans a delisting or a relisting gap."""
    syms = tuple(symbols if symbols is not None else sorted(hourly))
    # ``hourly[sym]`` is read once per symbol and not kept (the real run streams the archives)
    n_hours = stage.n_hours
    nb = n_hours // 4
    grid_start_s = int(stage.grid_start.timestamp())
    end_s = int(stage.end.timestamp())
    grid_ms = grid_start_s * 1000
    times = grid_start_s + BAR_S * (np.arange(nb) + 1)
    n_sym = len(syms)
    close = np.full((nb, n_sym), np.nan)
    mark_px = np.full((nb, n_sym), np.nan)
    fill_qv = np.full((nb, n_sym), np.nan)
    regime = np.full((nb, n_sym), np.nan)
    vol = np.full((nb, n_sym), np.nan)
    forced = np.zeros((nb, n_sym), dtype=bool)
    age = np.zeros((nb, n_sym), dtype=np.int8)
    fund = np.zeros((nb, n_sym))
    events: list[tuple[int, int]] = []
    info: dict[str, Any] = {
        "episodes": {},
        "notice_matched": 0,
        "notice_fallback": 0,
        "notice_late": 0,
        "funding_events_unpriced": 0,
        "funding_mark_fallbacks": 0,
    }
    ev_times: dict[int, np.ndarray] = {}
    for k, sym in enumerate(syms):
        h = hourly[sym]
        traded = (h["qv"] > 0) | (h["vol"] > 0)
        eps = find_episodes(h["close"], traded)
        keep = np.zeros(n_hours, dtype=bool)
        for ep in eps:
            keep[ep.start : ep.end] = True
        clean = {
            key: np.where(keep, h[key], np.nan)
            for key in ("open", "high", "low", "close", "qv")
        }
        fr = build_frame(clean)
        close[:, k], regime[:, k], vol[:, k], fill_qv[:, k] = (
            fr.close,
            fr.regime,
            fr.vol_ann,
            fr.fill_qv,
        )
        releases = notice_times(notices, sym, end_s)
        recs = []
        for ep in eps:
            lo, hi = ep.start // 4, (ep.end - 1) // 4 + 1
            seg = fr.close[lo:hi]
            fin = np.nonzero(np.isfinite(seg))[0]
            if not fin.size:
                continue
            j_first, j_last = lo + int(fin[0]), lo + int(fin[-1])
            # an episode that is still listed at the stage end keeps its last close for marking
            # through a trailing hole of less than a day; a delisted one ends at its last bar
            j_mark = j_last if ep.delisted else nb - 1
            mark_px[j_first : j_mark + 1, k] = _ffill(fr.close[j_first : j_mark + 1])
            t0 = float(times[j_first]) - BAR_S
            ages = (times[j_first : j_mark + 1] - t0) / DAY_S
            age[j_first : j_mark + 1, k] = np.searchsorted(
                AGE_EDGES_DAYS, ages, side="right"
            )
            rec: dict[str, Any] = {
                "first_close": int(times[j_first]),
                "last_close": int(times[j_last]),
                "delisted": ep.delisted,
            }
            if ep.delisted:
                start, status = forced_start(times, j_first, j_last, releases)
                if start is not None:
                    forced[start : j_last + 1, k] = True
                events.append((j_last, k))
                info[f"notice_{'matched' if status == 'notice' else status}"] += 1
                rec["notice"] = status
            recs.append(rec)
        info["episodes"][sym] = recs
        ev = sorted(funding.get(sym, []))
        ev_times[k] = np.array([e[0] / 1000.0 for e in ev])
        c0_ms = int(times[0]) * 1000
        for t_ms, rate, mark in ev:
            if t_ms < c0_ms or t_ms >= int(times[-1]) * 1000:
                continue
            j = int((t_ms - c0_ms) // BAR_MS) + 1
            price = mark
            if not (math.isfinite(price) and price > 0.0):
                op = clean["open"][(t_ms - grid_ms) // HOUR_MS]
                price = float(op) if math.isfinite(op) else float(mark_px[j - 1, k])
                if math.isfinite(price) and price > 0.0:
                    info["funding_mark_fallbacks"] += 1
            if not (math.isfinite(price) and price > 0.0):
                info["funding_events_unpriced"] += (
                    1  # outside every episode: no position
                )
                continue
            fund[j, k] += price * rate
    panel = Panel(
        times,
        syms,
        close,
        mark_px,
        fill_qv,
        regime,
        vol,
        forced,
        age,
        fund,
        tuple(sorted(events)),
    )
    info["_ev_times"] = ev_times
    return panel, info


def panel_problems(panel: Panel, info: dict[str, Any]) -> list[str]:
    """Data problems that make the stage INVALID: a non-positive price and funding coverage."""
    problems: list[str] = []
    px = panel.mark_px
    if np.any(np.isfinite(px) & (px <= 0.0)):
        problems.append("non-positive price in an episode")
    elig = eligible_matrix(panel)
    total = int(elig.sum())
    uncovered = 0
    for k, ev in info["_ev_times"].items():
        rows = np.nonzero(elig[:, k])[0]
        if not rows.size:
            continue
        t = panel.times[rows].astype(float)
        pos = np.searchsorted(ev, t, side="right") - 1
        last = np.where(
            pos >= 0, ev[np.maximum(pos, 0)] if len(ev) else -np.inf, -np.inf
        )
        uncovered += int(np.sum(t - last > MAX_FUNDING_GAP_S))
    info["funding_uncovered_fraction"] = uncovered / total if total else 0.0
    if total and uncovered / total > FUNDING_UNCOVERED_MAX:
        problems.append(
            f"funding coverage: {uncovered / total:.4f} of eligible symbol-bars have no "
            f"funding event in the last {MAX_FUNDING_GAP_S} s (limit {FUNDING_UNCOVERED_MAX})"
        )
    return problems


# ----------------------------------------------------------------------------- weights


def eligible_matrix(panel: Panel) -> np.ndarray:
    """A symbol is eligible at a close when its regime and 30-day volatility exist (which needs
    200 consecutive complete daily bars of its current listing episode), it has a price and it is
    not inside a forced-close window."""
    return (
        np.isfinite(panel.regime)
        & np.isfinite(panel.vol)
        & np.isfinite(panel.mark_px)
        & ~panel.forced
    )


def regime_direction(regime: np.ndarray) -> np.ndarray:
    """Rule direction: +1 if regime >= 0.5, -1 if <= -0.5, else 0 (NaN -> 0)."""
    return np.where(
        regime >= REGIME_THRESHOLD, 1, np.where(regime <= -REGIME_THRESHOLD, -1, 0)
    ).astype(np.int8)


def _size(vol_ann: float) -> float:
    return 1.0 if vol_ann <= 0.0 else min(1.0, SIZE_VOL / vol_ann)


def symbol_weights(
    regime: np.ndarray,
    vol: np.ndarray,
    close: np.ndarray,
    eligible: np.ndarray,
    n_t: np.ndarray,
    n_min: int = N_MIN,
    budget: float = SIDE_BUDGET,
) -> np.ndarray:
    """Target weight of one symbol per decision row (fraction of equity).

    Direction from the regime vote, size ``min(1, 0.5/vol_ann)`` fixed while the direction is
    unchanged, the previous position held on a row whose close or volatility is not finite, flat
    on an ineligible row. A new entry or flip fixes the sleeve ``budget / N_t`` at that row and
    needs ``N_t >= n_min``; the sleeve never changes while the direction is unchanged."""
    rule = regime_direction(regime)
    out = np.zeros(len(rule))
    pos = 0.0
    sleeve = 0.0
    for t in range(len(rule)):
        if not eligible[t]:
            pos = 0.0
            continue
        va, c = float(vol[t]), float(close[t])
        if math.isfinite(va) and math.isfinite(c):
            newpos = float(rule[t]) * _size(va)
            if pos != 0.0 and np.sign(newpos) == np.sign(pos):
                newpos = pos
            elif newpos != 0.0:
                if n_t[t] < n_min:
                    newpos = 0.0
                else:
                    sleeve = budget / float(n_t[t])
            pos = newpos
        out[t] = pos * sleeve
    return out


def trend_weights(
    panel: Panel,
    n_min: int = N_MIN,
    budget: float = SIDE_BUDGET,
    long_only: bool = False,
) -> np.ndarray:
    elig = eligible_matrix(panel)
    n_t = elig.sum(axis=1)
    w = np.zeros(panel.close.shape)
    for k in range(len(panel.symbols)):
        w[:, k] = symbol_weights(
            panel.regime[:, k],
            panel.vol[:, k],
            panel.close[:, k],
            elig[:, k],
            n_t,
            n_min,
            budget,
        )
    return np.maximum(w, 0.0) if long_only else w


def rebalance_rows(times: np.ndarray) -> np.ndarray:
    """Decision rows at 00:00 UTC of the first day of a calendar month."""
    t = np.asarray(times).astype("datetime64[s]")
    month_start = t.astype("datetime64[M]").astype("datetime64[s]")
    return np.asarray(t == month_start)


def ew_weights(
    panel: Panel, n_min: int = N_MIN, budget: float = SIDE_BUDGET
) -> tuple[np.ndarray, np.ndarray]:
    """Equal-weight long control: ``budget / N_t`` on every eligible symbol at each monthly
    rebalance row (flat while ``N_t < n_min``); between rebalances a symbol that leaves the
    eligible set is closed and nobody is added. Returns ``(weights, forced re-target mask)``."""
    elig = eligible_matrix(panel)
    n_t = elig.sum(axis=1)
    reb = rebalance_rows(panel.times)
    w = np.zeros(panel.close.shape)
    force = np.zeros(panel.close.shape, dtype=bool)
    cur = np.zeros(panel.close.shape[1])
    for j in range(len(w)):
        if reb[j]:
            cur = (
                np.where(elig[j], budget / n_t[j], 0.0)
                if n_t[j] >= n_min
                else cur * 0.0
            )
            force[j] = True
        else:
            cur = np.where(elig[j], cur, 0.0)
        w[j] = cur
    return w, force


# ----------------------------------------------------------------------------- stats


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


def yearly(returns: np.ndarray, times: np.ndarray) -> dict[str, dict[str, float]]:
    """Per calendar year (of the close that ends the return interval) compounded return and max
    drawdown of the 4h simple returns; ``returns[i]`` ends at close ``c_i``."""
    n = min(len(returns), len(times))
    years = (
        np.asarray(times[:n] - 1).astype("datetime64[s]").astype("datetime64[Y]")
    ).astype(int) + 1970
    out: dict[str, dict[str, float]] = {}
    for y in sorted(set(years.tolist())):
        r = returns[:n][years == y]
        eq = np.concatenate(([1.0], np.cumprod(1.0 + r)))
        peak = np.maximum.accumulate(eq)
        out[str(y)] = {
            "return": float(eq[-1] - 1.0),
            "max_dd": float(np.max((peak - eq) / peak)),
            "bars": int(len(r)),
        }
    return out


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
    """Actual contracts and cash (``equity = cash + sum(units * price)``). A symbol whose target
    weight is unchanged keeps its units. A changed (or forced) target is converted to units ONCE,
    at the fill, from the equity marked at the fill prices before that event's fills. A fill is
    at the close (no price slippage), is capped at ``MAX_PARTICIPATION`` of the last completed 1h
    quote volume, pays ``unit_cost_rate`` on the filled notional and carries an unfilled residual
    while the target is unchanged. Reducing fills go before increasing fills. An increasing fill
    is clamped so that its side's notional stays within ``SIDE_CAP`` x equity after the fill and
    its costs (hence no fill can leave cash below zero). A forced-close fill pays an extra
    ``FORCED_EXTRA`` of its notional; a position still held at the last available bar of its
    listing is settled at that close and pays the same extra charge. Funding is
    ``-units * mark * rate`` on the units held over the bar."""

    def __init__(self, panel: Panel, multiplier: float) -> None:
        self.p, self.mult = panel, multiplier
        n = len(panel.symbols)
        self.u = np.zeros(n)
        self.w_prev = np.zeros(n)
        self.pending = np.full(n, np.nan)
        self.pend_forced = np.zeros(n, dtype=bool)
        self.cash = INITIAL_CAPITAL
        self.eq = INITIAL_CAPITAL
        self.post: list[float] = []
        self.reasons: list[str] = []
        self.err = 0.0
        self.fills = 0
        self.capped = 0
        self.clamped = 0
        self.forced_fills = 0
        self.forced_charge = 0.0
        self.write_offs: list[dict[str, Any]] = []
        self.turnover = 0.0
        self.max_side_mark = 0.0
        self.min_cash_frac = math.inf
        self.sum_long = 0.0
        self.sum_short = 0.0
        self.marks = 0
        self.long_n = 0.0
        self.short_n = 0.0
        self._event_cost = 0.0
        self._inc = {1: False, -1: False}
        self.cum_price = {1: 0.0, -1: 0.0}
        self.cum_fund = {1: 0.0, -1: 0.0}
        self.cum_cost = {1: 0.0, -1: 0.0}
        self.bucket = np.zeros(len(AGE_LABELS))

    def _equity(self, px: np.ndarray) -> float:
        idx = np.nonzero(self.u)[0]
        return self.cash + (float(np.dot(self.u[idx], px[idx])) if idx.size else 0.0)

    def mark(self, j: int) -> None:
        p0, p1 = self.p.mark_px[j - 1], self.p.mark_px[j]
        idx = np.nonzero(self.u)[0]
        if idx.size:
            a, b = p0[idx], p1[idx]
            if (
                not (np.isfinite(a).all() and np.isfinite(b).all())
                or (a <= 0).any()
                or (b <= 0).any()
            ):
                self.reasons.append(
                    f"non-positive or missing price of a held symbol at bar {j}"
                )
                self.eq = math.nan
                return
            u = self.u[idx]
            pnl = u * (b - a)
            fund = -u * self.p.fund[j][idx]
            self.cash += float(fund.sum())
            lg = u > 0
            self.cum_price[1] += float(pnl[lg].sum())
            self.cum_price[-1] += float(pnl[~lg].sum())
            self.cum_fund[1] += float(fund[lg].sum())
            self.cum_fund[-1] += float(fund[~lg].sum())
            np.add.at(self.bucket, self.p.age_bucket[j][idx], pnl + fund)
        px = p1
        idx = np.nonzero(self.u)[0]
        eq = self.cash + (float(np.dot(self.u[idx], px[idx])) if idx.size else 0.0)
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
        notion = self.u[idx] * px[idx]
        ln = float(notion[notion > 0].sum())
        sn = float(-notion[notion < 0].sum())
        self.long_n, self.short_n = ln, sn
        self.sum_long += ln / eq
        self.sum_short += sn / eq
        self.marks += 1
        self.max_side_mark = max(self.max_side_mark, ln / eq, sn / eq)
        frac = self.cash / eq
        self.min_cash_frac = min(self.min_cash_frac, frac)
        if frac < -BORROW_TOL:
            self.reasons.append(
                f"negative_cash_borrowing: cash {self.cash:.2f} is {frac:.4f} of equity "
                f"at bar {j} (limit -{BORROW_TOL})"
            )

    def _reducing(self, k: int) -> bool:
        tgt, u = float(self.pending[k]), float(self.u[k])
        return abs(tgt) <= abs(u) and tgt * u >= 0.0

    def _try_fill(self, j: int, k: int, px: float, e_pre: float) -> bool:
        tgt = float(self.pending[k])
        pre = float(self.u[k])
        delta = tgt - pre
        want = abs(delta) * px
        if want <= 1e-15 * max(e_pre, 1.0):
            self.pending[k] = np.nan
            return False
        vol = float(self.p.fill_qv[j, k])
        if not (math.isfinite(vol) and vol > 0.0 and math.isfinite(self.p.close[j, k])):
            return False  # not tradable this bar: the target stays pending
        cap = MAX_PARTICIPATION * vol
        x = min(want, cap)
        reduce_part = min(x, abs(pre) * px) if delta * pre < 0.0 else 0.0
        open_part = x - reduce_part
        side = 1 if delta > 0 else -1
        e_now = e_pre - self._event_cost
        clamped = False
        if open_part > 0.0:
            held = self.long_n if side > 0 else self.short_n
            r = sum(unit_cost_rate(x / vol, self.mult))
            room = max(0.0, SIDE_CAP * e_now - held) / (1.0 + SIDE_CAP * r)
            if open_part > room:
                open_part, clamped = room, True
        x = reduce_part + open_part
        if x <= 1e-15 * max(e_pre, 1.0):
            if clamped:
                self.clamped += 1
            return False
        full = x >= want * (1.0 - 1e-15)
        fee_r, spr_r, imp_r = unit_cost_rate(x / vol, self.mult)
        total = x * (fee_r + spr_r + imp_r)
        extra = FORCED_EXTRA * x if self.pend_forced[k] else 0.0
        if full:
            qty = tgt - pre
            self.u[k] = tgt
            self.pending[k] = np.nan
        else:
            qty = math.copysign(x / px, delta)
            self.u[k] = pre + qty
            if want > cap:
                self.capped += 1
            if clamped:
                self.clamped += 1
        post = float(self.u[k])
        self.cash -= qty * px + total + extra
        sg = 1 if post > 0 else (-1 if post < 0 else (1 if pre > 0 else -1))
        self.cum_cost[sg] += total + extra
        self.bucket[int(self.p.age_bucket[j, k])] -= total + extra
        if extra:
            self.forced_fills += 1
            self.forced_charge += extra
        self.long_n += max(post, 0.0) * px - max(pre, 0.0) * px
        self.short_n += max(-post, 0.0) * px - max(-pre, 0.0) * px
        if open_part > 0.0:
            self._inc[side] = True
        self.turnover += x
        self.fills += 1
        self._event_cost += total + extra
        return True

    def fill_event(
        self, j: int, w: np.ndarray, force: np.ndarray, forced: np.ndarray
    ) -> None:
        px = self.p.mark_px[j]
        e_pre = self._equity(px)
        if not math.isfinite(e_pre) or e_pre <= 0.0:
            self.reasons.append(f"equity {e_pre!r} not positive before fill at bar {j}")
            return
        self._event_cost = 0.0
        self._inc = {1: False, -1: False}
        filled_any = False
        for k in np.nonzero((np.abs(w - self.w_prev) > TRADE_EPS) | force)[0]:
            self.w_prev[k] = float(w[k])
            if not (math.isfinite(px[k]) and px[k] > 0.0):
                self.pending[k] = np.nan  # no listing at this close: nothing to hold
                continue
            self.pending[k] = float(w[k]) * e_pre / px[k]
            self.pend_forced[k] = bool(forced[k])
        pend = np.nonzero(~np.isnan(self.pending))[0]
        self.pend_forced[pend] |= forced[pend]
        order = [k for k in pend if self._reducing(k)] + [
            k for k in pend if not self._reducing(k)
        ]
        for k in order:
            filled_any |= self._try_fill(j, int(k), float(px[k]), e_pre)
        e_post = self._equity(px)
        self.eq = e_post
        self.err = max(
            self.err, abs(e_post - (e_pre - self._event_cost)) / max(abs(e_pre), 1.0)
        )
        if not math.isfinite(e_post) or e_post <= 0.0:
            self.reasons.append(f"equity {e_post!r} not positive after fill at bar {j}")
            return
        if filled_any:
            for sg, notion in ((1, self.long_n), (-1, self.short_n)):
                if self._inc[sg] and notion > SIDE_CAP * e_post * (1.0 + IDENTITY_TOL):
                    self.reasons.append(
                        f"side notional {notion / e_post!r} > {SIDE_CAP!r} after an increasing fill, bar {j}"
                    )
            if self.cash < -IDENTITY_TOL * max(e_post, 1.0) and (
                self._inc[1] or self._inc[-1]
            ):
                self.reasons.append(
                    f"negative_cash_borrowing: a fill left cash {self.cash!r}, bar {j}"
                )

    def settle(self, j: int, k: int) -> None:
        """Write-off: a position held at the last available close of its listing is settled at
        that close and pays the extra charge."""
        self.pending[k] = np.nan
        self.w_prev[k] = 0.0
        pre = float(self.u[k])
        if pre == 0.0:
            return
        px = float(self.p.mark_px[j, k])
        e_pre = self._equity(self.p.mark_px[j])
        notional = abs(pre) * px
        charge = FORCED_EXTRA * notional
        self.cash += pre * px - charge
        self.u[k] = 0.0
        sg = 1 if pre > 0 else -1
        self.cum_cost[sg] += charge
        self.bucket[int(self.p.age_bucket[j, k])] -= charge
        self.forced_charge += charge
        self.long_n -= max(pre, 0.0) * px
        self.short_n -= max(-pre, 0.0) * px
        e_post = self._equity(self.p.mark_px[j])
        self.eq = e_post
        self.err = max(self.err, abs(e_post - (e_pre - charge)) / max(abs(e_pre), 1.0))
        self.write_offs.append(
            {
                "symbol": self.p.symbols[k],
                "bar": int(j),
                "side": sg,
                "notional": notional,
                "charge": charge,
            }
        )


def _side(d: dict[int, float]) -> dict[str, float]:
    return {"long": d[1] / INITIAL_CAPITAL, "short": d[-1] / INITIAL_CAPITAL}


def simulate_weights(
    panel: Panel,
    w: np.ndarray,
    multiplier: float,
    delay: int,
    force: np.ndarray | None = None,
) -> dict[str, Any]:
    """Replay one cell. Decisions at closes c_0..c_{n-1}; marks at c_1..c_n; the decision of
    close c_j fills at c_{j+delay}. The book starts flat with ``INITIAL_CAPITAL``."""
    n = panel.bars
    book = _Book(panel, multiplier)
    no_force = np.zeros(len(panel.symbols), dtype=bool)
    settle_at: dict[int, list[int]] = {}
    for j_e, k in panel.events:
        settle_at.setdefault(j_e, []).append(k)
    for j in range(n + 1):
        if j >= 1:
            book.mark(j)
        d = j - delay
        if not book.reasons and 0 <= d <= n - 1:
            book.fill_event(
                j, w[d], force[d] if force is not None else no_force, panel.forced[d]
            )
        if not book.reasons:
            for k in settle_at.get(j, []):
                book.settle(j, k)
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
    mean_eq = float(np.nanmean(equity)) * INITIAL_CAPITAL
    marks = max(book.marks, 1)
    out.update(
        fills=book.fills,
        capped_fills=book.capped,
        clamped_fills=book.clamped,
        forced_fills=book.forced_fills,
        forced_charge=book.forced_charge / INITIAL_CAPITAL,
        write_off_count=len(book.write_offs),
        write_off_notional=sum(x["notional"] for x in book.write_offs)
        / INITIAL_CAPITAL,
        write_offs=book.write_offs[:50],
        turnover_x=book.turnover / mean_eq if mean_eq > 0.0 else 0.0,
        price_pnl=_side(book.cum_price),
        funding_pnl=_side(book.cum_fund),
        cost_paid=_side(book.cum_cost),
        costs_total=sum(book.cum_cost.values()) / INITIAL_CAPITAL,
        funding_total=sum(book.cum_fund.values()) / INITIAL_CAPITAL,
        mean_long_gross=book.sum_long / marks,
        mean_short_gross=book.sum_short / marks,
        max_side_at_marks=book.max_side_mark,
        min_cash_frac=book.min_cash_frac if math.isfinite(book.min_cash_frac) else 0.0,
        age_bucket_pnl=dict(
            zip(AGE_LABELS, (book.bucket / INITIAL_CAPITAL).tolist(), strict=True)
        ),
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
        clamped_fills=0,
        forced_fills=0,
        forced_charge=0.0,
        write_off_count=0,
        write_off_notional=0.0,
        write_offs=[],
        turnover_x=0.0,
        price_pnl=dict(zero),
        funding_pnl=dict(zero),
        cost_paid=dict(zero),
        costs_total=0.0,
        funding_total=0.0,
        mean_long_gross=0.0,
        mean_short_gross=0.0,
        max_side_at_marks=0.0,
        min_cash_frac=1.0,
        age_bucket_pnl=dict.fromkeys(AGE_LABELS, 0.0),
        max_identity_err=0.0,
        invalid_reasons=[],
        valid=True,
    )
    return out


def summarize_cell(
    cell: dict[str, Any], times: np.ndarray | None = None
) -> dict[str, Any]:
    """JSON-safe cell record: stats plus digests of the equity series and per-year results."""
    out = {k: v for k, v in cell.items() if k not in ("returns", "equity")}
    out["bars"] = int(len(cell["returns"]))
    out["equity_sha256"] = hashlib.sha256(
        np.asarray(cell["equity"], dtype="<f8").tobytes()
    ).hexdigest()
    if times is not None and cell["valid"]:
        out["per_year"] = yearly(np.asarray(cell["returns"], dtype=float), times)
    return out


def evaluate_cells(panel: Panel, n_min: int = N_MIN) -> dict[str, dict[str, Any]]:
    w_ls = trend_weights(panel, n_min)
    w_long = np.maximum(w_ls, 0.0)
    w_ew, force = ew_weights(panel, n_min)
    cells: dict[str, dict[str, Any]] = {}
    cells[PRIMARY] = simulate_weights(panel, w_ls, COST_BASE, 0)
    cells[LONG_CELL] = simulate_weights(panel, w_long, COST_BASE, 0)
    cells[HIGH_COST_CELL] = simulate_weights(panel, w_ls, COST_HIGH, 0)
    cells[DELAY_CELL] = simulate_weights(panel, w_ls, COST_BASE, 1)
    cells[EW_CELL] = simulate_weights(panel, w_ew, COST_BASE, 0, force)
    cells[CASH_CELL] = cash_cell(panel.bars)
    return cells


# ----------------------------------------------------------------------------- decision


def block_bootstrap_means(
    series: dict[str, np.ndarray],
    block: int = BOOT_BLOCK,
    draws: int = BOOT_DRAWS,
    seed: int = BOOT_SEED,
) -> dict[str, np.ndarray]:
    """Paired moving-block bootstrap of the mean: overlapping blocks of ``block`` rows (shortened
    to the series length), one shared set of block starts for every series."""
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


def analyze(cells: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """The decision rule's statistics. Needs valid cells. ``SUPPORTED`` requires all of: base
    Sharpe >= 0.5; the paired bootstrap one-sided lower 95% bound of the mean 4h return
    difference ``TREND_LS__BASE - EW_LONG`` above 0; positive HIGH_COST and DELAY total returns;
    base max drawdown below 40%."""
    base = cells[PRIMARY]
    diff_r = np.asarray(base["returns"], dtype=float) - np.asarray(
        cells[EW_CELL]["returns"], dtype=float
    )
    means = block_bootstrap_means({"d": diff_r})["d"]
    lower = float(np.quantile(means, ALPHA))
    checks = {
        "all_cells_valid": all(c["valid"] for c in cells.values()),
        "base_sharpe_at_least_0.5": float(base["sharpe"]) >= SHARPE_MIN,
        "bootstrap_lower95_vs_ew_long_above_0": lower > 0.0,
        "high_cost_total_return_positive": float(cells[HIGH_COST_CELL]["total_return"])
        > 0.0,
        "delay_total_return_positive": float(cells[DELAY_CELL]["total_return"]) > 0.0,
        "base_max_dd_below_40pct": float(base["max_dd"]) < MAX_DD_LIMIT,
    }
    return {
        "mean_diff_vs_ew_long": float(np.mean(diff_r)),
        "bootstrap_lower95": lower,
        "p_one_sided": float((1 + np.sum(means <= 0.0)) / (len(means) + 1)),
        "base_sharpe": float(base["sharpe"]),
        "base_max_dd": float(base["max_dd"]),
        "base_total_return": float(base["total_return"]),
        "high_cost_total_return": float(cells[HIGH_COST_CELL]["total_return"]),
        "delay_total_return": float(cells[DELAY_CELL]["total_return"]),
        "ew_long_total_return": float(cells[EW_CELL]["total_return"]),
        "checks": checks,
        "supported": all(checks.values()),
    }


def decide(
    cells: dict[str, dict[str, Any]], problems: list[str] | None = None
) -> dict[str, Any]:
    """Label. Priority: INVALID (any problem or invalid cell), then the frozen rule."""
    problems = list(problems or [])
    invalid = sorted(k for k, c in cells.items() if not c["valid"])
    out: dict[str, Any] = {"invalid_cells": invalid, "problems": problems}
    if invalid or problems or not cells:
        out["label"] = LABEL_INVALID
        return out
    out["analysis"] = analyze(cells)
    out["label"] = (
        LABEL_SUPPORTED if out["analysis"]["supported"] else LABEL_NOT_SUPPORTED
    )
    return out


# ----------------------------------------------------------------------------- archives

_KLINE_MONTH = re.compile(r"^(?P<s>[A-Z0-9]+)-1h-(?P<y>\d{4})-(?P<m>\d{2})\.zip$")
_KLINE_DAY = re.compile(
    r"^(?P<s>[A-Z0-9]+)-1h-(?P<y>\d{4})-(?P<m>\d{2})-(?P<d>\d{2})\.zip$"
)
_FUND_MONTH = re.compile(
    r"^(?P<s>[A-Z0-9]+)-fundingRate-(?P<y>\d{4})-(?P<m>\d{2})\.zip$"
)
_FUND_REST = re.compile(
    r"^(?P<s>[A-Z0-9]+)-fundingRate-(?P<y>\d{4})-(?P<m>\d{2})\.json$"
)

KLINE_COLS = 12


@dataclass
class SymbolFiles:
    monthly: dict[tuple[int, int], Path]
    daily: dict[tuple[int, int], list[Path]]
    funding: dict[tuple[int, int], Path]
    rest: dict[tuple[int, int], Path]
    embargoed: int = 0


def _in_window(y: int, m: int, stage: Stage, day: int = 1) -> bool:
    start = datetime(y, m, day, tzinfo=UTC)
    month_end = datetime(y + (m == 12), 1 if m == 12 else m + 1, 1, tzinfo=UTC)
    return start < stage.end and month_end > stage.grid_start


def scan_symbol(root: Path, symbol: str, stage: Stage) -> SymbolFiles:
    """Files of one symbol that may be read. A file whose month (or day) lies at or after the
    stage end is never opened (the embargo) and only counted."""
    um = root / "futures" / "um"
    out = SymbolFiles({}, {}, {}, {})

    def listing(path: Path) -> list[Path]:
        return sorted(path.iterdir()) if path.is_dir() else []

    for p in listing(um / "monthly" / "klines" / symbol / "1h"):
        mt = _KLINE_MONTH.match(p.name)
        if mt and mt["s"] == symbol:
            y, m = int(mt["y"]), int(mt["m"])
            if _in_window(y, m, stage):
                out.monthly[(y, m)] = p
            elif datetime(y, m, 1, tzinfo=UTC) >= stage.end:
                out.embargoed += 1
    for p in listing(um / "daily" / "klines" / symbol / "1h"):
        mt = _KLINE_DAY.match(p.name)
        if mt and mt["s"] == symbol:
            y, m, d = int(mt["y"]), int(mt["m"]), int(mt["d"])
            if datetime(y, m, d, tzinfo=UTC) >= stage.end:
                out.embargoed += 1
            elif _in_window(y, m, stage, d):
                out.daily.setdefault((y, m), []).append(p)
    for p in listing(um / "monthly" / "fundingRate" / symbol):
        mt = _FUND_MONTH.match(p.name)
        if mt and mt["s"] == symbol:
            y, m = int(mt["y"]), int(mt["m"])
            if _in_window(y, m, stage):
                out.funding[(y, m)] = p
    for p in listing(root / "rest" / "fundingRate" / symbol):
        mt = _FUND_REST.match(p.name)
        if mt and mt["s"] == symbol:
            y, m = int(mt["y"]), int(mt["m"])
            if _in_window(y, m, stage):
                out.rest[(y, m)] = p
    return out


def used_files(files: SymbolFiles) -> list[Path]:
    """Every file the reader may open: all monthly and daily kline zips and funding files."""
    out = (
        list(files.monthly.values())
        + list(files.funding.values())
        + list(files.rest.values())
    )
    for v in files.daily.values():
        out.extend(v)
    return sorted(out)


def _checksum_line(path: Path) -> str | None:
    side = path.with_name(path.name + ".CHECKSUM")
    if not side.is_file():
        return None
    parts = side.read_text(encoding="utf-8").split()
    return parts[0].lower() if parts else None


def archive_preflight(
    root: Path, symbols: tuple[str, ...], stage: Stage
) -> dict[str, Any]:
    """Result-blind pre-claim check of the archives: every symbol has monthly kline files; every
    zip matches its ``.CHECKSUM`` sidecar and holds one csv member. No archive value is parsed.
    Returns the manifest digest (relative path + sha256 of every file that may be read)."""
    lines: list[str] = []
    n_files = 0
    embargoed = 0
    for sym in symbols:
        files = scan_symbol(root, sym, stage)
        embargoed += files.embargoed
        if not files.monthly and not files.daily:
            raise SystemExit(f"{sym}: no kline archive in the stage window")
        for p in used_files(files):
            digest = sha256_file(p)
            if p.suffix == ".zip":
                want = _checksum_line(p)
                if want is None:
                    raise SystemExit(f"missing .CHECKSUM sidecar for {p}")
                if want != digest:
                    raise SystemExit(f"checksum mismatch for {p}")
                try:
                    with zipfile.ZipFile(p) as z:
                        if len(z.namelist()) != 1:
                            raise SystemExit(f"{p} does not hold exactly one member")
                except zipfile.BadZipFile:
                    # a malformed monthly archive may be rebuilt from daily files at read time
                    pass
            lines.append(f"{p.relative_to(root).as_posix()}\0{digest}")
            n_files += 1
    manifest = hashlib.sha256("\n".join(sorted(lines)).encode("utf-8")).hexdigest()
    return {
        "files": n_files,
        "manifest_sha256": manifest,
        "embargoed_files_ignored": embargoed,
    }


def parse_kline_csv(text: str) -> np.ndarray:
    """Binance Vision 1h kline csv -> (rows, 12) float array. A header line is skipped."""
    text = text.strip()
    if not text:
        return np.empty((0, KLINE_COLS))
    first, _, rest = text.partition("\n")
    if first[:1].isalpha():
        text = rest
    if not text.strip():
        return np.empty((0, KLINE_COLS))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        try:
            arr = np.fromstring(text.replace("\n", ",").replace("\r", ""), sep=",")
        except (ValueError, DeprecationWarning):
            raise ValueError("malformed kline csv") from None
    if arr.size % KLINE_COLS:
        raise ValueError("kline csv row width is not 12")
    return arr.reshape(-1, KLINE_COLS)


def _zip_text(path: Path) -> str:
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        if len(names) != 1:
            raise ValueError("archive must hold one member")
        return z.read(names[0]).decode("utf-8")


def read_klines(
    files: SymbolFiles, stage: Stage
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Hourly grid arrays (NaN holes) of one symbol. A month comes from its monthly zip; if that
    is missing or malformed, from the daily zips of the month. A row at or after the stage end
    is refused; there is no forward fill."""
    n = stage.n_hours
    start_ms = int(stage.grid_start.timestamp()) * 1000
    end_ms = int(stage.end.timestamp()) * 1000
    keys = ("open", "high", "low", "close", "qv", "vol")
    arrs = {k: np.full(n, np.nan) for k in keys}
    stats = {"monthly": 0, "daily_months": 0, "malformed_months": 0, "rows": 0}
    months = sorted(set(files.monthly) | set(files.daily))
    for ym in months:
        blocks: list[np.ndarray] = []
        done = False
        if ym in files.monthly:
            try:
                blocks = [parse_kline_csv(_zip_text(files.monthly[ym]))]
                done = True
                stats["monthly"] += 1
            except (ValueError, zipfile.BadZipFile):
                stats["malformed_months"] += 1
        if not done and ym in files.daily:
            for p in files.daily[ym]:
                try:
                    blocks.append(parse_kline_csv(_zip_text(p)))
                except (ValueError, zipfile.BadZipFile):
                    stats["malformed_months"] += 1
            stats["daily_months"] += 1
        for b in blocks:
            if not b.size:
                continue
            if np.any(b[:, 0] >= 1e14):
                raise SystemExit("kline open time is not in milliseconds")
            open_ms = b[:, 0].astype(np.int64)
            if np.any(open_ms % HOUR_MS):
                raise SystemExit("kline open time is not on the hour")
            if np.any(open_ms >= end_ms):
                raise SystemExit("archive row at or after the stage end (embargo)")
            idx = (open_ms - start_ms) // HOUR_MS
            ok = idx >= 0
            idx = idx[ok]
            if np.isfinite(arrs["close"][idx]).any():
                raise SystemExit("duplicate hourly bar in the archives")
            for key, col in zip(keys, (1, 2, 3, 4, 7, 5), strict=True):
                arrs[key][idx] = b[ok, col]
            stats["rows"] += int(idx.size)
    return arrs, stats


def read_funding(
    files: SymbolFiles, stage: Stage
) -> tuple[list[tuple[int, float, float]], dict[str, int]]:
    """Funding events ``(time ms, rate, mark price or NaN)`` strictly before the stage end. A
    month comes from its Vision zip, else from the cached REST json (which carries a mark)."""
    end_ms = int(stage.end.timestamp()) * 1000
    events: dict[int, tuple[float, float]] = {}
    stats = {"vision_months": 0, "rest_months": 0}
    for ym in sorted(set(files.funding) | set(files.rest)):
        vision_ok = False
        if ym in files.funding:
            try:
                text = _zip_text(files.funding[ym]).strip().splitlines()
                rows = [ln.split(",") for ln in text if ln[:1].isdigit()]
                parsed = {int(r[0]): (float(r[2]), math.nan) for r in rows}
            except (ValueError, IndexError, zipfile.BadZipFile):
                parsed = {}  # malformed month: fall back to REST when cached
            else:
                vision_ok = True
                events.update(parsed)
                stats["vision_months"] += 1
        if not vision_ok and ym in files.rest:
            payload = json.loads(files.rest[ym].read_text(encoding="utf-8"))
            stats["rest_months"] += 1
            if not isinstance(payload, list):
                raise SystemExit("malformed REST funding payload")
            for it in payload:
                raw = it.get("markPrice")
                events[int(it["fundingTime"])] = (
                    float(it["fundingRate"]),
                    float(raw) if raw not in (None, "") else math.nan,
                )
    return [(t, r, m) for t, (r, m) in sorted(events.items()) if t < end_ms], stats


class LazyArchive(Mapping[str, dict[str, np.ndarray]]):
    """Streams the hourly grids of the symbols from the archives, one at a time (the full set of
    441 grids would need about 1 GB). Read statistics are recorded into ``stats``."""

    def __init__(
        self, root: Path, stage: Stage, symbols: tuple[str, ...], stats: dict[str, Any]
    ) -> None:
        self.root, self.stage, self.symbols, self.stats = root, stage, symbols, stats

    def __getitem__(self, symbol: str) -> dict[str, np.ndarray]:
        arrs, ks = read_klines(scan_symbol(self.root, symbol, self.stage), self.stage)
        self.stats.setdefault(symbol, {}).update(ks)
        return arrs

    def __iter__(self) -> Iterator[str]:
        return iter(self.symbols)

    def __len__(self) -> int:
        return len(self.symbols)


def load_universe(
    path: Path = UNIVERSE_PATH, sha: str = UNIVERSE_SHA256
) -> tuple[str, ...]:
    """The frozen 441-symbol DISJOINT candidate list, verified against its pinned sha256."""
    if sha256_file(path) != sha:
        raise SystemExit(f"universe file {path} does not match its pinned sha256")
    symbols = tuple(json.loads(path.read_text(encoding="utf-8"))["perp_symbols"])
    if len(symbols) != N_UNIVERSE or len(set(symbols)) != N_UNIVERSE:
        raise SystemExit("universe must hold 441 distinct symbols")
    return symbols


def load_notices(
    path: Path = NOTICES_PATH, sha: str = NOTICES_SHA256
) -> dict[str, list[float]]:
    """Recorded delisting notices per symbol (release time, epoch seconds), sha256-verified."""
    if sha256_file(path) != sha:
        raise SystemExit(f"notices file {path} does not match its pinned sha256")
    out: dict[str, list[float]] = {}
    for e in json.loads(path.read_text(encoding="utf-8"))["entries"]:
        stamp = datetime.fromisoformat(e["release_utc"].replace("Z", "+00:00"))
        out.setdefault(e["symbol"], []).append(stamp.timestamp())
    return {k: sorted(v) for k, v in out.items()}


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


# ----------------------------------------------------------------------------- ledger


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


def check_cross_digest(root: Path, name: str) -> None:
    """Single use is keyed by protocol id + stage, across ALL digest namespaces under ``root``: a
    harness commit changes the digest but never reopens a consumed stage. Refuses before any
    claim, read or replay."""
    if (root / f"{name}.claim").exists():
        raise SystemExit(
            f"stage {name} was already claimed at the protocol root: "
            "single use applies across digests"
        )
    for ns in _namespaces(root):
        results = ledger_results(ns)
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


def library_versions() -> dict[str, str | None]:  # pragma: no cover - environment
    out: dict[str, str | None] = {}
    for pkg in ("numpy",):
        try:
            out[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            out[pkg] = None
    return out


# ----------------------------------------------------------------------------- run


def breadth_report(panel: Panel, info: dict[str, Any], n_min: int) -> dict[str, Any]:
    """Descriptive, non-deciding facts about the cross-section (no return is used)."""
    elig = eligible_matrix(panel)
    n_t = elig.sum(axis=1)
    live = n_t >= n_min
    first = int(np.argmax(live)) if live.any() else None
    return {
        "eligible_min": int(n_t.min()),
        "eligible_median": float(np.median(n_t)),
        "eligible_max": int(n_t.max()),
        "bars_with_book": int(live.sum()),
        "first_book_close": None if first is None else int(panel.times[first]),
        "symbols_ever_eligible": int((elig.sum(axis=0) > 0).sum()),
        "events": len(panel.events),
        "notice_matched": info["notice_matched"],
        "notice_fallback": info["notice_fallback"],
        "notice_late": info["notice_late"],
        "funding_mark_fallbacks": info["funding_mark_fallbacks"],
        "funding_events_unpriced": info["funding_events_unpriced"],
        "funding_uncovered_fraction": info.get("funding_uncovered_fraction"),
    }


def run_stage(
    stage_name: Stage | str,
    authorization_url: str,
    output_dir: Path,
    archive_root: Path = ARCHIVE_ROOT,
    now: datetime | None = None,
    gh_fetch: Callable[[int], dict[str, Any]] = _gh_comment,
    ledger_dir: Path = LEDGER_DIR,
    symbols: tuple[str, ...] | None = None,
    notices: dict[str, list[float]] | None = None,
    n_min: int = N_MIN,
) -> dict[str, Any]:
    stage = STAGES[stage_name] if isinstance(stage_name, str) else stage_name
    now = now or datetime.now(UTC)
    if output_dir.exists():
        raise SystemExit(f"output directory exists: {output_dir}")
    syms = symbols if symbols is not None else load_universe()
    if notices is None:
        notices = load_notices()
    digest = protocol_digest()
    name = stage.name
    ledger_root = ledger_dir / PROTOCOL_ID
    check_cross_digest(ledger_root, name)
    ledger_dir = ledger_root / digest
    commit = harness_commit()
    auth = authorization_receipt(authorization_url, digest, commit, gh_fetch)
    preflight = archive_preflight(archive_root, syms, stage)  # refuses WITHOUT claiming
    # The protocol-root claim is the atomic single-use lock across digests (exclusive create);
    # the per-digest copy only records provenance.
    claim_stage(name, digest, commit, auth["g4_relay"]["comment_id"], now, ledger_root)
    claim = claim_stage(
        name, digest, commit, auth["g4_relay"]["comment_id"], now, ledger_dir
    )

    funding: dict[str, list[tuple[int, float, float]]] = {}
    read_stats: dict[str, Any] = {}
    for sym in syms:
        files = scan_symbol(archive_root, sym, stage)
        funding[sym], fs = read_funding(files, stage)
        read_stats[sym] = dict(fs)
    hourly = LazyArchive(archive_root, stage, syms, read_stats)
    panel, info = build_panel(hourly, stage, funding, notices, syms)
    problems = panel_problems(panel, info)
    result: dict[str, Any] = {
        "protocol_id": PROTOCOL_ID,
        "protocol_digest": digest,
        "commit": commit,
        "stage": stage.name,
        "claim": str(claim),
        "authorization": auth,
        "libraries": library_versions(),
        "data": {
            "universe_sha256": UNIVERSE_SHA256 if symbols is None else None,
            "notices_sha256": NOTICES_SHA256,
            "symbols": len(syms),
            "preflight": preflight,
            "read": read_stats,
        },
        "window": {
            "grid_start": stage.grid_start.isoformat(),
            "first_decision_close": stage.first_close.isoformat(),
            "end_exclusive": stage.end.isoformat(),
            "bars": int(panel.bars),
        },
        "run_at": now.isoformat(),
    }
    cells: dict[str, dict[str, Any]] = {}
    if not problems:
        cells = evaluate_cells(panel, n_min)
        result["breadth"] = breadth_report(panel, info, n_min)
    decision = decide(cells, problems)
    result.update(
        {
            "cells": {k: summarize_cell(v, panel.times) for k, v in cells.items()},
            **decision,
        }
    )
    output_dir.mkdir(parents=True)
    _write_once(output_dir / "result.json", result)
    write_ledger_result(
        name,
        digest,
        result["label"],
        result["label"] == LABEL_SUPPORTED,
        sha256_file(output_dir / "result.json"),
        ledger_dir,
    )
    return result


# ----------------------------------------------------------------------------- CLI


def cmd_digest(_args: argparse.Namespace) -> None:
    print(protocol_digest())


def cmd_run(args: argparse.Namespace) -> None:  # pragma: no cover - real archives
    result = run_stage(
        args.stage,
        args.authorization_url,
        Path(args.output_dir),
        archive_root=Path(args.archive_root),
    )
    print(
        json.dumps(
            {k: result[k] for k in ("stage", "label", "invalid_cells", "problems")},
            indent=1,
        )
    )
    for name, st in result["cells"].items():
        print(
            f"{name:22s} ret {st['total_return'] * 100:+8.2f}% dd {st['max_dd'] * 100:5.2f}% "
            f"sr {st['sharpe']:+.2f} valid {st['valid']}"
        )


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="trend_breadth")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("digest").set_defaults(fn=cmd_digest)
    p = sub.add_parser("run")
    p.add_argument("--stage", required=True, choices=sorted(STAGES))
    p.add_argument("--authorization-url", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--archive-root", default=str(ARCHIVE_ROOT))
    p.set_defaults(fn=cmd_run)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
