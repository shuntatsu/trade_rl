"""Synthetic fixtures of ``xs_lowvol_funding_v1`` (no market data, no network).

Every series is a deterministic function, so tests can compute expected numbers by hand:
the 4h log return of row ``r`` is ``s * e(r)`` with ``e(r) = +1`` for an even row and ``-1``
for an odd one, hence any window of an even number of rows has mean zero and a sample
variance of ``n s^2 / (n - 1)``.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np

from trade_rl.evaluation.core7_mtf_study.tierd import LegRules
from trade_rl.evaluation.xs_lowvol_funding import data, protocol

HOUR_MS = 3_600_000
LOAD_START = datetime(2022, 1, 3, tzinfo=UTC)  # a Monday
WINDOW = data.StageWindow(
    stage=protocol.STAGE_DEV,
    load_start=LOAD_START,
    first_decision=datetime(2022, 3, 7, tzinfo=UTC),
    last_decision_bar=datetime(2022, 3, 21, tzinfo=UTC),
    settles=datetime(2022, 3, 25, tzinfo=UTC),
)
SYMBOLS = (
    "AAAUSDT",
    "BBBUSDT",
    "CCCUSDT",
    "DDDUSDT",
    "EEEUSDT",
    "FFFUSDT",
    "GGGUSDT",
)
#: per-4h-bar log-return magnitude and per-event funding rate of each symbol: volatility and
#: funding both increase with the symbol order, so AAA/BBB are the long leg and GGG/FFF the
#: short leg of the candidate.
STEP = dict(
    zip(SYMBOLS, (0.005, 0.010, 0.015, 0.020, 0.025, 0.030, 0.035), strict=True)
)
RATE = dict(zip(SYMBOLS, (-3e-4, -2e-4, -1e-4, 0.0, 1e-4, 2e-4, 3e-4), strict=True))


def n_hours(window: data.StageWindow = WINDOW) -> int:
    return (window.last_row_index + 1) * protocol.BAR_HOURS


def hourly_leg(
    total_hours: int,
    *,
    step: float,
    rate: float,
    volume: float = 1e9,
    first_hour: int = 0,
    last_hour: int | None = None,
    grid_start: datetime = LOAD_START,
    funding_every_hours: int = 8,
    volume_from: tuple[int, float] | None = None,
) -> data.PerpHourly:
    """Hourly series on the grid: 4h bars alternate ``+step`` and ``-step`` log returns."""
    stop = total_hours if last_hour is None else last_hour + 1
    present = np.zeros(total_hours, dtype=np.bool_)
    present[first_hour:stop] = True
    open_ = np.full(total_hours, np.nan)
    close = np.full(total_hours, np.nan)
    price = 100.0
    for hour in range(first_hour, stop):
        block = hour // protocol.BAR_HOURS
        open_[hour] = price
        if hour % protocol.BAR_HOURS == protocol.BAR_HOURS - 1:
            price = price * float(np.exp(step * (1.0 if block % 2 == 0 else -1.0)))
        close[hour] = price
    high = np.fmax(open_, close)
    low = np.fmin(open_, close)
    vol = np.where(present, volume, np.nan)
    if volume_from is not None:
        hour_from, thin = volume_from
        vol[hour_from:stop] = thin
    start_ms = int(grid_start.timestamp() * 1_000)
    events = [
        start_ms + hour * HOUR_MS
        for hour in range(first_hour, stop)
        if hour % funding_every_hours == 0 and hour > first_hour
    ]
    return data.PerpHourly(
        present=present,
        halted=np.zeros(total_hours, dtype=np.bool_),
        open=open_,
        high=high,
        low=low,
        close=close,
        quote_volume=vol,
        funding_ms=np.asarray(events, dtype=np.int64),
        funding_rate=np.full(len(events), rate, dtype=np.float64),
    )


def legs(
    window: data.StageWindow = WINDOW,
    *,
    symbols: tuple[str, ...] = SYMBOLS,
    overrides: dict[str, dict[str, object]] | None = None,
) -> dict[str, data.PerpHourly]:
    total = n_hours(window)
    out: dict[str, data.PerpHourly] = {}
    for symbol in symbols:
        kwargs: dict[str, object] = {"step": STEP[symbol], "rate": RATE[symbol]}
        kwargs.update((overrides or {}).get(symbol, {}))
        out[symbol] = hourly_leg(total, grid_start=window.load_start, **kwargs)  # type: ignore[arg-type]
    return out


def stage(
    window: data.StageWindow = WINDOW,
    *,
    overrides: dict[str, dict[str, object]] | None = None,
    symbols: tuple[str, ...] = SYMBOLS,
    data_authorization_url: str | None = None,
) -> data.StageData:
    hourly = legs(window, symbols=symbols, overrides=overrides)
    names = tuple(sorted(hourly))
    clock = data.stage_clock(window, window.last_row_close)
    member = np.ones((clock.size, len(names)), dtype=np.bool_)
    return data.assemble_stage_dataset(
        window,
        hourly,
        member=member,
        universe={},
        static_universe=True,
        rules={name: LegRules(0.001, 5.0) for name in names},
        limit_close=window.last_row_close,
        data_authorization_url=data_authorization_url,
    )


def last_hour_of_row(window: data.StageWindow, row: int) -> int:
    """Index of the last hourly open of 4h row ``row``."""
    return (row + 1) * protocol.BAR_HOURS - 1


def close_of(row: int, window: data.StageWindow = WINDOW) -> datetime:
    return window.load_start + (row + 1) * timedelta(hours=protocol.BAR_HOURS)
