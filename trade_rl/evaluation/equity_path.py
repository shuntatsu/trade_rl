"""Daily UTC sampling and explicit annualization of native positive equity paths.

Read-only statistics; no replay, synthetic accounting, selection or data access.
Historical study-specific module bytes remain unchanged until source cutover.
"""

from __future__ import annotations

import math
from numbers import Real
from typing import TypedDict

import numpy as np


def _require_plain_native_array(value: np.ndarray, *, field_name: str) -> np.ndarray:
    """Reject masks, subclasses and opaque owners before reading numeric values.

    Exact ndarray/memoryview owner links can be inspected without invoking
    arbitrary user attributes. Owned arrays, native views, and exact raw-byte
    buffers remain supported. Missingness already erased by the caller cannot
    be recovered.
    """
    error = f"{field_name} must be a native array without opaque or masked ownership"
    if type(value) is not np.ndarray:
        raise ValueError(error)
    owner: object = value
    while True:
        if type(owner) is np.ndarray:
            owner = np.ndarray.base.__get__(owner)
        elif type(owner) is memoryview:
            owner = owner.obj
        else:
            break
    if owner is not None and type(owner) is not bytes and type(owner) is not bytearray:
        raise ValueError(error)
    return value


_NS_PER_DAY = 24 * 3_600 * 1_000_000_000


def _equity_path(equity: np.ndarray) -> np.ndarray:
    raw = _require_plain_native_array(equity, field_name="equity")
    if raw.dtype.kind not in "iuf":
        raise ValueError("equity must contain real numeric values")
    path = np.asarray(raw, dtype=np.float64)
    if (
        path.ndim != 1
        or path.size < 2
        or not np.isfinite(path).all()
        or np.any(path <= 0.0)
    ):
        raise ValueError("equity path must hold at least two positive finite values")
    return path


def _aligned_path(
    timestamps_ns: np.ndarray, equity: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    ts = _require_plain_native_array(timestamps_ns, field_name="timestamps")
    path = _equity_path(equity)
    if ts.dtype != np.dtype(np.int64) or ts.shape != path.shape:
        raise ValueError("timestamps must be aligned one-dimensional int64 nanoseconds")
    if np.any(ts == np.iinfo(np.int64).min) or np.any(ts[1:] <= ts[:-1]):
        raise ValueError("timestamps must be valid and strictly increasing")
    return ts, path


def max_drawdown(equity: np.ndarray) -> float:
    """Largest peak-to-trough decline, as a positive fraction."""
    path = _equity_path(equity)
    peaks = np.maximum.accumulate(path)
    return float(np.max(1.0 - path / peaks))


def daily_equity(
    timestamps_ns: np.ndarray, equity: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Sample actual UTC-midnight rows; never interpolate a missing day."""
    ts, path = _aligned_path(timestamps_ns, equity)
    mask = ts % _NS_PER_DAY == 0
    if int(mask.sum()) < 3:
        raise ValueError("equity path needs at least three 00:00 UTC samples")
    return ts[mask], path[mask]


def daily_returns(daily: np.ndarray) -> np.ndarray:
    """Simple returns between supplied equity observations."""
    values = _equity_path(daily)
    with np.errstate(over="ignore", invalid="ignore"):
        returns = values[1:] / values[:-1] - 1.0
    if not np.isfinite(returns).all():
        raise ValueError("equity returns must be finite")
    return returns


def _year_of(timestamp_ns: int) -> int:
    return (
        int(np.datetime64(timestamp_ns, "ns").astype("datetime64[Y]").astype(int))
        + 1970
    )


def per_year_returns(timestamps_ns: np.ndarray, equity: np.ndarray) -> dict[str, float]:
    """Historical calendar-segment convention, including the next year boundary if observed."""
    ts, path = _aligned_path(timestamps_ns, equity)
    result: dict[str, float] = {}
    first_year = _year_of(int(ts[0]))
    last_year = _year_of(int(ts[-1]))
    for year in range(first_year, last_year + 1):
        # Use the endpoint when the calendar boundary lies beyond ns range.
        year_start = (
            int(np.datetime64(f"{year}-01-01", "D").astype(np.int64)) * _NS_PER_DAY
        )
        year_end = (
            int(np.datetime64(f"{year + 1}-01-01", "D").astype(np.int64)) * _NS_PER_DAY
        )
        begin = int(np.searchsorted(ts, max(year_start, int(ts[0])), side="left"))
        end = int(np.searchsorted(ts, min(year_end, int(ts[-1])), side="right")) - 1
        if end <= begin:
            continue
        with np.errstate(over="ignore", invalid="ignore"):
            value = float(path[end] / path[begin] - 1.0)
        if not math.isfinite(value):
            raise ValueError("calendar-year return must be finite")
        result[str(year)] = value
    return result


class EquityPathSummary(TypedDict):
    """Explicit percentage fields and elapsed-year statistics."""

    terminal_net_profit_pct: float
    cagr_pct: float
    max_drawdown_pct: float
    sharpe_daily: float | None
    n_daily_returns: int
    years: float
    per_year_return_pct: dict[str, float]


def _annualization(value: object) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError("annualization scales must be finite and positive")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError("annualization scales must be finite and positive") from exc
    if not math.isfinite(result) or result <= 0:
        raise ValueError("annualization scales must be finite and positive")
    return result


def performance_summary(
    timestamps_ns: np.ndarray,
    equity: np.ndarray,
    *,
    cagr_days_per_year: float,
    trading_days_per_year: float,
) -> EquityPathSummary:
    """Historical daily convention, with explicit caller-owned annualization.

    CAGR uses elapsed ns; daily Sharpe uses sample std (ddof=1). Caller owns
    whether the provided series covers the entire requested scoring window.
    """
    ts, path = _aligned_path(timestamps_ns, equity)
    cagr_days_per_year = _annualization(cagr_days_per_year)
    trading_days_per_year = _annualization(trading_days_per_year)
    with np.errstate(over="ignore", invalid="ignore"):
        terminal = float(path[-1] / path[0] - 1.0)
        years = float(int(ts[-1]) - int(ts[0])) / _NS_PER_DAY / cagr_days_per_year
        if not math.isfinite(years) or years <= 0:
            raise ValueError("elapsed years must be finite and positive")
        cagr = float((path[-1] / path[0]) ** (1.0 / years) - 1.0)
        _, daily = daily_equity(ts, path)
        returns = daily_returns(daily)
        std = float(np.std(returns, ddof=1))
        sharpe = (
            float(np.mean(returns) / std * math.sqrt(trading_days_per_year))
            if std > 0.0 and math.isfinite(std)
            else None
        )
    if not all(math.isfinite(x) for x in (terminal, years, cagr, std)) or (
        sharpe is not None and not math.isfinite(sharpe)
    ):
        raise ValueError("equity performance statistics must be finite")
    year_pct = {
        year: 100.0 * value for year, value in per_year_returns(ts, path).items()
    }
    if not all(
        math.isfinite(value)
        for value in (100.0 * terminal, 100.0 * cagr, *year_pct.values())
    ):
        raise ValueError("percentage statistics must be finite")
    return {
        "terminal_net_profit_pct": 100.0 * terminal,
        "cagr_pct": 100.0 * cagr,
        "max_drawdown_pct": 100.0 * max_drawdown(path),
        "sharpe_daily": sharpe,
        "n_daily_returns": int(returns.size),
        "years": years,
        "per_year_return_pct": year_pct,
    }
