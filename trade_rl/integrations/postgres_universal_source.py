"""Read-only, fail-closed loader for the maintained universal PostgreSQL source."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Final, Protocol

import numpy as np

MAINTAINED_SYMBOLS: Final = (
    "ADAUSDT",
    "APTUSDT",
    "ARBUSDT",
    "AVAXUSDT",
    "BCHUSDT",
    "BNBUSDT",
    "BTCUSDT",
    "DOGEUSDT",
    "ETHUSDT",
    "LINKUSDT",
    "LTCUSDT",
    "OPUSDT",
    "SOLUSDT",
    "SUIUSDT",
    "XRPUSDT",
)
_MINUTE_NS: Final = 60_000_000_000


class _Cursor(Protocol):
    def execute(self, query: str, params: object = None) -> Any: ...

    def fetchall(self) -> Sequence[Sequence[object]]: ...

    def __enter__(self) -> _Cursor: ...

    def __exit__(self, *args: object) -> None: ...


class UniversalSourceConnection(Protocol):
    """Small DB-API surface required by the source loader."""

    def cursor(self) -> _Cursor: ...


def _aware_utc(value: datetime, *, field: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field} must include a timezone")
    return value.astimezone(UTC)


def _epoch_ns(value: datetime) -> int:
    return int(round(value.timestamp() * 1_000_000_000.0))


@dataclass(frozen=True, slots=True)
class UniversalSourceScope:
    """Exact symbol and half-open time scope authorized for source reads."""

    symbols: tuple[str, ...]
    start: datetime
    end: datetime
    source: str = "binance"

    def __post_init__(self) -> None:
        if not self.symbols or any(not symbol for symbol in self.symbols):
            raise ValueError("source symbols must contain non-empty strings")
        if len(set(self.symbols)) != len(self.symbols):
            raise ValueError("source symbols must be unique")
        if not self.source:
            raise ValueError("source identity must not be empty")
        start = _aware_utc(self.start, field="source start")
        end = _aware_utc(self.end, field="source end")
        if end <= start:
            raise ValueError("source end must be later than source start")
        if _epoch_ns(start) % _MINUTE_NS or _epoch_ns(end) % _MINUTE_NS:
            raise ValueError("source range must align to the one-minute clock")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)

    @classmethod
    def maintained(cls) -> UniversalSourceScope:
        """Return the frozen 15-symbol real-data research scope."""

        return cls(
            symbols=MAINTAINED_SYMBOLS,
            start=datetime(2024, 11, 13, tzinfo=UTC),
            end=datetime(2026, 7, 5, tzinfo=UTC),
        )


@dataclass(frozen=True, slots=True)
class RawSymbolSource:
    """Validated one-minute source and explicitly sparse auxiliary channels."""

    timestamps: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    base_volume: np.ndarray
    funding_timestamps: np.ndarray
    funding_rate: np.ndarray
    derivative_timestamps: np.ndarray
    derivative_values: np.ndarray
    orderflow_timestamps: np.ndarray
    orderflow_values: np.ndarray

    def __post_init__(self) -> None:
        for value in (
            self.timestamps,
            self.open,
            self.high,
            self.low,
            self.close,
            self.base_volume,
            self.funding_timestamps,
            self.funding_rate,
            self.derivative_timestamps,
            self.derivative_values,
            self.orderflow_timestamps,
            self.orderflow_values,
        ):
            value.setflags(write=False)


def _timestamp_ns(value: object, *, field: str) -> int:
    if not isinstance(value, datetime):
        raise ValueError(f"{field} must be a datetime")
    return _epoch_ns(_aware_utc(value, field=field))


def _finite_float(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _load_klines(
    cursor: _Cursor,
    *,
    scope: UniversalSourceScope,
    symbol: str,
    expected_ns: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cursor.execute(
        """
        SELECT timestamp, open, high, low, close, volume
        FROM public.rl_klines
        WHERE source = %s AND symbol = %s AND timeframe = '1m'
          AND timestamp >= %s AND timestamp < %s
        ORDER BY timestamp
        """,
        (scope.source, symbol, scope.start, scope.end),
    )
    rows = tuple(cursor.fetchall())
    if any(len(row) != 6 for row in rows):
        raise ValueError(f"raw OHLCV row contract failed for {symbol}")
    timestamps_ns = np.asarray(
        [_timestamp_ns(row[0], field=f"raw kline timestamp for {symbol}") for row in rows],
        dtype=np.int64,
    )
    if timestamps_ns.size > 1 and np.any(np.diff(timestamps_ns) <= 0):
        raise ValueError(f"raw one-minute timestamps are not unique for {symbol}")
    if not np.array_equal(timestamps_ns, expected_ns):
        raise ValueError(f"raw one-minute timestamps are not contiguous for {symbol}")
    ohlcv = np.asarray(
        [
            [
                _finite_float(value, field=f"raw OHLCV for {symbol}")
                for value in row[1:]
            ]
            for row in rows
        ],
        dtype=np.float64,
    )
    if not np.isfinite(ohlcv).all():
        raise ValueError(f"raw OHLCV is not finite for {symbol}")
    open_, high, low, close, base_volume = (ohlcv[:, index] for index in range(5))
    if np.any(high < np.maximum.reduce((open_, close, low))) or np.any(
        low > np.minimum.reduce((open_, close, high))
    ):
        raise ValueError(f"raw OHLCV invariant failed for {symbol}")
    if np.any(base_volume < 0.0):
        raise ValueError(f"raw OHLCV volume is negative for {symbol}")
    return open_, high, low, close, base_volume


def _load_sparse_rows(
    cursor: _Cursor,
    *,
    query: str,
    params: tuple[object, ...],
    symbol: str,
    channel: str,
    width: int,
) -> tuple[np.ndarray, np.ndarray]:
    cursor.execute(query, params)
    rows = tuple(cursor.fetchall())
    if any(len(row) != width + 1 for row in rows):
        raise ValueError(f"raw {channel} row contract failed for {symbol}")
    timestamps_ns = np.asarray(
        [
            _timestamp_ns(row[0], field=f"raw {channel} timestamp for {symbol}")
            for row in rows
        ],
        dtype=np.int64,
    )
    if timestamps_ns.size > 1 and np.any(np.diff(timestamps_ns) <= 0):
        raise ValueError(f"raw {channel} timestamps must be strictly increasing for {symbol}")
    values = np.asarray(
        [
            [
                _finite_float(value, field=f"raw {channel} for {symbol}")
                for value in row[1:]
            ]
            for row in rows
        ],
        dtype=np.float64,
    ).reshape(-1, width)
    return timestamps_ns.astype("datetime64[ns]"), values


def _load_funding(
    cursor: _Cursor,
    *,
    scope: UniversalSourceScope,
    symbol: str,
) -> tuple[np.ndarray, np.ndarray]:
    cursor.execute(
        """
        SELECT timestamp, funding_rate
        FROM public.rl_funding_rate
        WHERE source = %s AND symbol = %s
          AND timestamp >= %s AND timestamp < %s
        ORDER BY timestamp
        """,
        (scope.source, symbol, scope.start, scope.end),
    )
    rows = tuple(cursor.fetchall())
    if any(len(row) != 2 for row in rows):
        raise ValueError(f"raw funding row contract failed for {symbol}")
    timestamps_ns = np.asarray(
        [_timestamp_ns(row[0], field=f"raw funding timestamp for {symbol}") for row in rows],
        dtype=np.int64,
    )
    if timestamps_ns.size > 1 and np.any(np.diff(timestamps_ns) <= 0):
        raise ValueError(f"raw funding timestamps must be strictly increasing for {symbol}")
    rates = np.asarray(
        [_finite_float(row[1], field=f"raw funding rate for {symbol}") for row in rows],
        dtype=np.float64,
    )
    return timestamps_ns.astype("datetime64[ns]"), rates


def load_postgres_universal_source(
    connection: UniversalSourceConnection,
    *,
    scope: UniversalSourceScope,
) -> dict[str, RawSymbolSource]:
    """Load and validate the declared source without mutation or future fills."""

    expected_ns = np.arange(
        _epoch_ns(scope.start), _epoch_ns(scope.end), _MINUTE_NS, dtype=np.int64
    )
    result: dict[str, RawSymbolSource] = {}
    with connection.cursor() as cursor:
        for symbol in scope.symbols:
            open_, high, low, close, base_volume = _load_klines(
                cursor, scope=scope, symbol=symbol, expected_ns=expected_ns
            )
            funding_timestamps, funding_rate = _load_funding(
                cursor, scope=scope, symbol=symbol
            )
            params = (scope.source, symbol, scope.start, scope.end)
            derivative_timestamps, derivative_values = _load_sparse_rows(
                cursor,
                query="""
                    SELECT timestamp, open_interest, ls_ratio, liq_notional,
                           funding_predicted
                    FROM public.rl_derivatives
                    WHERE source = %s AND symbol = %s
                      AND timestamp >= %s AND timestamp < %s
                    ORDER BY timestamp
                """,
                params=params,
                symbol=symbol,
                channel="derivative",
                width=4,
            )
            orderflow_timestamps, orderflow_values = _load_sparse_rows(
                cursor,
                query="""
                    SELECT timestamp, buy_volume, sell_volume, trade_count,
                           avg_trade_size, volume_imbalance
                    FROM public.rl_orderflow_1m
                    WHERE source = %s AND symbol = %s
                      AND timestamp >= %s AND timestamp < %s
                    ORDER BY timestamp
                """,
                params=params,
                symbol=symbol,
                channel="orderflow",
                width=5,
            )
            result[symbol] = RawSymbolSource(
                timestamps=expected_ns.astype("datetime64[ns]"),
                open=open_,
                high=high,
                low=low,
                close=close,
                base_volume=base_volume,
                funding_timestamps=funding_timestamps,
                funding_rate=funding_rate,
                derivative_timestamps=derivative_timestamps,
                derivative_values=derivative_values,
                orderflow_timestamps=orderflow_timestamps,
                orderflow_values=orderflow_values,
            )
    return result


__all__ = [
    "MAINTAINED_SYMBOLS",
    "RawSymbolSource",
    "UniversalSourceConnection",
    "UniversalSourceScope",
    "load_postgres_universal_source",
]
