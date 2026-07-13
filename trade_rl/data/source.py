"""Raw market-data sources used by the maintained dataset builder."""

from __future__ import annotations

import csv
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

import numpy as np


def _readonly(
    value: np.ndarray,
    *,
    dtype: np.dtype[np.generic] | None = None,
) -> np.ndarray:
    array = np.asarray(value, dtype=dtype).copy(order="C")
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class RawMarketSeries:
    """One symbol's timestamped raw bars before cross-symbol alignment."""

    timestamps: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    funding_rate: np.ndarray
    tradable: np.ndarray
    funding_available: np.ndarray | None = None

    def __post_init__(self) -> None:
        timestamps = _readonly(self.timestamps)
        if timestamps.ndim != 1 or not np.issubdtype(timestamps.dtype, np.datetime64):
            raise ValueError("timestamps must be a one-dimensional datetime64 array")
        timestamp_ns = timestamps.astype("datetime64[ns]").astype(np.int64)
        if timestamp_ns.size == 0:
            raise ValueError("raw market series must not be empty")
        if np.any(timestamp_ns == np.iinfo(np.int64).min):
            raise ValueError("timestamps must not contain NaT")
        if np.any(np.diff(timestamp_ns) <= 0):
            raise ValueError("timestamps must be strictly increasing")

        funding_available = (
            np.ones(timestamps.shape, dtype=np.bool_)
            if self.funding_available is None
            else self.funding_available
        )
        arrays = {
            "open": _readonly(self.open, dtype=np.dtype(np.float64)),
            "high": _readonly(self.high, dtype=np.dtype(np.float64)),
            "low": _readonly(self.low, dtype=np.dtype(np.float64)),
            "close": _readonly(self.close, dtype=np.dtype(np.float64)),
            "volume": _readonly(self.volume, dtype=np.dtype(np.float64)),
            "funding_rate": _readonly(self.funding_rate, dtype=np.dtype(np.float64)),
            "tradable": _readonly(self.tradable, dtype=np.dtype(np.bool_)),
            "funding_available": _readonly(funding_available, dtype=np.dtype(np.bool_)),
        }
        expected = timestamps.shape
        for field_name, array in arrays.items():
            if array.shape != expected:
                raise ValueError(f"{field_name} shape must match timestamps")
        for field_name in ("open", "high", "low", "close", "volume", "funding_rate"):
            if not np.isfinite(arrays[field_name]).all():
                raise ValueError(f"{field_name} must contain only finite values")
        if any(
            np.any(arrays[name] <= 0.0) for name in ("open", "high", "low", "close")
        ):
            raise ValueError("OHLC prices must be strictly positive")
        if np.any(arrays["volume"] < 0.0):
            raise ValueError("volume must be non-negative")
        if (
            np.any(arrays["low"] > arrays["high"])
            or np.any(arrays["low"] > arrays["open"])
            or np.any(arrays["low"] > arrays["close"])
            or np.any(arrays["high"] < arrays["open"])
            or np.any(arrays["high"] < arrays["close"])
        ):
            raise ValueError("OHLC values violate bar price invariants")

        object.__setattr__(self, "timestamps", timestamps.astype("datetime64[ns]"))
        for field_name, array in arrays.items():
            object.__setattr__(self, field_name, array)


class MarketDataSource(Protocol):
    def load(self, symbol: str) -> RawMarketSeries: ...


class InMemoryMarketDataSource:
    """Deterministic source for tests and embedded adapters."""

    def __init__(self, values: Mapping[str, RawMarketSeries]) -> None:
        if not values:
            raise ValueError("in-memory market source must not be empty")
        self._values = dict(values)

    def load(self, symbol: str) -> RawMarketSeries:
        try:
            return self._values[symbol]
        except KeyError as exc:
            raise KeyError(f"market source has no symbol {symbol}") from exc


def _parse_timestamp(value: str) -> np.datetime64:
    raw = value.strip()
    if not raw:
        raise ValueError("timestamp must not be empty")
    try:
        numeric = int(raw)
    except ValueError:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
        return np.datetime64(parsed, "ns")
    scale = 1_000.0 if abs(numeric) >= 1_000_000_000_000 else 1.0
    parsed = datetime.fromtimestamp(numeric / scale, tz=timezone.utc).replace(
        tzinfo=None
    )
    return np.datetime64(parsed, "ns")


def _parse_bool(value: str | None, *, default: bool) -> bool:
    if value is None or not value.strip():
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"invalid boolean value: {value}")


class CsvMarketDataSource:
    """Read one maintained real-data CSV file per symbol."""

    _REQUIRED = ("timestamp", "open", "high", "low", "close", "volume")

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def load(self, symbol: str) -> RawMarketSeries:
        path = self.root / f"{symbol}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"market CSV for {symbol} does not exist: {path}")

        timestamps: list[np.datetime64] = []
        open_price: list[float] = []
        high: list[float] = []
        low: list[float] = []
        close: list[float] = []
        volume: list[float] = []
        funding: list[float] = []
        funding_available: list[bool] = []
        tradable: list[bool] = []
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            names = tuple(reader.fieldnames or ())
            missing = [name for name in self._REQUIRED if name not in names]
            if missing:
                raise ValueError(f"market CSV is missing columns: {missing}")
            for row_index, row in enumerate(reader, start=2):
                try:
                    timestamps.append(_parse_timestamp(row["timestamp"]))
                    open_price.append(float(row["open"]))
                    high.append(float(row["high"]))
                    low.append(float(row["low"]))
                    close.append(float(row["close"]))
                    volume.append(float(row["volume"]))
                    raw_funding = row.get("funding_rate")
                    has_funding = raw_funding is not None and bool(raw_funding.strip())
                    funding_available.append(has_funding)
                    if has_funding:
                        assert raw_funding is not None
                        funding.append(float(raw_funding))
                    else:
                        funding.append(0.0)
                    tradable.append(_parse_bool(row.get("tradable"), default=True))
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"invalid market CSV row {row_index}: {exc}"
                    ) from exc

        return RawMarketSeries(
            timestamps=np.asarray(timestamps, dtype="datetime64[ns]"),
            open=np.asarray(open_price, dtype=np.float64),
            high=np.asarray(high, dtype=np.float64),
            low=np.asarray(low, dtype=np.float64),
            close=np.asarray(close, dtype=np.float64),
            volume=np.asarray(volume, dtype=np.float64),
            funding_rate=np.asarray(funding, dtype=np.float64),
            tradable=np.asarray(tradable, dtype=np.bool_),
            funding_available=np.asarray(funding_available, dtype=np.bool_),
        )
