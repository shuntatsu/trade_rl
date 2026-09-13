"""Validated historical Binance Vision aggregate-trade evidence."""

from __future__ import annotations

import csv
import hashlib
import io
import math
import re
import zipfile
from array import array
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
from numpy.typing import DTypeLike

from trade_rl.integrations.binance.types import (
    BinanceMarket,
    BinanceTransportError,
    _aware_utc,
    _market,
)
from trade_rl.integrations.binance.vision import _VISION_ROOT

_AGG_TRADES_HEADER = (
    "agg_trade_id",
    "price",
    "quantity",
    "first_trade_id",
    "last_trade_id",
    "transact_time",
    "is_buyer_maker",
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _as_1d_copy(
    value: np.ndarray,
    *,
    name: str,
    dtype: DTypeLike,
) -> np.ndarray:
    array_value = np.asarray(value)
    if array_value.ndim != 1:
        raise ValueError(f"{name} must have one-dimensional shape")
    return np.asarray(value, dtype=dtype).copy()


@dataclass(frozen=True, slots=True)
class BinanceAggTradesSeries:
    """Immutable provider-specific aggregate trades with raw provenance."""

    aggregate_trade_ids: np.ndarray
    prices: np.ndarray
    quantities: np.ndarray
    first_trade_ids: np.ndarray
    last_trade_ids: np.ndarray
    timestamps: np.ndarray
    buyer_is_maker: np.ndarray
    source_uri: str
    raw_payload_sha256: str
    raw_payload_size_bytes: int
    header_present: bool

    def __post_init__(self) -> None:
        aggregate_ids = _as_1d_copy(
            self.aggregate_trade_ids,
            name="aggregate_trade_ids",
            dtype=np.int64,
        )
        prices = _as_1d_copy(self.prices, name="prices", dtype=np.float64)
        quantities = _as_1d_copy(
            self.quantities,
            name="quantities",
            dtype=np.float64,
        )
        first_ids = _as_1d_copy(
            self.first_trade_ids,
            name="first_trade_ids",
            dtype=np.int64,
        )
        last_ids = _as_1d_copy(
            self.last_trade_ids,
            name="last_trade_ids",
            dtype=np.int64,
        )
        timestamps = _as_1d_copy(
            self.timestamps,
            name="timestamps",
            dtype="datetime64[ns]",
        )
        buyer_is_maker = _as_1d_copy(
            self.buyer_is_maker,
            name="buyer_is_maker",
            dtype=np.bool_,
        )

        if not self.source_uri:
            raise ValueError("source_uri must be non-empty")
        if not _SHA256_RE.fullmatch(self.raw_payload_sha256):
            raise ValueError("raw_payload_sha256 must be a lowercase SHA-256 digest")
        if (
            isinstance(self.raw_payload_size_bytes, bool)
            or self.raw_payload_size_bytes <= 0
        ):
            raise ValueError("raw_payload_size_bytes must be positive")
        if not isinstance(self.header_present, bool):
            raise ValueError("header_present must be a boolean")

        size = aggregate_ids.size
        if size <= 0:
            raise ValueError("aggTrades series must contain at least one trade")
        for name, values in (
            ("prices", prices),
            ("quantities", quantities),
            ("first_trade_ids", first_ids),
            ("last_trade_ids", last_ids),
            ("timestamps", timestamps),
            ("buyer_is_maker", buyer_is_maker),
        ):
            if values.shape != (size,):
                raise ValueError(f"{name} must have shape {(size,)}")

        if np.any(aggregate_ids < 0):
            raise ValueError("aggregate_trade_ids must be non-negative")
        if np.any(first_ids < 0) or np.any(last_ids < 0):
            raise ValueError("underlying trade IDs must be non-negative")
        if np.any(first_ids > last_ids):
            raise ValueError("first_trade_ids cannot exceed last_trade_ids")
        if not np.all(np.isfinite(prices)):
            raise ValueError("prices must be finite")
        if np.any(prices <= 0.0):
            raise ValueError("prices must be positive")
        if not np.all(np.isfinite(quantities)):
            raise ValueError("quantities must be finite")
        if np.any(quantities <= 0.0):
            raise ValueError("quantities must be positive")
        if np.any(np.isnat(timestamps)):
            raise ValueError("timestamps must not contain NaT")
        if size > 1:
            if np.any(aggregate_ids[1:] <= aggregate_ids[:-1]):
                raise ValueError("aggregate_trade_ids must be strictly increasing")
            if np.any(timestamps[1:] < timestamps[:-1]):
                raise ValueError("timestamps must be nondecreasing")

        for values in (
            aggregate_ids,
            prices,
            quantities,
            first_ids,
            last_ids,
            timestamps,
            buyer_is_maker,
        ):
            values.setflags(write=False)
        object.__setattr__(self, "aggregate_trade_ids", aggregate_ids)
        object.__setattr__(self, "prices", prices)
        object.__setattr__(self, "quantities", quantities)
        object.__setattr__(self, "first_trade_ids", first_ids)
        object.__setattr__(self, "last_trade_ids", last_ids)
        object.__setattr__(self, "timestamps", timestamps)
        object.__setattr__(self, "buyer_is_maker", buyer_is_maker)


def vision_agg_trades_url(
    market: BinanceMarket | str,
    symbol: str,
    day: datetime,
) -> str:
    """Return one official USD-M daily aggregate-trade archive URL."""

    resolved = _market(market)
    if resolved is not BinanceMarket.USDS_M:
        raise ValueError(
            "aggTrades evidence is currently supported only for Binance USD-M"
        )
    if not symbol:
        raise ValueError("symbol must be non-empty")
    date = _aware_utc(day, field="day").strftime("%Y-%m-%d")
    return (
        f"{_VISION_ROOT}/futures/um/daily/aggTrades/{symbol}/"
        f"{symbol}-aggTrades-{date}.zip"
    )


def plan_vision_agg_trades_urls(
    market: BinanceMarket | str,
    symbol: str,
    start_time: datetime,
    end_time: datetime,
) -> tuple[str, ...]:
    """Plan every UTC daily aggregate-trade archive touched by a half-open window."""

    start = _aware_utc(start_time, field="start_time")
    end = _aware_utc(end_time, field="end_time")
    if end <= start:
        raise ValueError("end_time must be later than start_time")
    cursor = start.replace(hour=0, minute=0, second=0, microsecond=0)
    urls: list[str] = []
    while cursor < end:
        urls.append(vision_agg_trades_url(market, symbol, cursor))
        cursor += timedelta(days=1)
    return tuple(urls)


def _nonnegative_int(value: str, *, field: str, source: str) -> int:
    try:
        result = int(value.strip())
    except ValueError as error:
        raise BinanceTransportError(
            f"aggTrades {field} must be an integer: {source}"
        ) from error
    if result < 0:
        raise BinanceTransportError(
            f"aggTrades {field} must be non-negative: {source}"
        )
    return result


def _positive_float(value: str, *, field: str, source: str) -> float:
    try:
        result = float(value)
    except ValueError as error:
        raise BinanceTransportError(
            f"invalid aggTrades {field} {value!r}: {source}"
        ) from error
    if not math.isfinite(result):
        raise BinanceTransportError(f"aggTrades {field} must be finite: {source}")
    if result <= 0.0:
        raise BinanceTransportError(f"aggTrades {field} must be positive: {source}")
    return result


def _buyer_maker(value: str, *, source: str) -> bool:
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise BinanceTransportError(
        f"aggTrades buyer-is-maker value must be boolean: {source}"
    )


def _timestamp_array(values: array[int], *, source: str) -> np.ndarray:
    milliseconds = np.array(values, dtype=np.int64, copy=True)
    timestamps = milliseconds.astype("datetime64[ms]").astype("datetime64[ns]")
    if np.any(np.isnat(timestamps)):
        raise BinanceTransportError(
            f"aggTrades timestamp is not representable: {source}"
        )
    round_trip = timestamps.astype("datetime64[ms]").astype(np.int64)
    if not np.array_equal(round_trip, milliseconds):
        raise BinanceTransportError(
            f"aggTrades timestamp exceeds maintained datetime range: {source}"
        )
    return timestamps


def parse_vision_agg_trades_archive(
    payload: bytes,
    *,
    source: str,
) -> BinanceAggTradesSeries:
    """Stream one frozen Vision aggTrades ZIP into compact immutable arrays."""

    aggregate_ids: array[int] = array("q")
    prices: array[float] = array("d")
    quantities: array[float] = array("d")
    first_ids: array[int] = array("q")
    last_ids: array[int] = array("q")
    timestamp_ms: array[int] = array("q")
    buyer_maker = bytearray()
    header_present = False
    previous_aggregate_id: int | None = None
    previous_timestamp_ms: int | None = None

    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            members = archive.infolist()
            if len(members) != 1 or members[0].is_dir():
                raise BinanceTransportError(
                    "aggTrades archive must contain exactly one regular CSV file: "
                    f"{source}"
                )
            member = members[0]
            if not member.filename.lower().endswith(".csv"):
                raise BinanceTransportError(
                    f"aggTrades archive member must be CSV: {source}"
                )
            with archive.open(member) as raw:
                text = io.TextIOWrapper(raw, encoding="utf-8-sig", newline="")
                rows = csv.reader(text)
                try:
                    first = next(rows)
                except StopIteration as error:
                    raise BinanceTransportError(
                        f"aggTrades archive contains no rows: {source}"
                    ) from error

                if tuple(cell.strip() for cell in first) == _AGG_TRADES_HEADER:
                    header_present = True
                    pending: list[str] | None = None
                else:
                    try:
                        int(first[0].strip())
                    except (IndexError, ValueError) as error:
                        raise BinanceTransportError(
                            "Binance Vision aggTrades header is unsupported: "
                            f"{tuple(first)}: {source}"
                        ) from error
                    pending = first

                while True:
                    if pending is not None:
                        row = pending
                        pending = None
                    else:
                        try:
                            row = next(rows)
                        except StopIteration:
                            break
                    if len(row) != 7:
                        raise BinanceTransportError(
                            f"aggTrades row must contain exactly seven fields: {source}"
                        )

                    aggregate_id = _nonnegative_int(
                        row[0],
                        field="aggregate trade ID",
                        source=source,
                    )
                    price = _positive_float(row[1], field="price", source=source)
                    quantity = _positive_float(row[2], field="quantity", source=source)
                    first_trade_id = _nonnegative_int(
                        row[3],
                        field="first trade ID",
                        source=source,
                    )
                    last_trade_id = _nonnegative_int(
                        row[4],
                        field="last trade ID",
                        source=source,
                    )
                    timestamp = _nonnegative_int(
                        row[5],
                        field="timestamp",
                        source=source,
                    )
                    maker = _buyer_maker(row[6], source=source)

                    if first_trade_id > last_trade_id:
                        raise BinanceTransportError(
                            "aggTrades first trade ID cannot exceed last trade ID: "
                            f"{source}"
                        )
                    if (
                        previous_aggregate_id is not None
                        and aggregate_id <= previous_aggregate_id
                    ):
                        raise BinanceTransportError(
                            "aggTrades aggregate trade IDs must be strictly increasing: "
                            f"{source}"
                        )
                    if (
                        previous_timestamp_ms is not None
                        and timestamp < previous_timestamp_ms
                    ):
                        raise BinanceTransportError(
                            f"aggTrades timestamps must be nondecreasing: {source}"
                        )

                    aggregate_ids.append(aggregate_id)
                    prices.append(price)
                    quantities.append(quantity)
                    first_ids.append(first_trade_id)
                    last_ids.append(last_trade_id)
                    timestamp_ms.append(timestamp)
                    buyer_maker.append(1 if maker else 0)
                    previous_aggregate_id = aggregate_id
                    previous_timestamp_ms = timestamp
    except BinanceTransportError:
        raise
    except (UnicodeDecodeError, zipfile.BadZipFile, csv.Error, OSError) as error:
        raise BinanceTransportError(
            f"invalid Binance Vision aggTrades archive: {source}"
        ) from error

    if not aggregate_ids:
        raise BinanceTransportError(f"aggTrades archive contains no data rows: {source}")

    return BinanceAggTradesSeries(
        aggregate_trade_ids=np.array(aggregate_ids, dtype=np.int64, copy=True),
        prices=np.array(prices, dtype=np.float64, copy=True),
        quantities=np.array(quantities, dtype=np.float64, copy=True),
        first_trade_ids=np.array(first_ids, dtype=np.int64, copy=True),
        last_trade_ids=np.array(last_ids, dtype=np.int64, copy=True),
        timestamps=_timestamp_array(timestamp_ms, source=source),
        buyer_is_maker=np.frombuffer(buyer_maker, dtype=np.uint8).astype(np.bool_),
        source_uri=source,
        raw_payload_sha256=hashlib.sha256(payload).hexdigest(),
        raw_payload_size_bytes=len(payload),
        header_present=header_present,
    )


__all__ = [
    "BinanceAggTradesSeries",
    "parse_vision_agg_trades_archive",
    "plan_vision_agg_trades_urls",
    "vision_agg_trades_url",
]
