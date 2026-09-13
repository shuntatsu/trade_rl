"""Validated historical Binance Vision book-depth evidence.

This module intentionally stops at provider-specific evidence ingestion. It does not
infer top-of-book spread or feed liquidity into MarketDataset execution economics.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np

from trade_rl.integrations.binance.types import (
    BinanceMarket,
    BinanceTransportError,
    _aware_utc,
    _market,
)
from trade_rl.integrations.binance.vision import _VISION_ROOT, _csv_rows_from_zip

BOOK_DEPTH_PERCENTAGE_BANDS = (-5, -4, -3, -2, -1, 1, 2, 3, 4, 5)
_BOOK_DEPTH_HEADER = ("timestamp", "percentage", "depth", "notional")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class BinanceBookDepthSeries:
    """Immutable provider-specific cumulative depth snapshots and raw provenance."""

    timestamps: np.ndarray
    available_at: np.ndarray
    depth: np.ndarray
    notional: np.ndarray
    implied_average_price: np.ndarray
    source_uri: str
    raw_payload_sha256: str
    raw_payload_size_bytes: int
    percentage_bands: tuple[int, ...] = BOOK_DEPTH_PERCENTAGE_BANDS

    def __post_init__(self) -> None:
        timestamps = (
            np.asarray(self.timestamps, dtype="datetime64[ns]").reshape(-1).copy()
        )
        available_at = (
            np.asarray(self.available_at, dtype="datetime64[ns]").reshape(-1).copy()
        )
        depth = np.asarray(self.depth, dtype=np.float64).copy()
        notional = np.asarray(self.notional, dtype=np.float64).copy()
        implied = np.asarray(self.implied_average_price, dtype=np.float64).copy()

        if not self.source_uri:
            raise ValueError("source_uri must be non-empty")
        if not _SHA256_RE.fullmatch(self.raw_payload_sha256):
            raise ValueError("raw_payload_sha256 must be a lowercase SHA-256 digest")
        if (
            isinstance(self.raw_payload_size_bytes, bool)
            or self.raw_payload_size_bytes <= 0
        ):
            raise ValueError("raw_payload_size_bytes must be positive")
        if tuple(self.percentage_bands) != BOOK_DEPTH_PERCENTAGE_BANDS:
            raise ValueError(
                "percentage_bands must match the maintained bookDepth schema"
            )

        expected_shape = (timestamps.size, len(BOOK_DEPTH_PERCENTAGE_BANDS))
        if timestamps.size <= 0:
            raise ValueError("bookDepth series must contain at least one snapshot")
        if available_at.shape != timestamps.shape:
            raise ValueError("available_at must match timestamps")
        for name, values in (
            ("depth", depth),
            ("notional", notional),
            ("implied_average_price", implied),
        ):
            if values.shape != expected_shape:
                raise ValueError(f"{name} must have shape {expected_shape}")
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must be finite")
            if np.any(values < 0.0):
                raise ValueError(f"{name} must be non-negative")

        if timestamps.size > 1 and np.any(timestamps[1:] <= timestamps[:-1]):
            raise ValueError("timestamps must be strictly increasing")
        if np.any(available_at < timestamps):
            raise ValueError("available_at cannot precede timestamps")

        for values in (timestamps, available_at, depth, notional, implied):
            values.setflags(write=False)
        object.__setattr__(self, "timestamps", timestamps)
        object.__setattr__(self, "available_at", available_at)
        object.__setattr__(self, "depth", depth)
        object.__setattr__(self, "notional", notional)
        object.__setattr__(self, "implied_average_price", implied)
        object.__setattr__(self, "percentage_bands", BOOK_DEPTH_PERCENTAGE_BANDS)


def vision_book_depth_url(
    market: BinanceMarket | str,
    symbol: str,
    day: datetime,
) -> str:
    """Return the official USD-M daily Vision bookDepth archive URL."""

    resolved = _market(market)
    if resolved is not BinanceMarket.USDS_M:
        raise ValueError(
            "bookDepth evidence is currently supported only for Binance USD-M"
        )
    if not symbol:
        raise ValueError("symbol must be non-empty")
    date = _aware_utc(day, field="day").strftime("%Y-%m-%d")
    return (
        f"{_VISION_ROOT}/futures/um/daily/bookDepth/{symbol}/"
        f"{symbol}-bookDepth-{date}.zip"
    )


def plan_vision_book_depth_urls(
    market: BinanceMarket | str,
    symbol: str,
    start_time: datetime,
    end_time: datetime,
) -> tuple[str, ...]:
    """Plan every UTC daily archive touched by the half-open request window."""

    start = _aware_utc(start_time, field="start_time")
    end = _aware_utc(end_time, field="end_time")
    if end <= start:
        raise ValueError("end_time must be later than start_time")
    cursor = start.replace(hour=0, minute=0, second=0, microsecond=0)
    urls: list[str] = []
    while cursor < end:
        urls.append(vision_book_depth_url(market, symbol, cursor))
        cursor += timedelta(days=1)
    return tuple(urls)


def _timestamp(value: str, *, source: str) -> np.datetime64:
    text = value.strip()
    if not text:
        raise BinanceTransportError(f"bookDepth timestamp is empty: {source}")
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as error:
        raise BinanceTransportError(
            f"invalid bookDepth timestamp {value!r}: {source}"
        ) from error
    if parsed.tzinfo is not None and parsed.utcoffset() is not None:
        parsed = _aware_utc(parsed, field="bookDepth timestamp").replace(tzinfo=None)
    return np.datetime64(parsed, "ns")


def _band(value: str, *, source: str) -> int:
    try:
        band = int(value.strip())
    except ValueError as error:
        raise BinanceTransportError(
            f"invalid bookDepth percentage band {value!r}: {source}"
        ) from error
    if band not in BOOK_DEPTH_PERCENTAGE_BANDS:
        raise BinanceTransportError(
            f"unsupported bookDepth percentage band {band}: {source}"
        )
    return band


def _nonnegative_float(value: str, *, field: str, source: str) -> float:
    try:
        result = float(value)
    except ValueError as error:
        raise BinanceTransportError(
            f"invalid bookDepth {field} {value!r}: {source}"
        ) from error
    if not math.isfinite(result):
        raise BinanceTransportError(f"bookDepth {field} must be finite: {source}")
    if result < 0.0:
        raise BinanceTransportError(f"bookDepth {field} must be non-negative: {source}")
    return result


def _require_nondecreasing(
    values: np.ndarray,
    *,
    field: str,
    side: str,
    source: str,
) -> None:
    if values.size > 1 and np.any(values[1:] < values[:-1]):
        raise BinanceTransportError(
            f"bookDepth {field} must be nondecreasing away from mid on {side} side: "
            f"{source}"
        )


def _validate_implied_prices(implied: np.ndarray, *, source: str) -> None:
    bid = implied[[4, 3, 2, 1, 0]]
    ask = implied[[5, 6, 7, 8, 9]]
    bid = bid[bid > 0.0]
    ask = ask[ask > 0.0]
    if bid.size > 1 and np.any(bid[1:] > bid[:-1]):
        raise BinanceTransportError(
            f"bookDepth bid implied prices violate side ordering: {source}"
        )
    if ask.size > 1 and np.any(ask[1:] < ask[:-1]):
        raise BinanceTransportError(
            f"bookDepth ask implied prices violate side ordering: {source}"
        )
    if bid.size and ask.size and float(np.max(bid)) >= float(np.min(ask)):
        raise BinanceTransportError(
            f"bookDepth bid implied prices cross ask implied prices: {source}"
        )


def _finalize_snapshot(
    rows: dict[int, tuple[float, float]],
    *,
    timestamp: np.datetime64,
    source: str,
) -> tuple[np.datetime64, np.ndarray, np.ndarray, np.ndarray]:
    actual = set(rows)
    expected = set(BOOK_DEPTH_PERCENTAGE_BANDS)
    if actual != expected:
        raise BinanceTransportError(
            "bookDepth snapshot bands do not match maintained schema: "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}: {source}"
        )

    depth = np.asarray([rows[band][0] for band in BOOK_DEPTH_PERCENTAGE_BANDS])
    notional = np.asarray([rows[band][1] for band in BOOK_DEPTH_PERCENTAGE_BANDS])
    zero_depth = depth == 0.0
    if np.any(zero_depth & (notional != 0.0)):
        raise BinanceTransportError(
            f"bookDepth zero depth must have zero notional: {source}"
        )
    if np.any((depth > 0.0) & (notional <= 0.0)):
        raise BinanceTransportError(
            f"bookDepth positive depth requires positive notional: {source}"
        )

    implied = np.divide(
        notional,
        depth,
        out=np.zeros_like(notional, dtype=np.float64),
        where=depth > 0.0,
    )
    if not np.all(np.isfinite(implied)) or np.any(implied < 0.0):
        raise BinanceTransportError(
            f"bookDepth implied average price must be finite and non-negative: {source}"
        )

    bid_indices = np.asarray([4, 3, 2, 1, 0])
    ask_indices = np.asarray([5, 6, 7, 8, 9])
    for side, indices in (("bid", bid_indices), ("ask", ask_indices)):
        _require_nondecreasing(
            depth[indices],
            field="depth",
            side=side,
            source=source,
        )
        _require_nondecreasing(
            notional[indices],
            field="notional",
            side=side,
            source=source,
        )
    _validate_implied_prices(implied, source=source)
    return timestamp, depth, notional, implied


def parse_vision_book_depth_archive(
    payload: bytes,
    *,
    source: str,
) -> BinanceBookDepthSeries:
    """Parse one frozen Vision bookDepth ZIP and reject malformed snapshots."""

    rows = _csv_rows_from_zip(payload, source=source)
    if not rows or tuple(cell.strip() for cell in rows[0]) != _BOOK_DEPTH_HEADER:
        actual = () if not rows else tuple(cell.strip() for cell in rows[0])
        raise BinanceTransportError(
            f"Binance Vision bookDepth header is unsupported: {actual}: {source}"
        )
    if len(rows) == 1:
        raise BinanceTransportError(
            f"bookDepth archive contains no data rows: {source}"
        )

    snapshots: list[tuple[np.datetime64, np.ndarray, np.ndarray, np.ndarray]] = []
    current_timestamp: np.datetime64 | None = None
    current_rows: dict[int, tuple[float, float]] = {}

    for raw in rows[1:]:
        if len(raw) != 4:
            raise BinanceTransportError(
                f"bookDepth row must contain exactly four fields: {source}"
            )
        timestamp = _timestamp(raw[0], source=source)
        band = _band(raw[1], source=source)
        depth = _nonnegative_float(raw[2], field="depth", source=source)
        notional = _nonnegative_float(raw[3], field="notional", source=source)

        if current_timestamp is None:
            current_timestamp = timestamp
        elif timestamp != current_timestamp:
            snapshots.append(
                _finalize_snapshot(
                    current_rows,
                    timestamp=current_timestamp,
                    source=source,
                )
            )
            if timestamp <= current_timestamp:
                raise BinanceTransportError(
                    f"bookDepth snapshot timestamps must be strictly increasing: {source}"
                )
            current_timestamp = timestamp
            current_rows = {}

        if band in current_rows:
            raise BinanceTransportError(
                f"duplicate bookDepth percentage band {band} at {timestamp}: {source}"
            )
        current_rows[band] = (depth, notional)

    if current_timestamp is None:
        raise BinanceTransportError(
            f"bookDepth archive contains no snapshots: {source}"
        )
    snapshots.append(
        _finalize_snapshot(
            current_rows,
            timestamp=current_timestamp,
            source=source,
        )
    )

    timestamps = np.asarray(
        [snapshot[0] for snapshot in snapshots], dtype="datetime64[ns]"
    )
    depth = np.stack([snapshot[1] for snapshot in snapshots])
    notional = np.stack([snapshot[2] for snapshot in snapshots])
    implied = np.stack([snapshot[3] for snapshot in snapshots])
    return BinanceBookDepthSeries(
        timestamps=timestamps,
        available_at=timestamps,
        depth=depth,
        notional=notional,
        implied_average_price=implied,
        source_uri=source,
        raw_payload_sha256=hashlib.sha256(payload).hexdigest(),
        raw_payload_size_bytes=len(payload),
    )


def validate_book_depth_reference_alignment(
    series: BinanceBookDepthSeries,
    reference_prices: np.ndarray,
    *,
    reference_available_at: np.ndarray,
    max_relative_deviation_rate: float,
) -> None:
    """Fail when bookDepth implied prices disagree with a causal reference."""

    if (
        not math.isfinite(max_relative_deviation_rate)
        or max_relative_deviation_rate <= 0.0
        or max_relative_deviation_rate >= 1.0
    ):
        raise ValueError("max_relative_deviation_rate must be finite and in (0, 1)")
    references = np.asarray(reference_prices, dtype=np.float64).reshape(-1)
    if references.shape != series.timestamps.shape:
        raise ValueError(
            "reference_prices must contain one value per bookDepth snapshot"
        )
    if not np.all(np.isfinite(references)) or np.any(references <= 0.0):
        raise ValueError("reference_prices must be finite and positive")

    reference_times = np.asarray(
        reference_available_at,
        dtype="datetime64[ns]",
    ).reshape(-1)
    if reference_times.shape != series.timestamps.shape:
        raise ValueError(
            "reference_available_at must contain one value per bookDepth snapshot"
        )
    if np.any(np.isnat(reference_times)):
        raise ValueError("reference_available_at must not contain NaT")
    if np.any(reference_times > series.available_at):
        raise BinanceTransportError(
            "bookDepth reference availability is in the future relative to the "
            f"liquidity snapshot: source={series.source_uri}"
        )

    for index, reference in enumerate(references):
        implied = series.implied_average_price[index]
        observed = implied[implied > 0.0]
        if not observed.size:
            continue
        deviation = np.abs(observed / reference - 1.0)
        if np.any(deviation > max_relative_deviation_rate):
            worst = float(np.max(deviation))
            raise BinanceTransportError(
                "bookDepth reference-price alignment exceeded explicit bound: "
                f"snapshot={series.timestamps[index]}, worst_relative_deviation={worst}, "
                f"limit={max_relative_deviation_rate}, source={series.source_uri}"
            )
