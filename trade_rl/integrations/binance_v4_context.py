"""Causal Binance Spot/perpetual source contracts for Causal Alpha V4."""

from __future__ import annotations

import csv
import hashlib
import io
import math
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Final

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.data.identity import content_and_arrays_digest

_BINANCE_VISION_ROOT: Final = "https://data.binance.vision/data/"
_BINANCE_V4_KLINE_SCHEMA: Final = "binance_v4_kline_archive_v1"
_BINANCE_V4_FUNDING_SCHEMA: Final = "binance_v4_funding_archive_v1"
_BINANCE_V4_METRICS_SCHEMA: Final = "binance_v4_futures_metrics_archive_v1"
_BINANCE_V4_ALIGNED_METRICS_SCHEMA: Final = "binance_v4_aligned_metrics_v1"

BINANCE_V4_MAX_DERIVATIVES_STALENESS_HOURS: Final = 0.25

KLINE_OPEN_TIME: Final = 0
KLINE_OPEN: Final = 1
KLINE_HIGH: Final = 2
KLINE_LOW: Final = 3
KLINE_CLOSE: Final = 4
KLINE_BASE_VOLUME: Final = 5
KLINE_CLOSE_TIME: Final = 6
KLINE_QUOTE_VOLUME: Final = 7
KLINE_TRADE_COUNT: Final = 8
KLINE_TAKER_BUY_BASE_VOLUME: Final = 9
KLINE_TAKER_BUY_QUOTE_VOLUME: Final = 10
KLINE_IGNORE: Final = 11
KLINE_FIELD_COUNT: Final = 12

BINANCE_FUTURES_METRICS_COLUMNS: Final = (
    "create_time",
    "symbol",
    "sum_open_interest",
    "sum_open_interest_value",
    "count_toptrader_long_short_ratio",
    "sum_toptrader_long_short_ratio",
    "count_long_short_ratio",
    "sum_taker_long_short_vol_ratio",
)

_KLINE_COLUMNS: Final = (
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "count",
    "taker_buy_volume",
    "taker_buy_quote_volume",
    "ignore",
)
_KLINE_HEADER_ALIASES: Final = {
    "quote_asset_volume": "quote_volume",
    "number_of_trades": "count",
    "taker_buy_base_asset_volume": "taker_buy_volume",
    "taker_buy_quote_asset_volume": "taker_buy_quote_volume",
}


def _freeze_vector(value: object, *, dtype: np.dtype[object], field: str) -> np.ndarray:
    array = np.asarray(value, dtype=dtype).reshape(-1).copy(order="C")
    if array.size == 0:
        raise ValueError(f"{field} must not be empty")
    if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
        raise ValueError(f"{field} must be finite")
    array.setflags(write=False)
    return array


def _require_sha256(value: str, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lower-case SHA-256 digest")
    return value


def _source_digest(*, schema: str, source_uri: str, payload: bytes) -> str:
    if not source_uri.startswith(_BINANCE_VISION_ROOT):
        raise ValueError("V4 Binance source URI must be an official Binance Vision URL")
    raw_sha256 = hashlib.sha256(payload).hexdigest()
    return content_digest(
        {
            "raw_sha256": raw_sha256,
            "schema_version": schema,
            "source_uri": source_uri,
        }
    )


def _archive_rows(payload: bytes, *, source_uri: str) -> list[list[str]]:
    if not isinstance(payload, bytes) or not payload:
        raise ValueError("V4 Binance archive payload must be non-empty bytes")
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            members = tuple(
                name for name in archive.namelist() if not name.endswith("/")
            )
            if len(members) != 1:
                raise ValueError("V4 Binance archive must contain exactly one CSV file")
            text = archive.read(members[0]).decode("utf-8-sig")
    except (UnicodeDecodeError, zipfile.BadZipFile, KeyError) as error:
        raise ValueError(f"V4 Binance archive is invalid: {source_uri}") from error
    rows = [row for row in csv.reader(io.StringIO(text)) if row]
    if not rows:
        raise ValueError("V4 Binance archive contains no rows")
    return rows


def _normalize_epoch_ms(value: object) -> int:
    try:
        result = int(str(value).strip())
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid Binance timestamp: {value!r}") from error
    while abs(result) >= 10_000_000_000_000:
        result //= 1_000
    if result <= 0:
        raise ValueError("Binance timestamp must be positive")
    return result


def _finite_float(value: object, *, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be numeric") from error
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _normalized_header(value: str) -> str:
    normalized = value.strip().lower().replace(" ", "_")
    return _KLINE_HEADER_ALIASES.get(normalized, normalized)


def _looks_like_header(row: list[str]) -> bool:
    return bool(row) and not row[0].strip().lstrip("-").isdigit()


def _strictly_increasing(values: np.ndarray, *, field: str) -> None:
    if values.size > 1 and np.any(values[1:] <= values[:-1]):
        raise ValueError(f"{field} must be strictly increasing")


@dataclass(frozen=True, slots=True)
class BinanceV4KlineSeries:
    open_time_ms: np.ndarray
    close_time_ms: np.ndarray
    close: np.ndarray
    quote_volume: np.ndarray
    taker_buy_quote_volume: np.ndarray
    source_digest: str
    digest: str = ""

    def __post_init__(self) -> None:
        open_time = _freeze_vector(
            self.open_time_ms, dtype=np.dtype(np.int64), field="V4 kline open_time_ms"
        )
        close_time = _freeze_vector(
            self.close_time_ms,
            dtype=np.dtype(np.int64),
            field="V4 kline close_time_ms",
        )
        close = _freeze_vector(
            self.close, dtype=np.dtype(np.float64), field="V4 kline close"
        )
        quote = _freeze_vector(
            self.quote_volume,
            dtype=np.dtype(np.float64),
            field="V4 kline quote_volume",
        )
        taker = _freeze_vector(
            self.taker_buy_quote_volume,
            dtype=np.dtype(np.float64),
            field="V4 kline taker_buy_quote_volume",
        )
        size = open_time.size
        if any(array.shape != (size,) for array in (close_time, close, quote, taker)):
            raise ValueError("V4 kline arrays must be row aligned")
        _strictly_increasing(open_time, field="V4 kline open_time_ms")
        if np.any(close_time < open_time):
            raise ValueError("V4 kline close time cannot precede open time")
        if np.any(close <= 0.0):
            raise ValueError("V4 kline close must be positive")
        if np.any(quote < 0.0) or np.any(taker < 0.0):
            raise ValueError("V4 kline quote volumes must be non-negative")
        if np.any(taker > quote + 1e-12):
            raise ValueError("V4 kline taker buy quote volume exceeds quote volume")
        source = _require_sha256(self.source_digest, field="V4 kline source_digest")
        for name, array in (
            ("open_time_ms", open_time),
            ("close_time_ms", close_time),
            ("close", close),
            ("quote_volume", quote),
            ("taker_buy_quote_volume", taker),
        ):
            object.__setattr__(self, name, array)
        object.__setattr__(self, "source_digest", source)
        expected = content_and_arrays_digest(
            {
                "schema_version": "binance_v4_kline_series_v1",
                "source_digest": source,
            },
            (
                ("open_time_ms", open_time),
                ("close_time_ms", close_time),
                ("close", close),
                ("quote_volume", quote),
                ("taker_buy_quote_volume", taker),
            ),
        )
        if self.digest and self.digest != expected:
            raise ValueError("V4 kline series digest mismatch")
        object.__setattr__(self, "digest", expected)


@dataclass(frozen=True, slots=True)
class BinanceFundingEventSeries:
    event_time_ms: np.ndarray
    rate: np.ndarray
    source_digest: str
    digest: str = ""

    def __post_init__(self) -> None:
        timestamps = _freeze_vector(
            self.event_time_ms,
            dtype=np.dtype(np.int64),
            field="V4 funding event_time_ms",
        )
        rate = _freeze_vector(
            self.rate, dtype=np.dtype(np.float64), field="V4 funding rate"
        )
        if rate.shape != timestamps.shape:
            raise ValueError("V4 funding arrays must be row aligned")
        _strictly_increasing(timestamps, field="V4 funding event_time_ms")
        source = _require_sha256(self.source_digest, field="V4 funding source_digest")
        object.__setattr__(self, "event_time_ms", timestamps)
        object.__setattr__(self, "rate", rate)
        object.__setattr__(self, "source_digest", source)
        expected = content_and_arrays_digest(
            {
                "schema_version": "binance_v4_funding_event_series_v1",
                "source_digest": source,
            },
            (("event_time_ms", timestamps), ("rate", rate)),
        )
        if self.digest and self.digest != expected:
            raise ValueError("V4 funding series digest mismatch")
        object.__setattr__(self, "digest", expected)


@dataclass(frozen=True, slots=True)
class BinanceFuturesMetricsSeries:
    create_time: np.ndarray
    open_interest_value: np.ndarray
    global_long_short_ratio: np.ndarray
    top_position_long_short_ratio: np.ndarray
    source_digest: str
    digest: str = ""

    def __post_init__(self) -> None:
        create_time = (
            np.asarray(self.create_time, dtype="datetime64[ns]").reshape(-1).copy()
        )
        if create_time.size == 0 or np.any(np.isnat(create_time)):
            raise ValueError("V4 metrics create_time must be non-empty and finite")
        create_ns = create_time.astype(np.int64)
        _strictly_increasing(create_ns, field="V4 metrics create_time")
        arrays: dict[str, np.ndarray] = {}
        for name, value in (
            ("open_interest_value", self.open_interest_value),
            ("global_long_short_ratio", self.global_long_short_ratio),
            ("top_position_long_short_ratio", self.top_position_long_short_ratio),
        ):
            array = _freeze_vector(
                value, dtype=np.dtype(np.float64), field=f"V4 metrics {name}"
            )
            if array.shape != create_time.shape:
                raise ValueError("V4 metrics arrays must be row aligned")
            if np.any(array < 0.0):
                raise ValueError(f"V4 metrics {name} must be non-negative")
            arrays[name] = array
        if np.any(arrays["global_long_short_ratio"] <= 0.0) or np.any(
            arrays["top_position_long_short_ratio"] <= 0.0
        ):
            raise ValueError("V4 metrics long/short ratios must be positive")
        source = _require_sha256(self.source_digest, field="V4 metrics source_digest")
        create_time.setflags(write=False)
        object.__setattr__(self, "create_time", create_time)
        for name, array in arrays.items():
            object.__setattr__(self, name, array)
        object.__setattr__(self, "source_digest", source)
        expected = content_and_arrays_digest(
            {
                "schema_version": "binance_v4_futures_metrics_series_v1",
                "source_digest": source,
            },
            (
                ("create_time_ns", create_ns),
                ("open_interest_value", arrays["open_interest_value"]),
                ("global_long_short_ratio", arrays["global_long_short_ratio"]),
                (
                    "top_position_long_short_ratio",
                    arrays["top_position_long_short_ratio"],
                ),
            ),
        )
        if self.digest and self.digest != expected:
            raise ValueError("V4 metrics series digest mismatch")
        object.__setattr__(self, "digest", expected)


@dataclass(frozen=True, slots=True)
class AlignedFuturesMetrics:
    open_interest_value: np.ndarray
    global_long_short_ratio: np.ndarray
    top_position_long_short_ratio: np.ndarray
    available: np.ndarray
    staleness_hours: np.ndarray
    source_digest: str
    digest: str = ""

    def __post_init__(self) -> None:
        arrays: dict[str, np.ndarray] = {}
        for name, value, dtype in (
            ("open_interest_value", self.open_interest_value, np.float64),
            ("global_long_short_ratio", self.global_long_short_ratio, np.float64),
            (
                "top_position_long_short_ratio",
                self.top_position_long_short_ratio,
                np.float64,
            ),
            ("available", self.available, np.bool_),
            ("staleness_hours", self.staleness_hours, np.float64),
        ):
            array = np.asarray(value, dtype=dtype).reshape(-1).copy(order="C")
            if array.size == 0:
                raise ValueError(f"V4 aligned metrics {name} must not be empty")
            if dtype is not np.bool_ and not np.isfinite(array).all():
                raise ValueError(f"V4 aligned metrics {name} must be finite")
            array.setflags(write=False)
            arrays[name] = array
        size = arrays["available"].size
        if any(array.shape != (size,) for array in arrays.values()):
            raise ValueError("V4 aligned metrics arrays must be row aligned")
        if np.any(arrays["staleness_hours"] < 0.0):
            raise ValueError("V4 aligned metrics staleness must be non-negative")
        source = _require_sha256(
            self.source_digest, field="V4 aligned metrics source_digest"
        )
        for name, array in arrays.items():
            object.__setattr__(self, name, array)
        object.__setattr__(self, "source_digest", source)
        expected = content_and_arrays_digest(
            {
                "schema_version": _BINANCE_V4_ALIGNED_METRICS_SCHEMA,
                "source_digest": source,
            },
            tuple((name, array) for name, array in arrays.items()),
        )
        if self.digest and self.digest != expected:
            raise ValueError("V4 aligned metrics digest mismatch")
        object.__setattr__(self, "digest", expected)


def vision_futures_metrics_url(symbol: str, day: datetime) -> str:
    """Return the immutable Binance Vision daily USD-M metrics URL."""

    if not isinstance(symbol, str) or not symbol.strip():
        raise ValueError("metrics symbol must be non-empty")
    if day.tzinfo is None or day.utcoffset() is None:
        raise ValueError("metrics day must be timezone-aware")
    date = day.astimezone(UTC).strftime("%Y-%m-%d")
    resolved = symbol.strip().upper()
    return (
        f"{_BINANCE_VISION_ROOT}futures/um/daily/metrics/{resolved}/"
        f"{resolved}-metrics-{date}.zip"
    )


def parse_binance_v4_kline_archive(
    payload: bytes,
    *,
    source_uri: str,
) -> BinanceV4KlineSeries:
    """Parse exact public kline fields needed by the V4 context builder."""

    rows = _archive_rows(payload, source_uri=source_uri)
    data_rows = rows
    indices = {name: index for index, name in enumerate(_KLINE_COLUMNS)}
    if _looks_like_header(rows[0]):
        header = tuple(_normalized_header(value) for value in rows[0])
        if len(header) != KLINE_FIELD_COUNT or set(header) != set(_KLINE_COLUMNS):
            raise ValueError("V4 Binance kline header must contain the exact 12 fields")
        if len(set(header)) != len(header):
            raise ValueError("V4 Binance kline header contains duplicate fields")
        indices = {name: header.index(name) for name in _KLINE_COLUMNS}
        data_rows = rows[1:]
    if not data_rows:
        raise ValueError("V4 Binance kline archive has no data rows")

    open_times: list[int] = []
    close_times: list[int] = []
    closes: list[float] = []
    quote_volumes: list[float] = []
    taker_quote_volumes: list[float] = []
    for row in data_rows:
        if len(row) != KLINE_FIELD_COUNT:
            raise ValueError("V4 Binance kline row must contain exactly 12 fields")
        open_time = _normalize_epoch_ms(row[indices["open_time"]])
        close_time = _normalize_epoch_ms(row[indices["close_time"]])
        open_price = _finite_float(row[indices["open"]], field="kline open")
        high = _finite_float(row[indices["high"]], field="kline high")
        low = _finite_float(row[indices["low"]], field="kline low")
        close = _finite_float(row[indices["close"]], field="kline close")
        base_volume = _finite_float(row[indices["volume"]], field="kline base volume")
        quote_volume = _finite_float(
            row[indices["quote_volume"]], field="kline quote volume"
        )
        taker_base = _finite_float(
            row[indices["taker_buy_volume"]], field="kline taker buy base volume"
        )
        taker_quote = _finite_float(
            row[indices["taker_buy_quote_volume"]],
            field="kline taker buy quote volume",
        )
        if min(open_price, high, low, close) <= 0.0:
            raise ValueError("V4 Binance kline OHLC prices must be positive")
        if min(base_volume, quote_volume, taker_base, taker_quote) < 0.0:
            raise ValueError("V4 Binance kline volumes must be non-negative")
        if taker_quote > quote_volume + 1e-12:
            raise ValueError("V4 Binance kline taker buy quote exceeds quote volume")
        if close_time < open_time:
            raise ValueError("V4 Binance kline close time precedes open time")
        open_times.append(open_time)
        close_times.append(close_time)
        closes.append(close)
        quote_volumes.append(quote_volume)
        taker_quote_volumes.append(taker_quote)

    return BinanceV4KlineSeries(
        open_time_ms=np.asarray(open_times, dtype=np.int64),
        close_time_ms=np.asarray(close_times, dtype=np.int64),
        close=np.asarray(closes, dtype=np.float64),
        quote_volume=np.asarray(quote_volumes, dtype=np.float64),
        taker_buy_quote_volume=np.asarray(taker_quote_volumes, dtype=np.float64),
        source_digest=_source_digest(
            schema=_BINANCE_V4_KLINE_SCHEMA,
            source_uri=source_uri,
            payload=payload,
        ),
    )


def parse_binance_funding_archive(
    payload: bytes,
    *,
    source_uri: str,
) -> BinanceFundingEventSeries:
    """Parse actual funding events without carrying them onto decision rows."""

    rows = _archive_rows(payload, source_uri=source_uri)
    if not _looks_like_header(rows[0]):
        raise ValueError("V4 Binance funding archive requires a header")
    header = {name.strip(): index for index, name in enumerate(rows[0])}
    if len(header) != len(rows[0]):
        raise ValueError("V4 Binance funding header contains duplicate fields")
    if {"calc_time", "last_funding_rate"}.issubset(header):
        time_field = "calc_time"
        rate_field = "last_funding_rate"
    elif {"fundingTime", "fundingRate"}.issubset(header):
        time_field = "fundingTime"
        rate_field = "fundingRate"
    else:
        raise ValueError("V4 Binance funding header is unsupported")
    timestamps: list[int] = []
    rates: list[float] = []
    for row in rows[1:]:
        if len(row) <= max(header[time_field], header[rate_field]):
            raise ValueError("V4 Binance funding row is short")
        timestamps.append(_normalize_epoch_ms(row[header[time_field]]))
        rates.append(_finite_float(row[header[rate_field]], field="funding rate"))
    if not timestamps:
        raise ValueError("V4 Binance funding archive has no events")
    return BinanceFundingEventSeries(
        event_time_ms=np.asarray(timestamps, dtype=np.int64),
        rate=np.asarray(rates, dtype=np.float64),
        source_digest=_source_digest(
            schema=_BINANCE_V4_FUNDING_SCHEMA,
            source_uri=source_uri,
            payload=payload,
        ),
    )


def _parse_metrics_time(value: object) -> np.datetime64:
    text = str(value).strip()
    if not text:
        raise ValueError("V4 Binance metrics create_time is empty")
    if text.lstrip("-").isdigit():
        return np.datetime64(_normalize_epoch_ms(text), "ms").astype("datetime64[ns]")
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("V4 Binance metrics create_time is invalid") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        parsed = parsed.replace(tzinfo=UTC)
    else:
        parsed = parsed.astimezone(UTC)
    milliseconds = int(parsed.timestamp() * 1_000)
    return np.datetime64(milliseconds, "ms").astype("datetime64[ns]")


def parse_binance_futures_metrics_archive(
    payload: bytes,
    *,
    source_uri: str,
    expected_symbol: str,
) -> BinanceFuturesMetricsSeries:
    """Parse one Binance Vision USD-M daily metrics archive by header name."""

    if not isinstance(expected_symbol, str) or not expected_symbol.strip():
        raise ValueError("V4 metrics expected_symbol must be non-empty")
    symbol = expected_symbol.strip().upper()
    rows = _archive_rows(payload, source_uri=source_uri)
    if not _looks_like_header(rows[0]):
        raise ValueError("V4 Binance metrics archive requires a header")
    header_values = tuple(value.strip() for value in rows[0])
    if len(header_values) != len(BINANCE_FUTURES_METRICS_COLUMNS):
        raise ValueError("V4 Binance metrics header has an unexpected column count")
    if len(set(header_values)) != len(header_values):
        raise ValueError("V4 Binance metrics header contains duplicate columns")
    if set(header_values) != set(BINANCE_FUTURES_METRICS_COLUMNS):
        missing = sorted(set(BINANCE_FUTURES_METRICS_COLUMNS) - set(header_values))
        unknown = sorted(set(header_values) - set(BINANCE_FUTURES_METRICS_COLUMNS))
        raise ValueError(
            "V4 Binance metrics header columns mismatch; "
            f"missing={missing}, unknown={unknown}"
        )
    header = {
        name: header_values.index(name) for name in BINANCE_FUTURES_METRICS_COLUMNS
    }
    if len(rows) == 1:
        raise ValueError("V4 Binance metrics archive has no data rows")

    create_time: list[np.datetime64] = []
    oi_value: list[float] = []
    global_ratio: list[float] = []
    top_position_ratio: list[float] = []
    for row in rows[1:]:
        if len(row) != len(BINANCE_FUTURES_METRICS_COLUMNS):
            raise ValueError("V4 Binance metrics row has an unexpected field count")
        row_symbol = row[header["symbol"]].strip().upper()
        if row_symbol != symbol:
            raise ValueError(
                f"V4 Binance metrics symbol mismatch: expected {symbol}, got {row_symbol}"
            )
        values = {
            name: _finite_float(row[header[name]], field=f"metrics {name}")
            for name in BINANCE_FUTURES_METRICS_COLUMNS
            if name not in {"create_time", "symbol"}
        }
        if values["sum_open_interest"] < 0.0 or values["sum_open_interest_value"] < 0.0:
            raise ValueError("V4 Binance metrics open interest must be non-negative")
        for name in (
            "count_toptrader_long_short_ratio",
            "sum_toptrader_long_short_ratio",
            "count_long_short_ratio",
            "sum_taker_long_short_vol_ratio",
        ):
            if values[name] <= 0.0:
                raise ValueError(
                    "V4 Binance metrics long/short ratios must be positive"
                )
        create_time.append(_parse_metrics_time(row[header["create_time"]]))
        oi_value.append(values["sum_open_interest_value"])
        global_ratio.append(values["count_long_short_ratio"])
        top_position_ratio.append(values["sum_toptrader_long_short_ratio"])

    return BinanceFuturesMetricsSeries(
        create_time=np.asarray(create_time, dtype="datetime64[ns]"),
        open_interest_value=np.asarray(oi_value, dtype=np.float64),
        global_long_short_ratio=np.asarray(global_ratio, dtype=np.float64),
        top_position_long_short_ratio=np.asarray(top_position_ratio, dtype=np.float64),
        source_digest=_source_digest(
            schema=_BINANCE_V4_METRICS_SCHEMA,
            source_uri=source_uri,
            payload=payload,
        ),
    )


def align_futures_metrics_to_decisions(
    decision_timestamps: object,
    metrics: BinanceFuturesMetricsSeries,
) -> AlignedFuturesMetrics:
    """Backward as-of join immutable metrics onto decision closes."""

    decisions = np.asarray(decision_timestamps, dtype="datetime64[ns]").reshape(-1)
    if decisions.size == 0 or np.any(np.isnat(decisions)):
        raise ValueError("V4 decision timestamps must be non-empty and finite")
    decision_ns = decisions.astype(np.int64)
    _strictly_increasing(decision_ns, field="V4 decision timestamps")
    if not isinstance(metrics, BinanceFuturesMetricsSeries):
        raise TypeError("metrics must be BinanceFuturesMetricsSeries")
    metric_ns = metrics.create_time.astype("datetime64[ns]").astype(np.int64)

    size = decisions.size
    open_interest = np.zeros(size, dtype=np.float64)
    global_ratio = np.zeros(size, dtype=np.float64)
    top_ratio = np.zeros(size, dtype=np.float64)
    available = np.zeros(size, dtype=np.bool_)
    staleness = np.full(
        size,
        BINANCE_V4_MAX_DERIVATIVES_STALENESS_HOURS,
        dtype=np.float64,
    )
    indices = np.searchsorted(metric_ns, decision_ns, side="right") - 1
    for row, source_index in enumerate(indices):
        if source_index < 0:
            continue
        age_hours = (
            float(decision_ns[row] - metric_ns[source_index]) / 3_600_000_000_000.0
        )
        if age_hours < 0.0:
            raise RuntimeError("V4 metrics backward as-of join produced negative age")
        open_interest[row] = metrics.open_interest_value[source_index]
        global_ratio[row] = metrics.global_long_short_ratio[source_index]
        top_ratio[row] = metrics.top_position_long_short_ratio[source_index]
        staleness[row] = age_hours
        available[row] = age_hours <= BINANCE_V4_MAX_DERIVATIVES_STALENESS_HOURS

    source_digest = content_and_arrays_digest(
        {
            "maximum_staleness_hours": BINANCE_V4_MAX_DERIVATIVES_STALENESS_HOURS,
            "metrics_digest": metrics.digest,
            "schema_version": _BINANCE_V4_ALIGNED_METRICS_SCHEMA,
        },
        (("decision_timestamps_ns", decision_ns),),
    )
    return AlignedFuturesMetrics(
        open_interest_value=open_interest,
        global_long_short_ratio=global_ratio,
        top_position_long_short_ratio=top_ratio,
        available=available,
        staleness_hours=staleness,
        source_digest=source_digest,
    )


__all__ = [
    "BINANCE_FUTURES_METRICS_COLUMNS",
    "BINANCE_V4_MAX_DERIVATIVES_STALENESS_HOURS",
    "AlignedFuturesMetrics",
    "BinanceFundingEventSeries",
    "BinanceFuturesMetricsSeries",
    "BinanceV4KlineSeries",
    "align_futures_metrics_to_decisions",
    "parse_binance_funding_archive",
    "parse_binance_futures_metrics_archive",
    "parse_binance_v4_kline_archive",
    "vision_futures_metrics_url",
]
