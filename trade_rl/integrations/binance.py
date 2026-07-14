"""Public Binance market-data ingestion for deterministic research datasets."""

from __future__ import annotations

import csv
import io
import json
import math
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any

import numpy as np

from trade_rl.data.builder import MarketDatasetBuilder
from trade_rl.data.contracts import (
    FeatureKind,
    FeatureSpec,
    InstrumentContract,
    MarketBuildConfig,
    VolumeUnit,
)
from trade_rl.data.market import MarketDataset
from trade_rl.data.source import MarketDataSource, RawMarketSeries

_INTERVAL_MILLISECONDS = {
    "15m": 15 * 60 * 1_000,
    "30m": 30 * 60 * 1_000,
    "1h": 60 * 60 * 1_000,
    "2h": 2 * 60 * 60 * 1_000,
    "4h": 4 * 60 * 60 * 1_000,
    "6h": 6 * 60 * 60 * 1_000,
    "8h": 8 * 60 * 60 * 1_000,
    "12h": 12 * 60 * 60 * 1_000,
    "1d": 24 * 60 * 60 * 1_000,
}

_REST_BASE = {
    "spot": "https://api.binance.com",
    "usds-m": "https://fapi.binance.com",
    "coin-m": "https://dapi.binance.com",
}
_REST_KLINES = {
    "spot": "/api/v3/klines",
    "usds-m": "/fapi/v1/klines",
    "coin-m": "/dapi/v1/klines",
}
_REST_EXCHANGE_INFO = {
    "spot": "/api/v3/exchangeInfo",
    "usds-m": "/fapi/v1/exchangeInfo",
    "coin-m": "/dapi/v1/exchangeInfo",
}
_REST_FUNDING = {
    "usds-m": "/fapi/v1/fundingRate",
    "coin-m": "/dapi/v1/fundingRate",
}
_VISION_ROOT = "https://data.binance.vision/data"
_USER_AGENT = "trade-rl/0.3 public-market-data"


class BinanceMarket(StrEnum):
    SPOT = "spot"
    USDS_M = "usds-m"
    COIN_M = "coin-m"


class BinanceTransportMode(StrEnum):
    AUTO = "auto"
    REST = "rest"
    VISION = "vision"


class BinanceTransportError(RuntimeError):
    """Public Binance transport failed after bounded retries."""


class BinanceUnsupportedContractError(ValueError):
    """Requested instrument cannot be represented by the current accounting model."""


@dataclass(frozen=True, slots=True)
class BinanceInstrumentMetadata:
    symbol: str
    listed_at: datetime
    tick_size: float
    lot_size: float
    minimum_notional: float
    volume_unit: VolumeUnit
    contract_multiplier: float = 1.0

    def __post_init__(self) -> None:
        if not self.symbol:
            raise ValueError("Binance symbol must not be empty")
        if self.listed_at.tzinfo is None or self.listed_at.utcoffset() is None:
            raise ValueError("listed_at must be timezone-aware")
        for field_name, value in (
            ("tick_size", self.tick_size),
            ("lot_size", self.lot_size),
            ("minimum_notional", self.minimum_notional),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{field_name} must be finite and non-negative")
        if (
            not math.isfinite(self.contract_multiplier)
            or self.contract_multiplier <= 0.0
        ):
            raise ValueError("contract_multiplier must be finite and positive")

    def to_contract(self) -> InstrumentContract:
        return InstrumentContract(
            symbol=self.symbol,
            listed_at=self.listed_at.astimezone(UTC),
            volume_unit=self.volume_unit,
            contract_multiplier=self.contract_multiplier,
            tick_size=self.tick_size,
            lot_size=self.lot_size,
            minimum_notional=self.minimum_notional,
        )


@dataclass(frozen=True, slots=True)
class BinanceDatasetBuildResult:
    dataset: MarketDataset
    metadata: tuple[BinanceInstrumentMetadata, ...]
    sources_used: tuple[str, ...]
    feature_timeframes: tuple[str, ...] = ()


def _market(value: BinanceMarket | str) -> BinanceMarket:
    try:
        return BinanceMarket(value)
    except ValueError as error:
        raise ValueError(f"unsupported Binance market: {value}") from error


def _mode(value: BinanceTransportMode | str) -> BinanceTransportMode:
    try:
        return BinanceTransportMode(value)
    except ValueError as error:
        raise ValueError(f"unsupported Binance transport: {value}") from error


def _aware_utc(value: datetime, *, field: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware")
    return value.astimezone(UTC)


def _epoch_ms(value: datetime) -> int:
    return int(value.timestamp() * 1_000)


def _normalize_epoch_ms(value: object) -> int:
    try:
        numeric = int(str(value))
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid Binance timestamp: {value!r}") from error
    while abs(numeric) >= 10_000_000_000_000:
        numeric //= 1_000
    return numeric


def _interval_ms(interval: str) -> int:
    try:
        return _INTERVAL_MILLISECONDS[interval]
    except KeyError as error:
        raise ValueError(f"unsupported Binance interval: {interval}") from error


def _day_floor_ms(value: int) -> datetime:
    return datetime.fromtimestamp(value / 1_000, tz=UTC).replace(
        hour=0,
        minute=0,
        second=0,
        microsecond=0,
    )


def _iter_days(start_ms: int, end_ms: int) -> Iterable[datetime]:
    day = _day_floor_ms(start_ms)
    last = _day_floor_ms(end_ms - 1)
    while day <= last:
        yield day
        day += timedelta(days=1)


def _iter_months(start_ms: int, end_ms: int) -> Iterable[datetime]:
    month = _day_floor_ms(start_ms).replace(day=1)
    last = _day_floor_ms(end_ms - 1).replace(day=1)
    while month <= last:
        yield month
        if month.month == 12:
            month = month.replace(year=month.year + 1, month=1)
        else:
            month = month.replace(month=month.month + 1)


def vision_kline_url(
    market: BinanceMarket | str,
    symbol: str,
    interval: str,
    day: datetime,
) -> str:
    resolved = _market(market)
    _interval_ms(interval)
    date = _aware_utc(day, field="day").strftime("%Y-%m-%d")
    if resolved is BinanceMarket.SPOT:
        prefix = "spot/daily/klines"
    elif resolved is BinanceMarket.USDS_M:
        prefix = "futures/um/daily/klines"
    else:
        prefix = "futures/cm/daily/klines"
    return f"{_VISION_ROOT}/{prefix}/{symbol}/{interval}/{symbol}-{interval}-{date}.zip"


def vision_monthly_kline_url(
    market: BinanceMarket | str,
    symbol: str,
    interval: str,
    month: datetime,
) -> str:
    resolved = _market(market)
    _interval_ms(interval)
    period = _aware_utc(month, field="month").strftime("%Y-%m")
    if resolved is BinanceMarket.SPOT:
        prefix = "spot/monthly/klines"
    elif resolved is BinanceMarket.USDS_M:
        prefix = "futures/um/monthly/klines"
    else:
        prefix = "futures/cm/monthly/klines"
    return (
        f"{_VISION_ROOT}/{prefix}/{symbol}/{interval}/{symbol}-{interval}-{period}.zip"
    )


def _next_month(value: datetime) -> datetime:
    if value.month == 12:
        return value.replace(year=value.year + 1, month=1, day=1)
    return value.replace(month=value.month + 1, day=1)


def plan_vision_kline_urls(
    market: BinanceMarket | str,
    symbol: str,
    interval: str,
    start_time: datetime,
    end_time: datetime,
) -> tuple[str, ...]:
    start = _aware_utc(start_time, field="start_time")
    end = _aware_utc(end_time, field="end_time")
    if end <= start:
        raise ValueError("end_time must be later than start_time")
    cursor = start.replace(hour=0, minute=0, second=0, microsecond=0)
    urls: list[str] = []
    while cursor < end:
        month_start = cursor.replace(day=1)
        next_month = _next_month(month_start)
        if cursor == month_start and start <= cursor and next_month <= end:
            urls.append(vision_monthly_kline_url(market, symbol, interval, month_start))
            cursor = next_month
        else:
            urls.append(vision_kline_url(market, symbol, interval, cursor))
            cursor += timedelta(days=1)
    return tuple(urls)


def vision_funding_url(
    market: BinanceMarket | str,
    symbol: str,
    month: datetime,
) -> str:
    resolved = _market(market)
    if resolved is BinanceMarket.SPOT:
        raise ValueError("spot markets do not have funding rates")
    period = _aware_utc(month, field="month").strftime("%Y-%m")
    product = "um" if resolved is BinanceMarket.USDS_M else "cm"
    return (
        f"{_VISION_ROOT}/futures/{product}/monthly/fundingRate/{symbol}/"
        f"{symbol}-fundingRate-{period}.zip"
    )


def _csv_rows_from_zip(payload: bytes, *, source: str) -> list[list[str]]:
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            members = tuple(
                name for name in archive.namelist() if not name.endswith("/")
            )
            if len(members) != 1:
                raise BinanceTransportError(
                    f"{source} archive must contain exactly one file, found {members}"
                )
            raw = archive.read(members[0]).decode("utf-8-sig")
    except (UnicodeDecodeError, zipfile.BadZipFile, KeyError) as error:
        raise BinanceTransportError(
            f"invalid Binance Vision archive: {source}"
        ) from error
    return [row for row in csv.reader(io.StringIO(raw)) if row]


def _looks_like_header(row: Sequence[str]) -> bool:
    return bool(row) and not row[0].strip().lstrip("-").isdigit()


def _finite_float(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise ValueError(f"invalid {field}: {value!r}")
    try:
        result = float(value)
    except ValueError as error:
        raise ValueError(f"invalid {field}: {value!r}") from error
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


class BinancePublicTransport:
    """Bounded public HTTP transport for REST and Binance Vision archives."""

    def __init__(
        self,
        *,
        timeout_seconds: float = 30.0,
        max_attempts: int = 3,
        retry_backoff_seconds: float = 0.25,
    ) -> None:
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0.0:
            raise ValueError("timeout_seconds must be finite and positive")
        if isinstance(max_attempts, bool) or not isinstance(max_attempts, int):
            raise ValueError("max_attempts must be an integer")
        if max_attempts <= 0:
            raise ValueError("max_attempts must be positive")
        if not math.isfinite(retry_backoff_seconds) or retry_backoff_seconds < 0.0:
            raise ValueError("retry_backoff_seconds must be non-negative")
        self.timeout_seconds = timeout_seconds
        self.max_attempts = max_attempts
        self.retry_backoff_seconds = retry_backoff_seconds

    def _request_bytes(self, url: str) -> bytes:
        request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
        last_error: BaseException | None = None
        for attempt in range(self.max_attempts):
            try:
                with urllib.request.urlopen(  # noqa: S310 - fixed HTTPS endpoints
                    request,
                    timeout=self.timeout_seconds,
                ) as response:
                    return response.read()
            except urllib.error.HTTPError as error:
                last_error = error
                if error.code == 404:
                    break
                if error.code < 500 and error.code not in {418, 429, 451}:
                    break
            except (TimeoutError, urllib.error.URLError) as error:
                last_error = error
            if attempt + 1 < self.max_attempts and self.retry_backoff_seconds > 0.0:
                time.sleep(self.retry_backoff_seconds * (2**attempt))
        detail = "unknown transport error" if last_error is None else str(last_error)
        raise BinanceTransportError(f"Binance request failed for {url}: {detail}")

    def _request_json(self, url: str) -> object:
        payload = self._request_bytes(url)
        try:
            return json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise BinanceTransportError(
                f"Binance returned invalid JSON for {url}"
            ) from error

    @staticmethod
    def _query_url(base: str, path: str, values: Mapping[str, object]) -> str:
        return f"{base}{path}?{urllib.parse.urlencode(values)}"

    def _load_rest_klines(
        self,
        *,
        market: BinanceMarket,
        symbol: str,
        interval: str,
        start_ms: int,
        end_ms: int,
    ) -> list[list[object]]:
        rows: list[list[object]] = []
        cursor = start_ms
        step = _interval_ms(interval)
        while cursor < end_ms:
            url = self._query_url(
                _REST_BASE[market.value],
                _REST_KLINES[market.value],
                {
                    "symbol": symbol,
                    "interval": interval,
                    "startTime": cursor,
                    "endTime": end_ms - 1,
                    "limit": 1_000,
                },
            )
            payload = self._request_json(url)
            if not isinstance(payload, list):
                raise BinanceTransportError("Binance kline response must be a list")
            chunk: list[list[object]] = []
            for item in payload:
                if not isinstance(item, list):
                    raise BinanceTransportError("Binance kline row must be a list")
                chunk.append(item)
            if not chunk:
                break
            rows.extend(chunk)
            last_open = _normalize_epoch_ms(chunk[-1][0])
            next_cursor = last_open + step
            if next_cursor <= cursor:
                raise BinanceTransportError(
                    "Binance REST kline pagination did not advance"
                )
            cursor = next_cursor
            if len(chunk) < 1_000:
                break
        return rows

    def _load_vision_klines(
        self,
        *,
        market: BinanceMarket,
        symbol: str,
        interval: str,
        start_ms: int,
        end_ms: int,
    ) -> list[list[object]]:
        result: list[list[object]] = []
        start_time = datetime.fromtimestamp(start_ms / 1_000, tz=UTC)
        end_time = datetime.fromtimestamp(end_ms / 1_000, tz=UTC)
        for url in plan_vision_kline_urls(
            market, symbol, interval, start_time, end_time
        ):
            rows = _csv_rows_from_zip(self._request_bytes(url), source=url)
            if rows and _looks_like_header(rows[0]):
                rows = rows[1:]
            for row in rows:
                if len(row) < 8:
                    raise BinanceTransportError(
                        f"Binance Vision kline row is short: {url}"
                    )
                open_ms = _normalize_epoch_ms(row[0])
                if start_ms <= open_ms < end_ms:
                    result.append(list(row))
        return result

    def load_klines(
        self,
        *,
        market: BinanceMarket | str,
        symbol: str,
        interval: str,
        start_ms: int,
        end_ms: int,
        mode: BinanceTransportMode | str = BinanceTransportMode.AUTO,
    ) -> tuple[list[list[object]], str]:
        resolved_market = _market(market)
        resolved_mode = _mode(mode)
        if resolved_mode is BinanceTransportMode.REST:
            return (
                self._load_rest_klines(
                    market=resolved_market,
                    symbol=symbol,
                    interval=interval,
                    start_ms=start_ms,
                    end_ms=end_ms,
                ),
                "rest",
            )
        if resolved_mode is BinanceTransportMode.VISION:
            return (
                self._load_vision_klines(
                    market=resolved_market,
                    symbol=symbol,
                    interval=interval,
                    start_ms=start_ms,
                    end_ms=end_ms,
                ),
                "vision",
            )
        try:
            return (
                self._load_rest_klines(
                    market=resolved_market,
                    symbol=symbol,
                    interval=interval,
                    start_ms=start_ms,
                    end_ms=end_ms,
                ),
                "rest",
            )
        except BinanceTransportError:
            return (
                self._load_vision_klines(
                    market=resolved_market,
                    symbol=symbol,
                    interval=interval,
                    start_ms=start_ms,
                    end_ms=end_ms,
                ),
                "vision",
            )

    def _load_rest_funding(
        self,
        *,
        market: BinanceMarket,
        symbol: str,
        start_ms: int,
        end_ms: int,
    ) -> list[tuple[int, float]]:
        if market is BinanceMarket.SPOT:
            return []
        cursor = start_ms
        result: list[tuple[int, float]] = []
        while cursor < end_ms:
            url = self._query_url(
                _REST_BASE[market.value],
                _REST_FUNDING[market.value],
                {
                    "symbol": symbol,
                    "startTime": cursor,
                    "endTime": end_ms,
                    "limit": 1_000,
                },
            )
            payload = self._request_json(url)
            if not isinstance(payload, list):
                raise BinanceTransportError("Binance funding response must be a list")
            chunk: list[tuple[int, float]] = []
            for item in payload:
                if not isinstance(item, dict):
                    raise BinanceTransportError("Binance funding row must be an object")
                timestamp = _normalize_epoch_ms(item.get("fundingTime"))
                rate = _finite_float(item.get("fundingRate"), field="funding rate")
                if start_ms <= timestamp <= end_ms:
                    chunk.append((timestamp, rate))
            result.extend(chunk)
            if len(chunk) < 1_000:
                break
            next_cursor = chunk[-1][0] + 1
            if next_cursor <= cursor:
                raise BinanceTransportError(
                    "Binance funding pagination did not advance"
                )
            cursor = next_cursor
        return result

    def _load_vision_funding(
        self,
        *,
        market: BinanceMarket,
        symbol: str,
        start_ms: int,
        end_ms: int,
    ) -> list[tuple[int, float]]:
        if market is BinanceMarket.SPOT:
            return []
        result: list[tuple[int, float]] = []
        for month in _iter_months(start_ms, end_ms):
            url = vision_funding_url(market, symbol, month)
            rows = _csv_rows_from_zip(self._request_bytes(url), source=url)
            if not rows:
                continue
            if not _looks_like_header(rows[0]):
                raise BinanceTransportError(
                    f"Binance Vision funding archive lacks a supported header: {url}"
                )
            header = {name.strip(): index for index, name in enumerate(rows[0])}
            if {"calc_time", "last_funding_rate"}.issubset(header):
                time_field = "calc_time"
                rate_field = "last_funding_rate"
            elif {"fundingTime", "fundingRate"}.issubset(header):
                time_field = "fundingTime"
                rate_field = "fundingRate"
            else:
                raise BinanceTransportError(
                    f"Binance Vision funding header is unsupported: {tuple(header)}"
                )
            for row in rows[1:]:
                try:
                    timestamp = _normalize_epoch_ms(row[header[time_field]])
                    rate = _finite_float(row[header[rate_field]], field="funding rate")
                except IndexError as error:
                    raise BinanceTransportError(
                        f"Binance Vision funding row is short: {url}"
                    ) from error
                if start_ms <= timestamp <= end_ms:
                    result.append((timestamp, rate))
        return result

    def load_funding_rates(
        self,
        *,
        market: BinanceMarket | str,
        symbol: str,
        start_ms: int,
        end_ms: int,
        mode: BinanceTransportMode | str = BinanceTransportMode.AUTO,
    ) -> tuple[list[tuple[int, float]], str]:
        resolved_market = _market(market)
        if resolved_market is BinanceMarket.SPOT:
            return [], "spot:no-funding"
        resolved_mode = _mode(mode)
        if resolved_mode is BinanceTransportMode.REST:
            return (
                self._load_rest_funding(
                    market=resolved_market,
                    symbol=symbol,
                    start_ms=start_ms,
                    end_ms=end_ms,
                ),
                "rest",
            )
        if resolved_mode is BinanceTransportMode.VISION:
            return (
                self._load_vision_funding(
                    market=resolved_market,
                    symbol=symbol,
                    start_ms=start_ms,
                    end_ms=end_ms,
                ),
                "vision",
            )
        try:
            return (
                self._load_rest_funding(
                    market=resolved_market,
                    symbol=symbol,
                    start_ms=start_ms,
                    end_ms=end_ms,
                ),
                "rest",
            )
        except BinanceTransportError:
            return (
                self._load_vision_funding(
                    market=resolved_market,
                    symbol=symbol,
                    start_ms=start_ms,
                    end_ms=end_ms,
                ),
                "vision",
            )

    def load_exchange_information(
        self,
        *,
        market: BinanceMarket | str,
        mode: BinanceTransportMode | str = BinanceTransportMode.AUTO,
    ) -> tuple[dict[str, object], str]:
        resolved_market = _market(market)
        resolved_mode = _mode(mode)
        if resolved_mode is BinanceTransportMode.VISION:
            raise BinanceTransportError(
                "Binance Vision does not publish exchange metadata; provide static metadata"
            )
        url = (
            f"{_REST_BASE[resolved_market.value]}"
            f"{_REST_EXCHANGE_INFO[resolved_market.value]}"
        )
        payload = self._request_json(url)
        if not isinstance(payload, dict):
            raise BinanceTransportError(
                "Binance exchange information must be an object"
            )
        return dict(payload), "rest"


def _parse_kline_rows(
    rows: Sequence[Sequence[object]],
    *,
    interval_ms: int,
    start_ms: int,
    end_ms: int,
) -> tuple[np.ndarray, ...]:
    parsed: list[tuple[int, float, float, float, float, float]] = []
    for row in rows:
        if len(row) < 8:
            raise ValueError("Binance kline row must contain at least eight fields")
        open_ms = _normalize_epoch_ms(row[0])
        if not start_ms <= open_ms < end_ms:
            continue
        close_ms = open_ms + interval_ms
        if close_ms > end_ms:
            continue
        open_price = _finite_float(row[1], field="open")
        high = _finite_float(row[2], field="high")
        low = _finite_float(row[3], field="low")
        close = _finite_float(row[4], field="close")
        quote_volume = _finite_float(row[7], field="quote volume")
        parsed.append((close_ms, open_price, high, low, close, quote_volume))
    if len(parsed) < 2:
        raise ValueError("Binance range must contain at least two closed bars")
    timestamps = np.asarray([item[0] for item in parsed], dtype=np.int64)
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError("Binance kline timestamps must be strictly increasing")
    if np.any(np.diff(timestamps) != interval_ms):
        raise ValueError("Binance kline range must be complete and exactly regular")
    return (
        timestamps.astype("datetime64[ms]").astype("datetime64[ns]"),
        np.asarray([item[1] for item in parsed], dtype=np.float64),
        np.asarray([item[2] for item in parsed], dtype=np.float64),
        np.asarray([item[3] for item in parsed], dtype=np.float64),
        np.asarray([item[4] for item in parsed], dtype=np.float64),
        np.asarray([item[5] for item in parsed], dtype=np.float64),
    )


def _align_funding(
    timestamps: np.ndarray,
    events: Sequence[tuple[int, float]],
) -> tuple[np.ndarray, np.ndarray]:
    timestamp_ms = timestamps.astype("datetime64[ms]").astype(np.int64)
    index_by_timestamp = {int(value): index for index, value in enumerate(timestamp_ms)}
    funding = np.zeros(len(timestamps), dtype=np.float64)
    available = np.zeros(len(timestamps), dtype=np.bool_)
    previous: int | None = None
    for raw_timestamp, raw_rate in sorted(events):
        timestamp = _normalize_epoch_ms(raw_timestamp)
        if previous is not None and timestamp == previous:
            raise ValueError("Binance funding timestamps must be unique")
        previous = timestamp
        index = index_by_timestamp.get(timestamp)
        if index is None:
            continue
        funding[index] = _finite_float(raw_rate, field="funding rate")
        available[index] = True
    return funding, available


class BinanceMarketDataSource(MarketDataSource):
    """Load one fixed Binance range on one or more causal native clocks."""

    def __init__(
        self,
        *,
        market: BinanceMarket | str,
        interval: str,
        start_time: datetime,
        end_time: datetime,
        transport_mode: BinanceTransportMode | str = BinanceTransportMode.AUTO,
        transport: Any | None = None,
    ) -> None:
        self.market = _market(market)
        self.interval = interval
        self.interval_ms = _interval_ms(interval)
        self.start_time = _aware_utc(start_time, field="start_time")
        self.end_time = _aware_utc(end_time, field="end_time")
        if self.end_time <= self.start_time:
            raise ValueError("end_time must be later than start_time")
        start_ms = _epoch_ms(self.start_time)
        end_ms = _epoch_ms(self.end_time)
        if start_ms % self.interval_ms != 0 or end_ms % self.interval_ms != 0:
            raise ValueError("Binance range boundaries must align to the interval")
        self.transport_mode = _mode(transport_mode)
        self.transport = transport or BinancePublicTransport()
        self._sources_used: set[str] = set()
        self._series_cache: dict[tuple[str, str], RawMarketSeries] = {}
        self._funding_cache: dict[str, tuple[list[tuple[int, float]], object]] = {}

    @property
    def sources_used(self) -> tuple[str, ...]:
        return tuple(sorted(self._sources_used))

    def _record_source(self, source: object) -> None:
        if isinstance(source, str):
            self._sources_used.add(source)
            return
        if isinstance(source, Sequence):
            self._sources_used.update(str(item) for item in source)
            return
        self._sources_used.add(str(source))

    def _funding_events(self, symbol: str) -> list[tuple[int, float]]:
        cached = self._funding_cache.get(symbol)
        if cached is None:
            events, funding_source = self.transport.load_funding_rates(
                market=self.market,
                symbol=symbol,
                start_ms=_epoch_ms(self.start_time),
                end_ms=_epoch_ms(self.end_time),
                mode=self.transport_mode,
            )
            cached = (list(events), funding_source)
            self._funding_cache[symbol] = cached
            self._record_source(funding_source)
        return cached[0]

    def load(self, symbol: str) -> RawMarketSeries:
        return self.load_timeframe(symbol, self.interval)

    def load_timeframe(self, symbol: str, timeframe: str) -> RawMarketSeries:
        if not symbol:
            raise ValueError("Binance symbol must not be empty")
        interval_ms = _interval_ms(timeframe)
        start_ms = _epoch_ms(self.start_time)
        end_ms = _epoch_ms(self.end_time)
        if start_ms % interval_ms != 0 or end_ms % interval_ms != 0:
            raise ValueError(
                f"Binance range boundaries must align to native timeframe {timeframe}"
            )
        key = (symbol, timeframe)
        cached = self._series_cache.get(key)
        if cached is not None:
            return cached
        rows, kline_source = self.transport.load_klines(
            market=self.market,
            symbol=symbol,
            interval=timeframe,
            start_ms=start_ms,
            end_ms=end_ms,
            mode=self.transport_mode,
        )
        self._record_source(kline_source)
        timestamps, open_price, high, low, close, volume = _parse_kline_rows(
            rows,
            interval_ms=interval_ms,
            start_ms=start_ms,
            end_ms=end_ms,
        )
        if timeframe == self.interval:
            funding, funding_available = _align_funding(
                timestamps,
                self._funding_events(symbol),
            )
        else:
            funding = np.zeros(len(timestamps), dtype=np.float64)
            funding_available = np.zeros(len(timestamps), dtype=np.bool_)
        series = RawMarketSeries(
            timestamps=timestamps,
            available_at=timestamps,
            open=open_price,
            high=high,
            low=low,
            close=close,
            volume=volume,
            funding_rate=funding,
            funding_available=funding_available,
            tradable=np.ones(len(timestamps), dtype=np.bool_),
        )
        self._series_cache[key] = series
        return series


def _filter_value(
    values: Sequence[Mapping[str, object]],
    *,
    filter_type: str,
    fields: Sequence[str],
) -> float:
    for item in values:
        if item.get("filterType") != filter_type:
            continue
        for field in fields:
            raw = item.get(field)
            if raw is not None:
                return _finite_float(raw, field=f"{filter_type}.{field}")
    return 0.0


def _metadata_from_exchange_info(
    payload: Mapping[str, object],
    *,
    market: BinanceMarket,
    symbols: tuple[str, ...],
) -> tuple[BinanceInstrumentMetadata, ...]:
    raw_symbols = payload.get("symbols")
    if not isinstance(raw_symbols, list):
        raise ValueError("Binance exchange information lacks symbols")
    by_symbol: dict[str, Mapping[str, object]] = {}
    for item in raw_symbols:
        if isinstance(item, dict) and isinstance(item.get("symbol"), str):
            by_symbol[str(item["symbol"])] = item
    result: list[BinanceInstrumentMetadata] = []
    for symbol in symbols:
        item = by_symbol.get(symbol)
        if item is None:
            raise ValueError(f"Binance exchange information has no symbol {symbol}")
        status = item.get("status", item.get("contractStatus"))
        if status != "TRADING":
            raise ValueError(f"Binance symbol {symbol} is not trading: {status}")
        raw_filters = item.get("filters")
        if not isinstance(raw_filters, list):
            raise ValueError(f"Binance symbol {symbol} lacks filters")
        filters = tuple(value for value in raw_filters if isinstance(value, dict))
        listed_raw = item.get("onboardDate", 0)
        listed_ms = _normalize_epoch_ms(listed_raw)
        listed_at = datetime.fromtimestamp(listed_ms / 1_000, tz=UTC)
        result.append(
            BinanceInstrumentMetadata(
                symbol=symbol,
                listed_at=listed_at,
                tick_size=_filter_value(
                    filters,
                    filter_type="PRICE_FILTER",
                    fields=("tickSize",),
                ),
                lot_size=_filter_value(
                    filters,
                    filter_type="LOT_SIZE",
                    fields=("stepSize",),
                ),
                minimum_notional=max(
                    _filter_value(
                        filters,
                        filter_type="MIN_NOTIONAL",
                        fields=("notional", "minNotional"),
                    ),
                    _filter_value(
                        filters,
                        filter_type="NOTIONAL",
                        fields=("minNotional",),
                    ),
                ),
                volume_unit=VolumeUnit.QUOTE_NOTIONAL,
                contract_multiplier=1.0,
            )
        )
    return tuple(result)


def _optional_values(
    values: Sequence[float] | None,
    *,
    symbols: tuple[str, ...],
    field: str,
) -> tuple[float, ...] | None:
    if values is None or len(values) == 0:
        return None
    result = tuple(float(value) for value in values)
    if len(result) != len(symbols):
        raise ValueError(f"{field} must be provided once per Binance symbol")
    return result


def _optional_datetimes(
    values: Sequence[datetime] | None,
    *,
    symbols: tuple[str, ...],
) -> tuple[datetime, ...] | None:
    if values is None or len(values) == 0:
        return None
    result = tuple(_aware_utc(value, field="listed_at") for value in values)
    if len(result) != len(symbols):
        raise ValueError("listed_at must be provided once per Binance symbol")
    return result


def binance_multitimeframe_feature_specs(
    *,
    base_timeframe: str,
    feature_timeframes: Sequence[str],
) -> tuple[FeatureSpec, ...]:
    _interval_ms(base_timeframe)
    resolved = tuple(feature_timeframes)
    if len(set(resolved)) != len(resolved):
        raise ValueError("duplicate Binance feature timeframes are not allowed")
    if base_timeframe in resolved:
        raise ValueError("base timeframe must not be repeated as a feature timeframe")
    for timeframe in resolved:
        _interval_ms(timeframe)
    ordered = tuple(sorted((*resolved, base_timeframe), key=_interval_ms))
    features: list[FeatureSpec] = []
    for timeframe in ordered:
        native = None if timeframe == base_timeframe else timeframe
        native_hours = _interval_ms(timeframe) / 3_600_000.0
        staleness = max(
            native_hours * 2.0, base_timeframe == timeframe and 1.0 or native_hours
        )
        features.append(
            FeatureSpec(
                name=f"{timeframe}__log_return_1bar",
                kind=FeatureKind.LOG_RETURN,
                timeframe=native,
                lookback=1,
                max_staleness_hours=staleness,
            )
        )
        if timeframe == base_timeframe:
            one_day = max(1, int(round(24.0 / native_hours)))
            features.extend(
                (
                    FeatureSpec(
                        name=f"{timeframe}__log_return_1d",
                        kind=FeatureKind.LOG_RETURN,
                        lookback=one_day,
                        max_staleness_hours=staleness,
                    ),
                    FeatureSpec(
                        name=f"{timeframe}__volume_zscore_1d",
                        kind=FeatureKind.VOLUME_ZSCORE,
                        lookback=one_day,
                        min_periods=min(one_day, 2),
                        max_staleness_hours=staleness,
                    ),
                    FeatureSpec(
                        name=f"{timeframe}__funding_bps",
                        kind=FeatureKind.FUNDING_BPS,
                        max_staleness_hours=8.0,
                    ),
                )
            )
        elif timeframe == "1d":
            features.append(
                FeatureSpec(
                    name="1d__log_return_7bar",
                    kind=FeatureKind.LOG_RETURN,
                    timeframe="1d",
                    lookback=7,
                    max_staleness_hours=48.0,
                )
            )
        else:
            volatility_lookback = (
                4 if timeframe == "15m" else max(2, int(round(24.0 / native_hours)))
            )
            features.append(
                FeatureSpec(
                    name=(f"{timeframe}__realized_volatility_{volatility_lookback}bar"),
                    kind=FeatureKind.REALIZED_VOLATILITY,
                    timeframe=timeframe,
                    lookback=volatility_lookback,
                    max_staleness_hours=staleness,
                )
            )
    return tuple(features)


def _default_features(interval: str) -> tuple[FeatureSpec, ...]:
    bar_hours = _interval_ms(interval) / 3_600_000.0
    one_day = max(1, int(round(24.0 / bar_hours)))
    return (
        FeatureSpec(
            name="log_return_1bar",
            kind=FeatureKind.LOG_RETURN,
            lookback=1,
            max_staleness_hours=max(bar_hours * 2.0, 1.0),
        ),
        FeatureSpec(
            name="log_return_1d",
            kind=FeatureKind.LOG_RETURN,
            lookback=one_day,
            max_staleness_hours=max(bar_hours * 2.0, 1.0),
        ),
        FeatureSpec(
            name="realized_volatility_1d",
            kind=FeatureKind.REALIZED_VOLATILITY,
            lookback=one_day,
            max_staleness_hours=max(bar_hours * 2.0, 1.0),
        ),
        FeatureSpec(
            name="volume_zscore_1d",
            kind=FeatureKind.VOLUME_ZSCORE,
            lookback=one_day,
            min_periods=min(one_day, 2),
            max_staleness_hours=max(bar_hours * 2.0, 1.0),
        ),
        FeatureSpec(
            name="funding_bps",
            kind=FeatureKind.FUNDING_BPS,
            lookback=1,
            max_staleness_hours=8.0,
        ),
    )


def build_binance_market_dataset(
    *,
    market: BinanceMarket | str,
    symbols: Sequence[str],
    interval: str,
    start_time: datetime,
    end_time: datetime,
    transport_mode: BinanceTransportMode | str = BinanceTransportMode.AUTO,
    transport: Any | None = None,
    tick_sizes: Sequence[float] | None = None,
    lot_sizes: Sequence[float] | None = None,
    minimum_notionals: Sequence[float] | None = None,
    listed_ats: Sequence[datetime] | None = None,
    feature_timeframes: Sequence[str] | None = None,
) -> BinanceDatasetBuildResult:
    """Build one deterministic linear-product dataset from public Binance data."""

    resolved_market = _market(market)
    if resolved_market is BinanceMarket.COIN_M:
        raise BinanceUnsupportedContractError(
            "Binance COIN-M uses inverse contract value and PnL; the current linear "
            "BookState cannot represent it safely"
        )
    resolved_symbols = tuple(symbols)
    if not resolved_symbols or any(not symbol for symbol in resolved_symbols):
        raise ValueError("Binance symbols must not be empty")
    if len(set(resolved_symbols)) != len(resolved_symbols):
        raise ValueError("Binance symbols must be unique")
    resolved_mode = _mode(transport_mode)
    requested_feature_timeframes = tuple(feature_timeframes or ())
    resolved_features = (
        _default_features(interval)
        if not requested_feature_timeframes
        else binance_multitimeframe_feature_specs(
            base_timeframe=interval,
            feature_timeframes=requested_feature_timeframes,
        )
    )
    resolved_tick = _optional_values(
        tick_sizes,
        symbols=resolved_symbols,
        field="tick-size",
    )
    resolved_lot = _optional_values(
        lot_sizes,
        symbols=resolved_symbols,
        field="lot-size",
    )
    resolved_minimum = _optional_values(
        minimum_notionals,
        symbols=resolved_symbols,
        field="minimum-notional",
    )
    resolved_listed = _optional_datetimes(listed_ats, symbols=resolved_symbols)
    client = transport or BinancePublicTransport()
    metadata_source: str | None = None
    if all(
        value is not None
        for value in (resolved_tick, resolved_lot, resolved_minimum, resolved_listed)
    ):
        assert resolved_tick is not None
        assert resolved_lot is not None
        assert resolved_minimum is not None
        assert resolved_listed is not None
        metadata = tuple(
            BinanceInstrumentMetadata(
                symbol=symbol,
                listed_at=listed_at,
                tick_size=tick,
                lot_size=lot,
                minimum_notional=minimum,
                volume_unit=VolumeUnit.QUOTE_NOTIONAL,
            )
            for symbol, listed_at, tick, lot, minimum in zip(
                resolved_symbols,
                resolved_listed,
                resolved_tick,
                resolved_lot,
                resolved_minimum,
                strict=True,
            )
        )
    else:
        payload, metadata_source = client.load_exchange_information(
            market=resolved_market,
            mode=resolved_mode,
        )
        metadata = _metadata_from_exchange_info(
            payload,
            market=resolved_market,
            symbols=resolved_symbols,
        )
        metadata = tuple(
            BinanceInstrumentMetadata(
                symbol=item.symbol,
                listed_at=(
                    item.listed_at
                    if resolved_listed is None
                    else resolved_listed[index]
                ),
                tick_size=(
                    item.tick_size if resolved_tick is None else resolved_tick[index]
                ),
                lot_size=item.lot_size if resolved_lot is None else resolved_lot[index],
                minimum_notional=(
                    item.minimum_notional
                    if resolved_minimum is None
                    else resolved_minimum[index]
                ),
                volume_unit=item.volume_unit,
                contract_multiplier=item.contract_multiplier,
            )
            for index, item in enumerate(metadata)
        )
    source = BinanceMarketDataSource(
        market=resolved_market,
        interval=interval,
        start_time=start_time,
        end_time=end_time,
        transport_mode=resolved_mode,
        transport=client,
    )
    dataset = MarketDatasetBuilder(
        MarketBuildConfig(
            base_timeframe=interval,
            features=resolved_features,
        )
    ).build(source, tuple(item.to_contract() for item in metadata))
    sources = set(source.sources_used)
    if metadata_source is not None:
        sources.add(metadata_source)
    return BinanceDatasetBuildResult(
        dataset=dataset,
        metadata=metadata,
        sources_used=tuple(sorted(sources)),
        feature_timeframes=tuple(
            sorted(
                {spec.resolved_timeframe(interval) for spec in resolved_features},
                key=_interval_ms,
            )
        ),
    )


__all__ = [
    "BinanceDatasetBuildResult",
    "BinanceInstrumentMetadata",
    "BinanceMarket",
    "BinanceMarketDataSource",
    "BinancePublicTransport",
    "BinanceTransportError",
    "BinanceTransportMode",
    "BinanceUnsupportedContractError",
    "binance_multitimeframe_feature_specs",
    "build_binance_market_dataset",
    "plan_vision_kline_urls",
    "vision_funding_url",
    "vision_kline_url",
    "vision_monthly_kline_url",
]
