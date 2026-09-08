"""Pure Binance Vision URL planning and archive parsing."""

from __future__ import annotations

import csv
import io
import zipfile
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime, timedelta

from trade_rl.integrations.binance.types import (
    BinanceMarket,
    BinanceTransportError,
    _aware_utc,
    _market,
)

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

_VISION_ROOT = "https://data.binance.vision/data"


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
