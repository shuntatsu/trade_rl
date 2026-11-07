"""Exact parsing of Binance Vision kline and funding-rate archives.

Prices, quantities and rates stay :class:`~decimal.Decimal` read from the archive text,
never floats. Every row is checked against the sealed-window embargo before anything
is returned, so a caller can never hold (let alone write) an embargoed row.

Exactly one exchange-side artefact is recognised and skipped instead of rejected: the
zero-volume placeholder candle Binance published for the spot ``1h`` bar opening
2020-12-21 14:00 UTC (see :func:`is_known_exchange_placeholder`). It is never stored,
never silently dropped (it is returned and counted), and every other malformed row
still raises :class:`ArchiveContentError`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from typing import Final

from trade_rl.integrations.binance import (
    BinanceTransportError,
    binance_interval_milliseconds,
)
from trade_rl.integrations.binance.vision import (
    _csv_rows_from_zip,
    _looks_like_header,
    _normalize_epoch_ms,
)
from trade_rl.integrations.market_store.contracts import (
    ArchiveKind,
    ArchiveSpec,
    FundingRow,
    KlineRow,
    StoreMarket,
    ms_from_utc,
)
from trade_rl.integrations.market_store.embargo import EmbargoPolicy

# The one published exchange-side placeholder candle, as observed in the public spot
# monthly archives of BTC/ETH/BNB/XRP/ADA/SOL USDT 1h for 2020-12 (identical shape in
# every one): the bar opening 2020-12-21 14:00 UTC is flat (open = high = low = close),
# has zero volume, quote volume, taker volumes and trades, and carries a close_time
# about 12.6 minutes BEFORE its open_time (about 13:47:20 UTC), which is what the
# generic close-after-open check rejects. It is an outage placeholder, not
# a trade record, so it holds no information beyond "no bar was published".
KNOWN_PLACEHOLDER_MARKET: Final = StoreMarket.SPOT
KNOWN_PLACEHOLDER_INTERVAL: Final = "1h"
KNOWN_PLACEHOLDER_SYMBOLS: Final[frozenset[str]] = frozenset(
    {"BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "SOLUSDT"}
)
KNOWN_PLACEHOLDER_OPEN_MS: Final = ms_from_utc(datetime(2020, 12, 21, 14, tzinfo=UTC))


class ArchiveContentError(ValueError):
    """An archive's content is not a valid, in-period series of the declared kind."""


def _decimal(value: str, *, field: str, url: str) -> Decimal:
    try:
        parsed = Decimal(value.strip())
    except InvalidOperation as error:
        raise ArchiveContentError(f"{url}: invalid {field}: {value!r}") from error
    if not parsed.is_finite():
        raise ArchiveContentError(f"{url}: {field} must be finite")
    return parsed


def _integer(value: str, *, field: str, url: str) -> int:
    try:
        return int(value.strip())
    except ValueError as error:
        raise ArchiveContentError(f"{url}: invalid {field}: {value!r}") from error


def _timestamp(value: str, *, field: str, url: str) -> int:
    try:
        return _normalize_epoch_ms(value)
    except ValueError as error:
        raise ArchiveContentError(f"{url}: invalid {field}: {value!r}") from error


def _rows(payload: bytes, *, url: str) -> list[list[str]]:
    try:
        return _csv_rows_from_zip(payload, source=url)
    except BinanceTransportError as error:
        raise ArchiveContentError(str(error)) from error


def validate_kline(row: KlineRow, *, url: str) -> None:
    """The same invariants the database CHECK constraints enforce, loudly and early."""

    prices = (row.open, row.high, row.low, row.close)
    if any(price <= 0 for price in prices):
        raise ArchiveContentError(f"{url}: non-positive price at {row.open_time_ms}")
    if row.high < max(row.open, row.close) or row.low > min(row.open, row.close):
        raise ArchiveContentError(f"{url}: OHLC bounds violated at {row.open_time_ms}")
    quantities = (
        row.volume,
        row.quote_volume,
        row.taker_buy_base_volume,
        row.taker_buy_quote_volume,
    )
    if any(value < 0 for value in quantities) or row.trades < 0:
        raise ArchiveContentError(f"{url}: negative volume at {row.open_time_ms}")
    if row.close_time_ms <= row.open_time_ms:
        raise ArchiveContentError(
            f"{url}: close_time not after open at {row.open_time_ms}"
        )


@dataclass(frozen=True, slots=True)
class SkippedPlaceholder:
    """A recognised exchange placeholder candle that was skipped, kept as evidence."""

    url: str
    open_time_ms: int
    close_time_ms: int
    price: Decimal

    def to_dict(self) -> dict[str, object]:
        return {
            "close_time_ms": self.close_time_ms,
            "open_time_ms": self.open_time_ms,
            "price": str(self.price),
            "url": self.url,
        }


@dataclass(frozen=True, slots=True)
class ParsedKlines:
    """Valid rows to store plus the recognised placeholders that were skipped."""

    rows: list[KlineRow]
    placeholders: tuple[SkippedPlaceholder, ...] = ()


def is_known_exchange_placeholder(
    row: KlineRow, spec: ArchiveSpec, *, interval_ms: int
) -> bool:
    """True only for the exact, documented 2020-12-21 14:00 UTC spot ``1h`` placeholder.

    Every condition must hold, so a nonzero volume or trade count, a non-flat candle,
    an unverified symbol, a different market, interval or bar, or a close_time that is
    not strictly before the open (by less than one bar) is NOT a placeholder and fails
    validation as usual.
    """

    return (
        spec.market is KNOWN_PLACEHOLDER_MARKET
        and spec.interval == KNOWN_PLACEHOLDER_INTERVAL
        and spec.symbol in KNOWN_PLACEHOLDER_SYMBOLS
        and row.open_time_ms == KNOWN_PLACEHOLDER_OPEN_MS
        and row.open > 0
        and row.open == row.high == row.low == row.close
        and row.volume == row.quote_volume == 0
        and row.taker_buy_base_volume == row.taker_buy_quote_volume == 0
        and row.trades == 0
        and 0 < row.open_time_ms - row.close_time_ms < interval_ms
    )


def parse_kline_archive(
    payload: bytes, spec: ArchiveSpec, *, embargo: EmbargoPolicy
) -> ParsedKlines:
    """Parse a verified kline archive; refuse embargoed, out-of-period or invalid rows.

    The one recognised exchange placeholder (:func:`is_known_exchange_placeholder`) is
    returned in ``placeholders`` instead of ``rows``; it passes every other check
    (embargo, period, alignment, ordering) first.
    """

    if spec.kind is not ArchiveKind.KLINES or spec.interval is None:
        raise ValueError("spec is not a klines archive")
    interval_ms = binance_interval_milliseconds(spec.interval)
    rows = _rows(payload, url=spec.url)
    if rows and _looks_like_header(rows[0]):
        rows = rows[1:]
    start_ms = ms_from_utc(spec.period_start)
    end_ms = ms_from_utc(spec.period_end)
    result: list[KlineRow] = []
    placeholders: list[SkippedPlaceholder] = []
    previous = -1
    for raw in rows:
        if len(raw) < 11:
            raise ArchiveContentError(f"{spec.url}: kline row is short ({len(raw)})")
        open_ms = _timestamp(raw[0], field="open_time", url=spec.url)
        close_ms = _timestamp(raw[6], field="close_time", url=spec.url)
        embargo.check_ms(open_ms, context=f"{spec.url} open_time")
        embargo.check_ms(close_ms, context=f"{spec.url} close_time")
        if not start_ms <= open_ms < end_ms:
            raise ArchiveContentError(
                f"{spec.url}: row at {open_ms} lies outside the archive period"
            )
        if open_ms % interval_ms != 0:
            raise ArchiveContentError(f"{spec.url}: row at {open_ms} is not aligned")
        if open_ms <= previous:
            raise ArchiveContentError(f"{spec.url}: rows are not strictly increasing")
        previous = open_ms
        row = KlineRow(
            open_time_ms=open_ms,
            open=_decimal(raw[1], field="open", url=spec.url),
            high=_decimal(raw[2], field="high", url=spec.url),
            low=_decimal(raw[3], field="low", url=spec.url),
            close=_decimal(raw[4], field="close", url=spec.url),
            volume=_decimal(raw[5], field="volume", url=spec.url),
            close_time_ms=close_ms,
            quote_volume=_decimal(raw[7], field="quote_volume", url=spec.url),
            trades=_integer(raw[8], field="trades", url=spec.url),
            taker_buy_base_volume=_decimal(
                raw[9], field="taker_buy_base", url=spec.url
            ),
            taker_buy_quote_volume=_decimal(
                raw[10], field="taker_buy_quote", url=spec.url
            ),
        )
        if is_known_exchange_placeholder(row, spec, interval_ms=interval_ms):
            placeholders.append(
                SkippedPlaceholder(
                    url=spec.url,
                    open_time_ms=row.open_time_ms,
                    close_time_ms=row.close_time_ms,
                    price=row.open,
                )
            )
            continue
        validate_kline(row, url=spec.url)
        result.append(row)
    return ParsedKlines(result, tuple(placeholders))


def _funding_columns(header: Sequence[str], *, url: str) -> dict[str, int | None]:
    index = {name.strip(): position for position, name in enumerate(header)}
    if {"calc_time", "last_funding_rate"} <= set(index):
        time_name, rate_name = "calc_time", "last_funding_rate"
    elif {"fundingTime", "fundingRate"} <= set(index):
        time_name, rate_name = "fundingTime", "fundingRate"
    else:
        raise ArchiveContentError(f"{url}: unsupported funding header {tuple(index)}")
    mark_name = next(
        (name for name in ("markPrice", "mark_price") if name in index), None
    )
    return {
        "time": index[time_name],
        "rate": index[rate_name],
        "mark": None if mark_name is None else index[mark_name],
        "hours": index.get("funding_interval_hours"),
    }


def parse_funding_archive(
    payload: bytes, spec: ArchiveSpec, *, embargo: EmbargoPolicy
) -> list[FundingRow]:
    if spec.kind is not ArchiveKind.FUNDING_RATE:
        raise ValueError("spec is not a funding archive")
    rows = _rows(payload, url=spec.url)
    if not rows:
        return []
    if not _looks_like_header(rows[0]):
        raise ArchiveContentError(f"{spec.url}: funding archive lacks a header")
    columns = _funding_columns(rows[0], url=spec.url)
    start_ms = ms_from_utc(spec.period_start)
    end_ms = ms_from_utc(spec.period_end)
    result: list[FundingRow] = []
    previous = -1
    for raw in rows[1:]:
        time_index = columns["time"]
        rate_index = columns["rate"]
        assert time_index is not None and rate_index is not None
        try:
            time_text, rate_text = raw[time_index], raw[rate_index]
        except IndexError as error:
            raise ArchiveContentError(f"{spec.url}: funding row is short") from error
        funding_ms = _timestamp(time_text, field="funding_time", url=spec.url)
        embargo.check_ms(funding_ms, context=f"{spec.url} funding_time")
        if not start_ms <= funding_ms < end_ms:
            raise ArchiveContentError(
                f"{spec.url}: funding row at {funding_ms} lies outside the archive period"
            )
        if funding_ms <= previous:
            raise ArchiveContentError(f"{spec.url}: funding rows are not increasing")
        previous = funding_ms
        rate = _decimal(rate_text, field="funding_rate", url=spec.url)
        if not -1 < rate < 1:
            raise ArchiveContentError(f"{spec.url}: implausible funding rate {rate}")
        mark_index = columns["mark"]
        hours_index = columns["hours"]
        mark = None
        if mark_index is not None and raw[mark_index].strip():
            mark = _decimal(raw[mark_index], field="mark_price", url=spec.url)
            if mark <= 0:
                raise ArchiveContentError(f"{spec.url}: non-positive mark price")
        hours = None
        if hours_index is not None and raw[hours_index].strip():
            hours = _integer(
                raw[hours_index], field="funding_interval_hours", url=spec.url
            )
        result.append(
            FundingRow(
                funding_time_ms=funding_ms,
                rate=rate,
                mark_price=mark,
                interval_hours=hours,
            )
        )
    return result


__all__ = [
    "KNOWN_PLACEHOLDER_INTERVAL",
    "KNOWN_PLACEHOLDER_MARKET",
    "KNOWN_PLACEHOLDER_OPEN_MS",
    "ArchiveContentError",
    "ParsedKlines",
    "SkippedPlaceholder",
    "is_known_exchange_placeholder",
    "parse_funding_archive",
    "parse_kline_archive",
    "validate_kline",
]
