"""Dependency-light value types shared by the market-data store modules."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from enum import StrEnum
from typing import Final, Protocol

from trade_rl.integrations.binance import BinanceMarket

DEFAULT_BASE_ASSETS: Final = ("BTC", "ETH", "BNB", "XRP", "ADA", "SOL")
DEFAULT_QUOTE_ASSET: Final = "USDT"
DEFAULT_INTERVAL: Final = "1h"
MARKET_STORE_URL_ENV: Final = "TRADE_RL_MARKET_STORE_URL"


class StoreMarket(StrEnum):
    SPOT = "spot"
    USDS_M = "usds_m"

    @property
    def binance(self) -> BinanceMarket:
        return BinanceMarket.SPOT if self is StoreMarket.SPOT else BinanceMarket.USDS_M

    @property
    def contract_type(self) -> str | None:
        return None if self is StoreMarket.SPOT else "PERPETUAL"

    @classmethod
    def from_binance(cls, market: BinanceMarket | str) -> StoreMarket:
        resolved = BinanceMarket(market)
        if resolved is BinanceMarket.SPOT:
            return cls.SPOT
        if resolved is BinanceMarket.USDS_M:
            return cls.USDS_M
        raise ValueError("the market store supports spot and USD-M perpetuals only")


class ArchiveKind(StrEnum):
    KLINES = "klines"
    FUNDING_RATE = "funding_rate"


class PeriodKind(StrEnum):
    MONTHLY = "monthly"
    DAILY = "daily"


@dataclass(frozen=True, slots=True)
class Universe:
    """Symbols and markets to maintain; the default is the carry-research roster."""

    symbols: tuple[str, ...] = tuple(
        f"{base}{DEFAULT_QUOTE_ASSET}" for base in DEFAULT_BASE_ASSETS
    )
    interval: str = DEFAULT_INTERVAL
    markets: tuple[StoreMarket, ...] = (StoreMarket.USDS_M, StoreMarket.SPOT)

    def __post_init__(self) -> None:
        if not self.symbols or len(set(self.symbols)) != len(self.symbols):
            raise ValueError("universe symbols must be non-empty and unique")
        for symbol in self.symbols:
            if not symbol or not symbol.isalnum() or symbol != symbol.upper():
                raise ValueError(f"invalid symbol: {symbol!r}")
        if not self.markets or len(set(self.markets)) != len(self.markets):
            raise ValueError("universe markets must be non-empty and unique")
        if not self.interval:
            raise ValueError("interval must not be empty")

    @classmethod
    def from_bases(
        cls,
        bases: Sequence[str],
        *,
        quote: str = DEFAULT_QUOTE_ASSET,
        interval: str = DEFAULT_INTERVAL,
        markets: Sequence[StoreMarket] = (StoreMarket.USDS_M, StoreMarket.SPOT),
    ) -> Universe:
        return cls(
            symbols=tuple(f"{base.strip().upper()}{quote.upper()}" for base in bases),
            interval=interval,
            markets=tuple(markets),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "interval": self.interval,
            "markets": [market.value for market in self.markets],
            "symbols": list(self.symbols),
        }


@dataclass(frozen=True, slots=True)
class ArchiveSpec:
    """One Binance Vision archive to ingest, with its official checksum URL."""

    kind: ArchiveKind
    market: StoreMarket
    symbol: str
    interval: str | None
    period_kind: PeriodKind
    period_start: datetime
    period_end: datetime
    url: str
    checksum_url: str

    def __post_init__(self) -> None:
        if (self.kind is ArchiveKind.KLINES) != (self.interval is not None):
            raise ValueError("klines archives need an interval, funding archives none")
        if self.period_start >= self.period_end:
            raise ValueError("archive period must be non-empty")
        if self.checksum_url != f"{self.url}.CHECKSUM":
            raise ValueError("checksum URL must be the archive URL plus .CHECKSUM")

    @property
    def filename(self) -> str:
        return self.url.rsplit("/", 1)[1]

    def to_dict(self) -> dict[str, object]:
        return {
            "checksum_url": self.checksum_url,
            "interval": self.interval,
            "kind": self.kind.value,
            "market": self.market.value,
            "period_end": self.period_end.isoformat(),
            "period_kind": self.period_kind.value,
            "period_start": self.period_start.isoformat(),
            "symbol": self.symbol,
            "url": self.url,
        }


@dataclass(frozen=True, slots=True)
class KlineRow:
    """One exact kline; every price/quantity is a Decimal read from the archive text."""

    open_time_ms: int
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    close_time_ms: int
    quote_volume: Decimal
    trades: int
    taker_buy_base_volume: Decimal
    taker_buy_quote_volume: Decimal

    def comparable(self) -> dict[str, object]:
        """Value-comparison view (Decimal equality ignores trailing zeros)."""

        return {
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "close_time": self.close_time_ms,
            "quote_volume": self.quote_volume,
            "trades": self.trades,
            "taker_buy_base_volume": self.taker_buy_base_volume,
            "taker_buy_quote_volume": self.taker_buy_quote_volume,
        }

    def vision_row(self) -> list[object]:
        """The Vision CSV column layout the existing Dataset builder consumes."""

        return [
            self.open_time_ms,
            _plain(self.open),
            _plain(self.high),
            _plain(self.low),
            _plain(self.close),
            _plain(self.volume),
            self.close_time_ms,
            _plain(self.quote_volume),
            self.trades,
            _plain(self.taker_buy_base_volume),
            _plain(self.taker_buy_quote_volume),
            "0",
        ]


@dataclass(frozen=True, slots=True)
class FundingRow:
    funding_time_ms: int
    rate: Decimal
    mark_price: Decimal | None = None
    interval_hours: int | None = None

    def comparable(self) -> dict[str, object]:
        return {
            "rate": self.rate,
            "mark_price": self.mark_price,
            "funding_interval_hours": self.interval_hours,
        }


@dataclass(frozen=True, slots=True)
class StoredExchangeInfo:
    """One raw exchangeInfo response exactly as retrieved, with its evidence."""

    market: StoreMarket
    raw_payload: bytes
    raw_payload_sha256: str
    source_uri: str
    retrieved_at: datetime


@dataclass(frozen=True, slots=True)
class EmbargoState:
    """The database-side boundaries (the second guard behind the Python layers)."""

    fs1_warmup_start: datetime
    sealed_from: datetime


class MarketReader(Protocol):
    """Read access the Dataset export needs; PostgreSQL and test fakes implement it."""

    def read_klines(
        self,
        market: StoreMarket,
        symbol: str,
        interval: str,
        start_ms: int,
        end_ms: int,
    ) -> list[KlineRow]:
        """Klines with ``start_ms <= open_time < end_ms``, ascending."""
        ...

    def read_funding(
        self, market: StoreMarket, symbol: str, start_ms: int, end_ms: int
    ) -> list[FundingRow]:
        """Funding rows with ``start_ms <= funding_time <= end_ms`` (inclusive, as
        the Vision transport), ascending."""
        ...

    def load_exchange_info(
        self, market: StoreMarket, sha256: str | None = None
    ) -> StoredExchangeInfo:
        """The named snapshot, or the latest one when ``sha256`` is None."""
        ...

    def embargo_state(self) -> EmbargoState: ...


def _plain(value: Decimal) -> str:
    return format(value, "f")


def _canonical(value: object) -> str:
    if isinstance(value, Decimal):
        return format(value.normalize(), "f")
    return str(value)


def klines_digest(rows: Sequence[KlineRow]) -> str:
    """Order-sensitive, trailing-zero-insensitive digest of exact kline values."""

    digest = hashlib.sha256()
    for row in rows:
        fields = (row.open_time_ms, *row.comparable().values())
        digest.update("|".join(_canonical(value) for value in fields).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def funding_digest(rows: Sequence[FundingRow]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        fields = (row.funding_time_ms, *row.comparable().values())
        digest.update("|".join(_canonical(value) for value in fields).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def utc_from_ms(timestamp_ms: int) -> datetime:
    """Exact UTC datetime from integer epoch milliseconds (no float rounding)."""

    seconds, millis = divmod(timestamp_ms, 1_000)
    return datetime.fromtimestamp(seconds, tz=UTC).replace(microsecond=millis * 1_000)


def ms_from_utc(value: datetime) -> int:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("datetime must be timezone-aware")
    delta = value.astimezone(UTC) - datetime(1970, 1, 1, tzinfo=UTC)
    return (delta.days * 86_400 + delta.seconds) * 1_000 + delta.microseconds // 1_000


__all__ = [
    "DEFAULT_BASE_ASSETS",
    "DEFAULT_INTERVAL",
    "DEFAULT_QUOTE_ASSET",
    "MARKET_STORE_URL_ENV",
    "ArchiveKind",
    "ArchiveSpec",
    "EmbargoState",
    "FundingRow",
    "KlineRow",
    "MarketReader",
    "PeriodKind",
    "StoredExchangeInfo",
    "StoreMarket",
    "Universe",
    "funding_digest",
    "klines_digest",
    "ms_from_utc",
    "utc_from_ms",
]
