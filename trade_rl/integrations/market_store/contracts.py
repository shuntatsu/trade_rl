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


class StoreExportPolicy(StrEnum):
    """Explicit opt-in for a new export identity; legacy exports stay unchanged."""

    LEGACY_V1 = "legacy_v1"
    STRICT_SPOT_LINEAGE_V2 = "strict_spot_lineage_v2"


class ArchiveKind(StrEnum):
    KLINES = "klines"
    FUNDING_RATE = "funding_rate"
    #: USD-M daily ``metrics`` archive: 5-minute open-interest / long-short snapshots.
    METRICS = "metrics"
    #: USD-M daily ``bookDepth`` archive: cumulative depth within signed percent bands.
    BOOK_DEPTH = "book_depth"


#: The kinds every plan and ingest run covered before the opt-in kinds existed. The
#: default stays exactly this pair so existing runs and plan identities are unchanged.
DEFAULT_ARCHIVE_KINDS: Final = (ArchiveKind.KLINES, ArchiveKind.FUNDING_RATE)
#: Opt-in kinds that exist only as USD-M **daily** Vision archives.
DAILY_USDS_M_ONLY_KINDS: Final = frozenset(
    {ArchiveKind.METRICS, ArchiveKind.BOOK_DEPTH}
)


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
            raise ValueError(
                "klines archives need an interval; other archive kinds have none"
            )
        if self.period_start >= self.period_end:
            raise ValueError("archive period must be non-empty")
        if self.kind in DAILY_USDS_M_ONLY_KINDS:
            if self.market is not StoreMarket.USDS_M:
                raise ValueError(
                    f"{self.kind.value} archives exist only for USD-M futures, "
                    f"not {self.market.value}"
                )
            if self.period_kind is not PeriodKind.DAILY:
                raise ValueError(
                    f"{self.kind.value} archives exist only as daily archives, "
                    f"not {self.period_kind.value}"
                )
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
class OpenInterestMetricRow:
    """One exact 5-minute USD-M ``metrics`` snapshot, read from the archive text.

    ``create_time_ms`` is the archive's ``create_time`` (UTC, exact seconds). Values are
    stored exactly as published; their economic meaning and any availability lag are
    the consumer's to establish.
    """

    create_time_ms: int
    sum_open_interest: Decimal
    sum_open_interest_value: Decimal
    count_toptrader_long_short_ratio: Decimal
    sum_toptrader_long_short_ratio: Decimal
    count_long_short_ratio: Decimal
    sum_taker_long_short_vol_ratio: Decimal

    @property
    def create_time(self) -> datetime:
        return utc_from_ms(self.create_time_ms)

    def comparable(self) -> dict[str, object]:
        return {
            "sum_open_interest": self.sum_open_interest,
            "sum_open_interest_value": self.sum_open_interest_value,
            "count_toptrader_long_short_ratio": self.count_toptrader_long_short_ratio,
            "sum_toptrader_long_short_ratio": self.sum_toptrader_long_short_ratio,
            "count_long_short_ratio": self.count_long_short_ratio,
            "sum_taker_long_short_vol_ratio": self.sum_taker_long_short_vol_ratio,
        }


@dataclass(frozen=True, slots=True)
class BookDepthRow:
    """One exact ``bookDepth`` observation: cumulative depth inside a signed % band.

    A negative ``percentage`` is the bid side (that far below the mid), a positive one
    the ask side; ``depth`` is the cumulative base quantity and ``notional`` the
    cumulative quote value inside the band, both exactly as published.
    """

    timestamp_ms: int
    percentage: int
    depth: Decimal
    notional: Decimal

    @property
    def timestamp(self) -> datetime:
        return utc_from_ms(self.timestamp_ms)

    @property
    def key(self) -> tuple[int, int]:
        return (self.timestamp_ms, self.percentage)

    def comparable(self) -> dict[str, object]:
        return {"depth": self.depth, "notional": self.notional}


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


@dataclass(frozen=True, slots=True)
class WarmupRun:
    """An ingestion run that stored FS1 warm-up rows, with its recorded authorization."""

    run_id: int
    authorization_url: str

    def to_dict(self) -> dict[str, object]:
        return {"authorization_url": self.authorization_url, "run_id": self.run_id}


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

    def warmup_provenance(
        self,
        market: StoreMarket,
        symbols: Sequence[str],
        interval: str,
        start_ms: int,
        end_ms: int,
    ) -> list[WarmupRun]:
        """Runs whose rows in ``[start_ms, end_ms)`` lie in the FS1 warm-up span."""
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
    "DAILY_USDS_M_ONLY_KINDS",
    "DEFAULT_ARCHIVE_KINDS",
    "DEFAULT_BASE_ASSETS",
    "DEFAULT_INTERVAL",
    "DEFAULT_QUOTE_ASSET",
    "MARKET_STORE_URL_ENV",
    "ArchiveKind",
    "ArchiveSpec",
    "BookDepthRow",
    "EmbargoState",
    "FundingRow",
    "KlineRow",
    "MarketReader",
    "OpenInterestMetricRow",
    "PeriodKind",
    "StoredExchangeInfo",
    "WarmupRun",
    "StoreMarket",
    "StoreExportPolicy",
    "Universe",
    "funding_digest",
    "klines_digest",
    "ms_from_utc",
    "utc_from_ms",
]
