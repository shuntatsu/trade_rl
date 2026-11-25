"""Versioned raw Binance candle finality admission (opt-in new Dataset only)."""

from __future__ import annotations

from collections.abc import Sequence
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from typing import Final

from trade_rl.integrations.binance.vision import _normalize_epoch_ms

_KlineRow = Sequence[object]


class BinanceSourceClosePolicy(StrEnum):
    """Versioned admission of Binance source bar finality for new builds only."""

    RECONSTRUCTED_V1 = "reconstructed_v1"
    STRICT_SOURCE_CLOSE_V2 = "strict_source_close_v2"
    STRICT_SPOT_PLACEHOLDER_V3 = "strict_spot_placeholder_v3"


def _native_timestamp(value: object) -> int:
    # Strict v2 accepts the exact integer fields supplied in Binance klines.
    # The generic legacy normalizer deliberately accepts/coerces more formats.
    raw = str(value)
    if not raw.isascii() or not raw.isdecimal():
        raise ValueError("Binance source close_time requires integer native timestamps")
    return int(raw)


def _check_source_close_time(
    row: _KlineRow,
    *,
    open_ms: int,
    interval_ms: int,
    source_close_policy: BinanceSourceClosePolicy,
) -> None:
    if source_close_policy is not BinanceSourceClosePolicy.RECONSTRUCTED_V1:
        # Source formats use millisecond time fields or (SPOT Vision from
        # 2025-01-01, and opt-in REST) microsecond fields. Converting the raw
        # close to ms first discards up to 999 microseconds of invalidity.
        # The raw open must agree exactly with the already-normalized grid;
        # native units cannot be mixed within the same kline row.
        source_open = _native_timestamp(row[0])
        source_close = _native_timestamp(row[6])
        if source_open == open_ms:
            unit = 1
        elif source_open == open_ms * 1000:
            unit = 1000
        else:
            raise ValueError("Binance source close_time has invalid native open time")
        expected_close = (open_ms + interval_ms) * unit - 1
        if source_close != expected_close:
            raise ValueError(
                "Binance source close_time differs from expected interval end"
            )


# Shared historical exception, not a new authorization to inspect market data.
KNOWN_PLACEHOLDER_SYMBOLS: Final[frozenset[str]] = frozenset(
    {
        "BTCUSDT",
        "ETHUSDT",
        "BNBUSDT",
        "XRPUSDT",
        "ADAUSDT",
        "SOLUSDT",
        # Additional symbols established by the prior result-blind coverage run.
        "LTCUSDT",
        "AVAXUSDT",
        "DOGEUSDT",
        "DOTUSDT",
        "LINKUSDT",
    }
)
KNOWN_PLACEHOLDER_OPEN_MS: Final = 1_608_559_200_000


def _is_known_spot_placeholder(
    *,
    market: str,
    interval: str,
    symbol: str,
    interval_ms: int,
    open_ms: int,
    close_ms: int,
    prices: tuple[Decimal, Decimal, Decimal, Decimal],
    quantities: tuple[Decimal, Decimal, Decimal, Decimal],
    trades: int,
) -> bool:
    """One shared semantic predicate for file and store row admission."""
    return (
        market == "spot"
        and interval == "1h"
        and symbol in KNOWN_PLACEHOLDER_SYMBOLS
        and open_ms == KNOWN_PLACEHOLDER_OPEN_MS
        and prices[0] > 0
        and prices[0] == prices[1] == prices[2] == prices[3]
        and all(value == 0 for value in quantities)
        and trades == 0
        and 0 < open_ms - close_ms < interval_ms
    )


def _spot_v3_rows(
    rows: Sequence[_KlineRow],
    *,
    symbol: str,
    interval_ms: int,
    start_ms: int,
    end_ms: int,
) -> tuple[list[_KlineRow], int]:
    """Admit native 1h spot rows; count the exact documented missing bar.

    Validate every in-range raw row before any gap fill, including ordering
    of the skipped row, so duplicate or off-grid input cannot disappear.
    The caller has already restricted v3 to direct 1h Vision input.
    """
    result: list[_KlineRow] = []
    skipped = 0
    previous: int | None = None
    for row in rows:
        if len(row) < 11 or any(
            type(value) is not str and type(value) is not int for value in row[:11]
        ):
            raise ValueError("spot Vision v3 requires raw kline fields")
        open_ms = _normalize_epoch_ms(row[0])
        if not start_ms <= open_ms < end_ms:
            continue
        close_ms = _normalize_epoch_ms(row[6])
        if open_ms % interval_ms:
            raise ValueError("spot Vision v3 kline must align to interval")
        if previous is not None and open_ms <= previous:
            raise ValueError(
                "spot Vision v3 kline timestamps must be strictly increasing"
            )
        previous = open_ms
        try:
            prices = (
                Decimal(str(row[1]).strip()),
                Decimal(str(row[2]).strip()),
                Decimal(str(row[3]).strip()),
                Decimal(str(row[4]).strip()),
            )
            quantities = (
                Decimal(str(row[5]).strip()),
                Decimal(str(row[7]).strip()),
                Decimal(str(row[9]).strip()),
                Decimal(str(row[10]).strip()),
            )
            trades = int(str(row[8]).strip())
        except (InvalidOperation, ValueError) as error:
            raise ValueError("invalid spot Vision v3 kline numeric fields") from error
        if not all(value.is_finite() for value in (*prices, *quantities)):
            raise ValueError("spot Vision v3 kline numeric fields must be finite")
        if _is_known_spot_placeholder(
            market="spot",
            interval="1h",
            symbol=symbol,
            interval_ms=interval_ms,
            open_ms=open_ms,
            close_ms=close_ms,
            prices=prices,
            quantities=quantities,
            trades=trades,
        ):
            if str(row[0]) != str(open_ms) or str(row[6]) != str(close_ms):
                raise ValueError(
                    "source close_time placeholder requires native milliseconds"
                )
            skipped += 1
            continue
        if (
            any(value <= 0 for value in prices)
            or prices[1] < max(prices[0], prices[3])
            or prices[2] > min(prices[0], prices[3])
            or any(value < 0 for value in quantities)
            or trades < 0
        ):
            raise ValueError("spot Vision v3 kline violates OHLC or volume invariants")
        _check_source_close_time(
            row,
            open_ms=open_ms,
            interval_ms=interval_ms,
            source_close_policy=BinanceSourceClosePolicy.STRICT_SPOT_PLACEHOLDER_V3,
        )
        result.append(row)
    return result, skipped
