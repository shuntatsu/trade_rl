"""Exact exchange-rule extraction from a raw Binance ``exchangeInfo`` payload.

The raw bytes are stored verbatim (the Dataset builder re-reads them through the
existing metadata parser); this module only derives the queryable per-instrument rule
columns, as exact :class:`~decimal.Decimal` values. The rules are a snapshot valid
from the retrieval time, not historical point-in-time rules.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation

from trade_rl.integrations.binance.vision import _normalize_epoch_ms
from trade_rl.integrations.market_store.contracts import StoreMarket, utc_from_ms


class ExchangeInfoError(ValueError):
    """The payload cannot be read as exchange rules for the requested symbols."""


@dataclass(frozen=True, slots=True)
class SymbolRule:
    symbol: str
    trading_status: str
    onboard_date: datetime | None
    tick_size: Decimal
    lot_size: Decimal
    min_notional: Decimal


def _filter_decimal(
    filters: Sequence[Mapping[str, object]], filter_type: str, fields: Sequence[str]
) -> Decimal:
    for item in filters:
        if item.get("filterType") != filter_type:
            continue
        for field in fields:
            raw = item.get(field)
            if raw is None:
                continue
            try:
                value = Decimal(str(raw))
            except InvalidOperation as error:
                raise ExchangeInfoError(
                    f"invalid {filter_type}.{field}: {raw!r}"
                ) from error
            if not value.is_finite() or value < 0:
                raise ExchangeInfoError(f"{filter_type}.{field} must be non-negative")
            return value
    return Decimal(0)


def parse_symbol_rules(
    raw_payload: bytes, *, market: StoreMarket, symbols: Sequence[str]
) -> list[SymbolRule]:
    """Rules for ``symbols``; a symbol that is absent (or a non-perpetual) is refused."""

    try:
        payload = json.loads(raw_payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ExchangeInfoError("exchangeInfo payload is not valid JSON") from error
    raw_symbols = payload.get("symbols") if isinstance(payload, dict) else None
    if not isinstance(raw_symbols, list):
        raise ExchangeInfoError("exchangeInfo payload lacks a symbols list")
    by_symbol = {
        str(item["symbol"]): item
        for item in raw_symbols
        if isinstance(item, dict) and isinstance(item.get("symbol"), str)
    }
    result: list[SymbolRule] = []
    for symbol in symbols:
        item = by_symbol.get(symbol)
        if item is None:
            raise ExchangeInfoError(f"exchangeInfo has no symbol {symbol}")
        if market is StoreMarket.USDS_M and item.get("contractType") != "PERPETUAL":
            raise ExchangeInfoError(f"{symbol} is not a USD-M perpetual")
        raw_filters = item.get("filters")
        if not isinstance(raw_filters, list):
            raise ExchangeInfoError(f"{symbol} lacks filters")
        filters = [entry for entry in raw_filters if isinstance(entry, dict)]
        onboard = item.get("onboardDate")
        try:
            onboard_date = (
                None
                if onboard in (None, 0)
                else utc_from_ms(_normalize_epoch_ms(onboard))
            )
        except ValueError as error:
            raise ExchangeInfoError(f"{symbol} has an invalid onboardDate") from error
        notional = max(
            _filter_decimal(filters, "MIN_NOTIONAL", ("notional", "minNotional")),
            _filter_decimal(filters, "NOTIONAL", ("minNotional",)),
        )
        result.append(
            SymbolRule(
                symbol=symbol,
                trading_status=str(item.get("status", item.get("contractStatus", ""))),
                onboard_date=onboard_date,
                tick_size=_filter_decimal(filters, "PRICE_FILTER", ("tickSize",)),
                lot_size=_filter_decimal(filters, "LOT_SIZE", ("stepSize",)),
                min_notional=notional,
            )
        )
    return result


__all__ = ["ExchangeInfoError", "SymbolRule", "parse_symbol_rules"]
