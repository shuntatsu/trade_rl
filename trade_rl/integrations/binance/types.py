"""Dependency-neutral Binance adapter contracts and validation helpers."""

from __future__ import annotations

import math
from datetime import UTC, datetime
from enum import StrEnum


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
