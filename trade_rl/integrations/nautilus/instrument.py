"""Frozen single-instrument definitions for the Nautilus execution boundary."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from trade_rl.integrations.nautilus.runtime_identity import require_nautilus_runtime


@dataclass(frozen=True, slots=True)
class FrozenCryptoPerpetualSpec:
    """Serializable inputs required to create a Nautilus ``CryptoPerpetual``."""

    instrument_id: str
    raw_symbol: str
    base_currency: str
    quote_currency: str
    settlement_currency: str
    price_precision: int
    size_precision: int
    price_increment: str
    size_increment: str
    min_quantity: str
    min_notional: str
    margin_init: str
    margin_maint: str
    maker_fee: str
    taker_fee: str


MAINTAINED_BTCUSDT_PERPETUAL = FrozenCryptoPerpetualSpec(
    instrument_id="BTCUSDT-PERP.BINANCE",
    raw_symbol="BTCUSDT",
    base_currency="BTC",
    quote_currency="USDT",
    settlement_currency="USDT",
    price_precision=1,
    size_precision=3,
    price_increment="0.1",
    size_increment="0.001",
    min_quantity="0.001",
    min_notional="5.0",
    margin_init="0.01",
    margin_maint="0.005",
    maker_fee="0.0002",
    taker_fee="0.0004",
)


def build_crypto_perpetual(spec: FrozenCryptoPerpetualSpec) -> Any:
    """Build a Nautilus instrument only after exact-runtime validation."""

    require_nautilus_runtime()

    from nautilus_trader.model.currencies import Currency
    from nautilus_trader.model.identifiers import InstrumentId, Symbol
    from nautilus_trader.model.instruments import CryptoPerpetual
    from nautilus_trader.model.objects import Money, Price, Quantity

    base = Currency.from_str(spec.base_currency)
    quote = Currency.from_str(spec.quote_currency)
    settlement = Currency.from_str(spec.settlement_currency)
    return CryptoPerpetual(
        instrument_id=InstrumentId.from_str(spec.instrument_id),
        raw_symbol=Symbol(spec.raw_symbol),
        base_currency=base,
        quote_currency=quote,
        settlement_currency=settlement,
        is_inverse=False,
        price_precision=spec.price_precision,
        size_precision=spec.size_precision,
        price_increment=Price.from_str(spec.price_increment),
        size_increment=Quantity.from_str(spec.size_increment),
        min_quantity=Quantity.from_str(spec.min_quantity),
        min_notional=Money(Decimal(spec.min_notional), settlement),
        margin_init=Decimal(spec.margin_init),
        margin_maint=Decimal(spec.margin_maint),
        maker_fee=Decimal(spec.maker_fee),
        taker_fee=Decimal(spec.taker_fee),
        ts_event=0,
        ts_init=0,
    )


def build_maintained_btcusdt_perpetual() -> Any:
    """Create the frozen maintained Binance USDT-M BTC perpetual definition."""

    return build_crypto_perpetual(MAINTAINED_BTCUSDT_PERPETUAL)


__all__ = [
    "FrozenCryptoPerpetualSpec",
    "MAINTAINED_BTCUSDT_PERPETUAL",
    "build_crypto_perpetual",
    "build_maintained_btcusdt_perpetual",
]
