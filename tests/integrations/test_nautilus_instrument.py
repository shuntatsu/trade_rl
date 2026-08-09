from __future__ import annotations

from decimal import Decimal

import pytest

pytest.importorskip("nautilus_trader")

from trade_rl.integrations.nautilus.instrument import (
    MAINTAINED_BTCUSDT_PERPETUAL,
    build_maintained_btcusdt_perpetual,
)


@pytest.mark.nautilus
def test_maintained_btcusdt_perpetual_has_frozen_identity_and_grid() -> None:
    instrument = build_maintained_btcusdt_perpetual()

    assert str(instrument.id) == "BTCUSDT-PERP.BINANCE"
    assert str(instrument.raw_symbol) == "BTCUSDT"
    assert instrument.price_precision == 1
    assert instrument.size_precision == 3
    assert str(instrument.price_increment) == "0.1"
    assert str(instrument.size_increment) == "0.001"
    assert str(instrument.min_quantity) == "0.001"
    assert instrument.min_notional.as_decimal() == Decimal("5")
    assert instrument.min_notional.currency.code == "USDT"


def test_maintained_spec_is_serializable_framework_neutral_data() -> None:
    spec = MAINTAINED_BTCUSDT_PERPETUAL

    assert spec.instrument_id == "BTCUSDT-PERP.BINANCE"
    assert spec.raw_symbol == "BTCUSDT"
    assert spec.base_currency == "BTC"
    assert spec.quote_currency == "USDT"
    assert spec.settlement_currency == "USDT"
