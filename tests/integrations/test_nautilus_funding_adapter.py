from __future__ import annotations

from decimal import Decimal

import pytest

from trade_rl.integrations.nautilus.funding_adapter import (
    CanonicalFundingLedger,
    FundingSettlementInput,
)


def _settlement(
    *,
    signed_quantity: str = "1",
    rate: str = "0.0001",
    boundary_ns: int = 120,
) -> FundingSettlementInput:
    return FundingSettlementInput(
        instrument_id="BTCUSDT-PERP.BINANCE",
        settlement_currency="USDT",
        currency_precision=8,
        signed_quantity=Decimal(signed_quantity),
        settlement_price=Decimal("102"),
        contract_multiplier=Decimal("1"),
        funding_rate=Decimal(rate),
        boundary_ns=boundary_ns,
    )


def test_positive_rate_debits_long_and_preserves_quantity() -> None:
    ledger = CanonicalFundingLedger()

    settlement = ledger.settle(_settlement())

    assert settlement.amount == Decimal("-0.01020000")
    assert settlement.amount_minor == -1_020_000
    assert settlement.quantity_change is None
    assert settlement.signed_quantity == Decimal("1")


def test_positive_rate_credits_short() -> None:
    ledger = CanonicalFundingLedger()

    settlement = ledger.settle(_settlement(signed_quantity="-1"))

    assert settlement.amount == Decimal("0.01020000")
    assert settlement.amount_minor == 1_020_000


def test_negative_rate_reverses_long_direction() -> None:
    ledger = CanonicalFundingLedger()

    settlement = ledger.settle(_settlement(rate="-0.0001"))

    assert settlement.amount == Decimal("0.01020000")


def test_zero_position_produces_zero_settlement_but_closes_boundary() -> None:
    ledger = CanonicalFundingLedger()

    settlement = ledger.settle(_settlement(signed_quantity="0"))

    assert settlement.amount == Decimal("0E-8")
    assert settlement.amount_minor == 0
    assert ledger.settled_boundaries == (120,)


def test_duplicate_boundary_fails_closed() -> None:
    ledger = CanonicalFundingLedger()
    ledger.settle(_settlement())

    with pytest.raises(ValueError, match="already settled"):
        ledger.settle(_settlement())


def test_boundary_order_must_be_strictly_increasing() -> None:
    ledger = CanonicalFundingLedger()
    ledger.settle(_settlement(boundary_ns=120))

    with pytest.raises(ValueError, match="strictly increasing"):
        ledger.settle(_settlement(boundary_ns=100))


def test_invalid_precision_or_nonfinite_inputs_fail_closed() -> None:
    with pytest.raises(ValueError, match="currency_precision"):
        FundingSettlementInput(
            instrument_id="BTCUSDT-PERP.BINANCE",
            settlement_currency="USDT",
            currency_precision=-1,
            signed_quantity=Decimal("1"),
            settlement_price=Decimal("102"),
            contract_multiplier=Decimal("1"),
            funding_rate=Decimal("0.0001"),
            boundary_ns=120,
        )

    with pytest.raises(ValueError, match="settlement_price"):
        FundingSettlementInput(
            instrument_id="BTCUSDT-PERP.BINANCE",
            settlement_currency="USDT",
            currency_precision=8,
            signed_quantity=Decimal("1"),
            settlement_price=Decimal("NaN"),
            contract_multiplier=Decimal("1"),
            funding_rate=Decimal("0.0001"),
            boundary_ns=120,
        )
