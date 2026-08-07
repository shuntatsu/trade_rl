from __future__ import annotations

from decimal import Decimal

import pytest

import trade_rl.integrations.nautilus.funding_adapter as funding_adapter
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


def test_funding_settlement_becomes_canonical_equity_trace_record() -> None:
    ledger = CanonicalFundingLedger()
    settlement = ledger.settle(_settlement())

    record = funding_adapter.canonicalize_funding_settlement_record(
        settlement,
        sequence=3,
        price_tick=Decimal("0.01"),
        lot_size=Decimal("0.001"),
        equity_before_minor=10_000_000_000,
    )

    assert record.sequence == 3
    assert record.event_type == "funding"
    assert record.timestamp_ns == 120
    assert record.price_ticks == 10_200
    assert record.quantity_lots == 0
    assert record.fee_minor == 0
    assert record.funding_minor == -1_020_000
    assert record.position_lots == 1_000
    assert record.equity_minor == 9_998_980_000
    assert record.terminal_reason is None


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"sequence": 0}, "sequence"),
        ({"price_tick": Decimal("0.07")}, "price"),
        ({"lot_size": Decimal("0.003")}, "quantity"),
        ({"equity_before_minor": 1.5}, "equity_before_minor"),
    ],
)
def test_funding_trace_record_rejects_invalid_canonical_identity(
    kwargs: dict[str, object], message: str
) -> None:
    ledger = CanonicalFundingLedger()
    settlement = ledger.settle(_settlement())
    arguments: dict[str, object] = {
        "sequence": 1,
        "price_tick": Decimal("0.01"),
        "lot_size": Decimal("0.001"),
        "equity_before_minor": 10_000_000_000,
    }
    arguments.update(kwargs)

    with pytest.raises(ValueError, match=message):
        funding_adapter.canonicalize_funding_settlement_record(
            settlement,
            **arguments,
        )
