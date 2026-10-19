from decimal import Decimal
from fractions import Fraction

import pytest

from tools.market_account_oracle import (
    FillEconomics,
    ReferenceAccount,
    assert_observed_account,
)


def test_signed_non_dyadic_lots_keep_cash_and_marked_wealth_separate() -> None:
    account = ReferenceAccount(
        cash="1000",
        quantities=("0", "0"),
        marks=(10, 20),
        multipliers=(3, 5),
    )
    no_cost = FillEconomics()

    account.apply_fill(
        symbol_index=0,
        reported_quantity=0.3,
        execution_price=10,
        economics=no_cost,
        market_notional=1000,
        lot_count=3,
        lot_size="0.1",
    )
    account.apply_fill(
        symbol_index=1,
        reported_quantity=-0.6,
        execution_price=20,
        economics=no_cost,
        market_notional=1000,
        lot_count=-2,
        lot_size="0.3",
    )
    account.revalue((12, 25))
    snapshot = account.observe()

    assert account.quantities == (Fraction(3, 10), Fraction(-3, 5))
    assert account.cash == Decimal("1051")
    assert snapshot.wealth == Decimal("986.8")
    assert snapshot.total_cost == Decimal("0")


def test_maker_and_taker_apply_distinct_fees_and_spread_conventions() -> None:
    rates = {
        "fee_rate": "0.01",
        "row_fee_rate": "0.02",
        "maker_fee_rate": "0.03",
        "row_maker_fee_rate": "0.04",
        "taker_fee_rate": "0.05",
        "row_taker_fee_rate": "0.06",
        "spread_rate": "0.2",
        "row_spread_rate": "0.1",
    }
    maker = ReferenceAccount(
        cash="1000", quantities=("0",), marks=(100,), multipliers=(1,)
    )
    taker = ReferenceAccount(
        cash="1000", quantities=("0",), marks=(100,), multipliers=(1,)
    )

    maker_fill = maker.apply_fill(
        symbol_index=0,
        reported_quantity=1,
        execution_price=100,
        economics=FillEconomics(**rates, maker=True),
        market_notional=1000,
    )
    taker_fill = taker.apply_fill(
        symbol_index=0,
        reported_quantity=1,
        execution_price=100,
        economics=FillEconomics(**rates, maker=False),
        market_notional=1000,
    )

    # Maker: $10 combined fees plus $15 half-spread; taker: $14 fees plus $30 spread.
    assert maker_fill.notional == Decimal("100")
    assert maker_fill.cost == Decimal("25")
    assert maker.cash == Decimal("875")
    assert taker_fill.cost == Decimal("44")
    assert taker.cash == Decimal("856")


def test_funding_mask_rate_sign_and_borrow_multiplier_have_literal_results() -> None:
    account = ReferenceAccount(
        cash="1000",
        quantities=("2", "-3", "1", "-1"),
        marks=(100, 20, 10, 30),
        multipliers=(2, 1, 1, 1),
    )

    funding = account.apply_funding(
        price_rates=("0.25", "-0.1", "0", "0.2"),
        due=(False, True, True, True),
    )
    borrow = account.apply_borrow(
        annual_rates=("0", "0.1", "0", "0.2"),
        year_fraction="0.5",
        rate_multiplier="1.5",
    )
    snapshot = account.observe()

    assert funding == Decimal("-0.1")
    assert borrow == Decimal("9")
    assert snapshot.funding_pnl == Decimal("-0.1")
    assert snapshot.borrow_cost == Decimal("9")
    assert snapshot.cash == Decimal("990.9")
    assert snapshot.wealth == Decimal("1310.9")


def test_observed_account_rejects_missing_and_extra_fields() -> None:
    account = ReferenceAccount(
        cash="1000", quantities=("0",), marks=(1,), multipliers=(1,)
    )
    snapshot = account.observe()
    complete = snapshot.comparison_values()
    assert_observed_account(snapshot, complete)

    missing = dict(complete)
    del missing["funding_pnl"]
    with pytest.raises(AssertionError, match="field closure"):
        assert_observed_account(snapshot, missing)

    extra = dict(complete, unexpected=0)
    with pytest.raises(AssertionError, match="field closure"):
        assert_observed_account(snapshot, extra)
