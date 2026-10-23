from decimal import Decimal
from fractions import Fraction

import pytest

from tools.market_account_oracle import (
    FillEconomics,
    ReferenceAccount,
    assert_observed_account,
)


def _account(*, cash="1000", quantities=("0",), marks=(100.0,), multipliers=(1.0,)):
    return ReferenceAccount(
        cash=cash,
        quantities=quantities,
        marks=marks,
        multipliers=multipliers,
    )


def test_flat_round_trip_loses_exactly_the_two_fill_fees() -> None:
    account = _account()
    economics = FillEconomics(fee_rate=0.01)

    buy = account.apply_fill(
        symbol_index=0,
        reported_quantity=1.0,
        execution_price=100.0,
        economics=economics,
        market_notional=1000.0,
    )
    sell = account.apply_fill(
        symbol_index=0,
        reported_quantity=-1.0,
        execution_price=100.0,
        economics=economics,
        market_notional=1000.0,
    )

    assert buy.notional == Decimal("100")
    assert sell.notional == Decimal("100")
    assert buy.cost == Decimal("1")
    assert sell.cost == Decimal("1")
    assert account.quantities == (Fraction(0),)
    assert account.cash == Decimal("998")
    assert account.observe().wealth == Decimal("998")


def test_three_decimal_lots_then_no_lot_close_consumes_exact_inventory() -> None:
    account = _account(marks=(10.0,))
    economics = FillEconomics()

    for _ in range(3):
        account.apply_fill(
            symbol_index=0,
            reported_quantity=0.1,
            execution_price=10.0,
            economics=economics,
            market_notional=1000.0,
            lot_count=1,
            lot_size=0.1,
        )

    assert account.quantities == (Fraction(3, 10),)
    account.apply_fill(
        symbol_index=0,
        reported_quantity=-0.3,
        execution_price=10.0,
        economics=economics,
        market_notional=1000.0,
    )
    assert account.quantities == (Fraction(0),)
    assert account.cash == Decimal("1000")


def test_cost_multiplier_and_participation_do_not_scale_inventory_or_funding() -> None:
    account = _account(marks=(10.0,), multipliers=(3.0,))
    fill = account.apply_fill(
        symbol_index=0,
        reported_quantity=2.0,
        execution_price=10.0,
        economics=FillEconomics(
            fee_rate=0.01,
            spread_rate=0.02,
            impact_rate=0.04,
            cost_multiplier=2.0,
        ),
        market_notional=600.0,
        quantity_capacity=8.0,
        lot_count=2,
        lot_size=1.0,
    )

    assert fill.participation == Decimal("0.25")
    assert fill.notional == Decimal("60")
    assert fill.cost == Decimal("6")
    assert account.quantities == (Fraction(2),)
    account.apply_funding(price_rates=(0.1,), due=(True,))
    snapshot = account.observe()
    assert snapshot.funding_pnl == Decimal("-0.6")
    assert snapshot.total_cost == Decimal("6")
    assert snapshot.cash == Decimal("933.4")


def test_signed_cash_interest_funding_mask_and_borrow_multiplier_are_separate() -> None:
    account = _account(
        cash="-100",
        quantities=("2", "-3"),
        marks=(100.0, 20.0),
        multipliers=(2.0, 1.0),
    )

    account.apply_cash_interest(annual_rate=0.1, year_fraction=0.5)
    account.apply_funding(price_rates=(0.25, 0.5), due=(True, False))
    account.apply_borrow(
        annual_rates=(0.1, 0.04),
        year_fraction=0.5,
        rate_multiplier=2.0,
    )
    snapshot = account.observe()

    assert snapshot.cash_interest == Decimal("-5")
    assert snapshot.funding_pnl == Decimal("-1")
    assert snapshot.borrow_cost == Decimal("2.4")
    assert snapshot.cash == Decimal("-108.4")


def test_split_dividend_and_settlement_use_separate_account_components() -> None:
    account = _account(
        cash="100",
        quantities=("3/2",),
        marks=(10.0,),
        multipliers=(2.0,),
    )

    account.apply_split((2.0,))
    assert account.quantities == (Fraction(3),)
    assert account.observe().wealth == Decimal("130")
    account.apply_dividend((0.5,))
    settlement = account.settle(
        mask=(True,),
        prices=(4.0,),
        recovery=(0.25,),
    )
    snapshot = account.observe()

    assert settlement == Decimal("6")
    assert snapshot.dividend == Decimal("3")
    assert snapshot.settlement_proceeds == Decimal("6")
    assert snapshot.cash == Decimal("109")
    assert account.quantities == (Fraction(0),)


def test_repeated_termination_preserves_debt_or_floors_at_zero_by_mode() -> None:
    def marked_short() -> ReferenceAccount:
        account = _account()
        account.apply_fill(
            symbol_index=0,
            reported_quantity=-5.0,
            execution_price=100.0,
            economics=FillEconomics(fee_rate=0.002),
            market_notional=5000.0,
        )
        account.revalue((400.0,))
        return account

    retained = marked_short()
    retained.terminate(mode="retain_debt")
    retained.terminate(mode="retain_debt")
    assert retained.observe().wealth == Decimal("-501")
    assert retained.cash == Decimal("-501")
    assert retained.quantities == (Fraction(0),)
    assert retained.observe().total_cost == Decimal("1")
    assert retained.observe().max_signed_drawdown == Decimal("1.501")
    assert retained.observe().max_native_drawdown == Decimal("1")
    with pytest.raises(ValueError, match="terminal"):
        retained.apply_cash_interest(annual_rate=0.1, year_fraction=1.0)
    assert retained.cash == Decimal("-501")

    floored = marked_short()
    floored.terminate(mode="floor_zero")
    floored.terminate(mode="floor_zero")
    assert floored.observe().wealth == Decimal("0")
    assert floored.cash == Decimal("0")
    assert floored.observe().total_cost == Decimal("1")


def test_observed_account_matching_is_closed_against_coherent_cash_wealth_cost_tamper() -> (
    None
):
    account = _account(marks=(10.0,))
    account.apply_fill(
        symbol_index=0,
        reported_quantity=1.0,
        execution_price=10.0,
        economics=FillEconomics(fee_rate=0.01),
        market_notional=100.0,
    )
    snapshot = account.observe()
    observed = dict(snapshot.comparison_values())
    assert_observed_account(snapshot, observed)

    observed["cash"] = Decimal(str(observed["cash"])) - Decimal("1")
    observed["wealth"] = Decimal(str(observed["wealth"])) - Decimal("1")
    observed["total_cost"] = Decimal(str(observed["total_cost"])) + Decimal("1")
    with pytest.raises(AssertionError):
        assert_observed_account(snapshot, observed)


@pytest.mark.parametrize("cash", [True, float("nan"), float("inf")])
def test_account_rejects_boolean_and_nonfinite_cash(cash: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        _account(cash=cash)


def test_account_rejects_noncanonical_quantity_and_mismatched_vectors() -> None:
    with pytest.raises((TypeError, ValueError)):
        _account(quantities=("2/4",))
    with pytest.raises((TypeError, ValueError)):
        _account(quantities=("0", "0"))


@pytest.mark.parametrize(
    "stochastic",
    [
        {"slippage_std": 0.01},
        {"tail_slippage_probability": 0.01},
    ],
)
def test_nonzero_stochastic_slippage_fails_closed(stochastic: dict[str, float]) -> None:
    account = _account(marks=(10.0,))
    with pytest.raises((TypeError, ValueError, NotImplementedError)):
        economics = FillEconomics(**stochastic)
        account.apply_fill(
            symbol_index=0,
            reported_quantity=1.0,
            execution_price=10.0,
            economics=economics,
            market_notional=100.0,
        )
