from __future__ import annotations

import math
from dataclasses import replace
from decimal import Decimal
from fractions import Fraction
from typing import cast

import numpy as np
import pytest

from tools.market_account_oracle.account import ReferenceAccount
from tools.market_account_oracle.costs import (
    FillEconomics,
    calculate_fill,
)
from tools.market_account_oracle.numbers import (
    projection as reference_projection,
)
from tools.market_b4_oracle.binary64_account import (
    BorrowEvent,
    CashInterestEvent,
    DividendEvent,
    DrawdownObservation,
    ExecutionEvent,
    FundingMarkEvent,
    MarginEvent,
    OwnedBinary64Account,
    RevaluationEvent,
    SettlementEvent,
    TerminationEvent,
    UnsupportedTransition,
)
from trade_rl.simulation.accounting import BookState
from trade_rl.simulation.execution import ExecutionCostConfig, MarketExecutor


def _book(
    *,
    cash: float,
    quantities: tuple[Fraction, Fraction],
    marks: tuple[float, float],
    multipliers: tuple[float, float],
    peak: float,
) -> BookState:
    book = BookState(
        quantities=np.zeros(2, dtype=np.float64),
        cash=cash,
        mark_prices=np.asarray(marks, dtype=np.float64),
        peak_value=peak,
        contract_multipliers=np.asarray(multipliers, dtype=np.float64),
    )
    book.execute_quantity_deltas(
        fill_prices=np.asarray(marks, dtype=np.float64),
        quantity_deltas=quantities,
        cost_amount=0.0,
        turnover=0.0,
    )
    return book


def _assert_same_account(actual: OwnedBinary64Account, expected: BookState) -> None:
    observed = actual.snapshot()
    assert observed.cash.hex() == float(expected.cash).hex()
    assert observed.wealth.hex() == expected.portfolio_value.hex()
    assert observed.exact_quantities == expected.exact_quantities
    assert observed.peak_value.hex() == expected.peak_value.hex()
    assert observed.max_drawdown.hex() == expected.max_drawdown.hex()
    assert observed.total_cost.hex() == expected.total_cost.hex()
    assert observed.funding_pnl.hex() == expected.funding_pnl.hex()
    assert observed.borrow_cost.hex() == expected.borrow_cost.hex()
    assert observed.turnover_total.hex() == expected.turnover_total.hex()
    assert tuple(value.hex() for value in observed.returns_history) == tuple(
        value.hex() for value in expected.returns_history
    )
    assert observed.fill_count == expected.fill_count
    assert observed.rebalance_events == expected.rebalance_events
    assert observed.insolvent is expected.insolvent
    reason = (
        None
        if expected.termination_reason is None
        else expected.termination_reason.value
    )
    assert observed.termination_reason == reason
    for name in (
        "margin_used",
        "maintenance_margin",
        "maintenance_requirement",
        "margin_deficit",
    ):
        assert getattr(observed, name).hex() == float(getattr(expected, name)).hex()


def _reference(
    *,
    cash: str,
    quantities: tuple[Fraction, Fraction],
    marks: tuple[float, float],
    multipliers: tuple[float, float],
    peak: str,
) -> ReferenceAccount:
    return ReferenceAccount(
        cash=cash,
        quantities=tuple(str(value) for value in quantities),
        marks=marks,
        multipliers=multipliers,
        peak_value=peak,
    )


def test_one_ulp_wealth_is_owned_binary64_not_decimal_seeded_from_native() -> None:
    cash = 99987.65432110002
    quantities = (Fraction(1, 10), Fraction(0))
    marks = (123.456789, 1.0)
    multipliers = (1.0, 1.0)
    expected = BookState(
        quantities=np.array([0.1, 0.0]),
        cash=cash,
        mark_prices=np.array(marks),
        peak_value=100000.00000000003,
        contract_multipliers=np.array(multipliers),
    )
    owned = OwnedBinary64Account(
        cash=cash,
        exact_quantities=quantities,
        marks=marks,
        multipliers=multipliers,
        peak_value=100000.00000000003,
    )
    decimal = _reference(
        cash=str(cash),
        quantities=quantities,
        marks=marks,
        multipliers=multipliers,
        peak="100000.00000000003",
    ).snapshot()

    assert owned.wealth.hex() == expected.portfolio_value.hex()
    assert owned.wealth.hex() != float(decimal.wealth).hex()
    assert owned.exact_quantities == quantities


def test_two_symbol_float_reduction_and_signed_zero_match_public_book() -> None:
    marks = (1.0, 1.0)
    multipliers = (1.0, 1.0)
    cases = ((-0.0, -0.0), (-0.0, 0.0), (0.0, -0.0), (0.0, 0.0))
    for quantities in cases:
        expected = BookState(
            quantities=np.array(quantities),
            cash=-0.0,
            mark_prices=np.array(marks),
            peak_value=1.0,
            contract_multipliers=np.array(multipliers),
        )
        owned = OwnedBinary64Account(
            cash=-0.0,
            exact_quantities=(Fraction(0), Fraction(0)),
            marks=marks,
            multipliers=multipliers,
            peak_value=1.0,
        )

        native_positions = expected.position_values
        native_sum: float = float(np.sum(native_positions))
        direct_pair_sum: float = float(native_positions[0] + native_positions[1])
        assert native_positions[0].hex() == quantities[0].hex()
        assert native_positions[1].hex() == quantities[1].hex()
        assert native_sum.hex() == (0.0).hex()
        assert math.copysign(1.0, native_sum) == 1.0
        assert expected.portfolio_value.hex() == (0.0).hex()
        both_negative_zero = all(
            math.copysign(1.0, value) == -1.0 for value in quantities
        )
        if both_negative_zero:
            # A pairwise Python add preserves -0 here; the native full-array
            # reduction returns +0. The scalar prototype must retain that shape.
            assert direct_pair_sum.hex() == (-0.0).hex()
            assert math.copysign(1.0, direct_pair_sum) == -1.0
        else:
            assert direct_pair_sum.hex() == (0.0).hex()
        _assert_same_account(owned, expected)
        assert owned.wealth.hex() == expected.portfolio_value.hex()
        assert math.copysign(1.0, owned.cash) == -1.0
        assert math.copysign(1.0, owned.wealth) == 1.0


def test_execution_batch_uses_fsum_and_keeps_exact_large_quantities() -> None:
    q = Fraction(2**53 + 1)
    deltas = (q, -q)
    marks = (1.0, 1.0)
    multipliers = (1.0, 1.0)
    expected = BookState.zero(2, 100.0, np.array(marks), np.array(multipliers))
    expected.execute_quantity_deltas(
        fill_prices=np.array(marks),
        quantity_deltas=deltas,
        cost_amount=0.125,
        turnover=0.25,
    )
    owned = OwnedBinary64Account(
        cash=100.0,
        exact_quantities=(Fraction(0), Fraction(0)),
        marks=marks,
        multipliers=multipliers,
        peak_value=100.0,
    )
    owned.apply(
        ExecutionEvent(
            deltas=deltas,
            fill_prices=marks,
            cost_amount=0.125,
            turnover=0.25,
        )
    )

    assert owned.exact_quantities == deltas
    assert owned.projected_quantities == (float(2**53), -float(2**53))
    _assert_same_account(owned, expected)


def test_fee_fill_cash_and_two_step_carry_match_reference_and_native() -> None:
    marks = (100.0, 200.0)
    multipliers = (1.0, 2.0)
    quantities = (Fraction(0), Fraction(0))
    economics = FillEconomics(fee_rate="0.001", spread_rate="0.0002")
    accepted = (Fraction(3, 10), Fraction(-1, 5))
    reports = tuple(float(reference_projection(value)) for value in accepted)
    fill_inputs = (100.01, 199.99)
    calculations = tuple(
        calculate_fill(
            reported_quantity=reference_projection(quantity),
            price=Decimal(str(price)),
            contract_multiplier=Decimal(str(multiplier)),
            economics=economics,
            market_notional="1000000",
            quantity_capacity=None,
        )
        for quantity, price, multiplier in zip(
            accepted, fill_inputs, multipliers, strict=True
        )
    )
    total_cost = float(
        sum((item.cost for item in calculations), start=calculations[0].cost * 0)
    )

    reference = _reference(
        cash="100000",
        quantities=quantities,
        marks=marks,
        multipliers=multipliers,
        peak="100000",
    )
    for index, quantity in enumerate(accepted):
        reference.apply_fill(
            symbol_index=index,
            reported_quantity=reports[index],
            execution_price=fill_inputs[index],
            economics=economics,
            market_notional="1000000",
            lot_count=(3 if index == 0 else -4),
            lot_size=("0.1" if index == 0 else "0.05"),
        )

    expected = BookState.zero(2, 100000.0, np.array(marks), np.array(multipliers))
    expected.execute_quantity_deltas(
        fill_prices=np.array(fill_inputs),
        quantity_deltas=accepted,
        cost_amount=total_cost,
        turnover=0.5,
    )
    owned = OwnedBinary64Account(
        cash=100000.0,
        exact_quantities=quantities,
        marks=marks,
        multipliers=multipliers,
        peak_value=100000.0,
    )
    owned.apply(
        ExecutionEvent(
            deltas=accepted,
            fill_prices=fill_inputs,
            cost_amount=total_cost,
            turnover=0.5,
        )
    )
    _assert_same_account(owned, expected)
    assert abs(float(reference.cash) - owned.cash) <= 1e-8
    assert (
        abs(float(reference.snapshot().total_cost) - owned.snapshot().total_cost)
        <= 1e-8
    )

    interest = CashInterestEvent(annual_rate=0.03, year_fraction=0.125)
    expected.apply_cash_interest(0.03, year_fraction=0.125)
    reference.apply_cash_interest(annual_rate=0.03, year_fraction=0.125)
    owned.apply(interest)
    _assert_same_account(owned, expected)

    dividends = (0.02, -0.01)
    expected.apply_dividend(np.array(dividends))
    reference.apply_dividend(dividends)
    owned.apply(DividendEvent(per_unit=dividends))
    _assert_same_account(owned, expected)
    assert abs(float(reference.snapshot().dividend) - owned.snapshot().dividend) <= 1e-8

    borrow_rates = (0.0, 0.04)
    year_fraction = 0.125
    expected_borrow = float(
        np.sum(np.maximum(-expected.position_values, 0.0) * np.array(borrow_rates))
        * year_fraction
        * 1.5
    )
    expected.charge_borrow(expected_borrow)
    reference_borrow = reference.apply_borrow(
        annual_rates=borrow_rates,
        year_fraction=year_fraction,
        rate_multiplier=1.5,
    )
    owned.apply(
        BorrowEvent(
            annual_rates=borrow_rates,
            year_fraction=year_fraction,
            rate_multiplier=1.5,
        )
    )
    _assert_same_account(owned, expected)
    assert abs(float(reference_borrow) - expected_borrow) <= 1e-8

    funding_rates = (0.001, -0.002)
    due = (True, True)
    funding = -float(
        np.dot(
            expected.quantities * expected.contract_multipliers,
            np.array(funding_rates) * np.array(due, dtype=np.float64),
        )
    )
    new_marks = (101.0, 198.0)
    expected.mark_to_market(
        mark_prices=np.array(new_marks),
        funding_amount=funding,
        period_start_value=100000.0,
    )
    reference.apply_funding(price_rates=funding_rates, due=due)
    reference.revalue(new_marks)
    funding_result = owned.apply(
        FundingMarkEvent(
            price_rates=funding_rates,
            due=due,
            mark_prices=new_marks,
            period_start_value=100000.0,
        )
    )
    _assert_same_account(owned, expected)
    assert funding_result == funding
    assert funding_result != expected.returns_history[-1]
    assert (
        abs(float(reference.snapshot().funding_pnl) - owned.snapshot().funding_pnl)
        <= 1e-8
    )


def test_drawdown_nextafter_boundaries_match_native_two_symbol_book() -> None:
    quantities = (Fraction(100), Fraction(0))
    marks = (20.0, 1.0)
    multipliers = (1.0, 1.0)
    expected = BookState(
        quantities=np.array([100.0, 0.0]),
        cash=-1000.0,
        mark_prices=np.array(marks),
        peak_value=1000.0,
        contract_multipliers=np.array(multipliers),
    )
    reference = _reference(
        cash="-1000",
        quantities=quantities,
        marks=marks,
        multipliers=multipliers,
        peak="1000",
    )
    owned = OwnedBinary64Account(
        cash=-1000.0,
        exact_quantities=quantities,
        marks=marks,
        multipliers=multipliers,
        peak_value=1000.0,
    )
    prices = (
        math.nextafter(19.0, math.inf),
        19.0,
        math.nextafter(19.0, -math.inf),
        math.nextafter(18.0, math.inf),
        18.0,
        math.nextafter(18.0, -math.inf),
        17.0,
    )
    observed_drawdowns: dict[float, float] = {}
    for price in prices:
        next_marks = (price, 1.0)
        expected.revalue(np.array(next_marks))
        expected.refresh_drawdown()
        reference.revalue(next_marks)
        reference.observe()
        owned.apply(RevaluationEvent(mark_prices=next_marks))
        owned.apply(DrawdownObservation())
        observed_drawdowns[price] = owned.max_drawdown
        assert owned.max_drawdown.hex() == expected.max_drawdown.hex()
        assert float(reference.snapshot().max_native_drawdown) == pytest.approx(
            expected.max_drawdown, abs=1e-15
        )
        _assert_same_account(owned, expected)

    assert (
        observed_drawdowns[math.nextafter(19.0, math.inf)]
        < observed_drawdowns[19.0]
        < observed_drawdowns[math.nextafter(19.0, -math.inf)]
    )
    assert (
        observed_drawdowns[math.nextafter(18.0, math.inf)]
        < observed_drawdowns[18.0]
        < observed_drawdowns[math.nextafter(18.0, -math.inf)]
        < observed_drawdowns[17.0]
    )


def test_fill_exhaustion_and_borrow_insolvency_are_distinct() -> None:
    marks = (100.0, 1.0)
    multipliers = (1.0, 1.0)
    owned = OwnedBinary64Account(
        cash=100.0,
        exact_quantities=(Fraction(0), Fraction(0)),
        marks=marks,
        multipliers=multipliers,
        peak_value=100.0,
    )
    owned.apply(
        ExecutionEvent(
            deltas=(Fraction(1), Fraction(0)),
            fill_prices=marks,
            cost_amount=100.0,
            turnover=1.0,
        )
    )
    assert owned.insolvent
    assert owned.termination_reason == "execution_cost_exhaustion"
    assert owned.wealth == 0.0

    debt = OwnedBinary64Account(
        cash=100.0,
        exact_quantities=(Fraction(0), Fraction(0)),
        marks=marks,
        multipliers=multipliers,
        peak_value=100.0,
    )
    debt.apply(
        ExecutionEvent(
            deltas=(Fraction(-1), Fraction(0)),
            fill_prices=marks,
            cost_amount=0.0,
            turnover=1.0,
        )
    )
    debt.apply(
        BorrowEvent(
            annual_rates=(2.0, 0.0),
            year_fraction=1.0,
            rate_multiplier=1.0,
        )
    )
    assert debt.insolvent
    assert debt.termination_reason == "insolvency"
    debt.apply(DrawdownObservation())
    assert debt.max_drawdown == 1.0


def test_borrow_cash_overflow_rejects_before_mutating_account() -> None:
    owned = OwnedBinary64Account(
        cash=-1.5e308,
        exact_quantities=(Fraction("-1e307"), Fraction("1.7e308")),
        marks=(1.0, 1.0),
        multipliers=(1.0, 1.0),
        peak_value=1.0e307,
    )
    before = owned.snapshot()

    with pytest.raises(
        ValueError, match="borrow.*finite|account values|portfolio value"
    ):
        owned.apply(
            BorrowEvent(
                annual_rates=(4.0, 0.0),
                year_fraction=1.0,
                rate_multiplier=1.0,
            )
        )

    assert owned.snapshot() == before


def test_funding_cash_overflow_rejects_before_mutating_account() -> None:
    owned = OwnedBinary64Account(
        cash=1.6e308,
        exact_quantities=(Fraction("-1.5e308"), Fraction(0)),
        marks=(1.0, 1.0),
        multipliers=(1.0, 1.0),
        peak_value=1.0e308,
    )
    before = owned.snapshot()

    with pytest.raises(ValueError, match="funding|cash|portfolio value"):
        owned.apply(
            FundingMarkEvent(
                price_rates=(1.0, 0.0),
                due=(True, False),
                mark_prices=(1.0, 1.0),
                period_start_value=1.0e308,
            )
        )

    assert owned.snapshot() == before


def test_signed_large_position_mark_overflow_rejects_without_changing_snapshot() -> (
    None
):
    owned = OwnedBinary64Account(
        cash=1000.0,
        exact_quantities=(Fraction("1.2e308"), Fraction("-1.2e308")),
        marks=(1.0, 1.0),
        multipliers=(1.0, 1.0),
        peak_value=1000.0,
    )
    before = owned.snapshot()

    with pytest.raises(ValueError, match="position values|portfolio value"):
        owned.apply(RevaluationEvent(mark_prices=(2.0, 1.0)))

    assert owned.snapshot() == before


def test_signed_large_position_gross_overflow_rejects_atomically() -> None:
    owned = OwnedBinary64Account(
        cash=1000.0,
        exact_quantities=(Fraction("1.2e308"), Fraction("-1.2e308")),
        marks=(1.0, 1.0),
        multipliers=(1.0, 1.0),
        peak_value=1000.0,
    )
    before = owned.snapshot()

    with pytest.raises(ValueError, match="margin calculation"):
        owned.apply(
            MarginEvent(
                max_leverage=2.0,
                maintenance_margin_rate=0.25,
                collateral_haircut=1.0,
            )
        )

    assert owned.snapshot() == before


def test_funding_return_ratio_overflow_rejects_without_account_mutation() -> None:
    # The start and ending wealth are individually finite binary64 values,
    # but the ratio itself overflows: a reporting failure must be atomic.
    account = OwnedBinary64Account(
        cash=1e308,
        exact_quantities=(Fraction(0), Fraction(0)),
        marks=(1.0, 1.0),
        multipliers=(1.0, 1.0),
        peak_value=1e308,
    )
    before = account.snapshot()
    assert math.isfinite(before.wealth)
    with pytest.raises(
        ValueError, match="funding mark must preserve finite account values"
    ):
        account.apply(
            FundingMarkEvent(
                price_rates=(0.0, 0.0),
                due=(False, False),
                mark_prices=(1.0, 1.0),
                period_start_value=1e-308,
            )
        )
    assert account.snapshot() == before


def test_cumulative_funding_pnl_overflow_rejects_without_mutation() -> None:
    # Each funding transfer is finite; borrowing between the transfers keeps
    # cash/wealth finite but their running funding-PnL sum would overflow.
    account = OwnedBinary64Account(
        cash=-1e308,
        exact_quantities=(Fraction(13 * 10**307), Fraction(-(10**307))),
        marks=(1.0, 1.0),
        multipliers=(1.0, 1.0),
        peak_value=1e308,
    )
    assert math.isfinite(account.wealth) and account.wealth > 0.0
    account.apply(
        FundingMarkEvent(
            price_rates=(0.0, 9.0),
            due=(False, True),
            mark_prices=(1.0, 1.0),
            period_start_value=account.wealth,
        )
    )
    account.apply(BorrowEvent(annual_rates=(0.0, 4.0), year_fraction=1.0))
    before = account.snapshot()
    assert math.isfinite(before.cash) and math.isfinite(before.wealth)
    assert before.funding_pnl == 9e307
    with pytest.raises(
        ValueError, match="funding mark must preserve finite account values"
    ):
        account.apply(
            FundingMarkEvent(
                price_rates=(0.0, 9.0),
                due=(False, True),
                mark_prices=(1.0, 1.0),
                period_start_value=before.wealth,
            )
        )
    assert account.snapshot() == before


def test_settlement_uses_exact_products_then_fsum_and_closes_only_masked_slots() -> (
    None
):
    quantities = (Fraction(23, 10), Fraction(-2, 5))
    marks = (3.0, 5.0)
    multipliers = (2.0, 4.0)
    expected = BookState.zero(2, 100.0, np.array(marks), np.array(multipliers))
    owned = OwnedBinary64Account(
        cash=100.0,
        exact_quantities=(Fraction(0), Fraction(0)),
        marks=marks,
        multipliers=multipliers,
        peak_value=100.0,
    )
    reference = _reference(
        cash="100",
        quantities=(Fraction(0), Fraction(0)),
        marks=marks,
        multipliers=multipliers,
        peak="100",
    )
    expected.execute_quantity_deltas(
        fill_prices=np.array(marks),
        quantity_deltas=quantities,
        cost_amount=0.0,
        turnover=0.0,
    )
    owned.apply(
        ExecutionEvent(
            deltas=quantities,
            fill_prices=marks,
            cost_amount=0.0,
            turnover=0.0,
        )
    )
    economics = FillEconomics()
    for index, quantity in enumerate(quantities):
        reference.apply_fill(
            symbol_index=index,
            reported_quantity=float(reference_projection(quantity)),
            execution_price=marks[index],
            economics=economics,
            market_notional="1000000",
            lot_count=(23 if index == 0 else -8),
            lot_size=("0.1" if index == 0 else "0.05"),
        )

    event = SettlementEvent(
        mask=(True, False),
        prices=(2.5, 4.0),
        recovery=(0.7, 1.0),
    )
    proceeds = float(
        quantities[0] * Fraction(str(2.5)) * Fraction(str(2.0)) * Fraction(str(0.7))
    )
    expected.settle_positions(
        mask=np.array([True, False]),
        prices=np.array([2.5, 4.0]),
        recovery=np.array([0.7, 1.0]),
    )
    reference.settle(mask=(True, False), prices=(2.5, 4.0), recovery=(0.7, 1.0))
    actual = owned.apply(event)
    assert actual == proceeds
    _assert_same_account(owned, expected)
    assert owned.exact_quantities == (Fraction(0), quantities[1])
    assert abs(float(reference.snapshot().settlement_proceeds) - actual) < 1e-12


def test_unknown_or_unsupported_transitions_fail_closed() -> None:
    owned = OwnedBinary64Account(
        cash=10.0,
        exact_quantities=(Fraction(0), Fraction(0)),
        marks=(1.0, 1.0),
        multipliers=(1.0, 1.0),
        peak_value=10.0,
    )
    with pytest.raises(UnsupportedTransition):
        owned.apply(object())
    with pytest.raises(ValueError):
        owned.apply(
            ExecutionEvent(
                deltas=cast(tuple[Fraction, Fraction], (Fraction(1),)),
                fill_prices=cast(tuple[float, float], (1.0,)),
                cost_amount=0.0,
                turnover=0.0,
            )
        )


@pytest.mark.parametrize("valuation", ["floor_zero", "retain_debt"])
@pytest.mark.parametrize(
    ("cash", "quantities", "haircut", "prior_termination", "margin_call"),
    [
        (1000.0, (1, 0), 1.0, False, False),
        (-75.0, (1, 0), 1.0, False, False),
        (-80.0, (1, 0), 1.0, False, True),
        (-75.0, (1, 0), 0.95, False, True),
        (125.0, (-1, 0), 1.0, False, False),
        (50.0, (1, -1), 0.5, False, True),
        (50.0, (1, -1), 1.0, False, False),
        (1000.0, (1, 0), 1.0, True, False),
        (250.0, (-3, 0), 1.0, False, True),
        (math.nextafter(-75.000000000001, -math.inf), (1, 0), 1.0, False, True),
        (math.nextafter(-75.000000000001, math.inf), (1, 0), 1.0, False, False),
        (1e20, (2**53 + 1, 0), 1.0, False, False),
        (1e20, (0, -(2**53 + 1)), 1.0, False, False),
    ],
)
def test_cross_margin_and_termination_flatten_match_native_from_opening_facts(
    valuation, cash, quantities, haircut, prior_termination, margin_call
):
    from tests.evaluation.test_after_cost_allocation import market

    exact = tuple(Fraction(value) for value in quantities)
    marks = (100.0, 100.0)
    peak = 1e21
    native = BookState(
        quantities=np.asarray([reference_projection(value) for value in exact]),
        cash=cash,
        mark_prices=np.asarray(marks),
        contract_multipliers=np.ones(2),
        peak_value=peak,
        _exact_quantities=tuple(str(value) for value in exact),
    )
    owned = OwnedBinary64Account(
        cash=cash,
        exact_quantities=exact,
        marks=marks,
        multipliers=(1.0, 1.0),
        peak_value=peak,
    )
    if prior_termination:
        native.terminate("drawdown_stop")
        owned.apply(TerminationEvent("drawdown_stop"))
    executor = MarketExecutor(
        market(n_symbols=2),
        replace(
            ExecutionCostConfig.zero(),
            max_leverage=2.0,
            maintenance_margin_rate=0.25,
            collateral_haircut=haircut,
        ),
        insolvency_valuation=valuation,
    )
    executor._update_margin(native)
    assert (native.termination_reason == "margin_call") is margin_call
    owned.apply(
        MarginEvent(
            max_leverage=2.0,
            maintenance_margin_rate=0.25,
            collateral_haircut=haircut,
            insolvency_valuation=valuation,
        )
    )
    _assert_same_account(owned, native)
    if margin_call or prior_termination:
        assert owned.exact_quantities == (Fraction(0), Fraction(0))
        assert owned.snapshot().margin_used == 0.0


@pytest.mark.parametrize(
    "changes",
    [
        {"max_leverage": 0.0},
        {"maintenance_margin_rate": -0.1},
        {"maintenance_margin_rate": 1.1},
        {"collateral_haircut": float("nan")},
        {"collateral_haircut": 1.1},
        {"insolvency_valuation": "unknown"},
        {"margin_mode": "isolated"},
    ],
)
def test_margin_configuration_failure_preserves_owned_account(changes):
    owned = OwnedBinary64Account(
        cash=10.0,
        exact_quantities=(Fraction(0), Fraction(0)),
        marks=(1.0, 1.0),
        multipliers=(1.0, 1.0),
    )
    before = owned.snapshot()
    fields = {
        "max_leverage": 2.0,
        "maintenance_margin_rate": 0.25,
        "collateral_haircut": 1.0,
    }
    with pytest.raises(ValueError):
        owned.apply(MarginEvent(**(fields | changes)))
    assert owned.snapshot() == before


def test_margin_deficit_is_refreshed_after_collateral_recovers():
    from tests.evaluation.test_after_cost_allocation import market

    cash = -75.0000000000007
    native = BookState(
        quantities=np.asarray([1.0, 0.0]),
        cash=cash,
        mark_prices=np.asarray([100.0, 100.0]),
        peak_value=100.0,
    )
    owned = OwnedBinary64Account(
        cash=cash,
        exact_quantities=(Fraction(1), Fraction(0)),
        marks=(100.0, 100.0),
        multipliers=(1.0, 1.0),
        peak_value=100.0,
    )
    executor = MarketExecutor(
        market(n_symbols=2),
        replace(ExecutionCostConfig.zero(), maintenance_margin_rate=0.25),
    )
    event = MarginEvent(1.0, 0.25, 1.0)
    executor._update_margin(native)
    owned.apply(event)
    _assert_same_account(owned, native)
    assert 0.0 < owned.snapshot().margin_deficit < 1e-12
    assert not owned.insolvent

    native.apply_dividend(np.asarray([0.01, 0.0]))
    owned.apply(DividendEvent((0.01, 0.0)))
    executor._update_margin(native)
    owned.apply(event)
    _assert_same_account(owned, native)
    assert owned.snapshot().margin_deficit == 0.0
