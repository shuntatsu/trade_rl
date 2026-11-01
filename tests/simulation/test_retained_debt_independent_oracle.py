"""Independent accounting counterexample for opt-in native retained debt.

The oracle deliberately does not call the production cost/equity helpers;
it uses Decimal for a fixed artificial short and terminal mark. It cannot
authenticate any saved R2/R3 dataset or real execution venue.
"""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
from fractions import Fraction

import pytest

from tests.simulation.test_execution_v2 import two_symbol_market
from tests.simulation.test_insolvency_valuation import cost, market, short_entry
from trade_rl.simulation.accounting import BookState
from trade_rl.simulation.execution import ExecutionCostConfig, MarketExecutor
from trade_rl.simulation.orders.model import (
    OrderBookState,
    OrderIntent,
    OrderType,
    TimeInForce,
)


def test_terminal_short_gap_debt_matches_independent_decimal_account() -> None:
    opening_capital = Decimal("1000")
    short_units = Decimal("5")
    entry_price = Decimal("100")
    terminal_price = Decimal("400")
    fee_rate = Decimal("0.002")

    independent_fill_notional = short_units * entry_price
    independent_fee = independent_fill_notional * fee_rate
    independent_cash_after_fill = opening_capital + independent_fill_notional
    independent_cash_after_fill -= independent_fee
    independent_terminal_equity = (
        independent_cash_after_fill - short_units * terminal_price
    )
    independent_return = (
        independent_terminal_equity - opening_capital
    ) / opening_capital

    assert independent_terminal_equity == Decimal("-501")
    assert independent_return == Decimal("-1.501")

    executor = MarketExecutor(market(), cost(), insolvency_valuation="retain_debt")
    entry = short_entry(executor)
    assert entry.book.exact_quantities == (Fraction(-5),)
    assert entry.book.cash == float(independent_cash_after_fill)
    assert entry.book.total_cost == float(independent_fee)

    terminal = executor.execute_orders(
        entry.book, entry.order_book, (), start_index=1, bars=2
    )
    assert terminal.bars_advanced == 1
    assert terminal.next_index == 2
    assert terminal.book.insolvent
    assert terminal.book.exact_quantities == (Fraction(0),)
    assert terminal.book.cash == float(independent_terminal_equity)
    assert terminal.book.portfolio_value == float(independent_terminal_equity)
    assert terminal.book.total_cost == float(independent_fee)
    assert terminal.book.fill_count == 1
    assert terminal.order_book.active_orders == ()
    assert terminal.interval_cost == 0.0
    assert terminal.interval_funding == 0.0
    assert terminal.interval_borrow_cost == 0.0
    expected_interval_return = (
        terminal.book.portfolio_value / entry.book.portfolio_value - 1.0
    )
    assert terminal.interval_net_return == pytest.approx(expected_interval_return)
    assert terminal.interval_net_return < -1.0
    assert terminal.book.max_drawdown == pytest.approx(1.501)
    assert terminal.book.returns_history[-1] == pytest.approx(expected_interval_return)
    realized_return = terminal.book.portfolio_value / float(opening_capital) - 1.0
    assert realized_return == pytest.approx(float(independent_return))


def test_default_floor_and_explicit_retained_debt_have_different_terminal_books() -> (
    None
):
    default = MarketExecutor(market(), cost())
    opted_in = MarketExecutor(market(), cost(), insolvency_valuation="retain_debt")
    assert default.execution_policy_digest != opted_in.execution_policy_digest
    entry_default = short_entry(default)
    entry_retained = short_entry(opted_in)

    floored = default.execute_orders(
        entry_default.book, entry_default.order_book, (), start_index=1, bars=2
    )
    retained = opted_in.execute_orders(
        entry_retained.book,
        entry_retained.order_book,
        (),
        start_index=1,
        bars=2,
    )
    assert floored.book.portfolio_value == 0.0
    assert floored.bars_advanced == 2
    assert retained.book.portfolio_value == -501.0
    assert retained.bars_advanced == 1


@pytest.mark.parametrize("reverse_submission", [False, True])
def test_retain_debt_stops_other_symbol_fills_with_same_bar_fee_bankruptcy(
    reverse_submission: bool,
) -> None:
    """Later symbol must never consume capacity or realize its planned fill."""
    data = two_symbol_market()
    equity = Decimal("1000")
    quantity = Decimal("5")
    price = Decimal("100")
    fee_rate = Decimal("3")
    independent_first_fee = quantity * price * fee_rate
    independent_terminal_equity = equity - independent_first_fee
    assert independent_first_fee == Decimal("1500")
    assert independent_terminal_equity == Decimal("-500")

    config = replace(
        ExecutionCostConfig.zero(),
        fee_rate=float(fee_rate),
        max_participation_rate=1.0,
    )
    retained = MarketExecutor(data, config, insolvency_valuation="retain_debt")
    floored = MarketExecutor(data, config)
    assert retained.execution_policy_digest != floored.execution_policy_digest

    def intents(executor: MarketExecutor) -> tuple[OrderIntent, ...]:
        result = tuple(
            OrderIntent.create(
                dataset_id=data.dataset_id,
                target_identity=f"independent-same-bar-fee-{symbol}",
                execution_policy_digest=executor.execution_policy_digest,
                symbol_index=symbol,
                requested_quantity=float(quantity),
                order_type=OrderType.MARKET,
                time_in_force=TimeInForce.GTC,
                limit_price=None,
                stop_price=None,
                submit_index=0,
                eligible_index=1,
                expiry_index=None,
                submission_reference_price=float(price),
                decision_equity=float(equity),
            )
            for symbol in range(2)
        )
        return tuple(reversed(result)) if reverse_submission else result

    retained_result = retained.execute_orders(
        BookState.zero(2, float(equity), data.close[0]),
        OrderBookState.empty(),
        intents(retained),
        start_index=0,
        bars=3,
    )
    assert retained_result.book.cash == float(independent_terminal_equity)
    assert retained_result.book.portfolio_value == float(independent_terminal_equity)
    assert retained_result.book.exact_quantities == (
        Fraction(0),
        Fraction(0),
    )
    assert retained_result.fill_count == 1
    assert retained_result.book.fill_count == 1
    assert retained_result.interval_cost == float(independent_first_fee)
    assert retained_result.book.total_cost == float(independent_first_fee)
    assert retained_result.filled_notional_by_symbol == pytest.approx([500.0, 0.0])
    assert retained_result.next_index == 1
    assert retained_result.bars_advanced == 1
    assert retained_result.order_book.active_orders == ()
    assert any(
        order.intent.symbol_index == 1
        and order.terminal_reason == "economic_termination"
        for order in retained_result.order_book.terminal_orders
    )
    assert retained_result.interval_cash_interest == 0.0
    assert retained_result.interval_funding == 0.0
    assert retained_result.interval_borrow_cost == 0.0

    floor_result = floored.execute_orders(
        BookState.zero(2, float(equity), data.close[0]),
        OrderBookState.empty(),
        intents(floored),
        start_index=0,
        bars=3,
    )
    assert floor_result.fill_count == 2
    assert floor_result.book.total_cost == float(2 * independent_first_fee)
    assert floor_result.book.portfolio_value == 0.0
    assert floor_result.next_index == 3
    assert floor_result.bars_advanced == 3
