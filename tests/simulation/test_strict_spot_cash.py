"""Opt-in native fill-time spot cash boundary; legacy digests remain intact."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from tests.simulation.test_stateful_execution import _intent, _market, _zero_book
from trade_rl.evaluation.replay import run_shared_portfolio_replay
from trade_rl.risk import PreTradeRisk, PreTradeRiskConfig
from trade_rl.simulation import ExecutionCostConfig, MarketExecutor
from trade_rl.simulation.orders.model import (
    OrderBookState,
    OrderIntent,
    OrderType,
    TimeInForce,
)
from trade_rl.strategies.allocation.buy_and_hold import BuyAndHoldPortfolioPolicy


def _cost(**overrides):
    return replace(
        ExecutionCostConfig.zero(), allow_short=False, fee_rate=0.001, **overrides
    )


def test_same_mark_full_buy_never_borrows_fee_cash():
    data = _market()
    normal = MarketExecutor(data, _cost())
    strict = MarketExecutor(data, _cost(), strict_spot_cash=True)
    assert normal.execution_policy_digest == _cost().execution_policy_digest
    assert normal.execution_policy_digest != strict.execution_policy_digest
    old = normal.execute_orders(
        _zero_book(data),
        OrderBookState.empty(),
        (_intent(normal, 10.0),),
        start_index=0,
        bars=1,
    )
    assert old.book.cash == -1.0
    book = _zero_book(data)
    result = strict.execute_orders(
        book,
        OrderBookState.empty(),
        (_intent(strict, 10.0, time_in_force=TimeInForce.IOC),),
        start_index=0,
        bars=1,
    )
    assert result.book.cash == 1000.0 and result.book.quantities[0] == 0
    assert book.cash == 1000.0 and book.quantities[0] == 0
    assert result.interval_cost == 0 and result.fill_count == 0
    assert result.capacity_evidence[0].consumed_capacity_notional == 0
    assert any(
        e.event_type == "no_fill" and e.reason == "insufficient_spot_cash"
        for e in result.order_events
    )
    assert not result.order_book.active_orders


def test_actual_gap_and_source_fee_cannot_escape_native_budget():
    c = np.full((6, 1), 100.0)
    c[1:] = 101.0
    rate = np.zeros_like(c)
    rate[1] = 0.01
    data = _market(open=c.copy(), close=c, high=c + 1, low=c - 1, fee_rate=rate)
    executor = MarketExecutor(data, _cost(), strict_spot_cash=True)
    # 9.8 * 101 is affordable without the processing-row fee, but not with it.
    result = executor.execute_orders(
        _zero_book(data),
        OrderBookState.empty(),
        (_intent(executor, 9.8, time_in_force=TimeInForce.IOC),),
        start_index=0,
        bars=1,
    )
    assert result.book.cash == 1000 and result.fill_count == 0
    assert any(e.reason == "insufficient_spot_cash" for e in result.order_events)


def test_pending_order_can_fill_later_when_actual_price_is_affordable():
    c = np.full((6, 1), 100.0)
    c[1] = 101.0
    c[2:] = 98.0
    data = _market(open=c.copy(), close=c, high=c + 1, low=c - 1)
    ex = MarketExecutor(data, _cost(), strict_spot_cash=True)
    # At 101 the principal fits but the fee does not; at 98 it fits.
    first = ex.execute_orders(
        _zero_book(data),
        OrderBookState.empty(),
        (_intent(ex, 9.9),),
        start_index=0,
        bars=1,
    )
    assert first.fill_count == 0 and first.book.cash == 1000.0
    assert len(first.order_book.active_orders) == 1
    second = ex.execute_orders(first.book, first.order_book, (), start_index=1, bars=1)
    assert second.fill_count == 1
    assert second.book.cash == pytest.approx(1000 - 9.9 * 98 * 1.001)
    assert not second.order_book.active_orders
    assert first.book.cash == 1000.0


def test_two_same_bar_orders_cannot_each_spend_the_same_cash():
    data = _market()
    ex = MarketExecutor(data, _cost(), strict_spot_cash=True)
    # Reservation checks gross only, so both 5-unit orders may be eligible.
    orders = (
        _intent(ex, 5.0, target_identity="a"),
        _intent(ex, 5.0, target_identity="b"),
    )
    result = ex.execute_orders(
        _zero_book(data), OrderBookState.empty(), orders, start_index=0, bars=1
    )
    assert result.fill_count == 1 and result.book.quantities[0] == 5.0
    assert result.book.cash == 499.5
    assert result.capacity_evidence[0].consumed_capacity_notional == 500.0
    assert any(e.reason == "insufficient_spot_cash" for e in result.order_events)


def test_lot_fill_and_reduce_only_sell_keep_cash_nonnegative():
    data = _market()
    ex = MarketExecutor(data, _cost(lot_size=0.1), strict_spot_cash=True)
    r = ex.execute_orders(
        _zero_book(data),
        OrderBookState.empty(),
        (_intent(ex, 9.9),),
        start_index=0,
        bars=1,
    )
    assert r.book.cash == pytest.approx(9.01)
    intent = OrderIntent.create(
        dataset_id=data.dataset_id,
        target_identity="close",
        execution_policy_digest=ex.execution_policy_digest,
        symbol_index=0,
        requested_quantity=-9.9,
        order_type=OrderType.MARKET,
        limit_price=None,
        stop_price=None,
        expiry_index=None,
        time_in_force=TimeInForce.IOC,
        submit_index=1,
        eligible_index=2,
        submission_reference_price=100.0,
        decision_equity=r.book.portfolio_value,
        reduce_only=True,
    )
    closed = ex.execute_orders(r.book, r.order_book, (intent,), start_index=1, bars=1)
    assert closed.book.quantities[0] == 0.0
    assert closed.book.cash == pytest.approx(998.02)


def test_strict_spot_rejects_financing_and_unsafe_starting_book():
    data = _market()
    for cost in (
        ExecutionCostConfig.zero(),
        replace(_cost(), max_leverage=2.0),
        replace(_cost(), collateral_haircut=0.5),
    ):
        with pytest.raises(ValueError, match="strict spot"):
            MarketExecutor(data, cost, strict_spot_cash=True)
    ex = MarketExecutor(data, _cost(), strict_spot_cash=True)
    book = _zero_book(data)
    book.cash = -1.0
    with pytest.raises(ValueError, match="nonnegative"):
        ex.execute_orders(book, OrderBookState.empty(), (), start_index=0, bars=1)
    financed = _market(funding_rate=np.ones((6, 1)) * 0.001)
    bad = MarketExecutor(financed, _cost(), strict_spot_cash=True)
    with pytest.raises(ValueError, match="funding"):
        bad.execute_orders(
            _zero_book(financed), OrderBookState.empty(), (), start_index=0, bars=1
        )
    with pytest.raises(ValueError, match="stateful"):
        ex.liquidate_at_close(_zero_book(data), index=1)


def test_portfolio_replay_strict_mode_binds_ledger_and_prevents_overdraft():
    data = _market()
    risk = PreTradeRisk(
        PreTradeRiskConfig(
            max_gross=1.0,
            max_abs_weight=1.0,
            max_turnover=None,
            drawdown_start=1.0,
            drawdown_stop=1.0,
        )
    )
    result = run_shared_portfolio_replay(
        data,
        BuyAndHoldPortfolioPolicy(symbols=data.symbols, gross_budget=1.0),
        start_index=0,
        stop_index=4,
        initial_capital=1000.0,
        execution_cost=_cost(),
        risk=risk,
        strict_spot_cash=True,
        capture_ledger_evidence=True,
    )
    ledger = result.ledger_evidence.intervals
    assert (
        ledger.execution_policy_digest
        == MarketExecutor(data, _cost(), strict_spot_cash=True).execution_policy_digest
    )
    assert all(r.cash_after >= 0 and r.cash_before >= 0 for r in ledger.intervals)
    assert any(
        e.reason == "insufficient_spot_cash"
        for row in ledger.intervals
        for e in row.order_events
    )


@pytest.mark.parametrize("initial_slot,expected_buy", [(1, False), (0, True)])
def test_ordered_native_rotation_never_counts_a_future_sale(initial_slot, expected_buy):
    from tests.strategies.test_portfolio_policy_adapter import _dataset
    from trade_rl.simulation.accounting import BookState

    data = replace(_dataset(), volume=np.full((3, 3), 1e8))
    ex = MarketExecutor(data, _cost(), strict_spot_cash=True)
    w = np.zeros(3)
    w[initial_slot] = 0.6
    book = BookState.from_weights(weights=w, capital=1000.0, prices=data.close[0])
    buy_slot = 1 - initial_slot

    def order(slot, qty, reduce):
        return OrderIntent.create(
            dataset_id=data.dataset_id,
            target_identity=f"rotation-{slot}",
            execution_policy_digest=ex.execution_policy_digest,
            symbol_index=slot,
            requested_quantity=qty,
            order_type=OrderType.MARKET,
            time_in_force=TimeInForce.IOC,
            limit_price=None,
            stop_price=None,
            expiry_index=None,
            submit_index=0,
            eligible_index=1,
            submission_reference_price=100.0,
            decision_equity=1000.0,
            reduce_only=reduce,
        )

    r = ex.execute_orders(
        book,
        OrderBookState.empty(),
        (order(buy_slot, 4.0, False), order(initial_slot, -6.0, True)),
        start_index=0,
        bars=1,
    )
    assert r.book.cash >= 0
    assert r.book.quantities[initial_slot] == 0
    assert r.book.quantities[buy_slot] == (4.0 if expected_buy else 0.0)
    assert r.fill_count == (2 if expected_buy else 1)
    assert (any(e.reason == "insufficient_spot_cash" for e in r.order_events)) is (
        not expected_buy
    )
    assert book.cash == 400.0


def test_book_level_denial_preserves_exact_positions_cost_and_turnover():
    from trade_rl.simulation.accounting import SpotCashAdmissionError

    d = _market()
    book = _zero_book(d)
    before = book.clone()
    for qty, cost, reason in (
        (10.0, 1.0, "insufficient_spot_cash"),
        (-0.1, 0.0, "insufficient_spot_inventory"),
    ):
        with pytest.raises(SpotCashAdmissionError, match=reason):
            book.execute_fill(
                symbol_index=0,
                quantity=qty,
                fill_prices=d.close[0],
                cost_amount=cost,
                turnover=1.0,
                require_spot_cash=True,
            )
        assert book.exact_quantities == before.exact_quantities
        assert book.cash == before.cash
        assert book.total_cost == before.total_cost
        assert book.turnover_total == before.turnover_total
        assert book.fill_count == before.fill_count


def test_strict_replay_settles_through_the_same_native_cash_gate():
    data = _market()
    risk = PreTradeRisk(
        PreTradeRiskConfig(
            max_gross=1.0,
            max_abs_weight=1.0,
            max_turnover=None,
            drawdown_start=1.0,
            drawdown_stop=1.0,
        )
    )
    r = run_shared_portfolio_replay(
        data,
        BuyAndHoldPortfolioPolicy(symbols=data.symbols, gross_budget=0.8),
        start_index=0,
        stop_index=5,
        initial_capital=1000.0,
        execution_cost=_cost(),
        risk=risk,
        strict_spot_cash=True,
        capture_ledger_evidence=True,
        settle_terminal_position=True,
    )
    assert r.book.quantities[0] == 0
    assert r.book.cash == pytest.approx(998.4)
    assert r.book.fill_count == 2
    assert all(row.cash_after >= 0 for row in r.ledger_evidence.intervals.intervals)


def test_strict_cash_is_causal_and_legacy_order_digest_cannot_enter_strict_mode():
    data = _market()
    normal = MarketExecutor(data, _cost())
    strict = MarketExecutor(data, _cost(), strict_spot_cash=True)
    old_order = _intent(normal, 1.0, time_in_force=TimeInForce.IOC)
    wrong = strict.execute_orders(
        _zero_book(data), OrderBookState.empty(), (old_order,), start_index=0, bars=1
    )
    assert wrong.fill_count == 0
    assert any(
        e.event_type == "rejected" and e.reason == "identity_mismatch"
        for e in wrong.order_events
    )
    # A later funding row is not an input to the earlier execution.
    rates = np.zeros((6, 1))
    rates[3] = 0.01
    future = _market(funding_rate=rates)
    future_ex = MarketExecutor(future, _cost(), strict_spot_cash=True)
    first = future_ex.execute_orders(
        _zero_book(future),
        OrderBookState.empty(),
        (_intent(future_ex, 1.0),),
        start_index=0,
        bars=1,
    )
    assert first.book.cash == 899.9
    with pytest.raises(ValueError, match="funding"):
        future_ex.execute_orders(
            first.book, first.order_book, (), start_index=1, bars=2
        )
    assert first.book.cash == 899.9


@pytest.mark.parametrize("field", ["cash_rate", "dividend"])
def test_negative_nontrade_cash_flows_refuse_the_processed_row(field):
    value = np.zeros(6) if field == "cash_rate" else np.zeros((6, 1))
    value[1] = -0.01
    data = _market(**{field: value})
    ex = MarketExecutor(data, _cost(), strict_spot_cash=True)
    with pytest.raises(ValueError, match="negative non-trade"):
        ex.execute_orders(
            _zero_book(data), OrderBookState.empty(), (), start_index=0, bars=1
        )
