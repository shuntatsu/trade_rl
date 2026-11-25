"""Literal accounting oracles for the opt-in native fill observer."""

from __future__ import annotations

from dataclasses import asdict, replace
from fractions import Fraction

import numpy as np
import pytest

from tests.simulation.test_stateful_execution import _intent, _market, _zero_book
from trade_rl.artifacts.canonical import canonical_json_bytes
from trade_rl.evaluation.replay import run_shared_portfolio_replay
from trade_rl.risk import PreTradeRisk, PreTradeRiskConfig
from trade_rl.simulation import ExecutionCostConfig, MarketExecutor
from trade_rl.simulation.diagnostics.fill_cash import NativeFillCashObservation
from trade_rl.simulation.orders.model import OrderBookState, TimeInForce
from trade_rl.strategies.allocation.buy_and_hold import BuyAndHoldPortfolioPolicy


def _cost(**overrides):
    return replace(
        ExecutionCostConfig.zero(), allow_short=False, fee_rate=0.001, **overrides
    )


def test_exact_per_fill_cash_cost_order_and_legacy_event_bytes():
    market = _market()
    observed: list[NativeFillCashObservation] = []
    traced = MarketExecutor(market, _cost(), fill_cash_observer=observed.append)
    plain = MarketExecutor(market, _cost())
    intents = (
        _intent(traced, 2.0, target_identity="a"),
        _intent(traced, 3.0, target_identity="b"),
    )
    result = traced.execute_orders(
        _zero_book(market), OrderBookState.empty(), intents, start_index=0, bars=1
    )
    old = plain.execute_orders(
        _zero_book(market), OrderBookState.empty(), intents, start_index=0, bars=1
    )
    assert traced.execution_policy_digest == plain.execution_policy_digest
    assert canonical_json_bytes(result.order_events) == canonical_json_bytes(
        old.order_events
    )
    assert result.book.cash == old.book.cash
    assert len(observed) == 1
    batch = observed[0]
    assert batch.start_index == 0 and batch.next_index == 1
    assert batch.dataset_id == market.dataset_id
    assert batch.execution_policy_digest == traced.execution_policy_digest
    assert len(batch.fills) == result.fill_count == 2
    cash = Fraction(1000)
    qty = Fraction(0)
    costs = Fraction(0)
    # Equal priority and eligible index are ordered by canonical order ID,
    # not by the caller's input tuple order (LiquidityRequest.priority_key).
    ordered = sorted(intents, key=lambda intent: intent.order_id)
    assert tuple(row.event.order_id for row in batch.fills) == tuple(
        intent.order_id for intent in ordered
    )
    for row, intent in zip(batch.fills, ordered, strict=True):
        amount = int(intent.requested_quantity)
        assert row.cash_before == pytest.approx(float(cash))
        assert Fraction(row.quantity_before) == qty
        charge = Fraction(amount, 10)
        assert row.cost_amount == float(charge)
        assert row.signed_notional == amount * 100
        cash -= amount * 100 + charge
        qty += amount
        costs += charge
        assert Fraction(row.quantity_after) == qty
        assert row.cash_after == pytest.approx(float(cash))
        assert row.total_cost_after == pytest.approx(float(costs))
        assert row.event == result.order_events[row.event.sequence]
    assert batch.interval_cost == result.interval_cost == 0.5
    assert result.book.cash == pytest.approx(499.5)
    assert result.book.exact_quantities == (Fraction(5),)
    # The immutable observation is a separate schema, not an OrderEvent change.
    assert batch.schema_version == "native_fill_cash_observation_v1"
    assert "cash_before" not in asdict(batch.fills[0].event)


def test_rejected_strict_fill_emits_no_cash_transition():
    data = _market()
    observed = []
    executor = MarketExecutor(
        data, _cost(), strict_spot_cash=True, fill_cash_observer=observed.append
    )
    result = executor.execute_orders(
        _zero_book(data),
        OrderBookState.empty(),
        (_intent(executor, 10.0, time_in_force=TimeInForce.IOC),),
        start_index=0,
        bars=1,
    )
    assert len(observed) == 1
    assert observed[0].fills == ()
    assert observed[0].interval_cost == result.interval_cost == 0
    assert any(e.reason == "insufficient_spot_cash" for e in result.order_events)
    assert result.book.cash == 1000


def test_processing_fee_is_observed_and_exact_lot_quantity_is_preserved():
    rate = np.zeros((6, 1))
    rate[1] = 0.003
    data = _market(fee_rate=rate)
    observed = []
    executor = MarketExecutor(
        data, _cost(lot_size=0.1), fill_cash_observer=observed.append
    )
    result = executor.execute_orders(
        _zero_book(data),
        OrderBookState.empty(),
        (_intent(executor, 1.3),),
        start_index=0,
        bars=1,
    )
    row = observed[0].fills[0]
    assert row.quantity_after == "13/10"
    assert row.cost_amount == pytest.approx(130 * 0.004)
    assert row.cash_after == pytest.approx(1000 - 130 - 0.52)
    assert row.cash_after == result.book.cash


def test_common_replay_includes_terminal_sells_without_changing_ledger():
    data = _market()
    observed = []

    def run(observer=None):
        return run_shared_portfolio_replay(
            data,
            BuyAndHoldPortfolioPolicy(symbols=data.symbols, gross_budget=0.5),
            start_index=0,
            stop_index=4,
            initial_capital=1000,
            execution_cost=_cost(),
            risk=PreTradeRisk(
                PreTradeRiskConfig(max_gross=1, max_abs_weight=1, max_turnover=None)
            ),
            strict_spot_cash=True,
            settle_terminal_position=True,
            capture_ledger_evidence=True,
            fill_cash_observer=observer,
        )

    plain = run()
    traced = run(observed.append)
    assert canonical_json_bytes(plain.ledger_evidence) == canonical_json_bytes(
        traced.ledger_evidence
    )
    assert plain.book.cash == traced.book.cash
    assert traced.book.exact_quantities == (Fraction(0),)
    assert len(observed) == 4
    rows = [row for batch in observed for row in batch.fills]
    assert rows and rows[0].signed_notional > 0 and rows[-1].signed_notional < 0
    assert sum(row.cost_amount for row in rows) == pytest.approx(traced.book.total_cost)
    assert all(row.cash_before >= 0 and row.cash_after >= 0 for row in rows)


def test_observer_failure_returns_no_book_and_does_not_mutate_input():
    data = _market()

    def fail(_):
        raise RuntimeError("observer refused")

    executor = MarketExecutor(data, _cost(), fill_cash_observer=fail)
    book = _zero_book(data)
    with pytest.raises(RuntimeError, match="observer refused"):
        executor.execute_orders(
            book,
            OrderBookState.empty(),
            (_intent(executor, 1.0),),
            start_index=0,
            bars=1,
        )
    assert book.cash == 1000 and book.exact_quantities == (Fraction(0),)
    assert book.total_cost == 0


def test_observer_must_be_callable():
    with pytest.raises(TypeError, match="fill_cash_observer"):
        MarketExecutor(_market(), _cost(), fill_cash_observer=True)


def test_partial_fill_records_survive_later_native_invocation():
    from trade_rl.simulation.orders.model import OrderType

    volume = np.full((6, 1), 1000.0)
    volume[1:3, 0] = 8.0
    data = _market(volume=volume)
    observed = []
    cost = _cost(
        max_participation_rate=1.0,
        lot_size=1.0,
        processing_bar_volume_capacity=True,
        partial_fill_carry=True,
    )
    executor = MarketExecutor(data, cost, fill_cash_observer=observed.append)
    intent = _intent(executor, 3.0, order_type=OrderType.LIMIT, limit_price=99.0)
    first = executor.execute_orders(
        _zero_book(data), OrderBookState.empty(), (intent,), start_index=0, bars=1
    )
    second = executor.execute_orders(
        first.book, first.order_book, (), start_index=1, bars=1
    )
    a, b = (batch.fills[0] for batch in observed)
    assert (a.event.event_type, b.event.event_type) == ("partial_fill", "filled")
    assert a.event.order_id == b.event.order_id == intent.order_id
    assert (
        a.quantity_before,
        a.quantity_after,
        b.quantity_before,
        b.quantity_after,
    ) == ("0", "2", "2", "3")
    assert a.cost_amount == pytest.approx(0.198)
    assert b.cost_amount == pytest.approx(0.099)
    assert a.cash_after == b.cash_before
    assert b.cash_after == second.book.cash
    assert second.book.exact_quantities == (Fraction(3),)


def test_exact_sell_fees_do_not_precharge_later_large_sale():
    from trade_rl.simulation.accounting import BookState
    from trade_rl.simulation.orders.model import OrderIntent, OrderType

    shape = (6, 2)
    prices = np.full(shape, 100.0)
    data = _market(
        symbols=("SMALL", "LARGE"),
        features=np.zeros((6, 2, 1), dtype=np.float32),
        feature_available=np.ones((6, 2, 1), dtype=bool),
        open=prices,
        high=prices,
        low=prices,
        close=prices,
        volume=np.full(shape, 10000.0),
        funding_rate=np.zeros(shape),
        tradable=np.ones(shape, dtype=bool),
    )
    book = BookState.zero(2, 10100.0, prices[0])
    for symbol, qty in enumerate((1.0, 100.0)):
        book.execute_fill(
            symbol_index=symbol,
            quantity=qty,
            fill_prices=prices[0],
            cost_amount=0.0,
            turnover=0.0,
        )
    assert book.cash == 0.0
    seen = []
    executor = MarketExecutor(
        data,
        replace(_cost(), fee_rate=0.02),
        strict_spot_cash=True,
        fill_cash_observer=seen.append,
    )
    intents = tuple(
        OrderIntent.create(
            dataset_id=data.dataset_id,
            target_identity=f"sell-{symbol}",
            execution_policy_digest=executor.execution_policy_digest,
            symbol_index=symbol,
            requested_quantity=-qty,
            order_type=OrderType.MARKET,
            limit_price=None,
            stop_price=None,
            expiry_index=None,
            time_in_force=TimeInForce.IOC,
            submit_index=0,
            eligible_index=1,
            submission_reference_price=100.0,
            decision_equity=10100.0,
            reduce_only=True,
        )
        for symbol, qty in enumerate((1.0, 100.0))
    )
    result = executor.execute_orders(
        book, OrderBookState.empty(), intents, start_index=0, bars=1
    )
    a, b = seen[0].fills
    assert (a.event.symbol_index, b.event.symbol_index) == (0, 1)
    assert (a.cash_before, a.signed_notional, a.cost_amount, a.cash_after) == (
        0,
        -100,
        2,
        98,
    )
    assert (b.cash_before, b.signed_notional, b.cost_amount, b.cash_after) == (
        98,
        -10000,
        200,
        9898,
    )
    assert result.interval_cost == 202
    # The old conservative interval-fee envelope produces a false violation.
    assert a.cash_before - a.signed_notional - result.interval_cost == -102
    assert result.book.cash == 9898 and result.book.exact_quantities == (0, 0)


def test_non_order_cash_flow_is_outside_fill_observation():
    funding = np.zeros((6, 1))
    funding[1, 0] = 0.01
    data = _market(funding_rate=funding)
    seen = []
    executor = MarketExecutor(data, _cost(), fill_cash_observer=seen.append)
    result = executor.execute_orders(
        _zero_book(data),
        OrderBookState.empty(),
        (_intent(executor, 2.0),),
        start_index=0,
        bars=1,
    )
    row = seen[0].fills[0]
    assert row.cash_after == pytest.approx(799.8)
    assert result.interval_funding == pytest.approx(-2.0)
    assert result.book.cash == pytest.approx(row.cash_after - 2.0)
    assert result.interval_cost == pytest.approx(row.cost_amount)


def test_observer_not_called_on_failed_native_invocation(monkeypatch):
    from trade_rl.simulation.stateful.bar_lifecycle import StatefulBarLifecycle

    market = _market()
    seen = []
    executor = MarketExecutor(market, _cost(), fill_cash_observer=seen.append)
    book = _zero_book(market)

    def fail(*args, **kwargs):
        raise RuntimeError("later bar failure")

    monkeypatch.setattr(StatefulBarLifecycle, "finish_bar", fail)
    with pytest.raises(RuntimeError, match="later bar failure"):
        executor.execute_orders(
            book,
            OrderBookState.empty(),
            (_intent(executor, 2.0),),
            start_index=0,
            bars=1,
        )
    assert not seen
    assert book.cash == 1000.0 and book.total_cost == 0.0


def test_margin_flatten_is_not_misrepresented_as_the_order_fill():
    data = _market()
    seen = []
    # The fill debits 500 principal plus 1000 cost and exhausts equity;
    # legacy margin processing flattens the account immediately afterwards.
    executor = MarketExecutor(
        data, replace(_cost(), fee_rate=2.0), fill_cash_observer=seen.append
    )
    result = executor.execute_orders(
        _zero_book(data),
        OrderBookState.empty(),
        (_intent(executor, 5.0),),
        start_index=0,
        bars=1,
    )
    row = seen[0].fills[0]
    assert row.cash_before == 1000.0
    assert row.cash_after == -500.0
    assert row.quantity_after == "5"
    assert row.cost_amount == 1000.0
    assert result.book.insolvent
    assert result.book.cash == 0.0 and result.book.exact_quantities == (0,)


def test_tampered_observation_rejected():
    data = _market()
    seen = []
    executor = MarketExecutor(data, _cost(), fill_cash_observer=seen.append)
    executor.execute_orders(
        _zero_book(data),
        OrderBookState.empty(),
        (_intent(executor, 1.0),),
        start_index=0,
        bars=1,
    )
    batch = seen[0]
    row = batch.fills[0]
    with pytest.raises(ValueError, match="cash transition"):
        replace(row, cash_after=row.cash_after + 1)
    with pytest.raises(ValueError, match="quantity delta"):
        replace(row, quantity_after="2")
    with pytest.raises(ValueError, match="cost transition"):
        replace(row, total_cost_after=row.total_cost_after + 1)
    with pytest.raises(ValueError, match="interval cost"):
        replace(batch, interval_cost=0.0)
    with pytest.raises(ValueError, match="identity or order"):
        replace(batch, fills=(row, row), interval_cost=row.cost_amount * 2)


def test_contract_multiplier_uses_actual_native_cash_principal():
    data = _market(contract_multipliers=np.array([2.0]))
    seen = []
    executor = MarketExecutor(data, _cost(), fill_cash_observer=seen.append)
    result = executor.execute_orders(
        _zero_book(data),
        OrderBookState.empty(),
        (_intent(executor, 1.25),),
        start_index=0,
        bars=1,
    )
    row = seen[0].fills[0]
    assert row.quantity_after == "5/4"
    assert row.contract_multiplier == 2.0
    assert row.signed_notional == 250
    assert row.cost_amount == 0.25
    assert row.cash_after == result.book.cash == 749.75
