from __future__ import annotations

from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest

from trade_rl.data.contracts import VolumeUnit
from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.allocation import execute_nonrl_proposal, propose_nonrl_target
from trade_rl.risk import PreTradeRisk, PreTradeRiskConfig
from trade_rl.simulation import BookState, EconomicTerminationReason, MarketExecutor
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.simulation.orders.model import (
    OrderBookState,
    OrderIntent,
    OrderType,
    PendingOrder,
    TimeInForce,
)
from trade_rl.simulation.stateful.runtime import StatefulExecutionRuntime
from trade_rl.strategies.allocation import AfterCostTargetAllocator, AllocationInputs


def market(*, next_close=100.0, volume=100_000.0, n_symbols=1, split=False):
    shape = (6, n_symbols)
    close = np.full(shape, 100.0)
    close[1:] = 50.0 if split else next_close
    opens = close.copy()
    opens[1] = 50.0 if split else 100.0
    splits = np.ones(shape)
    if split:
        splits[1] = 2.0
    return MarketDataset(
        dataset_id="d" * 64,
        symbols=tuple(f"S{i}" for i in range(n_symbols)),
        timestamps=np.datetime64("2026-01-01", "ns")
        + np.arange(6) * np.timedelta64(1, "h"),
        features=np.zeros((6, n_symbols, 1), dtype=np.float32),
        global_features=np.zeros((6, 1), dtype=np.float32),
        open=opens,
        high=np.maximum(opens, close),
        low=np.minimum(opens, close),
        close=close,
        mark_price=close,
        volume=np.full(shape, volume),
        volume_units=(VolumeUnit.BASE_ASSET,) * n_symbols,
        funding_rate=np.zeros(shape),
        tradable=np.ones(shape, dtype=np.bool_),
        feature_available=np.ones((6, n_symbols, 1), dtype=np.bool_),
        feature_names=("signal",),
        global_feature_names=("regime",),
        periods_per_year=8760,
        split_factor=splits,
    )


def executor(dataset, **cost_changes):
    return MarketExecutor(
        dataset,
        replace(ExecutionCostConfig.zero(), max_participation_rate=1.0, **cost_changes),
    )


def book(dataset, quantity=0.0):
    quantities = np.zeros(dataset.n_symbols)
    quantities[0] = quantity
    return BookState(
        quantities,
        1000.0 - quantity * 100.0,
        dataset.resolved_array("mark_price")[0],
        1000.0,
    )


def risk(cap=1.0):
    return PreTradeRisk(
        PreTradeRiskConfig(
            max_abs_weight=cap,
            max_turnover=None,
            drawdown_start=0.10,
            drawdown_stop=0.20,
        )
    )


def inputs(ex, *, index=0, signal=0.0, cost=0.01):
    time = ex.dataset.timestamps[index]
    return AllocationInputs(
        symbol="S0",
        decision_time=time,
        available_at=time,
        horizon_end=time + np.timedelta64(24, "h"),
        source_identity="a" * 64,
        expected_simple_return=signal,
        buy_cost=cost,
        sell_cost=cost,
    )


def pending(ex, quantity, *, symbol=0, reduce_only=False, order_type=OrderType.MARKET):
    return PendingOrder.from_intent(
        OrderIntent.create(
            dataset_id=ex.dataset.dataset_id,
            target_identity="prior-proposal",
            execution_policy_digest=ex.execution_policy_digest,
            symbol_index=symbol,
            requested_quantity=quantity,
            order_type=order_type,
            time_in_force=TimeInForce.GTC,
            limit_price=100.0 if order_type is OrderType.LIMIT else None,
            stop_price=None,
            submit_index=0,
            eligible_index=0,
            expiry_index=None,
            submission_reference_price=100.0,
            decision_equity=1000.0,
            reduce_only=reduce_only,
        )
    )


def propose(ex, bk, orders, *, signal=0.0, index=0, upper=1.0, controller=None):
    return propose_nonrl_target(
        ex,
        bk,
        orders,
        account_id="independent-S0",
        inputs=inputs(ex, index=index, signal=signal),
        allocator=AfterCostTargetAllocator(upper_weight=upper),
        pretrade_risk=risk() if controller is None else controller,
        symbol_index=0,
        start_index=index,
    )


def apply(
    ex, bk, orders, proposal, *, index=0, account="independent-S0", controller=None
):
    return execute_nonrl_proposal(
        ex,
        bk,
        orders,
        proposal,
        account_id=account,
        pretrade_risk=risk() if controller is None else controller,
        start_index=index,
    )


def test_fee_only_market_stays_cash_without_orders_or_costs():
    dataset = market()
    ex, bk, orders = (
        executor(dataset, fee_rate=0.002),
        book(dataset),
        OrderBookState.empty(),
    )
    result = apply(ex, bk, orders, propose(ex, bk, orders))
    assert result.execution.next_index == 1
    assert result.execution.book.cash == 1000.0
    assert result.execution.book.exact_quantities == (Fraction(0),)
    assert result.execution.book.total_cost == 0.0
    assert result.execution.order_events == ()


def test_useful_signal_executes_after_risk_with_independent_net_cash_oracle():
    dataset = market(next_close=110.0)
    ex, bk, orders = (
        executor(dataset, fee_rate=0.002),
        book(dataset),
        OrderBookState.empty(),
    )
    result = apply(ex, bk, orders, propose(ex, bk, orders, signal=0.10, upper=0.5))
    # Buy five at next open 100; fee 500*0.002=1; mark five at 110.
    assert result.execution.book.exact_quantities == (Fraction(5),)
    assert result.execution.book.cash == pytest.approx(499.0)
    assert result.execution.book.total_cost == pytest.approx(1.0)
    assert result.execution.book.portfolio_value == pytest.approx(1049.0)
    assert result.execution.book.fill_count == 1
    np.testing.assert_array_equal(result.risk_target.weights, [0.5])


def test_partial_fill_preserves_residual_then_changed_target_replaces_only_delta():
    dataset = market(volume=1.0)
    ex, bk, orders = executor(dataset), book(dataset, 2.0), OrderBookState.empty()
    original = pending(ex, 3.0)
    orders = orders.add(original)
    first = apply(ex, bk, orders, propose(ex, bk, orders, signal=0.10, upper=0.5))
    assert first.execution.book.exact_quantities == (Fraction(3),)
    assert first.execution.book.cash == 700.0
    residual = first.execution.order_book.active_orders[0]
    assert residual.order_id == original.order_id and residual.remaining_quantity == 2.0
    assert not any(
        event.event_type == "submitted" for event in first.execution.order_events
    )
    changed = propose(
        ex,
        first.execution.book,
        first.execution.order_book,
        signal=0.10,
        index=1,
        upper=0.4,
    )
    second = apply(
        ex, first.execution.book, first.execution.order_book, changed, index=1
    )
    submitted = [
        event
        for event in second.execution.order_events
        if event.event_type == "submitted"
    ]
    assert len(submitted) == 1 and submitted[0].requested_quantity == 1.0
    assert any(
        event.event_type == "cancelled" and event.order_id == original.order_id
        for event in second.execution.order_events
    )
    assert second.execution.book.exact_quantities == (Fraction(4),)
    assert second.execution.book.cash == 600.0
    assert second.execution.book.total_cost == 0.0


@pytest.mark.parametrize("split", [False, True])
@pytest.mark.parametrize("reduce_only", [False, True])
def test_exact_hold_cancels_all_market_remainders_without_fill_or_fee(
    split, reduce_only
):
    dataset = market(next_close=110.0, split=split)
    ex = executor(dataset, fee_rate=0.002)
    bk = book(dataset, float(Fraction(1, 3)))
    bk = replace(bk, _exact_quantities=("1/3",))
    old = pending(ex, -0.1 if reduce_only else 0.1, reduce_only=reduce_only)
    orders = OrderBookState.empty().add(old)
    result = apply(ex, bk, orders, propose(ex, bk, orders))
    assert result.execution.book.exact_quantities == (Fraction(2 if split else 1, 3),)
    assert result.execution.book.cash == bk.cash
    assert result.execution.book.total_cost == 0.0
    assert result.execution.book.fill_count == 0
    assert result.execution.order_book.active_orders == ()
    assert len(result.execution.order_events) == 1
    assert result.execution.order_events[0].event_type == "cancelled"
    assert result.execution.order_events[0].order_id == old.order_id


def test_hard_drawdown_risk_overrides_quantity_hold():
    dataset = market()
    ex, bk, orders = executor(dataset), book(dataset, 3.0), OrderBookState.empty()
    bk.max_drawdown = 0.20
    bk.peak_value = 1250.0
    result = apply(ex, bk, orders, propose(ex, bk, orders))
    np.testing.assert_array_equal(result.risk_target.weights, [0.0])
    assert result.execution.book.exact_quantities == (Fraction(0),)
    assert result.execution.book.cash == 1000.0
    assert result.execution.book.fill_count == 1


@pytest.mark.parametrize(
    "drift", ["holdings", "pending", "account", "risk", "policy", "dataset"]
)
def test_stale_proposals_reject_before_stateful_executor(monkeypatch, drift):
    dataset = market()
    ex, bk, orders = executor(dataset), book(dataset, 4.0), OrderBookState.empty()
    proposal = propose(ex, bk, orders, signal=0.10)
    account, controller = "independent-S0", risk()
    if drift == "holdings":
        bk.quantities[0], bk.cash = 5.0, 500.0
        assert bk.portfolio_value == 1000.0
    elif drift == "pending":
        orders = orders.add(pending(ex, 1.0))
    elif drift == "account":
        account = "another-account"
    elif drift == "risk":
        controller = risk(0.9)
    elif drift == "policy":
        ex = executor(dataset, fee_rate=0.002)
    elif drift == "dataset":
        ex = executor(replace(dataset, dataset_id="e" * 64))

    def forbidden_execution(*args, **kwargs):
        pytest.fail("stale proposal reached the stateful executor")

    monkeypatch.setattr(
        StatefulExecutionRuntime, "create", classmethod(forbidden_execution)
    )
    with pytest.raises(ValueError):
        apply(ex, bk, orders, proposal, account=account, controller=controller)


@pytest.mark.parametrize(
    "invalid",
    [
        "index",
        "terminal_index",
        "marks",
        "terminated",
        "other_holdings",
        "other_orders",
        "latency",
        "policy",
        "active_limit",
        "weak_risk",
    ],
)
def test_proposal_rejects_unsupported_account_or_execution_scope(invalid):
    dataset = market(n_symbols=2)
    ex, bk, orders, controller, index = (
        executor(dataset),
        book(dataset),
        OrderBookState.empty(),
        risk(),
        0,
    )
    if invalid == "index":
        index = -1
    elif invalid == "terminal_index":
        index = dataset.n_bars - 1
    elif invalid == "marks":
        bk.mark_prices[0] = 99.0
    elif invalid == "terminated":
        bk.termination_reason = EconomicTerminationReason.DRAWDOWN_STOP
    elif invalid == "other_holdings":
        bk.quantities[1] = 1.0
    elif invalid == "other_orders":
        orders = orders.add(pending(ex, 1.0, symbol=1))
    elif invalid == "latency":
        ex = executor(dataset, order_latency_bars=1)
    elif invalid == "policy":
        ex = executor(dataset, order_type="limit")
    elif invalid == "active_limit":
        orders = orders.add(pending(ex, 1.0, order_type=OrderType.LIMIT))
    elif invalid == "weak_risk":
        controller = PreTradeRisk.default_for_execution(max_leverage=1.0)
    with pytest.raises(ValueError):
        propose(ex, bk, orders, index=index, controller=controller)


@pytest.mark.parametrize("invalid", ["symbol", "decision", "late", "expired"])
def test_input_symbol_and_clock_must_match_the_exact_decision(invalid):
    dataset = market()
    ex, bk = executor(dataset), book(dataset)
    forecast = inputs(ex)
    changes = {
        "symbol": {"symbol": "S1"},
        "decision": {"decision_time": dataset.timestamps[1]},
        "late": {"available_at": dataset.timestamps[0] + np.timedelta64(1, "ns")},
        "expired": {"horizon_end": dataset.timestamps[0]},
    }
    with pytest.raises(ValueError):
        propose_nonrl_target(
            ex,
            bk,
            OrderBookState.empty(),
            account_id="independent-S0",
            inputs=replace(forecast, **changes[invalid]),
            allocator=AfterCostTargetAllocator(),
            pretrade_risk=risk(),
            symbol_index=0,
            start_index=0,
        )


@pytest.mark.parametrize("change", ["horizon", "source", "target", "terminal_archive"])
def test_decision_binding_rejects_tampering_even_if_optimal_weight_is_unchanged(
    monkeypatch, change
):
    dataset = market()
    ex, bk, orders = executor(dataset), book(dataset), OrderBookState.empty()
    proposal = propose(ex, bk, orders, signal=0.10)
    if change == "horizon":
        proposal = replace(
            proposal,
            inputs=replace(
                proposal.inputs,
                horizon_end=proposal.inputs.horizon_end + np.timedelta64(1, "h"),
            ),
        )
    elif change == "source":
        proposal = replace(
            proposal,
            inputs=replace(proposal.inputs, source_identity="different-recipe"),
        )
    elif change == "target":
        proposal = replace(proposal, target_weight=0.5)
    else:
        done = pending(ex, 1.0).cancel(processing_index=0, reason="prior_decision")
        orders = OrderBookState(active_orders=(), terminal_orders=(done,))

    def forbidden_execution(*args, **kwargs):
        pytest.fail("tampered decision reached the stateful executor")

    monkeypatch.setattr(
        StatefulExecutionRuntime, "create", classmethod(forbidden_execution)
    )
    with pytest.raises(ValueError):
        apply(ex, bk, orders, proposal)


def test_inconsistent_unrecorded_drawdown_rejects_before_risk_or_execution():
    dataset = market()
    ex, bk = executor(dataset), book(dataset)
    bk.peak_value = 1300.0
    assert bk.max_drawdown == 0.0
    with pytest.raises(ValueError, match="drawdown"):
        propose(ex, bk, OrderBookState.empty())


def test_caller_risk_tolerance_cannot_hide_unrecorded_drawdown():
    dataset = market()
    ex, bk = executor(dataset), book(dataset)
    bk.peak_value = 1300.0
    controller = PreTradeRisk(replace(risk().config, fail_closed_tolerance=0.25))
    with pytest.raises(ValueError, match="drawdown"):
        propose(ex, bk, OrderBookState.empty(), controller=controller, signal=0.1)


def test_zero_width_drawdown_guardrail_is_rejected_at_stop_boundary():
    dataset = market()
    ex, bk = executor(dataset), book(dataset, 3.0)
    bk.max_drawdown, bk.peak_value = 0.20, 1250.0
    controller = PreTradeRisk(
        replace(risk().config, drawdown_start=0.20, drawdown_stop=0.20)
    )
    with pytest.raises(ValueError, match="drawdown"):
        propose(ex, bk, OrderBookState.empty(), controller=controller)


@pytest.mark.parametrize("clock", ["latency", "expired", "future_processed"])
def test_active_order_clock_must_be_compatible_with_the_next_processing_bar(clock):
    dataset = market()
    ex, bk = executor(dataset), book(dataset)
    old = pending(ex, 5.0)
    if clock == "latency":
        intent = replace(old.intent, eligible_index=4)
        old = PendingOrder.from_intent(intent)
    elif clock == "expired":
        intent = replace(old.intent, expiry_index=0)
        old = PendingOrder.from_intent(intent)
    else:
        old = old.mark_latency_wait(processing_index=1)
    with pytest.raises(ValueError, match="clock"):
        propose(ex, bk, OrderBookState.empty().add(old), signal=0.10, upper=0.5)


def test_initial_margin_failure_is_rejected_by_canonical_margin_check():
    dataset = market()
    ex = executor(
        dataset, max_leverage=2.0, maintenance_margin_rate=0.9, collateral_haircut=0.1
    )
    bk = book(dataset, 8.0)
    assert bk.termination_reason is None
    canonical = bk.clone()
    ex._update_margin(canonical)
    assert canonical.termination_reason is EconomicTerminationReason.MARGIN_CALL
    with pytest.raises(ValueError, match="margin"):
        propose(ex, bk, OrderBookState.empty(), signal=-0.10)
    assert bk.termination_reason is None and bk.cash == 200.0


def test_return_horizon_cannot_end_before_first_processing_bar():
    dataset = market()
    ex, bk = executor(dataset), book(dataset)
    packet = replace(
        inputs(ex), horizon_end=dataset.timestamps[0] + np.timedelta64(1, "ns")
    )
    with pytest.raises(ValueError, match="horizon"):
        propose_nonrl_target(
            ex,
            bk,
            OrderBookState.empty(),
            account_id="a",
            inputs=packet,
            allocator=AfterCostTargetAllocator(),
            pretrade_risk=risk(),
            symbol_index=0,
            start_index=0,
        )


def test_future_terminal_transition_cannot_appear_in_a_past_decision():
    dataset = market()
    ex, bk = executor(dataset), book(dataset)
    future = pending(ex, 1).cancel(processing_index=4, reason="future")
    orders = OrderBookState(active_orders=(), terminal_orders=(future,))
    with pytest.raises(ValueError, match="clock"):
        propose(ex, bk, orders)


def test_quantity_hold_preserves_nonunit_contract_multiplier_account():
    dataset = replace(market(next_close=110.0), contract_multipliers=np.array([3.0]))
    ex = executor(dataset)
    bk = BookState(
        np.array([float(Fraction(1, 3))]),
        900.0,
        dataset.close[0],
        1000.0,
        contract_multipliers=np.array([3.0]),
        _exact_quantities=("1/3",),
    )
    result = apply(
        ex, bk, OrderBookState.empty(), propose(ex, bk, OrderBookState.empty())
    )
    assert result.execution.book.exact_quantities == (Fraction(1, 3),)
    assert result.execution.book.cash == 900.0
    assert result.execution.book.portfolio_value == pytest.approx(1010.0)
    assert result.execution.fill_count == 0
