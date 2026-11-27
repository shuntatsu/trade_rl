"""Literal synthetic oracles for a fine interval without a new actor intent."""

from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest

from tests.evaluation.test_shared_portfolio_replay import _market
from trade_rl.risk.pretrade import PreTradeRisk, PreTradeRiskConfig
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.simulation.stateful.portfolio_session import SharedPortfolioSession


def _session(
    *, close=None, volume=None, latency=3, partial=True, risk=None, **cost_changes
):
    if close is None:
        close = np.tile([100.0, 200.0], (12, 1))
    cost = replace(
        ExecutionCostConfig.zero(),
        **(
            {
                "order_latency_bars": latency,
                "partial_fill_carry": partial,
                "processing_bar_volume_capacity": False,
            }
            | cost_changes
        ),
    )
    events = []
    session = SharedPortfolioSession(
        _market(close=close, volume=volume),
        start_index=0,
        stop_index=len(close) - 1,
        initial_capital=1000.0,
        execution_cost=cost,
        risk=risk,
        execution_trace_observer=lambda observation: events.extend(
            observation.order_events
        ),
    )
    return session, events


def _risk(max_gross=1.0, max_abs_weight=1.0, max_turnover=None):
    return PreTradeRisk(
        PreTradeRiskConfig(
            max_gross=max_gross,
            max_abs_weight=max_abs_weight,
            max_turnover=max_turnover,
            drawdown_start=1.0,
            drawdown_stop=1.0,
        )
    )


@pytest.mark.parametrize("sign", [1, -1])
def test_continuation_keeps_original_latency_and_partial_fill_order(sign):
    volume = np.full((12, 2), 1.0)
    session, events = _session(volume=volume)
    session.advance((sign * 0.5, 0.0))
    (order,) = session.active_orders
    original_id = order.order_id
    assert order.intent.submit_index == 0
    assert order.intent.eligible_index == 3
    assert session.book.exact_quantities == (0, 0)

    session.advance_without_new_intent()
    assert session.current_index == 2
    assert session.active_orders[0].order_id == original_id
    assert session.book.exact_quantities == (0, 0)
    for quantity in range(1, 6):
        risk_target, execution, holds, status = session.advance_without_new_intent()
        assert status == "continued"
        assert execution.next_index == quantity + 2
        assert session.book.exact_quantities == (sign * quantity, 0)
        assert session.book.cash == pytest.approx(1000.0 - sign * 100.0 * quantity)
        assert session.book.portfolio_value == pytest.approx(1000.0)
        assert risk_target.weights == pytest.approx((sign * 0.5, 0.0))
        assert holds == ()
        if quantity < 5:
            (pending,) = session.active_orders
            assert pending.order_id == original_id
            assert pending.remaining_quantity == sign * (5 - quantity)
        else:
            assert session.active_orders == ()

    assert {event.order_id for event in events} == {original_id}
    assert [
        event.processing_index for event in events if event.event_type == "submitted"
    ] == [0]
    assert [
        event.processing_index
        for event in events
        if event.event_type in {"filled", "partial_fill"}
    ] == [3, 4, 5, 6, 7]
    assert not any(event.event_type == "cancelled" for event in events)
    session.advance_without_new_intent()
    assert session.book.exact_quantities == (sign * 5, 0)


def test_risk_cancels_pending_entry_when_actual_mark_changes_before_eligibility():
    close = np.tile([100.0, 200.0], (12, 1))
    close[1:, 0] = 150.0
    session, events = _session(close=close, risk=_risk(max_abs_weight=0.5))
    session.advance((0.5, 0.0))
    assert len(session.active_orders) == 1
    order_id = session.active_orders[0].order_id

    session.advance_without_new_intent()
    assert session.current_index == 2
    assert session.active_orders == ()
    assert session.book.exact_quantities == (0, 0)
    assert session.book.cash == 1000.0
    assert [
        (e.order_id, e.processing_index) for e in events if e.event_type == "cancelled"
    ] == [(order_id, 1)]
    session.advance_without_new_intent()
    assert not any(e.event_type in {"filled", "partial_fill"} for e in events)


def test_mixed_partial_fill_exposure_cancels_otherwise_safe_endpoint():
    session, events = _session(
        latency=2, risk=_risk(max_gross=0.75, max_abs_weight=0.75)
    )
    session.advance((0.5, 0.0))
    session.advance_without_new_intent()
    assert session.book.exact_quantities == (5, 0)
    # Full replacement has gross .5, but independent fills could hold both
    # old and new exposure at once, gross 1.0 > .75.
    session.advance((0.0, 0.5))
    assert len(session.active_orders) == 2
    ids = {order.order_id for order in session.active_orders}
    session.advance_without_new_intent()
    assert session.active_orders == ()
    assert session.book.exact_quantities == (5, 0)
    assert {e.order_id for e in events if e.event_type == "cancelled"} == ids


def test_hold_still_cancels_and_new_target_still_supersedes():
    session, events = _session()
    session.advance((0.5, 0.0))
    old = session.active_orders[0].order_id
    session.advance((0.0, 0.0), hold_current_quantity_symbols=(0, 1))
    assert session.active_orders == ()
    assert any(e.order_id == old and e.event_type == "cancelled" for e in events)
    session.advance((0.3, 0.0))
    second = session.active_orders[0].order_id
    session.advance((0.4, 0.0))
    (third,) = session.active_orders
    assert third.order_id != second
    session.advance_without_new_intent()
    assert session.active_orders[0].order_id == third.order_id


def test_continuation_with_no_orders_keeps_exact_hold_and_hard_risk():
    close = np.tile([100.0, 200.0], (12, 1))
    close[1:, 0] = 200.0
    session, events = _session(
        latency=0, close=close, risk=_risk(max_gross=0.6, max_abs_weight=0.6)
    )
    session.advance((0.5, 0.0))
    assert session.book.exact_quantities == (5, 0)
    assert session.book.weights[0] > 0.6
    risk, execution, holds, status = session.advance_without_new_intent()
    assert status == "hold"
    assert risk.weights[0] == pytest.approx(0.6)
    assert execution.book.weights[0] == pytest.approx(0.6)
    assert execution.book.exact_quantities == (Fraction(9, 2), 0)
    assert 0 not in holds
    assert any(e.event_type == "filled" for e in events)


@pytest.mark.parametrize(
    "changes",
    [
        {"order_type": "limit"},
        {"processing_bar_volume_capacity": True},
        {"max_leverage": 2.0},
        {"collateral_haircut": 0.9},
        {"slippage_std": 0.01},
    ],
)
def test_unsupported_continuation_refused_before_mutation(changes):
    session, events = _session(**changes)
    with pytest.raises(ValueError, match="pending continuation"):
        session.advance_without_new_intent()
    assert not events
    assert session.current_index == 0
    assert not session.poisoned


def test_continuation_cannot_enter_terminal_reserve():
    cost = replace(
        ExecutionCostConfig.zero(),
        order_latency_bars=2,
        processing_bar_volume_capacity=False,
    )
    session = SharedPortfolioSession(
        _market(close=np.tile([100.0, 200.0], (8, 1))),
        start_index=0,
        stop_index=7,
        execution_cost=cost,
        settle_terminal_position=True,
    )
    while session.can_decide:
        session.advance_without_new_intent()
    assert session.can_settle_terminal
    with pytest.raises(RuntimeError, match="not accepting"):
        session.advance_without_new_intent()
    while session.can_settle_terminal:
        session.settle_next_interval()
    assert session.finished


def test_repeated_no_fill_does_not_require_new_order_or_transition_version():
    volume = np.full((12, 2), 100.0)
    volume[0:3, 0] = 0.0
    session, events = _session(volume=volume, latency=0)
    session.advance((0.5, 0.0))
    (pending,) = session.active_orders
    for _ in range(2):
        session.advance_without_new_intent()
        assert session.active_orders[0].order_id == pending.order_id
        assert session.book.exact_quantities == (0, 0)
    session.advance_without_new_intent()
    assert session.active_orders == ()
    assert session.book.exact_quantities == (5, 0)
    assert [
        event.processing_index for event in events if event.event_type == "no_fill"
    ] == [1, 2, 3]
    assert [
        event.processing_index for event in events if event.event_type == "filled"
    ] == [4]


def test_turnover_drift_refuses_pending_order_without_repricing():
    close = np.tile([100.0, 200.0], (12, 1))
    close[1:, 0] = 150.0
    session, events = _session(close=close, risk=_risk(max_turnover=0.5))
    session.advance((0.5, 0.0))
    order_id = session.active_orders[0].order_id
    session.advance_without_new_intent()
    assert session.active_orders == ()
    assert session.book.exact_quantities == (0, 0)
    assert [
        (e.order_id, e.processing_index) for e in events if e.event_type == "cancelled"
    ] == [(order_id, 1)]


def test_drawdown_stop_cancels_pending_entry_and_reduces_actual_holding():
    close = np.tile([100.0, 200.0], (12, 1))
    close[1:, 0] = 50.0
    volume = np.full((12, 2), 1.0)
    risk = PreTradeRisk(
        PreTradeRiskConfig(
            max_gross=1.0,
            max_abs_weight=1.0,
            max_turnover=None,
            drawdown_start=0.01,
            drawdown_stop=0.04,
        )
    )
    session, events = _session(close=close, volume=volume, latency=0, risk=risk)
    session.advance((0.5, 0.0))
    assert session.book.exact_quantities == (1, 0)
    (pending,) = session.active_orders
    assert session.book.max_drawdown == pytest.approx(0.05)
    target, _, holds, status = session.advance_without_new_intent()
    assert status == "risk_fallback"
    assert target.risk_scale == 0.0
    assert target.weights == pytest.approx((0.0, 0.0))
    assert 0 not in holds
    assert session.book.exact_quantities == (0, 0)
    assert session.book.cash == pytest.approx(950.0)
    assert session.active_orders == ()
    assert any(
        e.order_id == pending.order_id and e.event_type == "cancelled" for e in events
    )


def test_continuation_observer_failure_poisoned_not_retryable():
    calls = []

    def observer(observation):
        calls.append(observation.next_index)
        if observation.next_index == 2:
            raise RuntimeError("observer stopped")

    session = SharedPortfolioSession(
        _market(close=np.tile([100.0, 200.0], (12, 1))),
        start_index=0,
        stop_index=11,
        execution_cost=replace(
            ExecutionCostConfig.zero(),
            order_latency_bars=3,
            processing_bar_volume_capacity=False,
        ),
        execution_trace_observer=observer,
    )
    session.advance((0.5, 0.0))
    with pytest.raises(RuntimeError, match="observer stopped"):
        session.advance_without_new_intent()
    assert calls == [1, 2]
    assert session.poisoned
    assert session.current_index == 1
    with pytest.raises(RuntimeError, match="not accepting"):
        session.advance_without_new_intent()
