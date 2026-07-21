from __future__ import annotations

import numpy as np
import pytest

from trade_rl.simulation.accounting import BookState
from trade_rl.simulation.order_reconciliation import (
    OrderReconciliationError,
    reconcile_target,
)
from trade_rl.simulation.orders import (
    OrderBookState,
    OrderIntent,
    OrderStatus,
    OrderType,
    PendingOrder,
    TimeInForce,
)


def _book(quantity: float = 0.0) -> BookState:
    return BookState(
        quantities=np.array([quantity]),
        cash=1_000.0 - quantity * 100.0,
        mark_prices=np.array([100.0]),
        peak_value=1_000.0,
        contract_multipliers=np.array([1.0]),
    )


def _active(quantity: float, *, target_identity: str = "old") -> PendingOrder:
    intent = OrderIntent.create(
        dataset_id="d" * 64,
        target_identity=target_identity,
        execution_policy_digest="e" * 64,
        symbol_index=0,
        requested_quantity=quantity,
        order_type=OrderType.MARKET,
        time_in_force=TimeInForce.GTC,
        limit_price=None,
        stop_price=None,
        submit_index=2,
        eligible_index=3,
        expiry_index=None,
        submission_reference_price=100.0,
        decision_equity=1_000.0,
    )
    return PendingOrder.from_intent(intent)


def _reconcile(
    *,
    target_weight: float,
    book: BookState | None = None,
    order_book: OrderBookState | None = None,
    submit_index: int = 4,
    latency_bars: int = 1,
    order_type: OrderType = OrderType.MARKET,
    time_in_force: TimeInForce = TimeInForce.GTC,
    expiry_index: int | None = None,
    limit_offset_rate: float = 0.01,
):
    return reconcile_target(
        dataset_id="d" * 64,
        target_identity=f"target-{target_weight}",
        execution_policy_digest="e" * 64,
        target_weights=np.array([target_weight]),
        book=_book() if book is None else book,
        order_book=OrderBookState.empty() if order_book is None else order_book,
        reference_prices=np.array([100.0]),
        decision_equity=1_000.0,
        submit_index=submit_index,
        latency_bars=latency_bars,
        order_type=order_type,
        time_in_force=time_in_force,
        expiry_index=expiry_index,
        limit_offset_rate=limit_offset_rate,
        maximum_gross=1.0,
    )


def test_reconciliation_does_not_double_submit_matching_active_residual() -> None:
    active = _active(3.0)
    result = _reconcile(
        target_weight=0.5,
        book=_book(2.0),
        order_book=OrderBookState(active_orders=(active,), terminal_orders=()),
    )

    assert result.new_intents == ()
    assert result.cancelled_orders == ()
    assert result.order_book.active_orders == (active,)
    np.testing.assert_allclose(result.desired_quantities, [5.0])
    np.testing.assert_allclose(result.residual_quantities, [0.0])


def test_changed_target_cancels_old_residual_and_submits_only_latest_delta() -> None:
    active = _active(3.0)
    result = _reconcile(
        target_weight=0.4,
        book=_book(2.0),
        order_book=OrderBookState(active_orders=(active,), terminal_orders=()),
    )

    assert result.cancelled_orders[0].status is OrderStatus.CANCELLED
    assert result.cancelled_orders[0].terminal_reason == "superseded"
    assert result.order_book.active_orders == ()
    assert result.order_book.terminal_orders[-1] == result.cancelled_orders[0]
    assert result.new_intents[0].requested_quantity == pytest.approx(2.0)
    assert result.new_intents[0].replaced_order_id == active.order_id


def test_reversal_cancels_same_direction_residual_and_uses_holdings_delta() -> None:
    active = _active(4.0)
    result = _reconcile(
        target_weight=-0.3,
        book=_book(2.0),
        order_book=OrderBookState(active_orders=(active,), terminal_orders=()),
    )

    assert result.new_intents[0].requested_quantity == pytest.approx(-5.0)
    assert result.cancelled_orders == (result.order_book.terminal_orders[-1],)


def test_quantity_is_fixed_from_submission_price_and_decision_equity() -> None:
    result = _reconcile(target_weight=0.5, latency_bars=3)

    intent = result.new_intents[0]
    assert intent.requested_quantity == pytest.approx(5.0)
    assert intent.submission_reference_price == pytest.approx(100.0)
    assert intent.decision_equity == pytest.approx(1_000.0)
    assert intent.submit_index == 4
    assert intent.eligible_index == 7


def test_limit_and_stop_prices_are_directional_and_submission_bound() -> None:
    buy_limit = _reconcile(target_weight=0.5, order_type=OrderType.LIMIT)
    sell_stop = _reconcile(
        target_weight=-0.5,
        order_type=OrderType.STOP_MARKET,
        limit_offset_rate=0.02,
    )

    assert buy_limit.new_intents[0].limit_price == pytest.approx(99.0)
    assert buy_limit.new_intents[0].stop_price is None
    assert sell_stop.new_intents[0].stop_price == pytest.approx(98.0)
    assert sell_stop.new_intents[0].limit_price is None


def test_day_orders_require_explicit_expiry_and_zero_delta_submits_nothing() -> None:
    with pytest.raises(OrderReconciliationError, match="expiry"):
        _reconcile(target_weight=0.5, time_in_force=TimeInForce.DAY)

    flat = _reconcile(target_weight=0.2, book=_book(2.0))
    assert flat.new_intents == ()
    np.testing.assert_allclose(flat.residual_quantities, [0.0])


def test_invalid_shapes_gross_latency_and_offsets_fail_closed() -> None:
    with pytest.raises(OrderReconciliationError, match="shape"):
        reconcile_target(
            dataset_id="d" * 64,
            target_identity="target",
            execution_policy_digest="e" * 64,
            target_weights=np.array([0.1, 0.2]),
            book=_book(),
            order_book=OrderBookState.empty(),
            reference_prices=np.array([100.0]),
            decision_equity=1_000.0,
            submit_index=4,
            latency_bars=0,
            order_type=OrderType.MARKET,
            time_in_force=TimeInForce.GTC,
            expiry_index=None,
            limit_offset_rate=0.0,
        )
    with pytest.raises(OrderReconciliationError, match="gross"):
        _reconcile(target_weight=1.1)
    with pytest.raises(OrderReconciliationError, match="latency"):
        _reconcile(target_weight=0.5, latency_bars=-1)
    with pytest.raises(OrderReconciliationError, match="offset"):
        _reconcile(target_weight=0.5, limit_offset_rate=1.0)
