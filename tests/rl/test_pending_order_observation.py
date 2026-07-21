from __future__ import annotations

import numpy as np
import pytest

from trade_rl.data.market import MarketDataset
from trade_rl.rl.observations import (
    ORDER_OBSERVATION_WIDTH,
    PendingOrderObservationState,
    PolicyObservationSnapshot,
    build_observation,
    observation_layout,
)
from trade_rl.simulation.accounting import BookState
from trade_rl.simulation.orders import (
    OrderBookState,
    OrderIntent,
    OrderStatus,
    OrderType,
    PendingOrder,
    TimeInForce,
)
from trade_rl.strategies.trend import TrendTargets


def _market() -> MarketDataset:
    n = 6
    close = np.column_stack([100.0 + np.arange(n), 200.0 + np.arange(n)])
    open_price = np.vstack([close[0], close[:-1]])
    return MarketDataset(
        dataset_id="d" * 64,
        symbols=("A", "B"),
        timestamps=np.datetime64("2026-01-01", "ns")
        + np.arange(n) * np.timedelta64(1, "h"),
        features=np.zeros((n, 2, 1), dtype=np.float32),
        global_features=np.zeros((n, 1), dtype=np.float32),
        open=open_price,
        high=np.maximum(open_price, close) + 1.0,
        low=np.minimum(open_price, close) - 1.0,
        close=close,
        volume=np.full_like(close, 1_000.0),
        funding_rate=np.zeros_like(close),
        tradable=np.ones_like(close, dtype=np.bool_),
        feature_available=np.ones((n, 2, 1), dtype=np.bool_),
        feature_names=("ret",),
        global_feature_names=("regime",),
        periods_per_year=8_760,
    )


def _pending(
    *,
    symbol_index: int = 0,
    requested_quantity: float = 4.0,
    order_type: OrderType = OrderType.LIMIT,
    status: OrderStatus = OrderStatus.PARTIALLY_FILLED,
    submit_index: int = 1,
    eligible_index: int = 3,
    expiry_index: int | None = 5,
) -> PendingOrder:
    intent = OrderIntent.create(
        dataset_id="d" * 64,
        target_identity=f"target-{symbol_index}-{requested_quantity}",
        execution_policy_digest="e" * 64,
        symbol_index=symbol_index,
        requested_quantity=requested_quantity,
        order_type=order_type,
        time_in_force=(
            TimeInForce.DAY if expiry_index is not None else TimeInForce.GTC
        ),
        limit_price=99.0 if order_type is OrderType.LIMIT else None,
        stop_price=105.0 if order_type is OrderType.STOP_MARKET else None,
        submit_index=submit_index,
        eligible_index=eligible_index,
        expiry_index=expiry_index,
        submission_reference_price=100.0,
        decision_equity=1_000.0,
    )
    order = PendingOrder.from_intent(intent)
    if status is OrderStatus.PARTIALLY_FILLED:
        return order.mark_eligible(processing_index=eligible_index).apply_fill(
            quantity=requested_quantity / 2.0,
            notional=abs(requested_quantity / 2.0) * 100.0,
            processing_index=eligible_index,
        )
    if status is OrderStatus.TRIGGERED:
        return order.mark_eligible(processing_index=eligible_index).mark_triggered(
            processing_index=eligible_index
        )
    if status is OrderStatus.ELIGIBLE:
        return order.mark_eligible(processing_index=eligible_index)
    if status is OrderStatus.LATENCY_WAIT:
        return order.mark_latency_wait(processing_index=submit_index + 1)
    return order


def test_pending_order_state_is_fixed_shape_and_causal() -> None:
    market = _market()
    order = _pending()
    state = PendingOrderObservationState.from_order_book(
        OrderBookState(active_orders=(order,), terminal_orders=()),
        n_symbols=2,
        current_index=4,
        reference_prices=market.close[4],
        contract_multipliers=np.ones(2),
        portfolio_value=1_000.0,
    )

    assert state.matrix.shape == (2, ORDER_OBSERVATION_WIDTH)
    assert state.remaining_notional_ratio[0] == pytest.approx(0.208)
    assert state.order_type_code[0] == 2.0
    assert state.status_code[0] == 5.0
    assert state.age_bars[0] == 3.0
    assert state.eligible_delay_bars[0] == 0.0
    assert state.triggered[0] == 0.0
    assert state.expiry_distance_bars[0] == 2.0
    np.testing.assert_array_equal(state.matrix[1], np.zeros(ORDER_OBSERVATION_WIDTH))


def test_triggered_stop_and_latency_fields_are_explicit() -> None:
    market = _market()
    triggered = _pending(
        requested_quantity=-2.0,
        order_type=OrderType.STOP_MARKET,
        status=OrderStatus.TRIGGERED,
        expiry_index=None,
    )
    waiting = _pending(
        symbol_index=1,
        requested_quantity=1.0,
        order_type=OrderType.MARKET,
        status=OrderStatus.LATENCY_WAIT,
        submit_index=3,
        eligible_index=5,
        expiry_index=None,
    )
    state = PendingOrderObservationState.from_order_book(
        OrderBookState(active_orders=(triggered, waiting), terminal_orders=()),
        n_symbols=2,
        current_index=4,
        reference_prices=market.close[4],
        contract_multipliers=np.ones(2),
        portfolio_value=1_000.0,
    )

    assert state.order_type_code.tolist() == [3.0, 1.0]
    assert state.triggered.tolist() == [1.0, 0.0]
    assert state.eligible_delay_bars.tolist() == [0.0, 1.0]
    assert state.expiry_distance_bars.tolist() == [0.0, 0.0]


def test_multiple_active_orders_per_symbol_fail_closed_for_policy_state() -> None:
    market = _market()
    first = _pending()
    second = _pending(requested_quantity=2.0, status=OrderStatus.ELIGIBLE)
    with pytest.raises(ValueError, match="one active order per symbol"):
        PendingOrderObservationState.from_order_book(
            OrderBookState(active_orders=(first, second), terminal_orders=()),
            n_symbols=2,
            current_index=4,
            reference_prices=market.close[4],
            contract_multipliers=np.ones(2),
            portfolio_value=1_000.0,
        )


def test_observation_appends_pending_order_state_per_symbol() -> None:
    market = _market()
    book = BookState.zero(2, 1_000.0, market.close[4])
    pending = PendingOrderObservationState.from_order_book(
        OrderBookState(active_orders=(_pending(),), terminal_orders=()),
        n_symbols=2,
        current_index=4,
        reference_prices=market.close[4],
        contract_multipliers=np.ones(2),
        portfolio_value=book.portfolio_value,
    )
    vector = build_observation(
        dataset=market,
        index=4,
        trends=TrendTargets(fast=np.zeros(2), base=np.zeros(2), slow=np.zeros(2)),
        alpha=np.zeros(2),
        hybrid=book,
        shadow=book.clone(),
        start_index=0,
        end_index=5,
        hybrid_risk_scale=1.0,
        shadow_risk_scale=1.0,
        pending_order_state=pending,
        previous_action=np.zeros(2),
        action_size=2,
    )
    layout = observation_layout(market, action_size=2)
    rows = vector[: market.n_symbols * layout.per_symbol_width].reshape(
        market.n_symbols, layout.per_symbol_width
    )

    np.testing.assert_allclose(rows[:, -ORDER_OBSERVATION_WIDTH:], pending.matrix)


def test_snapshot_digest_binds_order_state_and_execution_policy() -> None:
    market = _market()
    pending = PendingOrderObservationState.zero(2)
    snapshot = PolicyObservationSnapshot(
        dataset_id=market.dataset_id,
        index=4,
        symbols=market.symbols,
        feature_names=market.feature_names,
        global_feature_names=market.global_feature_names,
        availability_mask=np.ones(3),
        staleness=np.zeros(3),
        hybrid_book_state=np.ones(10),
        shadow_book_state=np.ones(10),
        pending_target=np.zeros(2),
        previous_action=np.zeros(2),
        pending_order_remaining=pending.remaining_notional_ratio,
        pending_order_type=pending.order_type_code,
        pending_order_status=pending.status_code,
        pending_order_age_bars=pending.age_bars,
        pending_order_eligible_delay=pending.eligible_delay_bars,
        pending_order_triggered=pending.triggered,
        pending_order_expiry_distance=pending.expiry_distance_bars,
        execution_policy_digest="e" * 64,
        raw_observation=np.ones(20),
        normalized_observation=np.ones(20),
    )
    changed = PolicyObservationSnapshot(
        dataset_id=market.dataset_id,
        index=4,
        symbols=market.symbols,
        feature_names=market.feature_names,
        global_feature_names=market.global_feature_names,
        availability_mask=np.ones(3),
        staleness=np.zeros(3),
        hybrid_book_state=np.ones(10),
        shadow_book_state=np.ones(10),
        pending_target=np.zeros(2),
        previous_action=np.zeros(2),
        pending_order_remaining=np.array([0.1, 0.0]),
        pending_order_type=pending.order_type_code,
        pending_order_status=pending.status_code,
        pending_order_age_bars=pending.age_bars,
        pending_order_eligible_delay=pending.eligible_delay_bars,
        pending_order_triggered=pending.triggered,
        pending_order_expiry_distance=pending.expiry_distance_bars,
        execution_policy_digest="e" * 64,
        raw_observation=np.ones(20),
        normalized_observation=np.ones(20),
    )

    assert snapshot.snapshot_digest != changed.snapshot_digest
    assert snapshot.execution_policy_digest == "e" * 64
