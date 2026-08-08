from __future__ import annotations

from dataclasses import replace

import numpy as np

from trade_rl.data.market import MarketDataset
from trade_rl.simulation import MarketExecutor
from trade_rl.simulation.accounting import BookState
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.simulation.orders import (
    OrderBookState,
    OrderIntent,
    OrderStatus,
    OrderType,
    TimeInForce,
)


def test_partially_filled_remainder_expires_when_market_disables_buying() -> None:
    n_bars = 4
    shape = (n_bars, 1)
    close = np.full(shape, 100.0)
    volume = np.full(shape, 1_000.0)
    volume[1, 0] = 1.0
    buy_allowed = np.ones(shape, dtype=np.bool_)
    buy_allowed[2, 0] = False
    dataset = MarketDataset(
        dataset_id="d" * 64,
        symbols=("S0",),
        timestamps=np.datetime64("2026-01-01", "ns")
        + np.arange(n_bars) * np.timedelta64(1, "h"),
        features=np.zeros((n_bars, 1, 1), dtype=np.float32),
        global_features=np.zeros((n_bars, 1), dtype=np.float32),
        open=close.copy(),
        high=close + 1.0,
        low=close - 1.0,
        close=close.copy(),
        volume=volume,
        funding_rate=np.zeros(shape),
        tradable=np.ones(shape, dtype=np.bool_),
        feature_available=np.ones((n_bars, 1, 1), dtype=np.bool_),
        feature_names=("probe",),
        global_feature_names=("probe",),
        periods_per_year=8_760,
        buy_allowed=buy_allowed,
    )
    executor = MarketExecutor(
        dataset,
        replace(
            ExecutionCostConfig.zero(),
            path_mode="conservative",
            processing_bar_volume_capacity=True,
            partial_fill_carry=True,
            max_participation_rate=1.0,
        ),
    )
    book = BookState.zero(
        1,
        1_000.0,
        dataset.close[0],
        dataset.resolved_array("contract_multipliers"),
    )
    intent = OrderIntent.create(
        dataset_id=dataset.dataset_id,
        target_identity="target-1",
        execution_policy_digest=executor.execution_policy_digest,
        symbol_index=0,
        requested_quantity=2.0,
        order_type=OrderType.MARKET,
        time_in_force=TimeInForce.GTC,
        limit_price=None,
        stop_price=None,
        submit_index=0,
        eligible_index=1,
        expiry_index=None,
        submission_reference_price=100.0,
        decision_equity=1_000.0,
    )

    first = executor.execute_orders(
        book,
        OrderBookState.empty(),
        (intent,),
        start_index=0,
        bars=1,
    )
    active = first.order_book.active_orders[0]
    assert active.status is OrderStatus.PARTIALLY_FILLED
    assert active.remaining_quantity == 1.0

    second = executor.execute_orders(
        first.book,
        first.order_book,
        (),
        start_index=1,
        bars=1,
    )

    assert second.order_book.active_orders == ()
    expired = second.order_book.terminal_orders[-1]
    assert expired.status is OrderStatus.EXPIRED
    assert expired.terminal_reason == "buy_disabled"
    assert second.expired_count == 1
    assert second.rejected_count == 0
    assert second.order_events[-1].event_type == "expired"
    assert second.order_events[-1].reason == "buy_disabled"
