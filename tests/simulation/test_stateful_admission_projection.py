from __future__ import annotations

from dataclasses import replace

import numpy as np

from trade_rl.data.market import MarketDataset
from trade_rl.simulation import MarketExecutor
from trade_rl.simulation.accounting import BookState
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.simulation.orders.model import (
    OrderBookState,
    OrderIntent,
    OrderType,
    PendingOrder,
    TimeInForce,
)


def _two_symbol_market() -> MarketDataset:
    n_bars = 4
    shape = (n_bars, 2)
    close = np.tile(np.array([100.0, 200.0]), (n_bars, 1))
    volume = np.full(shape, 1_000.0)
    volume[1] = 0.0
    return MarketDataset(
        dataset_id="d" * 64,
        symbols=("S0", "S1"),
        timestamps=np.datetime64("2026-01-01", "ns")
        + np.arange(n_bars) * np.timedelta64(1, "h"),
        features=np.zeros((n_bars, 2, 1), dtype=np.float32),
        global_features=np.zeros((n_bars, 1), dtype=np.float32),
        open=close.copy(),
        high=close + 10.0,
        low=close - 10.0,
        close=close.copy(),
        volume=volume,
        funding_rate=np.zeros(shape),
        tradable=np.ones(shape, dtype=np.bool_),
        feature_available=np.ones((n_bars, 2, 1), dtype=np.bool_),
        feature_names=("ret",),
        global_feature_names=("regime",),
        periods_per_year=8_760,
        contract_multipliers=np.array([1.0, 0.8]),
    )


def test_same_bar_admission_projects_exposure_in_order_without_mutating_input() -> None:
    dataset = _two_symbol_market()
    config = replace(
        ExecutionCostConfig.zero(),
        path_mode="conservative",
        processing_bar_volume_capacity=True,
        partial_fill_carry=True,
        max_leverage=1.0,
    )
    executor = MarketExecutor(dataset, config)
    intents = tuple(
        OrderIntent.create(
            dataset_id=dataset.dataset_id,
            target_identity=f"target-{symbol_index}",
            execution_policy_digest=executor.execution_policy_digest,
            symbol_index=symbol_index,
            requested_quantity=6.0,
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
        for symbol_index in range(2)
    )
    orders = OrderBookState.empty().add(
        *(PendingOrder.from_intent(intent) for intent in intents)
    )
    initial_book = BookState.zero(
        2,
        1_000.0,
        dataset.close[0],
        dataset.resolved_array("contract_multipliers"),
    )

    result = executor.execute_orders(
        initial_book,
        orders,
        (),
        start_index=0,
        bars=1,
    )

    rejected = [
        event for event in result.order_events if event.event_type == "rejected"
    ]
    eligible = [
        event for event in result.order_events if event.event_type == "eligible"
    ]
    assert len(rejected) == 1
    assert rejected[0].reason == "pretrade_leverage_exceeded"
    assert len(eligible) == 1
    assert len(result.order_book.active_orders) == 1
    np.testing.assert_array_equal(initial_book.quantities, np.zeros(2))
    assert initial_book.cash == 1_000.0
