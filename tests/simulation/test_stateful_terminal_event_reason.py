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


def test_filled_event_preserves_terminal_order_reason() -> None:
    n_bars = 3
    shape = (n_bars, 1)
    close = np.full(shape, 100.0)
    market = MarketDataset(
        dataset_id="d" * 64,
        symbols=("BTCUSDT",),
        timestamps=np.datetime64("2026-01-01", "ns")
        + np.arange(n_bars) * np.timedelta64(15, "m"),
        features=np.zeros((n_bars, 1, 1), dtype=np.float32),
        global_features=np.zeros((n_bars, 1), dtype=np.float32),
        open=close.copy(),
        high=close + 1.0,
        low=close - 1.0,
        close=close.copy(),
        volume=np.full(shape, 1_000.0),
        funding_rate=np.zeros(shape),
        tradable=np.ones(shape, dtype=np.bool_),
        feature_available=np.ones((n_bars, 1, 1), dtype=np.bool_),
        feature_names=("probe",),
        global_feature_names=("probe",),
        periods_per_year=35_040,
    )
    cost = replace(
        ExecutionCostConfig.zero(),
        processing_bar_volume_capacity=True,
        max_participation_rate=1.0,
    )
    executor = MarketExecutor(market, cost)
    intent = OrderIntent.create(
        dataset_id=market.dataset_id,
        target_identity="filled-terminal-reason",
        execution_policy_digest=executor.execution_policy_digest,
        symbol_index=0,
        requested_quantity=1.0,
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
    book = BookState.zero(
        1,
        1_000.0,
        market.close[0],
        market.resolved_array("contract_multipliers"),
    )

    result = executor.execute_orders(
        book,
        OrderBookState.empty(),
        (intent,),
        start_index=0,
        bars=1,
    )

    terminal = result.order_book.terminal_orders[-1]
    filled = next(event for event in result.order_events if event.event_type == "filled")
    assert terminal.status is OrderStatus.FILLED
    assert terminal.terminal_reason == "filled"
    assert filled.new_status is OrderStatus.FILLED
    assert filled.reason == terminal.terminal_reason
