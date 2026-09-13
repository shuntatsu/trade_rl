from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from trade_rl.data.contracts import VolumeUnit
from trade_rl.data.market import MarketDataset
from trade_rl.simulation import MarketExecutor
from trade_rl.simulation.accounting import BookState
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.simulation.orders.model import (
    OrderBookState,
    OrderIntent,
    OrderType,
    TimeInForce,
)


def _market(
    *,
    previous_close: float = 50.0,
    processing_open: float = 100.0,
    previous_volume: float = 2.0,
    processing_volume: float = 1_000.0,
    volume_unit: VolumeUnit = VolumeUnit.BASE_ASSET,
) -> MarketDataset:
    close = np.array(
        [[previous_close], [processing_open], [processing_open], [processing_open]],
        dtype=np.float64,
    )
    open_price = close.copy()
    high = np.maximum(open_price, close) + 10.0
    low = np.minimum(open_price, close) - 10.0
    volume = np.array(
        [
            [previous_volume],
            [processing_volume],
            [processing_volume],
            [processing_volume],
        ],
        dtype=np.float64,
    )
    return MarketDataset(
        dataset_id="c" * 64,
        symbols=("BTC",),
        timestamps=np.datetime64("2026-01-01", "ns")
        + np.arange(4) * np.timedelta64(1, "h"),
        features=np.zeros((4, 1, 1), dtype=np.float32),
        global_features=np.zeros((4, 1), dtype=np.float32),
        open=open_price,
        high=high,
        low=low,
        close=close,
        volume=volume,
        funding_rate=np.zeros((4, 1), dtype=np.float64),
        tradable=np.ones((4, 1), dtype=np.bool_),
        feature_available=np.ones((4, 1, 1), dtype=np.bool_),
        feature_names=("ret",),
        global_feature_names=("regime",),
        periods_per_year=8_760,
        volume_units=(volume_unit,),
    )


def _executor(dataset: MarketDataset) -> MarketExecutor:
    return MarketExecutor(
        dataset,
        replace(
            ExecutionCostConfig.zero(),
            processing_bar_volume_capacity=False,
            max_participation_rate=1.0,
            path_mode="conservative",
        ),
    )


def _intent(executor: MarketExecutor, quantity: float = 10.0) -> OrderIntent:
    return OrderIntent.create(
        dataset_id=executor.dataset.dataset_id,
        target_identity="causal-capacity",
        execution_policy_digest=executor.execution_policy_digest,
        symbol_index=0,
        requested_quantity=quantity,
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


def _execute(dataset: MarketDataset):
    executor = _executor(dataset)
    return executor.execute_orders(
        BookState.zero(1, 1_000.0, dataset.close[0]),
        OrderBookState.empty(),
        (_intent(executor),),
        start_index=0,
        bars=1,
    )


def test_causal_capacity_evidence_binds_previous_completed_bar() -> None:
    result = _execute(
        _market(
            previous_close=50.0,
            processing_open=100.0,
            previous_volume=2.0,
            processing_volume=1_000.0,
        )
    )

    assert result.book.quantities[0] == pytest.approx(1.0)
    assert result.filled_notional == pytest.approx(100.0)
    evidence = result.capacity_evidence[0]
    assert evidence.processing_volume == pytest.approx(2.0)
    assert evidence.capacity_reference_price == pytest.approx(50.0)
    assert evidence.market_notional == pytest.approx(100.0)
    assert evidence.initial_capacity_notional == pytest.approx(100.0)
    assert evidence.consumed_capacity_notional == pytest.approx(100.0)


def test_causal_capacity_pool_ignores_processing_bar_volume_and_price() -> None:
    baseline = _execute(
        _market(
            previous_close=50.0,
            processing_open=100.0,
            previous_volume=2.0,
            processing_volume=1_000.0,
        )
    )
    changed_processing_bar = _execute(
        _market(
            previous_close=50.0,
            processing_open=200.0,
            previous_volume=2.0,
            processing_volume=10_000.0,
        )
    )

    assert baseline.filled_notional == pytest.approx(100.0)
    assert changed_processing_bar.filled_notional == pytest.approx(100.0)
    assert changed_processing_bar.book.quantities[0] == pytest.approx(0.5)
    assert changed_processing_bar.capacity_evidence[
        0
    ].processing_volume == pytest.approx(
        baseline.capacity_evidence[0].processing_volume
    )
    assert changed_processing_bar.capacity_evidence[0].market_notional == pytest.approx(
        baseline.capacity_evidence[0].market_notional
    )
    assert changed_processing_bar.capacity_evidence[
        0
    ].capacity_reference_price == pytest.approx(
        baseline.capacity_evidence[0].capacity_reference_price
    )


def test_causal_capacity_preserves_previous_quote_notional_semantics() -> None:
    result = _execute(
        _market(
            previous_close=50.0,
            processing_open=100.0,
            previous_volume=100.0,
            processing_volume=10_000.0,
            volume_unit=VolumeUnit.QUOTE_NOTIONAL,
        )
    )

    evidence = result.capacity_evidence[0]
    assert evidence.processing_volume == pytest.approx(100.0)
    assert evidence.capacity_reference_price == pytest.approx(50.0)
    assert evidence.market_notional == pytest.approx(100.0)
    assert result.filled_notional == pytest.approx(100.0)


def test_capacity_mode_is_already_execution_policy_identity_bound() -> None:
    legacy = ExecutionCostConfig.zero()
    causal = replace(legacy, processing_bar_volume_capacity=False)

    assert legacy.processing_bar_volume_capacity is True
    assert causal.processing_bar_volume_capacity is False
    assert legacy.execution_policy_digest != causal.execution_policy_digest
