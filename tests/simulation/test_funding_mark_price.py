from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from trade_rl.data.market import MarketDataset
from trade_rl.simulation import MarketExecutor
from trade_rl.simulation.accounting import BookState
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.simulation.orders import OrderBookState


def _funding_dataset() -> MarketDataset:
    n_bars = 4
    shape = (n_bars, 1)
    close = np.full(shape, 100.0, dtype=np.float64)
    mark_price = close.copy()
    mark_price[1, 0] = 120.0
    funding_rate = np.zeros(shape, dtype=np.float64)
    funding_rate[1, 0] = 0.001
    funding_due = np.zeros(shape, dtype=np.bool_)
    funding_due[1, 0] = True
    return MarketDataset(
        dataset_id="f" * 64,
        symbols=("BTCUSDT",),
        timestamps=np.datetime64("2026-01-01", "ns")
        + np.arange(n_bars) * np.timedelta64(1, "h"),
        features=np.zeros((n_bars, 1, 1), dtype=np.float32),
        global_features=np.zeros((n_bars, 1), dtype=np.float32),
        open=close.copy(),
        high=np.maximum(close, mark_price),
        low=np.minimum(close, mark_price),
        close=close,
        volume=np.full(shape, 1_000.0, dtype=np.float64),
        funding_rate=funding_rate,
        tradable=np.ones(shape, dtype=np.bool_),
        feature_available=np.ones((n_bars, 1, 1), dtype=np.bool_),
        feature_names=("probe",),
        global_feature_names=("probe",),
        periods_per_year=8_760,
        funding_due=funding_due,
        mark_price=mark_price,
        contract_multipliers=np.array([1.0], dtype=np.float64),
    )


def test_usds_m_funding_uses_mark_price_position_notional() -> None:
    dataset = _funding_dataset()
    executor = MarketExecutor(
        dataset,
        replace(
            ExecutionCostConfig.zero(),
            processing_bar_volume_capacity=True,
            partial_fill_carry=True,
        ),
    )
    book = BookState.from_weights(
        weights=np.array([1.0], dtype=np.float64),
        capital=1_000.0,
        prices=dataset.close[0],
        contract_multipliers=dataset.resolved_array("contract_multipliers"),
    )

    result = executor.execute_orders(
        book,
        OrderBookState.empty(),
        (),
        start_index=0,
        bars=1,
    )

    # Binance USDⓈ-M funding amount is position notional at mark price × rate.
    # Quantity is 10 BTC; mark notional at the funding boundary is 10 × 120.
    assert result.interval_funding == pytest.approx(-1.2)
    assert result.book.funding_pnl == pytest.approx(-1.2)
