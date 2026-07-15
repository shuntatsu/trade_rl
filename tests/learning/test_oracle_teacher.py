from __future__ import annotations

import numpy as np
import pytest

from trade_rl.data.market import MarketDataset
from trade_rl.learning.oracle_teacher import (
    OracleTeacherConfig,
    oracle_target_path,
)
from trade_rl.simulation.execution import ExecutionCostConfig


def _market(close_values: np.ndarray) -> MarketDataset:
    close = np.asarray(close_values, dtype=np.float64)
    if close.ndim == 1:
        close = close[:, None]
    n_bars, n_symbols = close.shape
    open_price = np.vstack([close[0], close[:-1]])
    return MarketDataset(
        dataset_id="a" * 64,
        symbols=tuple(f"S{index}" for index in range(n_symbols)),
        timestamps=np.datetime64("2026-01-01", "ns")
        + np.arange(n_bars) * np.timedelta64(15, "m"),
        features=np.zeros((n_bars, n_symbols, 1), dtype=np.float32),
        global_features=np.zeros((n_bars, 1), dtype=np.float32),
        open=open_price,
        high=np.maximum(open_price, close) * 1.001,
        low=np.minimum(open_price, close) * 0.999,
        close=close,
        volume=np.full_like(close, 1_000_000.0),
        funding_rate=np.zeros_like(close),
        tradable=np.ones_like(close, dtype=np.bool_),
        feature_available=np.ones((n_bars, n_symbols, 1), dtype=np.bool_),
        feature_names=("return",),
        global_feature_names=("regime",),
        periods_per_year=35_040,
    )


def test_oracle_path_is_deterministic_bounded_and_train_range_only() -> None:
    market = _market(
        np.column_stack(
            [
                100.0 * np.exp(np.arange(9) * 0.02),
                100.0 * np.exp(np.arange(9) * -0.02),
            ]
        )
    )
    config = OracleTeacherConfig(execution_cost=ExecutionCostConfig.zero())

    first = oracle_target_path(market, (1, 8), config)
    second = oracle_target_path(market, (1, 8), config)
    changed_close = market.close.copy()
    changed_close[8] *= 100.0
    changed_future = _market(changed_close)

    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(
        first,
        oracle_target_path(changed_future, (1, 8), config),
    )
    assert first.shape == (6, 2)
    assert np.max(np.abs(first)) <= 0.5
    assert np.max(np.abs(first).sum(axis=1)) <= 1.0
    assert np.mean(first[:, 0]) > 0.0
    assert np.mean(first[:, 1]) < 0.0


def test_execution_cost_creates_flat_or_hold_regions() -> None:
    market = _market(np.array([100.0, 100.1, 100.0, 100.1, 100.0, 100.1]))
    free = OracleTeacherConfig(execution_cost=ExecutionCostConfig.zero())
    expensive = OracleTeacherConfig(
        execution_cost=ExecutionCostConfig(
            fee_rate=0.01,
            spread_rate=0.01,
            impact_rate=0.0,
            max_participation_rate=1.0,
        )
    )

    free_path = oracle_target_path(market, (0, 6), free)
    expensive_path = oracle_target_path(market, (0, 6), expensive)

    assert np.count_nonzero(np.diff(free_path[:, 0])) > 0
    assert np.count_nonzero(expensive_path) == 0


@pytest.mark.parametrize("train_range", [(-1, 4), (0, 1), (3, 3), (0, 7)])
def test_oracle_rejects_ranges_outside_dataset(
    train_range: tuple[int, int],
) -> None:
    market = _market(np.linspace(100.0, 105.0, 6))

    with pytest.raises(ValueError, match="training range"):
        oracle_target_path(market, train_range, OracleTeacherConfig())
