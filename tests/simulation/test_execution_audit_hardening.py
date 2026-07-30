from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from trade_rl.data.market import MarketDataset
from trade_rl.simulation.execution import ExecutionCostConfig, MarketExecutor


def _market(n_symbols: int) -> MarketDataset:
    n_bars = 3
    shape = (n_bars, n_symbols)
    close = np.full(shape, 100.0)
    return MarketDataset(
        dataset_id="d" * 64,
        symbols=tuple(f"S{index}" for index in range(n_symbols)),
        timestamps=np.datetime64("2026-01-01", "ns")
        + np.arange(n_bars) * np.timedelta64(1, "h"),
        features=np.zeros((n_bars, n_symbols, 1), dtype=np.float32),
        global_features=np.zeros((n_bars, 1), dtype=np.float32),
        open=close.copy(),
        high=close + 1.0,
        low=close - 1.0,
        close=close,
        volume=np.full(shape, 1_000_000.0),
        funding_rate=np.zeros(shape),
        tradable=np.ones(shape, dtype=np.bool_),
        feature_available=np.ones((n_bars, n_symbols, 1), dtype=np.bool_),
        feature_names=("ret",),
        global_feature_names=("regime",),
        periods_per_year=8_760,
    )


def test_tail_slippage_stress_cannot_improve_execution() -> None:
    with pytest.raises(ValueError, match="tail_slippage_multiplier"):
        ExecutionCostConfig(
            tail_slippage_probability=0.1,
            tail_slippage_multiplier=0.5,
        )


def test_execution_policy_digest_binds_economic_costs() -> None:
    base = ExecutionCostConfig()
    changed = replace(base, fee_rate=base.fee_rate + 0.001)

    assert base.execution_policy_digest != changed.execution_policy_digest


def test_multi_asset_isolated_margin_is_rejected_without_a_collateral_ledger() -> None:
    with pytest.raises(ValueError, match="isolated margin"):
        MarketExecutor(
            _market(2),
            ExecutionCostConfig(margin_mode="isolated"),
        )


def test_single_asset_isolated_margin_remains_equivalent_to_cross() -> None:
    executor = MarketExecutor(
        _market(1),
        ExecutionCostConfig(margin_mode="isolated"),
    )

    assert executor.cost.margin_mode == "isolated"
