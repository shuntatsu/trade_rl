from __future__ import annotations

import numpy as np
import pytest

from trade_rl.data.market import MarketDataset
from trade_rl.simulation.accounting import BookState
from trade_rl.simulation import MarketExecutor
from trade_rl.simulation.execution import (
    ExecutionCostConfig,
    ExecutionRuleStress,
)


def _market(**overrides: object) -> MarketDataset:
    n = 4
    close = np.full((n, 1), 100.0)
    values: dict[str, object] = {
        "dataset_id": "c" * 64,
        "symbols": ("BTCUSDT",),
        "timestamps": np.datetime64("2026-01-01", "ns")
        + np.arange(n) * np.timedelta64(1, "h"),
        "features": np.zeros((n, 1, 1), dtype=np.float32),
        "global_features": np.zeros((n, 1), dtype=np.float32),
        "open": close.copy(),
        "high": close + 1.0,
        "low": close - 1.0,
        "close": close,
        "volume": np.full((n, 1), 1_000_000.0),
        "funding_rate": np.zeros_like(close),
        "tradable": np.ones_like(close, dtype=np.bool_),
        "feature_available": np.ones((n, 1, 1), dtype=np.bool_),
        "feature_names": ("ret",),
        "global_feature_names": ("regime",),
        "periods_per_year": 8_760,
        "tick_size": np.full((n, 1), 0.1),
        "lot_size": np.full((n, 1), 0.01),
        "minimum_notional": np.full((n, 1), 5.0),
    }
    values.update(overrides)
    return MarketDataset(**values)


def test_execution_rule_stress_rejects_empty_name() -> None:
    with pytest.raises(ValueError, match="name must be non-empty"):
        ExecutionRuleStress(name="")


def test_execution_rule_stress_rejects_factor_below_one() -> None:
    with pytest.raises(ValueError, match="tick_size_factor"):
        ExecutionRuleStress(tick_size_factor=0.5)


def test_execution_rule_stress_rejects_non_boolean_rounding_flag() -> None:
    with pytest.raises(ValueError, match="adverse_tick_rounding must be a boolean"):
        ExecutionRuleStress(adverse_tick_rounding=1)  # type: ignore[arg-type]


def test_execution_rule_stress_multiplies_point_in_time_rules_without_mutation() -> (
    None
):
    dataset = _market()
    stress = ExecutionRuleStress(
        name="joint_2x",
        tick_size_factor=2.0,
        lot_size_factor=2.0,
        minimum_notional_factor=2.0,
        adverse_tick_rounding=True,
    )
    executor = MarketExecutor(dataset, ExecutionCostConfig.zero(), rule_stress=stress)

    tick, lot, minimum = executor.effective_rule_arrays(index=1)

    assert tick == pytest.approx([0.2])
    assert lot == pytest.approx([0.02])
    assert minimum == pytest.approx([10.0])
    assert dataset.tick_size is not None
    assert dataset.tick_size[1] == pytest.approx([0.1])


def test_execution_rule_stress_rejects_zero_source_rule() -> None:
    dataset = _market(lot_size=np.zeros((4, 1)))

    with pytest.raises(ValueError, match="lot_size.*positive"):
        MarketExecutor(
            dataset,
            ExecutionCostConfig(lot_size=0.05),
            rule_stress=ExecutionRuleStress(name="lot_2x", lot_size_factor=2.0),
        )


def test_sensitivity_only_tick_rounding_is_adverse_to_trade_direction() -> None:
    open_price = np.full((4, 1), 100.05)
    dataset = _market(open=open_price)
    nominal = MarketExecutor(dataset, ExecutionCostConfig.zero())
    stressed = MarketExecutor(
        dataset,
        ExecutionCostConfig.zero(),
        rule_stress=ExecutionRuleStress(
            name="tick_adverse",
            adverse_tick_rounding=True,
        ),
    )
    nominal_result = nominal.execute_interval(
        BookState.zero(1, 1_000.0, dataset.close[0]),
        np.array([0.9]),
        start_index=0,
        bars=1,
    )
    stressed_result = stressed.execute_interval(
        BookState.zero(1, 1_000.0, dataset.close[0]),
        np.array([0.9]),
        start_index=0,
        bars=1,
    )

    assert stressed_result.book.cash < nominal_result.book.cash
    assert stressed_result.interval_net_return < nominal_result.interval_net_return


def test_rule_burden_reports_declared_ratio_percentiles() -> None:
    executor = MarketExecutor(
        _market(),
        ExecutionCostConfig.zero(),
        rule_stress=ExecutionRuleStress(
            name="joint_5x",
            tick_size_factor=5.0,
            lot_size_factor=5.0,
            minimum_notional_factor=5.0,
            adverse_tick_rounding=True,
        ),
    )
    burden = executor.rule_burden_percentiles(start=1, stop=3)

    assert burden["tick_size_ratio"] == pytest.approx(
        {"p50": 5.0, "p95": 5.0, "max": 5.0}
    )
    assert burden["lot_size_ratio"] == pytest.approx(
        {"p50": 5.0, "p95": 5.0, "max": 5.0}
    )
    assert burden["minimum_notional_ratio"] == pytest.approx(
        {"p50": 5.0, "p95": 5.0, "max": 5.0}
    )
    assert burden["adverse_tick_rounding"] is True
