from __future__ import annotations

import math

import pytest

from trade_rl.strategies.forecasts.controller import (
    CostAwareForecastIntentController,
    ForecastIntentConfig,
    ForecastIntentController,
)
from trade_rl.strategies.position_intent import PositionIntent


def controller() -> ForecastIntentController:
    return ForecastIntentController(
        ForecastIntentConfig(entry_threshold=0.10, exit_threshold=0.02)
    )


def test_forecast_controller_owns_entry_hold_exit_and_reversal() -> None:
    model = controller()

    assert model.decide(0.20, current=PositionIntent.FLAT) is PositionIntent.LONG
    assert model.decide(-0.20, current=PositionIntent.FLAT) is PositionIntent.SHORT
    assert model.decide(0.05, current=PositionIntent.FLAT) is PositionIntent.FLAT
    assert model.decide(0.05, current=PositionIntent.LONG) is PositionIntent.LONG
    assert model.decide(0.01, current=PositionIntent.LONG) is PositionIntent.FLAT
    assert model.decide(-0.20, current=PositionIntent.LONG) is PositionIntent.SHORT
    assert model.decide(-0.05, current=PositionIntent.SHORT) is PositionIntent.SHORT
    assert model.decide(-0.01, current=PositionIntent.SHORT) is PositionIntent.FLAT
    assert model.decide(0.20, current=PositionIntent.SHORT) is PositionIntent.LONG


def test_non_finite_forecast_fails_closed() -> None:
    model = controller()
    for value in (math.nan, math.inf, -math.inf):
        assert model.decide(value, current=PositionIntent.LONG) is PositionIntent.FLAT


def test_cost_aware_controller_vetoes_unprofitable_exit() -> None:
    model = CostAwareForecastIntentController(
        ForecastIntentConfig(entry_threshold=0.10, exit_threshold=0.02),
        one_way_switch_cost=0.005,
    )

    assert (
        model.decide(math.log1p(0.01), current=PositionIntent.LONG)
        is PositionIntent.LONG
    )
    assert (
        model.decide(math.log1p(-0.006), current=PositionIntent.LONG)
        is PositionIntent.FLAT
    )
    assert (
        model.decide(math.log1p(-0.005), current=PositionIntent.LONG)
        is PositionIntent.LONG
    )
    assert (
        model.decide(math.log1p(-0.01), current=PositionIntent.SHORT)
        is PositionIntent.SHORT
    )
    assert (
        model.decide(math.log1p(0.006), current=PositionIntent.SHORT)
        is PositionIntent.FLAT
    )
    assert (
        model.decide(math.log1p(0.005), current=PositionIntent.SHORT)
        is PositionIntent.SHORT
    )


def test_cost_aware_controller_compares_log_forecast_as_simple_return() -> None:
    model = CostAwareForecastIntentController(
        ForecastIntentConfig(entry_threshold=0.005, exit_threshold=0.001),
        one_way_switch_cost=0.01,
    )

    assert (
        model.decide(math.log1p(0.01), current=PositionIntent.FLAT)
        is PositionIntent.FLAT
    )
    assert (
        model.decide(math.log1p(0.010000000000005), current=PositionIntent.FLAT)
        is PositionIntent.LONG
    )
    assert (
        model.decide(math.log1p(-0.01), current=PositionIntent.LONG)
        is PositionIntent.LONG
    )
    assert (
        model.decide(math.log1p(-0.0101), current=PositionIntent.LONG)
        is PositionIntent.SHORT
    )


def test_cost_aware_controller_prices_entry_and_reversal_by_intent_distance() -> None:
    model = CostAwareForecastIntentController(
        ForecastIntentConfig(entry_threshold=0.10, exit_threshold=0.02),
        one_way_switch_cost=0.11,
    )

    assert (
        model.decide(math.log1p(0.20), current=PositionIntent.FLAT)
        is PositionIntent.LONG
    )
    assert (
        model.decide(math.log1p(-0.20), current=PositionIntent.FLAT)
        is PositionIntent.SHORT
    )
    assert (
        model.decide(math.log1p(-0.20), current=PositionIntent.LONG)
        is PositionIntent.SHORT
    )
    assert (
        model.decide(math.log1p(0.20), current=PositionIntent.SHORT)
        is PositionIntent.LONG
    )

    exact_cost = CostAwareForecastIntentController(
        ForecastIntentConfig(entry_threshold=0.10, exit_threshold=0.02),
        one_way_switch_cost=0.20,
    )
    assert (
        exact_cost.decide(math.log1p(0.20), current=PositionIntent.FLAT)
        is PositionIntent.FLAT
    )
    assert (
        exact_cost.decide(math.log1p(-0.20), current=PositionIntent.LONG)
        is PositionIntent.LONG
    )


def test_cost_aware_controller_nonfinite_forecast_still_fails_closed() -> None:
    model = CostAwareForecastIntentController(
        ForecastIntentConfig(entry_threshold=0.10, exit_threshold=0.02),
        one_way_switch_cost=1.0,
    )
    for value in (math.nan, math.inf, -math.inf):
        assert model.decide(value, current=PositionIntent.LONG) is PositionIntent.FLAT


@pytest.mark.parametrize("switch_cost", (-0.01, math.inf, math.nan, True))
def test_cost_aware_controller_rejects_invalid_switch_cost(switch_cost: float) -> None:
    with pytest.raises(ValueError, match="switch cost"):
        CostAwareForecastIntentController(
            ForecastIntentConfig(entry_threshold=0.10, exit_threshold=0.02),
            one_way_switch_cost=switch_cost,
        )


@pytest.mark.parametrize(
    ("entry", "exit_threshold"),
    ((0.0, 0.0), (0.1, -0.01), (0.1, 0.1), (0.05, 0.10)),
)
def test_forecast_controller_rejects_invalid_hysteresis(
    entry: float,
    exit_threshold: float,
) -> None:
    with pytest.raises(ValueError):
        ForecastIntentConfig(entry_threshold=entry, exit_threshold=exit_threshold)
