from __future__ import annotations

import math

import pytest

from trade_rl.strategies.forecasts.controller import (
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
