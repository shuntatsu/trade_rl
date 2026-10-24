from __future__ import annotations

import math
from fractions import Fraction
from sys import float_info

import numpy as np
import pytest

from trade_rl.strategies.forecasts.controller import (
    CostAwareForecastIntentController,
    ForecastIntentConfig,
    ForecastIntentController,
)
from trade_rl.strategies.interface import StrategyObservation
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


def test_cost_aware_controller_uses_filled_position_when_budget_is_known() -> None:
    model = CostAwareForecastIntentController(
        ForecastIntentConfig(entry_threshold=0.10, exit_threshold=0.02),
        one_way_switch_cost=0.01,
    )

    assert (
        model.decide_for_position(
            math.log1p(0.005),
            current_weight=0.0,
            current_position_quantity=0.0,
            gross_budget=0.5,
        )
        is PositionIntent.FLAT
    )


@pytest.mark.parametrize(
    (
        "current_weight",
        "quantity",
        "current_intent",
        "same_intent_target_weight",
        "forecast",
        "expected",
    ),
    (
        (
            0.45,
            5.0,
            PositionIntent.LONG,
            0.45,
            math.log1p(-0.0095),
            PositionIntent.LONG,
        ),
        (
            -0.45,
            -5.0,
            PositionIntent.SHORT,
            -0.45,
            math.log1p(0.0095),
            PositionIntent.SHORT,
        ),
        (
            0.45,
            4.5,
            PositionIntent.LONG,
            0.5,
            math.log1p(-0.0095),
            PositionIntent.FLAT,
        ),
        (
            0.45,
            5.0,
            PositionIntent.SHORT,
            -0.5,
            math.log1p(0.03),
            PositionIntent.LONG,
        ),
        (
            1.2,
            12.0,
            PositionIntent.LONG,
            1.3,
            math.log1p(-0.009),
            PositionIntent.FLAT,
        ),
    ),
)
def test_cost_aware_controller_prices_cached_same_intent_target(
    current_weight: float,
    quantity: float,
    current_intent: PositionIntent,
    same_intent_target_weight: float,
    forecast: float,
    expected: PositionIntent,
) -> None:
    model = CostAwareForecastIntentController(
        ForecastIntentConfig(entry_threshold=0.10, exit_threshold=0.02),
        one_way_switch_cost=0.01,
    )

    assert (
        model.decide_for_position(
            forecast,
            current_weight=current_weight,
            current_position_quantity=quantity,
            gross_budget=0.5,
            current_intent=current_intent,
            same_intent_target_weight=same_intent_target_weight,
        )
        is expected
    )


def test_cost_aware_controller_requires_logical_intent_for_cached_target() -> None:
    model = CostAwareForecastIntentController(
        ForecastIntentConfig(entry_threshold=0.10, exit_threshold=0.02),
        one_way_switch_cost=0.01,
    )

    with pytest.raises(ValueError, match="current_intent"):
        model.decide_for_position(
            0.0,
            current_weight=0.0,
            current_position_quantity=0.0,
            gross_budget=0.5,
            same_intent_target_weight=0.5,
        )
    with pytest.raises(TypeError, match="current_intent"):
        model.decide_for_position(
            0.0,
            current_weight=0.0,
            current_position_quantity=0.0,
            gross_budget=0.5,
            current_intent="long",  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("target_weight", (math.nan, math.inf, -math.inf))
def test_cost_aware_controller_rejects_non_finite_cached_target(
    target_weight: float,
) -> None:
    model = CostAwareForecastIntentController(
        ForecastIntentConfig(entry_threshold=0.10, exit_threshold=0.02),
        one_way_switch_cost=0.01,
    )

    with pytest.raises(ValueError, match="same_intent_target_weight"):
        model.decide_for_position(
            0.0,
            current_weight=0.0,
            current_position_quantity=0.0,
            gross_budget=0.5,
            current_intent=PositionIntent.FLAT,
            same_intent_target_weight=target_weight,
        )


def test_cached_exposure_comparison_preserves_positive_gap_below_float_ulp() -> None:
    model = CostAwareForecastIntentController(
        ForecastIntentConfig(entry_threshold=0.10, exit_threshold=0.02),
        one_way_switch_cost=1.0,
    )

    assert (
        model.decide_for_position(
            -1000.0,
            current_weight=-0.5,
            current_position_quantity=-5.0,
            gross_budget=0.5,
            current_intent=PositionIntent.SHORT,
            same_intent_target_weight=-1e308,
        )
        is PositionIntent.SHORT
    )


@pytest.mark.parametrize(
    ("current_weight", "quantity", "gross_budget", "cost", "simple_return"),
    (
        (Fraction(0), Fraction(0), Fraction(1, 2), Fraction(1, 100), Fraction(1, 200)),
        (
            Fraction(1, 20),
            Fraction(1, 2),
            Fraction(1, 10),
            Fraction(1, 1000),
            Fraction(1, 250),
        ),
        (
            Fraction(-1, 20),
            Fraction(-1, 2),
            Fraction(1, 10),
            Fraction(1, 1000),
            Fraction(-1, 20),
        ),
        (
            Fraction(1, 10),
            Fraction(1),
            Fraction(1, 10),
            Fraction(1, 100),
            Fraction(-1, 50),
        ),
        (
            Fraction(-1, 10),
            Fraction(-1),
            Fraction(1, 10),
            Fraction(1, 100),
            Fraction(1, 50),
        ),
        (
            Fraction(3, 20),
            Fraction(1),
            Fraction(1, 10),
            Fraction(1, 20),
            Fraction(-1, 25),
        ),
        (
            Fraction(-3, 20),
            Fraction(-1),
            Fraction(1, 10),
            Fraction(1, 20),
            Fraction(1, 25),
        ),
        (Fraction(0), Fraction(0), Fraction(1, 10), Fraction(1, 50), Fraction(3, 25)),
        (Fraction(0), Fraction(0), Fraction(1, 10), Fraction(1, 50), Fraction(-3, 25)),
    ),
)
def test_filled_position_decision_matches_fraction_utility_oracle(
    current_weight: Fraction,
    quantity: Fraction,
    gross_budget: Fraction,
    cost: Fraction,
    simple_return: Fraction,
) -> None:
    config = ForecastIntentConfig(entry_threshold=0.10, exit_threshold=0.02)
    model = CostAwareForecastIntentController(
        config,
        one_way_switch_cost=float(cost),
    )
    forecast = math.log1p(float(simple_return))
    current_weight_value = float(current_weight)
    gross_budget_value = float(gross_budget)
    quantity_value = float(quantity)
    filled_side = (
        PositionIntent.LONG
        if quantity_value > 0.0
        else PositionIntent.SHORT
        if quantity_value < 0.0
        else PositionIntent.FLAT
    )
    proposed = ForecastIntentController(config).decide(
        forecast,
        current=filled_side,
    )
    candidates = tuple(dict.fromkeys((filled_side, proposed, PositionIntent.FLAT)))
    utilities = {
        intent: (
            int(intent) * gross_budget * simple_return
            - cost * abs(int(intent) * gross_budget - current_weight)
        )
        for intent in candidates
    }
    expected = max(candidates, key=utilities.__getitem__)

    assert (
        model.decide_for_position(
            forecast,
            current_weight=current_weight_value,
            current_position_quantity=quantity_value,
            gross_budget=gross_budget_value,
        )
        is expected
    )


def test_filled_position_ties_preserve_actual_side_and_zero_cost_matches_legacy() -> (
    None
):
    config = ForecastIntentConfig(entry_threshold=0.10, exit_threshold=0.02)
    zero_cost = CostAwareForecastIntentController(config, one_way_switch_cost=0.0)
    aligned_cases = (
        (PositionIntent.FLAT, 0.0, 0.0, 0.20),
        (PositionIntent.LONG, 0.10, 1.0, 0.01),
        (PositionIntent.LONG, 0.10, 1.0, -0.01),
        (PositionIntent.LONG, 0.10, 1.0, -0.20),
        (PositionIntent.SHORT, -0.10, -1.0, -0.01),
        (PositionIntent.SHORT, -0.10, -1.0, 0.01),
        (PositionIntent.SHORT, -0.10, -1.0, 0.20),
    )
    for current, weight, quantity, forecast in aligned_cases:
        assert zero_cost.decide_for_position(
            forecast,
            current_weight=weight,
            current_position_quantity=quantity,
            gross_budget=0.10,
        ) is zero_cost.decide(forecast, current=current)

    exact_cost = CostAwareForecastIntentController(config, one_way_switch_cost=0.01)
    assert (
        exact_cost.decide_for_position(
            math.log1p(-0.01),
            current_weight=0.10,
            current_position_quantity=1.0,
            gross_budget=0.10,
        )
        is PositionIntent.LONG
    )


def test_filled_position_cost_gate_uses_partial_weight_and_opposite_fill_side() -> None:
    config = ForecastIntentConfig(entry_threshold=0.10, exit_threshold=0.02)
    partial_long = CostAwareForecastIntentController(
        config,
        one_way_switch_cost=0.001,
    )
    assert (
        partial_long.decide_for_position(
            math.log1p(0.001),
            current_weight=0.05,
            current_position_quantity=0.5,
            gross_budget=0.5,
        )
        is PositionIntent.LONG
    )

    actual_short = CostAwareForecastIntentController(
        config,
        one_way_switch_cost=0.01,
    )
    assert (
        actual_short.decide_for_position(
            math.log1p(0.005),
            current_weight=-0.5,
            current_position_quantity=-5.0,
            gross_budget=0.5,
        )
        is PositionIntent.SHORT
    )


def test_filled_position_cost_gate_handles_extreme_finite_inputs() -> None:
    model = CostAwareForecastIntentController(
        ForecastIntentConfig(entry_threshold=0.10, exit_threshold=0.02),
        one_way_switch_cost=float_info.max,
    )

    assert (
        model.decide_for_position(
            1e308,
            current_weight=0.0,
            current_position_quantity=0.0,
            gross_budget=1.0,
        )
        is PositionIntent.LONG
    )
    assert (
        model.decide_for_position(
            -1e308,
            current_weight=float_info.max,
            current_position_quantity=1.0,
            gross_budget=1e-300,
        )
        is PositionIntent.LONG
    )


@pytest.mark.parametrize("gross_budget", (0.0, -0.1, 1.0001, math.inf, math.nan))
def test_filled_position_decision_rejects_invalid_budget(
    gross_budget: float,
) -> None:
    model = CostAwareForecastIntentController(
        ForecastIntentConfig(entry_threshold=0.10, exit_threshold=0.02),
        one_way_switch_cost=0.01,
    )

    with pytest.raises(ValueError, match="gross_budget"):
        model.decide_for_position(
            0.0,
            current_weight=0.0,
            current_position_quantity=0.0,
            gross_budget=gross_budget,
        )


def test_strategy_observation_validates_optional_intent_gross_budget() -> None:
    def build_observation(
        intent_gross_budget: float | None,
        same_intent_target_weight: float | None = None,
    ) -> StrategyObservation:
        return StrategyObservation(
            index=0,
            timestamp=np.datetime64("2026-01-01T00:00:00", "ns"),
            symbol="SYNTH",
            features=np.asarray([0.0]),
            feature_available=np.asarray([True]),
            global_features=np.asarray([0.0]),
            global_feature_available=np.asarray([True]),
            current_intent=PositionIntent.FLAT,
            current_weight=0.0,
            intent_gross_budget=intent_gross_budget,
            same_intent_target_weight=same_intent_target_weight,
        )

    assert build_observation(None).intent_gross_budget is None
    assert build_observation(0.5).intent_gross_budget == 0.5
    with pytest.raises(ValueError, match="intent_gross_budget"):
        build_observation(0.0)
    assert build_observation(0.5, 0.45).same_intent_target_weight == 0.45
    with pytest.raises(ValueError, match="intent_gross_budget"):
        build_observation(None, 0.45)
    for invalid in (math.nan, math.inf, -math.inf):
        with pytest.raises(ValueError, match="same_intent_target_weight"):
            build_observation(0.5, invalid)


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
