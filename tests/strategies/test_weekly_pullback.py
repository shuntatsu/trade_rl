from dataclasses import replace

import numpy as np
import pytest

from tests.strategies.test_weekly_confirmation import _observation
from trade_rl.strategies.controls import ConstantIntentStrategy
from trade_rl.strategies.position_intent import PositionIntent
from trade_rl.strategies.rules.adaptive import (
    AdaptiveProfitConfig,
    RegimeAdaptiveStrategy,
)
from trade_rl.strategies.rules.weekly_confirmation import WeeklyConfirmationStrategy


def _pullback_observation(side, *, touch=True, short_term_weak=True):
    # Weekly cloud opposes the requested direction. The alternate package must
    # isolate recent band contact plus lower-timeframe weakness from permission.
    weekly = np.array((0.5, 1.1 if touch else 0.9, 0.2, 0.05, 0.1, -0.2, -0.1))
    if side == -1:
        weekly = -weekly[[0, 2, 1, 3, 4, 6, 5]]
    observation = _observation(side * 0.03, tuple(weekly))
    lower = -side * 0.02 if short_term_weak else side * 0.02
    return replace(
        observation,
        features=np.append(observation.features, lower),
        feature_available=np.ones(9, dtype=bool),
    )


@pytest.mark.parametrize("side", [-1, 1])
@pytest.mark.parametrize(
    "touch,weak", [(False, False), (False, True), (True, False), (True, True)]
)
def test_weekly_pullback_requires_both_band_touch_and_lower_timeframe_weakness(
    side, touch, weak
):
    requested = PositionIntent.LONG if side == 1 else PositionIntent.SHORT
    strategy = WeeklyConfirmationStrategy(
        ConstantIntentStrategy(requested), tuple(range(1, 8)), short_term_index=8
    )
    actual = strategy.decide(
        _pullback_observation(side, touch=touch, short_term_weak=weak)
    )
    assert actual is (PositionIntent.FLAT if touch and weak else requested)


@pytest.mark.parametrize("side", [-1, 1])
def test_weekly_pullback_touch_is_inclusive_but_zero_tenkan_distance_is_not_weak(side):
    requested = PositionIntent.LONG if side == 1 else PositionIntent.SHORT
    strategy = WeeklyConfirmationStrategy(
        ConstantIntentStrategy(requested), tuple(range(1, 8)), short_term_index=8
    )
    observation = _pullback_observation(side)
    values = observation.features.copy()
    values[2 if side == 1 else 3] = side
    assert strategy.decide(replace(observation, features=values)) is PositionIntent.FLAT
    values[8] = 0.0
    assert strategy.decide(replace(observation, features=values)) is requested


@pytest.mark.parametrize(
    "fault", ["missing_lower", "nonfinite_lower", "missing_weekly"]
)
def test_weekly_pullback_common_availability_and_lower_input_fail_closed(fault):
    strategy = WeeklyConfirmationStrategy(
        ConstantIntentStrategy(PositionIntent.LONG),
        tuple(range(1, 8)),
        short_term_index=8,
    )
    observation = _pullback_observation(1, touch=False)
    values = observation.features.copy()
    available = observation.feature_available.copy()
    if fault == "nonfinite_lower":
        values[8] = np.inf
    else:
        available[8 if fault == "missing_lower" else 6] = False
    assert (
        strategy.decide(
            replace(observation, features=values, feature_available=available)
        )
        is PositionIntent.FLAT
    )


@pytest.mark.parametrize("index", [True, -1, 1.0, "8", 4])
def test_weekly_pullback_index_requires_distinct_nonnegative_integer(index):
    with pytest.raises(ValueError, match="short-term"):
        WeeklyConfirmationStrategy(
            ConstantIntentStrategy(PositionIntent.LONG),
            tuple(range(1, 8)),
            short_term_index=index,
        )


def test_weekly_pullback_lower_index_out_of_range_is_rejected():
    strategy = WeeklyConfirmationStrategy(
        ConstantIntentStrategy(PositionIntent.LONG),
        tuple(range(1, 8)),
        short_term_index=9,
    )
    with pytest.raises(ValueError, match="out of range"):
        strategy.decide(_pullback_observation(1))


def test_weekly_pullback_preserves_protective_exit_with_missing_confirmation():
    strategy = WeeklyConfirmationStrategy(
        RegimeAdaptiveStrategy(
            AdaptiveProfitConfig(
                signal_index=0,
                volatility_index=0,
                volatility_regime_threshold=0,
                stop_loss_threshold=0.01,
            )
        ),
        tuple(range(1, 8)),
        short_term_index=8,
    )
    observation = replace(
        _pullback_observation(1),
        current_position_quantity=1.0,
        position_age_bars=1,
        gross_position_return=-0.02,
        feature_available=np.zeros(9, dtype=bool),
    )
    assert strategy.decide(observation) is PositionIntent.FLAT
    assert strategy.protective_exit_pending
    assert (
        strategy.decide(replace(observation, gross_position_return=0.02))
        is PositionIntent.FLAT
    )
    assert strategy.protective_exit_pending
