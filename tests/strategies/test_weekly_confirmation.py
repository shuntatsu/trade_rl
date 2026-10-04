from dataclasses import replace

import numpy as np
import pytest

from trade_rl.strategies.interface import StrategyObservation
from trade_rl.strategies.position_intent import PositionIntent
from trade_rl.strategies.rules.adaptive import (
    AdaptiveProfitConfig,
    RegimeAdaptiveStrategy,
)
from trade_rl.strategies.rules.weekly_confirmation import WeeklyConfirmationStrategy


def _observation(signal=0.03, weekly=(1.1, 1.2, 0.8, 0.05, 0.10, 0.1, 0.2)):
    values = np.asarray((signal, *weekly), dtype=np.float32)
    return StrategyObservation(
        index=0,
        timestamp=np.datetime64("2024-01-01"),
        symbol="ANY",
        features=values,
        feature_available=np.ones(8, dtype=bool),
        global_features=np.zeros(1),
        global_feature_available=np.ones(1, dtype=bool),
        current_intent=PositionIntent.FLAT,
        current_weight=0.0,
    )


def _strategy():
    return WeeklyConfirmationStrategy(
        RegimeAdaptiveStrategy(
            AdaptiveProfitConfig(
                signal_index=0, volatility_index=0, volatility_regime_threshold=0
            )
        ),
        tuple(range(1, 8)),
    )


def test_upper_band_touch_keeps_strong_weekly_uptrend():
    assert _strategy().decide(_observation()) is PositionIntent.LONG


@pytest.mark.parametrize("side", [-1, 1])
def test_cloud_direction_and_bb_tenkan_rejection_are_symmetric(side):
    weekly = np.asarray((0.5, 1.1, 0.2, -0.01, 0.1, 0.1, 0.2))
    if side == -1:
        weekly = -weekly[[0, 2, 1, 3, 4, 6, 5]]
    observation = _observation(side * 0.03, tuple(weekly))
    assert _strategy().decide(observation) is PositionIntent.FLAT
    weekly[1 if side == 1 else 2] = side * 0.8
    assert _strategy().decide(_observation(side * 0.03, tuple(weekly))) is (
        PositionIntent.LONG if side == 1 else PositionIntent.SHORT
    )
    weekly[5:7] = -side * 0.1
    assert (
        _strategy().decide(_observation(side * 0.03, tuple(weekly)))
        is PositionIntent.FLAT
    )


def test_missing_nonfinite_weekly_input_and_invalid_indices_fail_closed():
    observation = _observation()
    missing = observation.feature_available.copy()
    missing[4] = False
    assert (
        _strategy().decide(replace(observation, feature_available=missing))
        is PositionIntent.FLAT
    )
    values = observation.features.copy()
    values[4] = np.nan
    assert (
        _strategy().decide(replace(observation, features=values)) is PositionIntent.FLAT
    )
    with pytest.raises(ValueError):
        WeeklyConfirmationStrategy(_strategy(), (0, 1))


def test_protective_exit_latch_survives_weekly_mask_and_price_recovery():
    adaptive = RegimeAdaptiveStrategy(
        AdaptiveProfitConfig(
            signal_index=0,
            volatility_index=0,
            volatility_regime_threshold=0,
            stop_loss_threshold=0.01,
        )
    )
    strategy = WeeklyConfirmationStrategy(adaptive, tuple(range(1, 8)))
    observation = replace(
        _observation(),
        current_position_quantity=1.0,
        gross_position_return=-0.02,
        position_age_bars=1,
    )
    assert strategy.decide(observation) is PositionIntent.FLAT
    assert strategy.protective_exit_pending
    recovered = replace(
        observation,
        index=1,
        position_age_bars=2,
        gross_position_return=0.02,
        feature_available=np.zeros(8, dtype=bool),
    )
    assert strategy.decide(recovered) is PositionIntent.FLAT
    assert strategy.protective_exit_pending
