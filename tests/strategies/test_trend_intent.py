from __future__ import annotations

import numpy as np
import pytest

from trade_rl.strategies.interface import StrategyObservation
from trade_rl.strategies.position_intent import PositionIntent
from trade_rl.strategies.trend import TrendIntentConfig, TrendIntentStrategy


def observation(
    signal: float,
    *,
    current: PositionIntent = PositionIntent.FLAT,
    available: bool = True,
) -> StrategyObservation:
    return StrategyObservation(
        index=10,
        timestamp=np.datetime64("2026-01-01T10:00:00", "ns"),
        symbol="BTCUSDT",
        features=np.asarray([signal], dtype=np.float64),
        feature_available=np.asarray([available], dtype=np.bool_),
        global_features=np.asarray([0.0], dtype=np.float64),
        global_feature_available=np.asarray([True], dtype=np.bool_),
        current_intent=current,
        current_weight=0.0,
    )


def strategy() -> TrendIntentStrategy:
    return TrendIntentStrategy(
        TrendIntentConfig(
            signal_index=0,
            entry_threshold=0.10,
            exit_threshold=0.02,
        )
    )


def test_flat_position_enters_only_on_strong_trend() -> None:
    model = strategy()

    assert model.decide(observation(0.20)) is PositionIntent.LONG
    assert model.decide(observation(-0.20)) is PositionIntent.SHORT
    assert model.decide(observation(0.05)) is PositionIntent.FLAT


def test_existing_position_uses_hysteresis_for_hold_exit_and_reversal() -> None:
    model = strategy()

    assert (
        model.decide(observation(0.05, current=PositionIntent.LONG))
        is PositionIntent.LONG
    )
    assert (
        model.decide(observation(0.01, current=PositionIntent.LONG))
        is PositionIntent.FLAT
    )
    assert (
        model.decide(observation(-0.20, current=PositionIntent.LONG))
        is PositionIntent.SHORT
    )
    assert (
        model.decide(observation(-0.05, current=PositionIntent.SHORT))
        is PositionIntent.SHORT
    )
    assert (
        model.decide(observation(-0.01, current=PositionIntent.SHORT))
        is PositionIntent.FLAT
    )
    assert (
        model.decide(observation(0.20, current=PositionIntent.SHORT))
        is PositionIntent.LONG
    )


def test_unavailable_signal_fails_closed_to_flat() -> None:
    assert (
        strategy().decide(
            observation(0.20, current=PositionIntent.LONG, available=False)
        )
        is PositionIntent.FLAT
    )


@pytest.mark.parametrize(
    ("entry_threshold", "exit_threshold"),
    ((0.0, 0.0), (0.1, -0.01), (0.1, 0.1), (0.05, 0.10)),
)
def test_config_rejects_invalid_hysteresis(
    entry_threshold: float,
    exit_threshold: float,
) -> None:
    with pytest.raises(ValueError):
        TrendIntentConfig(
            signal_index=0,
            entry_threshold=entry_threshold,
            exit_threshold=exit_threshold,
        )
