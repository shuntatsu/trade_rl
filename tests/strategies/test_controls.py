from __future__ import annotations

import numpy as np
import pytest

from trade_rl.strategies.controls import ConstantIntentStrategy
from trade_rl.strategies.interface import StrategyObservation
from trade_rl.strategies.position_intent import PositionIntent


def observation() -> StrategyObservation:
    return StrategyObservation(
        index=0,
        timestamp=np.datetime64("2026-01-01", "ns"),
        symbol="BTCUSDT",
        features=np.array([0.0]),
        feature_available=np.array([False]),
        global_features=np.array([0.0]),
        global_feature_available=np.array([False]),
        current_intent=PositionIntent.FLAT,
        current_weight=0.0,
    )


@pytest.mark.parametrize(
    "intent", (PositionIntent.FLAT, PositionIntent.LONG, PositionIntent.SHORT)
)
def test_constant_intent_control_ignores_market_features(
    intent: PositionIntent,
) -> None:
    assert ConstantIntentStrategy(intent).decide(observation()) is intent


def test_constant_intent_control_rejects_non_intent() -> None:
    with pytest.raises(TypeError):
        ConstantIntentStrategy(1)  # type: ignore[arg-type]
