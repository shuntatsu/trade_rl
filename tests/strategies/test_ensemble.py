from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from trade_rl.strategies.interface import StrategyObservation
from trade_rl.strategies.position_intent import PositionIntent
from trade_rl.strategies.rules.ensemble import EnsembleIntentStrategy


@dataclass
class FixedIntent:
    intent: PositionIntent

    def decide(self, observation: StrategyObservation) -> PositionIntent:
        del observation
        return self.intent


def _observation() -> StrategyObservation:
    return StrategyObservation(
        index=0,
        timestamp=np.datetime64("2026-01-01T00:00:00", "ns"),
        symbol="BTCUSDT",
        features=np.zeros(1),
        feature_available=np.ones(1, dtype=np.bool_),
        global_features=np.zeros(1),
        global_feature_available=np.ones(1, dtype=np.bool_),
        current_intent=PositionIntent.FLAT,
        current_weight=0.0,
    )


@pytest.mark.parametrize("min_agreement", (True, 1.5, 2.0))
def test_ensemble_rejects_non_integer_minimum_agreement(
    min_agreement: object,
) -> None:
    with pytest.raises(ValueError, match="integer"):
        EnsembleIntentStrategy(
            (FixedIntent(PositionIntent.LONG), FixedIntent(PositionIntent.FLAT)),
            min_agreement=min_agreement,  # type: ignore[arg-type]
        )


def test_ensemble_uses_integer_vote_threshold() -> None:
    ensemble = EnsembleIntentStrategy(
        (FixedIntent(PositionIntent.LONG), FixedIntent(PositionIntent.FLAT)),
        min_agreement=2,
    )

    assert ensemble.min_agreement == 2
    assert ensemble.decide(_observation()) is PositionIntent.FLAT
