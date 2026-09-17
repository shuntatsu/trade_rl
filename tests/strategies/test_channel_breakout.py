import numpy as np

from trade_rl.strategies.interface import StrategyObservation
from trade_rl.strategies.position_intent import PositionIntent as Intent
from trade_rl.strategies.rules import channel_breakout


def observation(
    values: list[float], current: Intent, available: bool = True
) -> StrategyObservation:
    return StrategyObservation(
        index=1,
        timestamp=np.datetime64("2023-01-01", "ns"),
        symbol="BTCUSDT",
        features=np.array(values),
        feature_available=np.full(4, available),
        global_features=np.zeros(1),
        global_feature_available=np.ones(1, dtype=bool),
        current_intent=current,
        current_weight=0.0,
    )


def test_breakout_hold_exit_reversal_and_missing_input() -> None:
    assert hasattr(channel_breakout, "ChannelBreakoutStrategy")
    strategy = channel_breakout.ChannelBreakoutStrategy((0, 1, 2, 3))
    assert (
        strategy.decide(observation([0.01, 0.2, 0.02, 0.1], Intent.FLAT)) is Intent.LONG
    )
    assert (
        strategy.decide(observation([-0.01, 0.2, 0.02, 0.1], Intent.LONG))
        is Intent.LONG
    )
    assert (
        strategy.decide(observation([-0.1, 0.1, -0.02, -0.01], Intent.LONG))
        is Intent.FLAT
    )
    assert (
        strategy.decide(observation([-0.1, -0.02, -0.1, -0.02], Intent.LONG))
        is Intent.SHORT
    )
    assert (
        strategy.decide(observation([0.01, 0.2, 0.02, 0.1], Intent.LONG, False))
        is Intent.FLAT
    )
