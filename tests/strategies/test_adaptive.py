from __future__ import annotations

import numpy as np

from trade_rl.strategies.interface import StrategyObservation
from trade_rl.strategies.position_intent import PositionIntent
from trade_rl.strategies.rules.adaptive import (
    AdaptiveProfitConfig,
    RegimeAdaptiveStrategy,
)


def _make_obs(
    features: list[float],
    *,
    index: int = 0,
    current_intent: PositionIntent = PositionIntent.FLAT,
    current_weight: float = 0.0,
    position_age_bars: int = 0,
    gross_position_return: float | None = None,
) -> StrategyObservation:
    feat_arr = np.array(features, dtype=np.float32)
    feat_avail = np.ones(len(features), dtype=np.bool_)
    return StrategyObservation(
        index=index,
        timestamp=np.datetime64("2026-01-01T00:00:00", "ns"),
        symbol="CRYPTO_0",
        features=feat_arr,
        feature_available=feat_avail,
        global_features=np.zeros(1, dtype=np.float32),
        global_feature_available=np.ones(1, dtype=np.bool_),
        current_intent=current_intent,
        current_weight=current_weight,
        position_age_bars=position_age_bars,
        gross_position_return=gross_position_return,
    )


def test_adaptive_take_profit() -> None:
    cfg = AdaptiveProfitConfig(
        trend_entry_threshold=0.01,
        trend_exit_threshold=0.002,
        volatility_regime_threshold=0.005,
        take_profit_threshold=0.03,  # Take profit at +3%
    )
    strategy = RegimeAdaptiveStrategy(cfg)

    # First bar: strong momentum entry into LONG
    obs0 = _make_obs([0.02, 0.02], index=0, position_age_bars=0)
    decision0 = strategy.decide(obs0)
    assert decision0 is PositionIntent.LONG

    # Second bar: holding LONG, gained 2% (total +2%) -> below TP
    obs1 = _make_obs(
        [0.02, 0.02],
        index=1,
        current_intent=PositionIntent.LONG,
        current_weight=0.2,
        position_age_bars=1,
        gross_position_return=0.02,
    )
    decision1 = strategy.decide(obs1)
    assert decision1 is PositionIntent.LONG

    # Third bar: holding LONG, gained another 2% (total +4%) -> exceeds TP of 3%!
    obs2 = _make_obs(
        [0.02, 0.02],
        index=2,
        current_intent=PositionIntent.LONG,
        current_weight=0.2,
        position_age_bars=2,
        gross_position_return=0.04,
    )
    decision2 = strategy.decide(obs2)
    assert decision2 is PositionIntent.FLAT


def test_adaptive_stop_loss() -> None:
    cfg = AdaptiveProfitConfig(
        trend_entry_threshold=0.01,
        trend_exit_threshold=0.002,
        volatility_regime_threshold=0.005,
        stop_loss_threshold=0.02,  # Stop loss at -2%
    )
    strategy = RegimeAdaptiveStrategy(cfg)

    # Holding LONG, but market dropped 2.5%
    obs = _make_obs(
        [-0.025, 0.025],
        index=1,
        current_intent=PositionIntent.LONG,
        current_weight=0.2,
        position_age_bars=1,
        gross_position_return=-0.025,
    )
    decision = strategy.decide(obs)
    assert decision is PositionIntent.FLAT


def test_adaptive_trailing_stop() -> None:
    cfg = AdaptiveProfitConfig(
        trend_entry_threshold=0.01,
        trend_exit_threshold=0.002,
        volatility_regime_threshold=0.005,
        trailing_stop_threshold=0.015,  # Trailing stop 1.5% from peak
    )
    strategy = RegimeAdaptiveStrategy(cfg)

    # Holding LONG, price gained +3% (peak = 3%)
    obs1 = _make_obs(
        [0.03, 0.03],
        index=1,
        current_intent=PositionIntent.LONG,
        current_weight=0.2,
        position_age_bars=1,
        gross_position_return=0.03,
    )
    dec1 = strategy.decide(obs1)
    assert dec1 is PositionIntent.LONG

    # Next bar: price drops 2% (unrealized goes from 3% to 1%, drop = 2% >= 1.5% trailing stop threshold)
    obs2 = _make_obs(
        [-0.02, 0.03],
        index=2,
        current_intent=PositionIntent.LONG,
        current_weight=0.2,
        position_age_bars=2,
        gross_position_return=0.01,
    )
    dec2 = strategy.decide(obs2)
    assert dec2 is PositionIntent.FLAT
    assert strategy.protective_exit_pending

    still_open = _make_obs(
        [0.0, 0.0],
        index=3,
        current_intent=PositionIntent.LONG,
        current_weight=0.2,
        position_age_bars=3,
        gross_position_return=0.0,
    )
    assert strategy.decide(still_open) is PositionIntent.FLAT
    assert strategy.protective_exit_pending


def test_adaptive_trailing_peak_resets_on_direct_position_reversal() -> None:
    cfg = AdaptiveProfitConfig(
        trend_entry_threshold=0.01,
        trend_exit_threshold=0.002,
        volatility_regime_threshold=0.005,
        trailing_stop_threshold=0.015,
    )
    strategy = RegimeAdaptiveStrategy(cfg)

    prior_long = _make_obs(
        [0.02, 0.02],
        index=1,
        current_intent=PositionIntent.LONG,
        current_weight=0.2,
        position_age_bars=1,
        gross_position_return=0.04,
    )
    assert strategy.decide(prior_long) is PositionIntent.LONG

    reversed_short = _make_obs(
        [-0.02, 0.02],
        index=2,
        current_intent=PositionIntent.SHORT,
        current_weight=-0.2,
        position_age_bars=1,
        gross_position_return=0.0,
    )
    assert strategy.decide(reversed_short) is PositionIntent.SHORT
    assert not strategy.protective_exit_pending


def test_adaptive_max_holding_bars() -> None:
    cfg = AdaptiveProfitConfig(
        trend_entry_threshold=0.01,
        trend_exit_threshold=0.002,
        max_holding_bars=5,
    )
    strategy = RegimeAdaptiveStrategy(cfg)

    obs = _make_obs(
        [0.005, 0.005],
        index=5,
        current_intent=PositionIntent.LONG,
        current_weight=0.2,
        position_age_bars=5,
        gross_position_return=0.0,
    )
    decision = strategy.decide(obs)
    assert decision is PositionIntent.FLAT
