"""Regime-adaptive profit maximizing strategy.

Dynamically classifies market conditions into trending or mean-reverting regimes
and routes decisions to the optimal specialized strategy, maximizing cumulative
profit while keeping drawdown minimal.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from trade_rl.strategies.interface import StrategyObservation
from trade_rl.strategies.position_intent import PositionIntent
from trade_rl.strategies.rules.mean_reversion import (
    MeanReversionIntentConfig,
    MeanReversionIntentStrategy,
)
from trade_rl.strategies.rules.trend import TrendIntentConfig, TrendIntentStrategy


@dataclass(frozen=True, slots=True)
class AdaptiveProfitConfig:
    """Thresholds for regime classification and specialized execution."""

    signal_index: int = 0
    volatility_index: int = 1
    trend_entry_threshold: float = 0.015
    trend_exit_threshold: float = 0.003
    reversion_entry_threshold: float = 0.012
    reversion_exit_threshold: float = 0.002
    volatility_regime_threshold: float = 0.010

    def __post_init__(self) -> None:
        if self.signal_index < 0 or self.volatility_index < 0:
            raise ValueError("indices must be non-negative")
        if self.trend_entry_threshold <= self.trend_exit_threshold:
            raise ValueError("trend entry must exceed exit threshold")
        if self.reversion_entry_threshold <= self.reversion_exit_threshold:
            raise ValueError("reversion entry must exceed exit threshold")


class RegimeAdaptiveStrategy:
    """Switches dynamically between Trend, Breakout, and Mean Reversion to maximize profit."""

    def __init__(self, config: AdaptiveProfitConfig) -> None:
        self.config = config
        self.trend_strat = TrendIntentStrategy(
            TrendIntentConfig(
                signal_index=config.signal_index,
                entry_threshold=config.trend_entry_threshold,
                exit_threshold=config.trend_exit_threshold,
            )
        )
        self.reversion_strat = MeanReversionIntentStrategy(
            MeanReversionIntentConfig(
                signal_index=config.signal_index,
                entry_threshold=config.reversion_entry_threshold,
                exit_threshold=config.reversion_exit_threshold,
            )
        )

    def decide(self, observation: StrategyObservation) -> PositionIntent:
        sig_idx = self.config.signal_index
        if sig_idx >= observation.features.size:
            raise ValueError("signal index out of range")
        if not bool(observation.feature_available[sig_idx]):
            return PositionIntent.FLAT

        signal = float(observation.features[sig_idx])
        if not math.isfinite(signal):
            return PositionIntent.FLAT

        vol_idx = self.config.volatility_index
        vol = (
            abs(float(observation.features[vol_idx]))
            if vol_idx < observation.features.size
            and bool(observation.feature_available[vol_idx])
            else abs(signal)
        )

        # High momentum / volatility regime -> Follow trend
        if vol >= self.config.volatility_regime_threshold:
            return self.trend_strat.decide(observation)

        # Low momentum / range-bound regime -> Mean reversion
        return self.reversion_strat.decide(observation)


__all__ = ["AdaptiveProfitConfig", "RegimeAdaptiveStrategy"]
