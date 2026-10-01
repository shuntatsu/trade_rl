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
    """Thresholds for regime classification, specialized execution, and profit management."""

    signal_index: int = 0
    volatility_index: int = 1
    trend_entry_threshold: float = 0.015
    trend_exit_threshold: float = 0.003
    reversion_entry_threshold: float = 0.012
    reversion_exit_threshold: float = 0.002
    volatility_regime_threshold: float = 0.010
    take_profit_threshold: float = 0.0
    stop_loss_threshold: float = 0.0
    trailing_stop_threshold: float = 0.0
    max_holding_bars: int = 0

    def __post_init__(self) -> None:
        if self.signal_index < 0 or self.volatility_index < 0:
            raise ValueError("indices must be non-negative")
        if self.trend_entry_threshold <= self.trend_exit_threshold:
            raise ValueError("trend entry must exceed exit threshold")
        if self.reversion_entry_threshold <= self.reversion_exit_threshold:
            raise ValueError("reversion entry must exceed exit threshold")
        if self.take_profit_threshold < 0.0:
            raise ValueError("take_profit_threshold must be non-negative")
        if self.stop_loss_threshold < 0.0:
            raise ValueError("stop_loss_threshold must be non-negative")
        if self.trailing_stop_threshold < 0.0:
            raise ValueError("trailing_stop_threshold must be non-negative")
        if self.max_holding_bars < 0:
            raise ValueError("max_holding_bars must be non-negative")


class RegimeAdaptiveStrategy:
    """Switches dynamically between Trend, Breakout, and Mean Reversion with active profit maximization."""

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
        self._unrealized_return: float = 0.0
        self._peak_unrealized_return: float = 0.0
        self._last_index: int = -1

    def decide(self, observation: StrategyObservation) -> PositionIntent:
        sig_idx = self.config.signal_index
        if sig_idx >= observation.features.size:
            raise ValueError("signal index out of range")
        if not bool(observation.feature_available[sig_idx]):
            self._reset_pnl_tracking()
            return PositionIntent.FLAT

        signal = float(observation.features[sig_idx])
        if not math.isfinite(signal):
            self._reset_pnl_tracking()
            return PositionIntent.FLAT

        # Track P&L of active position if holding
        if (
            observation.current_intent is not PositionIntent.FLAT
            and observation.position_age_bars > 0
        ):
            if observation.index != self._last_index:
                # Accumulate bar return proxy
                bar_ret = signal
                if observation.current_intent is PositionIntent.SHORT:
                    bar_ret = -bar_ret
                self._unrealized_return += bar_ret
                if self._unrealized_return > self._peak_unrealized_return:
                    self._peak_unrealized_return = self._unrealized_return
                self._last_index = observation.index

            # 1. Take Profit check: lock in gains when target is reached
            if (
                self.config.take_profit_threshold > 0.0
                and self._unrealized_return >= self.config.take_profit_threshold
            ):
                self._reset_pnl_tracking()
                return PositionIntent.FLAT

            # 2. Stop Loss check: cut losses quickly
            if (
                self.config.stop_loss_threshold > 0.0
                and self._unrealized_return <= -self.config.stop_loss_threshold
            ):
                self._reset_pnl_tracking()
                return PositionIntent.FLAT

            # 3. Trailing Stop check: protect unrealized profit from reversal
            if (
                self.config.trailing_stop_threshold > 0.0
                and self._peak_unrealized_return > 0.0
                and (self._peak_unrealized_return - self._unrealized_return)
                >= self.config.trailing_stop_threshold
            ):
                self._reset_pnl_tracking()
                return PositionIntent.FLAT

            # 4. Max Holding Bars check: prevent capital lockup in stagnant trades
            if (
                self.config.max_holding_bars > 0
                and observation.position_age_bars >= self.config.max_holding_bars
            ):
                self._reset_pnl_tracking()
                return PositionIntent.FLAT
        else:
            self._reset_pnl_tracking()
            self._last_index = observation.index

        vol_idx = self.config.volatility_index
        vol = (
            abs(float(observation.features[vol_idx]))
            if vol_idx < observation.features.size
            and bool(observation.feature_available[vol_idx])
            else abs(signal)
        )

        # High momentum / volatility regime -> Follow trend
        if vol >= self.config.volatility_regime_threshold:
            decision = self.trend_strat.decide(observation)
        else:
            # Low momentum / range-bound regime -> Mean reversion
            decision = self.reversion_strat.decide(observation)

        if decision is PositionIntent.FLAT:
            self._reset_pnl_tracking()

        return decision

    def _reset_pnl_tracking(self) -> None:
        self._unrealized_return = 0.0
        self._peak_unrealized_return = 0.0


__all__ = ["AdaptiveProfitConfig", "RegimeAdaptiveStrategy"]
