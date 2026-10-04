"""Regime-adaptive directional intents and execution-based gross-return exits."""

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
    """Thresholds for regime classification and gross mark-to-fill exit triggers."""

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
        for field_name, integer_value in (
            ("signal_index", self.signal_index),
            ("volatility_index", self.volatility_index),
            ("max_holding_bars", self.max_holding_bars),
        ):
            if (
                isinstance(integer_value, bool)
                or not isinstance(integer_value, int)
                or integer_value < 0
            ):
                raise ValueError(f"{field_name} must be a non-negative integer")
        for field_name, threshold in (
            ("trend_entry_threshold", self.trend_entry_threshold),
            ("trend_exit_threshold", self.trend_exit_threshold),
            ("reversion_entry_threshold", self.reversion_entry_threshold),
            ("reversion_exit_threshold", self.reversion_exit_threshold),
            ("volatility_regime_threshold", self.volatility_regime_threshold),
            ("take_profit_threshold", self.take_profit_threshold),
            ("stop_loss_threshold", self.stop_loss_threshold),
            ("trailing_stop_threshold", self.trailing_stop_threshold),
        ):
            if (
                isinstance(threshold, bool)
                or not math.isfinite(threshold)
                or threshold < 0.0
            ):
                raise ValueError(f"{field_name} must be finite and non-negative")
        if self.trend_entry_threshold <= 0.0:
            raise ValueError("trend_entry_threshold must be positive")
        if self.reversion_entry_threshold <= 0.0:
            raise ValueError("reversion_entry_threshold must be positive")
        if self.trend_entry_threshold <= self.trend_exit_threshold:
            raise ValueError("trend entry must exceed exit threshold")
        if self.reversion_entry_threshold <= self.reversion_exit_threshold:
            raise ValueError("reversion entry must exceed exit threshold")


class RegimeAdaptiveStrategy:
    """Switch between directional rules and latch gross-return exit requests."""

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
        self._tracked_position_side: PositionIntent | None = None
        self._last_index: int = -1
        self._protective_exit_pending = False

    @property
    def protective_exit_pending(self) -> bool:
        """Whether an execution-based protective exit still needs a fill."""

        return self._protective_exit_pending

    def decide(self, observation: StrategyObservation) -> PositionIntent:
        sig_idx = self.config.signal_index
        gross_position_return = observation.gross_position_return
        if gross_position_return is not None:
            # The intent records the most recent target request and can be FLAT
            # while a missed/partial exit leaves the filled book invested.
            position_quantity = observation.current_position_quantity
            if position_quantity is None:
                # Keep direct StrategyObservation callers source-compatible;
                # canonical replay always supplies the exact filled quantity.
                position_quantity = observation.current_weight
            position_side = (
                PositionIntent.LONG
                if position_quantity > 0.0
                else PositionIntent.SHORT
                if position_quantity < 0.0
                else PositionIntent.FLAT
            )
            new_position = self._tracked_position_side is not position_side or (
                observation.index != self._last_index
                and observation.position_age_bars == 1
            )
            if new_position:
                self._reset_pnl_tracking()
                self._protective_exit_pending = False
                self._tracked_position_side = position_side
            if self._protective_exit_pending:
                return PositionIntent.FLAT
            if new_position or observation.index != self._last_index:
                self._unrealized_return = gross_position_return
                if self._unrealized_return > self._peak_unrealized_return:
                    self._peak_unrealized_return = self._unrealized_return
                self._last_index = observation.index

            if (
                self.config.take_profit_threshold > 0.0
                and self._unrealized_return >= self.config.take_profit_threshold
            ):
                self._protective_exit_pending = True
                return PositionIntent.FLAT

            if (
                self.config.stop_loss_threshold > 0.0
                and self._unrealized_return <= -self.config.stop_loss_threshold
            ):
                self._protective_exit_pending = True
                return PositionIntent.FLAT

            if (
                self.config.trailing_stop_threshold > 0.0
                and self._peak_unrealized_return > 0.0
                and (self._peak_unrealized_return - self._unrealized_return)
                >= self.config.trailing_stop_threshold
            ):
                self._protective_exit_pending = True
                return PositionIntent.FLAT

            if (
                self.config.max_holding_bars > 0
                and observation.position_age_bars >= self.config.max_holding_bars
            ):
                self._protective_exit_pending = True
                return PositionIntent.FLAT
        else:
            self._reset_pnl_tracking()
            self._protective_exit_pending = False
            self._last_index = observation.index

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
            decision = self.trend_strat.decide(observation)
        else:
            # Low momentum / range-bound regime -> Mean reversion
            decision = self.reversion_strat.decide(observation)

        return decision

    def _reset_pnl_tracking(self) -> None:
        self._unrealized_return = 0.0
        self._peak_unrealized_return = 0.0
        self._tracked_position_side = None


__all__ = ["AdaptiveProfitConfig", "RegimeAdaptiveStrategy"]
