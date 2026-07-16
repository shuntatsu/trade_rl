"""Causal market-risk triggers that can bypass ordinary rebalance controls."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from trade_rl.data.market import MarketDataset


@dataclass(frozen=True, slots=True)
class EmergencyRiskConfig:
    stop_loss_return: float = 0.0
    stop_loss_hours: float = 1.0
    gap_return: float = 0.0
    volatility_ratio: float = 0.0
    volatility_short_hours: float = 24.0
    volatility_long_hours: float = 720.0
    flatten_untradable: bool = False

    def __post_init__(self) -> None:
        for name, value in (
            ("stop_loss_return", self.stop_loss_return),
            ("stop_loss_hours", self.stop_loss_hours),
            ("gap_return", self.gap_return),
            ("volatility_ratio", self.volatility_ratio),
            ("volatility_short_hours", self.volatility_short_hours),
            ("volatility_long_hours", self.volatility_long_hours),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.stop_loss_return > 1.0 or self.gap_return > 1.0:
            raise ValueError("emergency return thresholds must not exceed one")
        if self.stop_loss_return > 0.0 and self.stop_loss_hours <= 0.0:
            raise ValueError("enabled stop loss requires a positive window")
        if self.volatility_ratio > 0.0 and (
            self.volatility_short_hours <= 0.0
            or self.volatility_long_hours <= self.volatility_short_hours
        ):
            raise ValueError("volatility windows must be positive and ordered")
        if not isinstance(self.flatten_untradable, bool):
            raise ValueError("flatten_untradable must be a boolean")


@dataclass(frozen=True, slots=True)
class EmergencyRiskAssessment:
    flatten_mask: np.ndarray
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        mask = np.asarray(self.flatten_mask, dtype=np.bool_).reshape(-1).copy()
        mask.setflags(write=False)
        object.__setattr__(self, "flatten_mask", mask)


class CausalEmergencyRiskMonitor:
    """Assess only completed observations at or before ``index``."""

    def __init__(self, config: EmergencyRiskConfig | None = None) -> None:
        self.config = config or EmergencyRiskConfig()

    def assess(
        self,
        dataset: MarketDataset,
        *,
        index: int,
        weights: np.ndarray,
    ) -> EmergencyRiskAssessment:
        if not 0 <= index < dataset.n_bars:
            raise ValueError("emergency assessment index is outside dataset")
        position = np.asarray(weights, dtype=np.float64).reshape(-1)
        if position.shape != (dataset.n_symbols,) or not np.isfinite(position).all():
            raise ValueError("emergency weights do not match dataset symbols")
        mask = np.zeros(dataset.n_symbols, dtype=np.bool_)
        reasons: list[str] = []

        if self.config.flatten_untradable:
            unavailable = ~dataset.tradable[index]
            if dataset.asset_active is not None:
                unavailable |= ~dataset.asset_active[index]
            for symbol_index in np.flatnonzero(unavailable):
                mask[symbol_index] = True
                reasons.append(f"untradable:{dataset.symbols[symbol_index]}")

        if self.config.stop_loss_return > 0.0:
            window = max(1, dataset.bars_for_hours(self.config.stop_loss_hours))
            if index >= window:
                horizon_return = (
                    dataset.close[index] / dataset.close[index - window] - 1.0
                )
                signed_return = np.sign(position) * horizon_return
                triggered = signed_return <= -self.config.stop_loss_return
                for symbol_index in np.flatnonzero(
                    triggered & (np.abs(position) > 0.0)
                ):
                    mask[symbol_index] = True
                    reasons.append(f"stop_loss:{dataset.symbols[symbol_index]}")

        if self.config.gap_return > 0.0 and index > 0:
            gap = dataset.open[index] / dataset.close[index - 1] - 1.0
            triggered = np.abs(gap) >= self.config.gap_return
            for symbol_index in np.flatnonzero(triggered):
                mask[symbol_index] = True
                reasons.append(f"gap:{dataset.symbols[symbol_index]}")

        if self.config.volatility_ratio > 0.0:
            short = max(2, dataset.bars_for_hours(self.config.volatility_short_hours))
            long = max(
                short + 1, dataset.bars_for_hours(self.config.volatility_long_hours)
            )
            if index >= long:
                log_prices = np.log(dataset.close[index - long : index + 1])
                returns = np.diff(log_prices, axis=0)
                short_vol = np.std(returns[-short:], axis=0)
                long_vol = np.std(returns, axis=0)
                triggered = short_vol >= self.config.volatility_ratio * np.maximum(
                    long_vol, 1e-12
                )
                for symbol_index in np.flatnonzero(triggered):
                    mask[symbol_index] = True
                    reasons.append(f"volatility_spike:{dataset.symbols[symbol_index]}")

        return EmergencyRiskAssessment(
            flatten_mask=mask,
            reasons=tuple(dict.fromkeys(reasons)),
        )


__all__ = [
    "CausalEmergencyRiskMonitor",
    "EmergencyRiskAssessment",
    "EmergencyRiskConfig",
]
