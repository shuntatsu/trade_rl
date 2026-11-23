"""Opt-in 1h/4h divergence entries under a daily visible Ichimoku cloud.

Decisions are passed to the existing Risk/Execution/Accounting owner. In
unchanged regimes it requests exact-quantity HOLD instead of free rebalancing.
This module never places orders or starts an economic experiment.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from trade_rl.data.features.divergence_mtf import DivergenceCloudMtfPack
from trade_rl.strategies.portfolio import (
    PortfolioDecisionSnapshot,
    PortfolioTargetProposal,
)


@dataclass(frozen=True, slots=True)
class DivergenceCloudMtfConfig:
    target_abs_weight: float = 0.25
    confirm_1h_lookback: int = 4
    confirm_4h_lookback: int = 8
    require_new_signal: bool = True

    def __post_init__(self) -> None:
        if (
            isinstance(self.target_abs_weight, bool)
            or not isinstance(self.target_abs_weight, (int, float))
            or not math.isfinite(self.target_abs_weight)
            or not 0 < self.target_abs_weight <= 1
        ):
            raise ValueError("target_abs_weight must be finite in (0, 1]")
        for name in ("confirm_1h_lookback", "confirm_4h_lookback"):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= 24:
                raise ValueError(f"{name} must be an integer in [1, 24]")
        if type(self.require_new_signal) is not bool:
            raise ValueError("require_new_signal must be a boolean")


class DivergenceCloudMtfPolicy:
    """Single-symbol native shared-portfolio policy bound to 1h timestamps."""

    def __init__(
        self,
        *,
        symbol: str,
        signals: DivergenceCloudMtfPack,
        config: DivergenceCloudMtfConfig | None = None,
    ) -> None:
        if not isinstance(symbol, str) or not symbol:
            raise ValueError("symbol must be a non-empty string")
        if not isinstance(signals, DivergenceCloudMtfPack):
            raise TypeError("signals must be a DivergenceCloudMtfPack")
        self.symbol = symbol
        self.signals = signals
        self.config = config or DivergenceCloudMtfConfig()

    def decide_portfolio(
        self, snapshot: PortfolioDecisionSnapshot
    ) -> PortfolioTargetProposal:
        if snapshot.symbols != (self.symbol,):
            raise ValueError("strategy requires exactly its bound single symbol")
        i = snapshot.index
        pack = self.signals
        if i >= len(pack.timestamps) or i < 0:
            raise ValueError("decision is outside precomputed signal window")
        signal_time = int(pack.timestamps[i].astype("datetime64[ns]").astype(np.int64))
        if snapshot.timestamp_ns != signal_time:
            raise ValueError("signal clock and shared-account decision clock differ")

        quantity = snapshot.account.quantities[0]
        regime = int(pack.cloud_state[i]) if bool(pack.cloud_ready[i]) else 0
        if quantity > 0 and regime != 1:
            return PortfolioTargetProposal((0.0,))
        if quantity < 0 and regime != -1:
            return PortfolioTargetProposal((0.0,))
        if regime == 0:
            return (
                PortfolioTargetProposal((0.0,))
                if quantity != 0
                else self._hold(snapshot)
            )

        first_1h = max(0, i - self.config.confirm_1h_lookback + 1)
        first_4h = max(0, i - self.config.confirm_4h_lookback + 1)
        bull_ready = (
            np.max(pack.bull_1h[first_1h : i + 1]) > 0
            and np.max(pack.bull_4h[first_4h : i + 1]) > 0
        )
        bear_ready = (
            np.max(pack.bear_1h[first_1h : i + 1]) > 0
            and np.max(pack.bear_4h[first_4h : i + 1]) > 0
        )
        if self.config.require_new_signal:
            bull_ready = bull_ready and bool(pack.bull_1h[i] or pack.bull_4h[i])
            bear_ready = bear_ready and bool(pack.bear_1h[i] or pack.bear_4h[i])

        if quantity > 0:
            return (
                PortfolioTargetProposal((0.0,)) if bear_ready else self._hold(snapshot)
            )
        if quantity < 0:
            return (
                PortfolioTargetProposal((0.0,)) if bull_ready else self._hold(snapshot)
            )
        if snapshot.active_orders:
            return self._hold(snapshot)
        weight = min(
            self.config.target_abs_weight,
            max(snapshot.risk.max_abs_weight, 0.0),
            max(snapshot.risk.max_gross, 0.0),
        )
        if regime > 0 and bull_ready and not bear_ready and weight > 0:
            return PortfolioTargetProposal((float(weight),))
        if regime < 0 and bear_ready and not bull_ready and weight > 0:
            return PortfolioTargetProposal((-float(weight),))
        return self._hold(snapshot)

    @staticmethod
    def _hold(snapshot: PortfolioDecisionSnapshot) -> PortfolioTargetProposal:
        return PortfolioTargetProposal(
            (float(snapshot.account.weights[0]),),
            hold_current_quantity_symbols=(0,),
        )


__all__ = ["DivergenceCloudMtfConfig", "DivergenceCloudMtfPolicy"]
