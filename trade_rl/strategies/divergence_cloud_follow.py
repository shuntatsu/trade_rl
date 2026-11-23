"""Research-only trend-following policies using the existing causal MTF pack.

C2 keeps daily visible-cloud direction with a six-hour entry confirmation.
C3 adds opposing STRICT 4h divergence exits and a 24-hour re-entry cooldown.
Both are explicit opt-in strategies, not yet economically validated, with
every order passed through the native risk/execution/accounting owner.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

from trade_rl.data.features.divergence_mtf import DivergenceCloudMtfPack
from trade_rl.strategies.portfolio import (
    PortfolioDecisionSnapshot,
    PortfolioTargetProposal,
)

CloudFollowMode = Literal["confirmed_cloud", "divergence_exit_cooldown"]


@dataclass(frozen=True, slots=True)
class CloudFollowConfig:
    mode: CloudFollowMode
    target_abs_weight: float = 0.25
    entry_confirmation_hours: int = 6
    divergence_exit_cooldown_hours: int = 24

    def __post_init__(self) -> None:
        if self.mode not in ("confirmed_cloud", "divergence_exit_cooldown"):
            raise ValueError("unsupported cloud-follow research mode")
        if (
            isinstance(self.target_abs_weight, bool)
            or not isinstance(self.target_abs_weight, (int, float))
            or not math.isfinite(self.target_abs_weight)
            or not 0 < self.target_abs_weight <= 1
        ):
            raise ValueError("target_abs_weight must be finite within (0, 1]")
        for key, low, high in (
            ("entry_confirmation_hours", 1, 24),
            ("divergence_exit_cooldown_hours", 0, 240),
        ):
            value = getattr(self, key)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"{key} must be a native integer in [{low}, {high}]")


class CloudFollowPolicy:
    """One-symbol PortfolioPolicy; instantiate a fresh policy for each replay."""

    def __init__(
        self,
        *,
        symbol: str,
        signals: DivergenceCloudMtfPack,
        config: CloudFollowConfig,
    ) -> None:
        if type(symbol) is not str or not symbol.strip():
            raise ValueError("symbol must be nonempty")
        if not isinstance(signals, DivergenceCloudMtfPack):
            raise TypeError("signals must be a causal DivergenceCloudMtfPack")
        if not isinstance(config, CloudFollowConfig):
            raise TypeError("config must be CloudFollowConfig")
        self._symbol = symbol
        self._signals = signals
        self._config = config
        self._cooldown_until = -1
        self._last_decision_index = -1

    @staticmethod
    def _hold(snapshot: PortfolioDecisionSnapshot) -> PortfolioTargetProposal:
        return PortfolioTargetProposal(
            target_weights=(float(snapshot.account.weights[0]),),
            hold_current_quantity_symbols=(0,),
        )

    def decide_portfolio(
        self, snapshot: PortfolioDecisionSnapshot
    ) -> PortfolioTargetProposal:
        if not isinstance(snapshot, PortfolioDecisionSnapshot):
            raise TypeError("snapshot must be PortfolioDecisionSnapshot")
        if snapshot.symbols != (self._symbol,):
            raise ValueError("snapshot symbols must match the strategy symbol")
        index = snapshot.index
        signals = self._signals
        if index < 0 or index >= len(signals.timestamps):
            raise ValueError("index outside signal history")
        timestamp_ns = int(
            signals.timestamps[index].astype("datetime64[ns]").astype(np.int64)
        )
        if snapshot.timestamp_ns != timestamp_ns:
            raise ValueError("snapshot timestamp differs from signal timestamp")
        if index <= self._last_decision_index:
            raise ValueError("decision indexes must increase monotonically")
        self._last_decision_index = index

        if (
            snapshot.account.insolvent
            or snapshot.account.termination_reason is not None
        ):
            return self._hold(snapshot)

        regime = (
            int(signals.cloud_state[index]) if bool(signals.cloud_ready[index]) else 0
        )
        if regime not in (-1, 0, 1):
            raise ValueError("cloud regime must be in -1, 0, 1")
        quantity = float(snapshot.account.quantities[0])
        current_direction = int(np.sign(quantity))

        if current_direction != 0:
            if regime != current_direction:
                return PortfolioTargetProposal((0.0,))
            if self._config.mode == "divergence_exit_cooldown":
                opposing = (
                    signals.bear_4h[index]
                    if current_direction > 0
                    else signals.bull_4h[index]
                )
                if int(opposing) > 0:
                    self._cooldown_until = (
                        index + self._config.divergence_exit_cooldown_hours
                    )
                    return PortfolioTargetProposal((0.0,))
            return self._hold(snapshot)

        if snapshot.active_orders or regime == 0 or index < self._cooldown_until:
            return self._hold(snapshot)

        confirmation = self._config.entry_confirmation_hours
        start = index - confirmation + 1
        if start < 0:
            return self._hold(snapshot)
        if not (
            np.all(signals.cloud_ready[start : index + 1])
            and np.all(signals.cloud_state[start : index + 1] == regime)
        ):
            return self._hold(snapshot)

        weight = min(
            float(self._config.target_abs_weight),
            max(float(snapshot.risk.max_gross), 0.0),
            max(float(snapshot.risk.max_abs_weight), 0.0),
        )
        if weight <= 0:
            return self._hold(snapshot)
        return PortfolioTargetProposal((regime * weight,))


__all__ = ["CloudFollowConfig", "CloudFollowPolicy"]
