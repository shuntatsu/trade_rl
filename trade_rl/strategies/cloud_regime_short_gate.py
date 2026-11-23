"""C2 wrapper that conditionally permits or exits short exposure.

The filter is explicitly opt-in research and only observes the current bar's
causal context. Native shared portfolio risk/execution/book owns all P&L.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from trade_rl.data.features.cloud_regime_context import CloudRegimeContext
from trade_rl.data.features.divergence_mtf import DivergenceCloudMtfPack
from trade_rl.strategies.divergence_cloud_follow import (
    CloudFollowConfig,
    CloudFollowPolicy,
)
from trade_rl.strategies.portfolio import (
    PortfolioDecisionSnapshot,
    PortfolioTargetProposal,
)


@dataclass(frozen=True, slots=True)
class CloudRegimeShortGateConfig:
    mode: Literal["momentum", "momentum_and_cloud_slope"]

    def __post_init__(self) -> None:
        if self.mode not in ("momentum", "momentum_and_cloud_slope"):
            raise ValueError("unsupported registered short gate")


class CloudRegimeShortGatePolicy:
    """C2 with momentum/slope short eligibility, long decisions unchanged."""

    def __init__(
        self,
        *,
        symbol: str,
        signals: DivergenceCloudMtfPack,
        context: CloudRegimeContext,
        config: CloudRegimeShortGateConfig,
    ) -> None:
        if not isinstance(context, CloudRegimeContext):
            raise TypeError("context must be CloudRegimeContext")
        if not isinstance(config, CloudRegimeShortGateConfig):
            raise TypeError("config must be CloudRegimeShortGateConfig")
        if not np.array_equal(context.timestamps, signals.timestamps):
            raise ValueError("context and signal clocks must match")
        n = len(signals.timestamps)
        if any(
            getattr(context, k).shape != (n,)
            for k in (
                "ready",
                "momentum_30d",
                "cloud_mid_slope_7d",
                "distance_to_cloud_mid_pct",
                "realized_vol_30d",
            )
        ):
            raise ValueError("context must have one row per signal timestamp")
        self._base = CloudFollowPolicy(
            symbol=symbol,
            signals=signals,
            config=CloudFollowConfig(mode="confirmed_cloud"),
        )
        self._context = context
        self._config = config
        self.blocked_short_proposals = 0
        self.ineligible_short_flat_requests = 0

    def decide_portfolio(
        self, snapshot: PortfolioDecisionSnapshot
    ) -> PortfolioTargetProposal:
        proposal = self._base.decide_portfolio(snapshot)
        # Preserve the native account termination fail-closed exact HOLD.
        if (
            snapshot.account.insolvent
            or snapshot.account.termination_reason is not None
        ):
            return proposal
        i = snapshot.index
        context = self._context
        eligible = bool(
            context.ready[i]
            and context.momentum_30d[i] < 0
            and (self._config.mode == "momentum" or context.cloud_mid_slope_7d[i] < 0)
        )
        if eligible:
            return proposal
        if snapshot.account.quantities[0] < 0:
            self.ineligible_short_flat_requests += 1
            return PortfolioTargetProposal((0.0,))
        if proposal.target_weights[0] < 0:
            self.blocked_short_proposals += 1
            return PortfolioTargetProposal(
                (float(snapshot.account.weights[0]),),
                hold_current_quantity_symbols=(0,),
            )
        return proposal


__all__ = ["CloudRegimeShortGateConfig", "CloudRegimeShortGatePolicy"]
