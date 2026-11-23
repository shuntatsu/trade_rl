"""Opt-in negative controls to falsify R2 short-gate information value.

N0 never takes short exposure; N1-N3 follow a causal weekly, frequency-matched
random gate. Both reuse C2's long logic and the single shared account owner.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

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
class ShortNullControlConfig:
    mode: Literal["long_only", "weekly_random"]

    def __post_init__(self) -> None:
        if self.mode not in ("long_only", "weekly_random"):
            raise ValueError("unregistered short-null control mode")


class ShortNullControlPolicy:
    """Wrap C2 and suppress disallowed short proposals or short holdings."""

    def __init__(
        self,
        *,
        symbol: str,
        signals: DivergenceCloudMtfPack,
        short_permission: np.ndarray,
        config: ShortNullControlConfig,
    ) -> None:
        if not isinstance(config, ShortNullControlConfig):
            raise TypeError("config must be a registered null control")
        mask = np.asarray(short_permission)
        if mask.dtype != np.bool_ or mask.shape != signals.timestamps.shape:
            raise ValueError("short permission must be a boolean mask on signal clock")
        if config.mode == "long_only" and bool(np.any(mask)):
            raise ValueError("long-only control cannot permit shorts")
        self._base = CloudFollowPolicy(
            symbol=symbol,
            signals=signals,
            config=CloudFollowConfig(mode="confirmed_cloud"),
        )
        self._mask = mask.copy()
        self._mask.setflags(write=False)
        self.blocked_short_proposals = 0
        self.ineligible_short_flat_requests = 0

    def decide_portfolio(
        self,
        snapshot: PortfolioDecisionSnapshot,
    ) -> PortfolioTargetProposal:
        proposal = self._base.decide_portfolio(snapshot)
        if (
            snapshot.account.insolvent
            or snapshot.account.termination_reason is not None
        ):
            return proposal
        i = snapshot.index
        if bool(self._mask[i]):
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


__all__ = ["ShortNullControlConfig", "ShortNullControlPolicy"]
