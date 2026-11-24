"""Research-only cloud trend-strength exits and entry volatility sizing."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import numpy as np

from trade_rl.data.features.cloud_trend_context import CloudTrendContext
from trade_rl.data.features.divergence_mtf import DivergenceCloudMtfPack
from trade_rl.simulation.quantities import parse_quantity
from trade_rl.strategies.divergence_cloud_follow import (
    CloudFollowConfig,
    CloudFollowPolicy,
)
from trade_rl.strategies.portfolio import (
    PortfolioDecisionSnapshot,
    PortfolioTargetProposal,
)

Mode = Literal["all_exits", "weak_trend_exits", "weak_trend_atr_entry"]
_ADX_THRESHOLD = 0.25
_ATR_BUDGET = 0.0025
_COOLDOWN_HOURS = 24


@dataclass(frozen=True, slots=True)
class CloudTrendConfig:
    """Fixed synthetic mechanism alternatives; no economic default."""

    mode: Mode

    def __post_init__(self) -> None:
        if self.mode not in ("all_exits", "weak_trend_exits", "weak_trend_atr_entry"):
            raise ValueError("unsupported cloud trend research mode")


class CloudTrendPolicy:
    """Reuse C2 entry; keep risk, fills and accounting in the native owner.

    The matched control and gated modes share a latched exit contract. A
    requested exit persists through partial fills until the account is flat.
    """

    def __init__(
        self,
        *,
        symbol: str,
        signals: DivergenceCloudMtfPack,
        context: CloudTrendContext,
        config: CloudTrendConfig,
    ) -> None:
        if not isinstance(context, CloudTrendContext):
            raise TypeError("context must be CloudTrendContext")
        if not isinstance(signals, DivergenceCloudMtfPack):
            raise TypeError("signals must be DivergenceCloudMtfPack")
        if not isinstance(config, CloudTrendConfig):
            raise TypeError("config must be CloudTrendConfig")
        if not np.array_equal(context.timestamps, signals.timestamps):
            raise ValueError("context and signal clock must match")
        arrays = {}
        for name in ("cloud_ready", "cloud_state", "bull_4h", "bear_4h"):
            arr = np.asarray(getattr(signals, name))
            if arr.shape != context.timestamps.shape:
                raise ValueError("signal shape must match context clock")
            if name == "cloud_ready":
                if arr.dtype != np.bool_:
                    raise ValueError("cloud readiness must be boolean")
            elif arr.dtype.kind not in "iu" or (
                np.any(~np.isin(arr, (-1, 0, 1)))
                if name == "cloud_state"
                else np.any(arr < 0)
            ):
                raise ValueError("invalid cloud state or divergence count")
            arrays[name] = np.frombuffer(arr.tobytes(), dtype=arr.dtype)
        self._signals = replace(
            signals,
            timestamps=context.timestamps,
            cloud_ready=arrays["cloud_ready"],
            cloud_state=arrays["cloud_state"],
            bull_4h=arrays["bull_4h"],
            bear_4h=arrays["bear_4h"],
        )
        self._base = CloudFollowPolicy(
            symbol=symbol,
            signals=self._signals,
            config=CloudFollowConfig(mode="confirmed_cloud"),
        )
        self._context = context
        self._config = config
        self._exit_pending = False
        self._divergence_exit = False
        self._cooldown_until = -1

    @staticmethod
    def _hold(snapshot: PortfolioDecisionSnapshot) -> PortfolioTargetProposal:
        return PortfolioTargetProposal(
            (float(snapshot.account.weights[0]),),
            hold_current_quantity_symbols=(0,),
        )

    def decide_portfolio(
        self, snapshot: PortfolioDecisionSnapshot
    ) -> PortfolioTargetProposal:
        base = self._base.decide_portfolio(snapshot)
        if (
            snapshot.account.insolvent
            or snapshot.account.termination_reason is not None
        ):
            return base
        i = snapshot.index
        quantity = parse_quantity(snapshot.account.exact_quantities[0])
        direction = int((quantity > 0) - (quantity < 0))
        if self._exit_pending:
            if direction:
                return PortfolioTargetProposal((0.0,))
            if self._divergence_exit:
                self._cooldown_until = i + _COOLDOWN_HOURS
            self._exit_pending = False
            self._divergence_exit = False
            return self._hold(snapshot)

        ready = bool(self._context.ready[i])
        if direction:
            cloud = self._signals
            supported = (
                bool(cloud.cloud_ready[i]) and int(cloud.cloud_state[i]) == direction
            )
            if not ready or not supported:
                self._exit_pending = True
                return PortfolioTargetProposal((0.0,))
            opposite = cloud.bear_4h[i] if direction > 0 else cloud.bull_4h[i]
            permits_exit = (
                self._config.mode == "all_exits"
                or self._context.adx_fraction[i] < _ADX_THRESHOLD
            )
            if opposite > 0 and permits_exit:
                self._exit_pending = True
                self._divergence_exit = True
                return PortfolioTargetProposal((0.0,))
            return self._hold(snapshot)

        if not ready or i < self._cooldown_until or snapshot.active_orders:
            return self._hold(snapshot)
        if base.hold_current_quantity_symbols or base.target_weights[0] == 0.0:
            return base
        if self._config.mode != "weak_trend_atr_entry":
            return base
        target = base.target_weights[0]
        atr = float(self._context.atr_fraction[i])
        # Avoid overflow for tiny nonzero ATR: only divide when the cap
        # would actually bind. This is a sizing proxy, not a stop loss.
        magnitude = abs(target)
        if atr > _ATR_BUDGET / magnitude:
            magnitude = _ATR_BUDGET / atr
        return PortfolioTargetProposal((float(np.copysign(magnitude, target)),))


__all__ = ["CloudTrendConfig", "CloudTrendPolicy"]
