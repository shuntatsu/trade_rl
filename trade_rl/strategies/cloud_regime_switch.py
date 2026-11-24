"""Single-symbol trend/range routing, synthetic research only."""

from __future__ import annotations

from typing import Literal

import numpy as np

from trade_rl.data.features.cloud_regime_switch_context import CloudRegimeSwitchContext
from trade_rl.simulation.quantities import parse_quantity
from trade_rl.strategies.portfolio import (
    PortfolioDecisionSnapshot,
    PortfolioTargetProposal,
)

Mode = Literal["trend_only", "hybrid", "hybrid_divergence"]
Branch = Literal["trend", "range"]


class CloudRegimeSwitchPolicy:
    """One entry owner until native exposure is flat, with latched exits.

    All weights are proposals to the existing risk, execution and accounting
    owner. No position or fill is inferred from a submitted target.
    """

    def __init__(self, *, context: CloudRegimeSwitchContext, mode: Mode) -> None:
        if not isinstance(context, CloudRegimeSwitchContext):
            raise TypeError("context must be CloudRegimeSwitchContext")
        if mode not in ("trend_only", "hybrid", "hybrid_divergence"):
            raise ValueError("unsupported switch mode")
        self._ctx = context
        self._mode = mode
        self._last_index: int | None = None
        self._owner: Branch | None = None
        self._side = 0
        self._first_fill: int | None = None
        self._exit_pending = False
        self._cooldown_until = -1

    @staticmethod
    def _hold(snapshot: PortfolioDecisionSnapshot) -> PortfolioTargetProposal:
        return PortfolioTargetProposal(
            (snapshot.account.weights[0],), hold_current_quantity_symbols=(0,)
        )

    def _request_exit(self) -> PortfolioTargetProposal:
        self._exit_pending = True
        return PortfolioTargetProposal((0.0,))

    def _clear_entry(self) -> None:
        self._owner = None
        self._side = 0
        self._first_fill = None
        self._exit_pending = False

    def decide_portfolio(
        self, snapshot: PortfolioDecisionSnapshot
    ) -> PortfolioTargetProposal:
        if not isinstance(snapshot, PortfolioDecisionSnapshot):
            raise TypeError("snapshot must be PortfolioDecisionSnapshot")
        c = self._ctx
        i = snapshot.index
        if (
            snapshot.symbols != (c.symbol,)
            or not 0 <= i < len(c.timestamps)
            or snapshot.timestamp_ns != int(c.timestamps[i].astype(np.int64))
        ):
            raise ValueError("snapshot clock or symbol mismatch")
        if self._last_index is not None and i != self._last_index + 1:
            raise ValueError("decision clock must advance exactly one hour")
        self._last_index = i
        if (
            snapshot.account.insolvent
            or snapshot.account.termination_reason is not None
        ):
            return self._hold(snapshot)
        q = parse_quantity(snapshot.account.exact_quantities[0])
        side = int((q > 0) - (q < 0))
        if side:
            if self._exit_pending or self._owner is None or self._side != side:
                return self._request_exit()
            if self._first_fill is None:
                self._first_fill = i
            if not c.ready[i]:
                return self._request_exit()
            adx = float(c.adx_fraction[i])
            cloud = int(c.cloud_state[i])
            if self._owner == "trend":
                if adx <= 0.20 or cloud != side or int(c.exit_breakout[i]) == -side:
                    return self._request_exit()
            elif (
                adx >= 0.25
                or cloud != 0
                or side * c.bollinger_position[i] >= 0
                or i - self._first_fill >= 24
            ):
                return self._request_exit()
            return self._hold(snapshot)

        # Only actual native flatness clears an old position's owner.
        if self._exit_pending or self._first_fill is not None:
            self._clear_entry()
            self._cooldown_until = i + 4
            return self._hold(snapshot)
        if self._owner is not None:
            # An entry request produced no position. Cancel any remaining
            # order and never treat a rejected/unfilled request as a fill.
            self._clear_entry()
            return self._hold(snapshot)
        if snapshot.active_orders or i < self._cooldown_until or i < 5:
            return self._hold(snapshot)
        start = i - 5
        if not np.all(c.ready[start : i + 1]):
            return self._hold(snapshot)
        cloud = int(c.cloud_state[i])
        if not np.all(c.cloud_state[start : i + 1] == cloud):
            return self._hold(snapshot)
        adx = float(c.adx_fraction[i])
        owner: Branch
        if cloud and adx >= 0.25 and int(c.entry_breakout[i]) == cloud:
            owner = "trend"
            desired = cloud
        elif self._mode != "trend_only" and cloud == 0 and adx <= 0.20:
            rsi = float(c.rsi_centered[i])
            bb = float(c.bollinger_position[i])
            desired = (
                1 if rsi <= -0.4 and bb <= -1 else -1 if rsi >= 0.4 and bb >= 1 else 0
            )
            if desired == 0:
                return self._hold(snapshot)
            if self._mode == "hybrid_divergence":
                bull, bear = int(c.bull_1h[i]), int(c.bear_1h[i])
                if not (
                    bull > 0 and bear == 0 if desired > 0 else bear > 0 and bull == 0
                ):
                    return self._hold(snapshot)
            owner = "range"
        else:
            return self._hold(snapshot)
        magnitude = min(
            0.25,
            max(float(snapshot.risk.max_gross), 0.0),
            max(float(snapshot.risk.max_abs_weight), 0.0),
        )
        if magnitude <= 0:
            return self._hold(snapshot)
        atr = float(c.atr_fraction[i])
        if atr > 0.0025 / magnitude:
            magnitude = 0.0025 / atr
        if magnitude <= 0:
            return self._hold(snapshot)
        self._owner = owner
        self._side = desired
        return PortfolioTargetProposal((desired * magnitude,))


__all__ = ["CloudRegimeSwitchPolicy"]
