"""U1 single-instrument environment surface without changing base economics."""

from __future__ import annotations

from typing import Any

import numpy as np

from trade_rl.rl.actions import (
    BaselineResidualComposer,
    ResidualComposition,
    TargetWeightAction,
)
from trade_rl.rl.environment import ResidualMarketEnv
from trade_rl.rl.universal_trade_runtime import UniversalTradeRuntimeSnapshot
from trade_rl.strategies.trend import TrendTargets


class _UniversalTradeTargetComposer(BaselineResidualComposer):
    """Keep U1 target weights raw until the maintained Risk projection stage."""

    @staticmethod
    def _compose_target(
        action: TargetWeightAction,
        trends: TrendTargets,
        *,
        max_gross: float,
    ) -> ResidualComposition:
        if action.weights.shape != trends.base.shape:
            raise ValueError("target weight count does not match trend targets")
        if not np.isfinite(max_gross) or max_gross <= 0.0:
            raise ValueError("max_gross must be finite and positive")
        proposal = np.asarray(action.weights, dtype=np.float64).reshape(-1).copy()
        zeros = np.zeros_like(trends.base)
        raw_gross = float(np.abs(proposal).sum())
        return ResidualComposition(
            action=action,
            baseline=trends.base.copy(),
            trend_component=zeros.copy(),
            alpha_component=zeros.copy(),
            factor_component=zeros.copy(),
            residual_component=proposal - trends.base,
            proposal=proposal,
            raw_gross=raw_gross,
            target_gross=raw_gross,
        )


class UniversalTradeMarketEnv(ResidualMarketEnv):
    """Expose U1 semantics while retaining maintained Risk/Execution economics."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if "composer" in kwargs:
            raise ValueError("Universal Trade RL fixes the U1 target composer")
        super().__init__(*args, composer=_UniversalTradeTargetComposer(), **kwargs)

    def universal_trade_runtime_snapshot(self) -> UniversalTradeRuntimeSnapshot:
        """Return scalar U1 state from existing Risk/Execution/accounting state."""

        if not self._has_reset:
            raise RuntimeError("environment must be reset before exporting U1 runtime")

        index = self.current_index
        bar_hours = self.dataset.bar_hours
        pending_target = self._pending_hybrid_target
        pending_order = self._pending_order_observation_state()
        drawdown = self._drawdown(self.hybrid)
        mark_price = float(self.dataset.resolved_array("mark_price")[index, 0])
        index_price = float(self.dataset.resolved_array("index_price")[index, 0])

        return UniversalTradeRuntimeSnapshot(
            policy_requested_weight=float(self._previous_action[0]),
            pending_target_weight=(
                0.0 if pending_target is None else float(pending_target[0])
            ),
            pending_target_active=pending_target is not None,
            risk_projected_weight=float(self._execution_state.requested_weights[0]),
            current_weight=float(self.hybrid.weights[0]),
            previous_action=float(self._previous_action[0]),
            fill_ratio=float(self._execution_state.fill_ratio[0]),
            unfilled_turnover_ratio=float(self._execution_state.unfilled_turnover[0]),
            participation_ratio=float(self._execution_state.participation[0]),
            execution_cost_rate=float(self._execution_state.execution_cost[0]),
            position_age_hours=float(self._execution_state.position_age[0] * bar_hours),
            pending_notional_ratio=float(pending_order.remaining_notional_ratio[0]),
            pending_order_type_code=float(pending_order.order_type_code[0]),
            pending_order_status_code=float(pending_order.status_code[0]),
            pending_order_age_hours=float(pending_order.age_bars[0] * bar_hours),
            pending_order_eligible_delay_hours=float(
                pending_order.eligible_delay_bars[0] * bar_hours
            ),
            pending_order_triggered=bool(pending_order.triggered[0]),
            pending_order_expiry_distance_hours=float(
                pending_order.expiry_distance_bars[0] * bar_hours
            ),
            asset_active=bool(self.dataset.resolved_array("asset_active")[index, 0]),
            tradable=bool(self.dataset.observable_tradable(index)[0]),
            borrow_available=bool(
                self.dataset.resolved_array("borrow_available")[index, 0]
            ),
            borrow_rate=float(self.dataset.resolved_array("borrow_rate")[index, 0]),
            mark_index_basis=mark_price / index_price - 1.0,
            current_drawdown=drawdown,
            current_gross_exposure=float(self.hybrid.gross_exposure),
            current_net_exposure=float(self.hybrid.net_exposure),
            cash_weight=float(self.hybrid.cash_weight),
            risk_scale=float(self.pre_trade_risk.risk_scale(drawdown)),
            margin_utilization=float(self.hybrid.margin_utilization),
        )


__all__ = ["UniversalTradeMarketEnv"]
