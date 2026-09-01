"""U1 single-instrument environment surface without changing base economics."""

from __future__ import annotations

from trade_rl.rl.environment import ResidualMarketEnv
from trade_rl.rl.universal_trade_runtime import UniversalTradeRuntimeSnapshot


class UniversalTradeMarketEnv(ResidualMarketEnv):
    """Expose the U1 causal runtime projection on top of maintained economics."""

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
