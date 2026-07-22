"""Stable step and terminal information construction for the market environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from trade_rl.evaluation.metrics import PerformanceMetrics, evaluate_performance
from trade_rl.evaluation.series import ReturnKind, ReturnSeries
from trade_rl.rl.rewards import RewardBreakdown, RewardConfig, RewardContext
from trade_rl.simulation.accounting import BookState


class InfoDataset(Protocol):
    @property
    def periods_per_year(self) -> int: ...


class RewardInfoSource(Protocol):
    @property
    def config(self) -> RewardConfig: ...

    @property
    def last_context_before(self) -> RewardContext: ...

    @property
    def last_context_after(self) -> RewardContext: ...


class ExecutionInfo(Protocol):
    @property
    def bars_advanced(self) -> int: ...

    @property
    def interval_cost(self) -> float: ...

    @property
    def interval_funding(self) -> float: ...

    @property
    def interval_gross_return(self) -> float: ...

    @property
    def interval_net_return(self) -> float: ...


class RiskInfo(Protocol):
    @property
    def projection_l1(self) -> float: ...


@dataclass(frozen=True, slots=True)
class EnvironmentStepInfoRequest:
    action_delta_l1: float
    raw_max_abs: float
    saturated_count: int
    composition: object
    decision_step_index: int
    hybrid_log_return: float
    shadow_log_return: float
    emergency_deleverage: bool
    execution_delay_warmup: bool
    submitted_target: np.ndarray
    executed_target: np.ndarray
    hybrid: BookState
    reward_breakdown: RewardBreakdown
    hybrid_execution: ExecutionInfo
    hybrid_risk: RiskInfo
    hybrid_terminated: bool
    shadow_execution: ExecutionInfo
    shadow_risk: RiskInfo
    shadow_terminated: bool
    liquidation_complete: bool
    liquidation_terminal: bool
    termination_reason: object | None
    terminal_accounting_mode: str
    terminal_liquidation_cost: float
    pending_target_discarded: bool
    discarded_pending_target: np.ndarray | None
    hybrid_liquidation: object | None
    shadow_liquidation: object | None


@dataclass(frozen=True, slots=True)
class EnvironmentTerminalInfoRequest:
    episode_hours: float
    episode_seed: int
    action_diagnostics: object
    hybrid: BookState
    shadow: BookState
    initial_state_mode: str


class EnvironmentInfoBuilder:
    """Build fresh audit dictionaries while preserving the stable key contract."""

    def __init__(
        self,
        dataset: InfoDataset,
        reward_tracker: RewardInfoSource,
    ) -> None:
        self.dataset = dataset
        self.reward_tracker = reward_tracker

    @staticmethod
    def drawdown(book: BookState) -> float:
        value = max(book.portfolio_value, 0.0)
        return min(
            1.0,
            max(0.0, 1.0 - value / max(book.peak_value, value, 1e-12)),
        )

    def step_info(self, request: EnvironmentStepInfoRequest) -> dict[str, object]:
        reward = request.reward_breakdown
        before = self.reward_tracker.last_context_before
        after = self.reward_tracker.last_context_after
        baseline_weight = self.reward_tracker.config.baseline_underperformance_weight
        info: dict[str, object] = {
            "action_delta_l1": request.action_delta_l1,
            "action_raw_max_abs": request.raw_max_abs,
            "action_saturated_count": request.saturated_count,
            "bars_advanced": request.hybrid_execution.bars_advanced,
            "composition": request.composition,
            "decision_step_index": request.decision_step_index,
            "excess_log_return": (
                request.hybrid_log_return - request.shadow_log_return
            ),
            "emergency_deleverage": request.emergency_deleverage,
            "execution_delay_warmup": request.execution_delay_warmup,
            "submitted_target": np.asarray(
                request.submitted_target, dtype=np.float64
            ).copy(),
            "executed_target": np.asarray(
                request.executed_target, dtype=np.float64
            ).copy(),
            "drawdown_after": self.drawdown(request.hybrid),
            "portfolio_value_after": request.hybrid.portfolio_value,
            "reward_growth_raw": reward.absolute_log_growth,
            "reward_baseline_penalty_delta": (
                0.0
                if baseline_weight == 0.0
                else reward.baseline_penalty / baseline_weight
            ),
            "reward_baseline_penalty_weighted": reward.baseline_penalty,
            "reward_drawdown_penalty_delta": reward.incremental_drawdown,
            "reward_drawdown_penalty_weighted": reward.drawdown_penalty,
            "reward_total_raw": reward.unscaled_total,
            "reward_total_scaled": reward.scaled_total,
            "reward_context_before": before,
            "reward_context_after": after,
            "rolling_hybrid_log_growth": after.rolling_hybrid_log_growth,
            "rolling_baseline_log_growth": after.rolling_shadow_log_growth,
            "rolling_growth_gap": after.rolling_growth_gap,
            "hybrid_execution": request.hybrid_execution,
            "hybrid_risk": request.hybrid_risk,
            "hybrid_terminated": request.hybrid_terminated,
            "interval_cost": request.hybrid_execution.interval_cost,
            "interval_funding": request.hybrid_execution.interval_funding,
            "interval_gross_return": request.hybrid_execution.interval_gross_return,
            "interval_net_return": request.hybrid_execution.interval_net_return,
            "liquidation_complete": request.liquidation_complete,
            "liquidation_terminal": request.liquidation_terminal,
            "projection_distance_l1": request.hybrid_risk.projection_l1,
            "reward_breakdown": reward,
            "shadow_execution": request.shadow_execution,
            "shadow_interval_net_return": request.shadow_execution.interval_net_return,
            "shadow_risk": request.shadow_risk,
            "shadow_terminated": request.shadow_terminated,
            "termination_reason": request.termination_reason,
            "terminal_accounting_mode": request.terminal_accounting_mode,
            "terminal_liquidation_cost": request.terminal_liquidation_cost,
            "pending_target_discarded": request.pending_target_discarded,
        }
        if request.discarded_pending_target is not None:
            info["discarded_pending_target"] = np.asarray(
                request.discarded_pending_target, dtype=np.float64
            ).copy()
        if request.hybrid_liquidation is not None:
            info["hybrid_liquidation"] = request.hybrid_liquidation
        if request.shadow_liquidation is not None:
            info["shadow_liquidation"] = request.shadow_liquidation
        return info

    def book_metrics(self, book: BookState) -> PerformanceMetrics:
        return evaluate_performance(
            ReturnSeries(
                values=tuple(book.returns_history),
                kind=ReturnKind.BASE_BAR,
                periods_per_year=self.dataset.periods_per_year,
            ),
            turnover_total=book.turnover_total,
            total_cost=book.total_cost,
            funding_pnl=book.funding_pnl - book.borrow_cost,
            n_trades=book.fill_count,
        )

    def terminal_info(
        self,
        request: EnvironmentTerminalInfoRequest,
    ) -> dict[str, object]:
        hybrid_metrics = self.book_metrics(request.hybrid)
        shadow_metrics = self.book_metrics(request.shadow)
        return {
            "episode_hours": request.episode_hours,
            "episode_seed": request.episode_seed,
            "action_diagnostics": request.action_diagnostics,
            "hybrid_metrics": hybrid_metrics,
            "hybrid_rebalance_events": request.hybrid.rebalance_events,
            "initial_state_mode": request.initial_state_mode,
            "shadow_metrics": shadow_metrics,
            "shadow_rebalance_events": request.shadow.rebalance_events,
            "excess_total_return": (
                hybrid_metrics.total_return - shadow_metrics.total_return
            ),
        }


__all__ = [
    "EnvironmentInfoBuilder",
    "EnvironmentStepInfoRequest",
    "EnvironmentTerminalInfoRequest",
]
