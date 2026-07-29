"""Stable step and terminal information construction for the market environment."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from trade_rl.evaluation.metrics import PerformanceMetrics, evaluate_performance
from trade_rl.evaluation.series import ReturnKind, ReturnSeries
from trade_rl.rl.environment_constraints import (
    ActionPathDiagnostics,
    ConstraintCostRequest,
    ConstraintCostVector,
    calculate_constraint_costs,
)
from trade_rl.rl.rewards import RewardBreakdown, RewardConfig, RewardContext
from trade_rl.simulation.accounting import BookState

_TOLERANCE = 1e-12


class InfoDataset(Protocol):
    @property
    def periods_per_year(self) -> int: ...

    def elapsed_hours(self, start_index: int, end_index: int) -> float: ...


class RewardInfoSource(Protocol):
    @property
    def config(self) -> RewardConfig: ...

    @property
    def last_context_before(self) -> RewardContext: ...

    @property
    def last_context_after(self) -> RewardContext: ...


class ExecutionInfo(Protocol):
    @property
    def next_index(self) -> int: ...

    @property
    def bars_advanced(self) -> int: ...

    @property
    def interval_cost(self) -> float: ...

    @property
    def interval_funding(self) -> float: ...

    @property
    def interval_borrow_cost(self) -> float: ...

    @property
    def interval_gross_return(self) -> float: ...

    @property
    def interval_net_return(self) -> float: ...

    @property
    def filled_turnover(self) -> float: ...


class RiskInfo(Protocol):
    @property
    def projection_l1(self) -> float: ...

    @property
    def proposal_weights(self) -> np.ndarray | None: ...

    @property
    def pretrade_weights(self) -> np.ndarray | None: ...

    @property
    def weights(self) -> np.ndarray: ...

    @property
    def max_gross(self) -> float | None: ...

    @property
    def drawdown_budget(self) -> float | None: ...


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
    action_path: ActionPathDiagnostics | None = None
    constraint_costs: ConstraintCostVector | None = None


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
        *,
        initial_capital: float | None = None,
    ) -> None:
        if initial_capital is not None and (
            not math.isfinite(initial_capital) or initial_capital <= 0.0
        ):
            raise ValueError("initial_capital must be finite and positive")
        self.dataset = dataset
        self.reward_tracker = reward_tracker
        self.initial_capital = initial_capital

    @staticmethod
    def drawdown(book: BookState) -> float:
        value = max(book.portfolio_value, 0.0)
        return min(
            1.0,
            max(0.0, 1.0 - value / max(book.peak_value, value, 1e-12)),
        )

    def _decision_hours(self, execution: ExecutionInfo) -> float:
        start_index = execution.next_index - execution.bars_advanced
        value = self.dataset.elapsed_hours(start_index, execution.next_index)
        if not math.isfinite(value) or value <= 0.0:
            raise RuntimeError("transition duration must be finite and positive")
        return value

    @staticmethod
    def _liquidation_metric(liquidation: object | None, field_name: str) -> float:
        if liquidation is None:
            return 0.0
        value = float(getattr(liquidation, field_name, 0.0))
        if not math.isfinite(value):
            raise RuntimeError(f"liquidation {field_name} must be finite")
        return value

    @staticmethod
    def _target_vector(value: np.ndarray, *, field_name: str) -> np.ndarray:
        vector = np.asarray(value, dtype=np.float64).reshape(-1)
        if vector.size == 0 or not np.isfinite(vector).all():
            raise RuntimeError(f"{field_name} must be a non-empty finite vector")
        return vector

    @staticmethod
    def _action_path_info(
        diagnostics: ActionPathDiagnostics,
    ) -> dict[str, object]:
        return {
            "action_path": diagnostics,
            "action_path_policy_to_execution_intent_l1": (
                diagnostics.policy_to_execution_intent_l1
            ),
            "action_path_execution_intent_to_pretrade_l1": (
                diagnostics.execution_intent_to_pretrade_l1
            ),
            "action_path_policy_to_pretrade_l1": diagnostics.policy_to_pretrade_l1,
            "action_path_pretrade_to_feasible_l1": (
                diagnostics.pretrade_to_feasible_l1
            ),
            "action_path_feasible_to_submitted_l1": (
                diagnostics.feasible_to_submitted_l1
            ),
            "action_path_submitted_to_filled_l1": diagnostics.submitted_to_filled_l1,
            "action_path_execution_intent_to_filled_l1": (
                diagnostics.execution_intent_to_filled_l1
            ),
            "action_path_policy_to_filled_l1": diagnostics.policy_to_filled_l1,
            "action_path_policy_to_execution_intent_max_abs": (
                diagnostics.policy_to_execution_intent_max_abs
            ),
            "action_path_execution_intent_to_pretrade_max_abs": (
                diagnostics.execution_intent_to_pretrade_max_abs
            ),
            "action_path_policy_to_pretrade_max_abs": (
                diagnostics.policy_to_pretrade_max_abs
            ),
            "action_path_pretrade_to_feasible_max_abs": (
                diagnostics.pretrade_to_feasible_max_abs
            ),
            "action_path_feasible_to_submitted_max_abs": (
                diagnostics.feasible_to_submitted_max_abs
            ),
            "action_path_submitted_to_filled_max_abs": (
                diagnostics.submitted_to_filled_max_abs
            ),
            "action_path_policy_changed_by_execution_delay": (
                diagnostics.policy_changed_by_execution_delay
            ),
            "action_path_execution_intent_changed_by_pretrade": (
                diagnostics.execution_intent_changed_by_pretrade
            ),
            "action_path_policy_changed_by_pretrade": (
                diagnostics.policy_changed_by_pretrade
            ),
            "action_path_pretrade_changed_by_feasibility": (
                diagnostics.pretrade_changed_by_feasibility
            ),
            "action_path_feasible_changed_before_submission": (
                diagnostics.feasible_changed_before_submission
            ),
            "action_path_submission_changed_by_fill": (
                diagnostics.submission_changed_by_fill
            ),
        }

    @staticmethod
    def _constraint_cost_info(costs: ConstraintCostVector) -> dict[str, object]:
        return {
            "constraint_costs": costs,
            "constraint_cost_drawdown_excess": costs.drawdown_excess,
            "constraint_cost_drawdown_stop_event": costs.drawdown_stop_event,
            "constraint_cost_margin_deficit_fraction": costs.margin_deficit_fraction,
            "constraint_cost_forced_liquidation_event": (
                costs.forced_liquidation_event
            ),
            "constraint_cost_gross_exposure_request_excess": (
                costs.gross_exposure_request_excess
            ),
            "constraint_cost_daily_turnover": costs.daily_turnover,
            "constraint_cost_execution_fraction": costs.execution_cost_fraction,
            "constraint_cost_funding_credit_fraction": costs.funding_credit_fraction,
        }

    @classmethod
    def _risk_pipeline_targets(
        cls,
        request: EnvironmentStepInfoRequest,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        policy_target = cls._target_vector(
            request.submitted_target,
            field_name="submitted_target",
        )
        execution_intent = cls._target_vector(
            request.executed_target,
            field_name="executed_target",
        )
        risk_proposal = request.hybrid_risk.proposal_weights
        pretrade = request.hybrid_risk.pretrade_weights
        if risk_proposal is None or pretrade is None:
            raise RuntimeError("risk result lacks action-path stage metadata")
        risk_proposal_vector = cls._target_vector(
            risk_proposal,
            field_name="risk proposal",
        )
        if risk_proposal_vector.shape != execution_intent.shape or not np.allclose(
            risk_proposal_vector,
            execution_intent,
            atol=_TOLERANCE,
            rtol=0.0,
        ):
            raise RuntimeError("executed target disagrees with risk proposal")
        return (
            policy_target,
            execution_intent,
            cls._target_vector(pretrade, field_name="pretrade target"),
            cls._target_vector(
                request.hybrid_risk.weights,
                field_name="feasible target",
            ),
        )

    @classmethod
    def _derived_action_path(
        cls,
        request: EnvironmentStepInfoRequest,
    ) -> ActionPathDiagnostics:
        policy_target, execution_intent, pretrade, feasible = (
            cls._risk_pipeline_targets(request)
        )
        return ActionPathDiagnostics.from_stages(
            policy_target=policy_target,
            execution_intent_target=execution_intent,
            pretrade_target=pretrade,
            feasible_target=feasible,
            submitted_order_target=feasible,
            filled_weight=request.hybrid.weights,
        )

    def _derived_constraint_costs(
        self,
        request: EnvironmentStepInfoRequest,
    ) -> ConstraintCostVector:
        if self.initial_capital is None:
            raise RuntimeError("constraint costs require configured initial capital")
        _, execution_intent, _, _ = self._risk_pipeline_targets(request)
        max_gross = request.hybrid_risk.max_gross
        drawdown_budget = request.hybrid_risk.drawdown_budget
        if max_gross is None or drawdown_budget is None:
            raise RuntimeError("risk result lacks constraint-limit metadata")
        final_equity = max(
            request.hybrid.portfolio_value,
            float(np.finfo(np.float64).eps),
        )
        previous_equity = final_equity * math.exp(-request.hybrid_log_return)
        if not math.isfinite(previous_equity) or previous_equity <= 0.0:
            raise RuntimeError("transition previous equity could not be reconstructed")
        liquidation = request.hybrid_liquidation
        filled_turnover = (
            request.hybrid_execution.filled_turnover
            + self._liquidation_metric(liquidation, "filled_turnover")
        )
        interval_cost = (
            request.hybrid_execution.interval_cost
            + self._liquidation_metric(liquidation, "interval_cost")
        )
        interval_funding = (
            request.hybrid_execution.interval_funding
            + self._liquidation_metric(liquidation, "interval_funding")
        )
        interval_borrow_cost = (
            request.hybrid_execution.interval_borrow_cost
            + self._liquidation_metric(liquidation, "interval_borrow_cost")
        )
        return calculate_constraint_costs(
            ConstraintCostRequest(
                policy_target=execution_intent,
                max_gross=max_gross,
                decision_hours=self._decision_hours(request.hybrid_execution),
                drawdown=self.drawdown(request.hybrid),
                drawdown_budget=drawdown_budget,
                margin_deficit=request.hybrid.margin_deficit,
                initial_capital=self.initial_capital,
                previous_equity=previous_equity,
                filled_turnover=filled_turnover,
                interval_cost=interval_cost,
                interval_funding=interval_funding,
                interval_borrow_cost=interval_borrow_cost,
                termination_reason=request.termination_reason,
                emergency_deleverage=request.emergency_deleverage,
                liquidation_terminal=request.liquidation_terminal,
                liquidation_complete=request.liquidation_complete,
            )
        )

    def step_info(self, request: EnvironmentStepInfoRequest) -> dict[str, object]:
        reward = request.reward_breakdown
        before = self.reward_tracker.last_context_before
        after = self.reward_tracker.last_context_after
        baseline_weight = self.reward_tracker.config.baseline_underperformance_weight
        action_path = request.action_path or self._derived_action_path(request)
        constraint_costs = request.constraint_costs or self._derived_constraint_costs(
            request
        )
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
                request.submitted_target,
                dtype=np.float64,
            ).copy(),
            "executed_target": np.asarray(
                request.executed_target,
                dtype=np.float64,
            ).copy(),
            "sampled_policy_action": np.asarray(
                action_path.policy_target,
                dtype=np.float64,
            ).copy(),
            "effective_filled_weights": np.asarray(
                action_path.filled_weight,
                dtype=np.float64,
            ).copy(),
            "sampled_policy_to_filled_l1": action_path.policy_to_filled_l1,
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
        info.update(self._action_path_info(action_path))
        info.update(self._constraint_cost_info(constraint_costs))
        if request.discarded_pending_target is not None:
            info["discarded_pending_target"] = np.asarray(
                request.discarded_pending_target,
                dtype=np.float64,
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
