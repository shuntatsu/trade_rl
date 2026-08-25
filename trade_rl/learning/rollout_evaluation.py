"""Exact simulator rollouts used by Oracle and behavior-cloning audits."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from trade_rl.evaluation.closed_trades import ClosedTradeTracker
from trade_rl.learning.evaluation import (
    ActionPathCollapseEvidence,
    PathPerformanceMetrics,
    evaluate_path_performance,
)


class EvaluationEnvironment(Protocol):
    current_index: int
    dataset: Any

    def reset(
        self, *, options: dict[str, object]
    ) -> tuple[object, dict[str, object]]: ...

    def step(
        self, action: np.ndarray
    ) -> tuple[object, float, bool, bool, dict[str, object]]: ...


@dataclass(frozen=True, slots=True)
class ActionPathStepEconomics:
    """Simulator-authoritative step economics retained for attribution."""

    gross_returns: np.ndarray
    net_returns: np.ndarray
    costs: np.ndarray
    turnover: np.ndarray

    def __post_init__(self) -> None:
        arrays: dict[str, np.ndarray] = {}
        for name in ("gross_returns", "net_returns", "costs", "turnover"):
            array = np.asarray(getattr(self, name), dtype=np.float64).reshape(-1).copy()
            if array.size == 0 or not np.isfinite(array).all():
                raise ValueError("action path step economics must be non-empty and finite")
            array.setflags(write=False)
            arrays[name] = array
        if len({array.size for array in arrays.values()}) != 1:
            raise ValueError("action path step economics arrays are not aligned")
        if np.any(arrays["gross_returns"] <= -1.0) or np.any(
            arrays["net_returns"] <= -1.0
        ):
            raise ValueError("action path step returns must be greater than -1")
        if np.any(arrays["costs"] < 0.0) or np.any(arrays["turnover"] < 0.0):
            raise ValueError("action path step costs and turnover must be non-negative")
        for name, array in arrays.items():
            object.__setattr__(self, name, array)


@dataclass(frozen=True, slots=True)
class ActionPathEvaluation:
    actions: np.ndarray
    performance: PathPerformanceMetrics
    collapse_evidence: ActionPathCollapseEvidence
    step_economics: ActionPathStepEconomics | None = None

    def __post_init__(self) -> None:
        actions = np.asarray(self.actions, dtype=np.float32).copy(order="C")
        if (
            actions.ndim != 2
            or len(actions) != self.performance.step_count
            or not np.isfinite(actions).all()
        ):
            raise ValueError("evaluated actions do not match path metrics")
        if self.collapse_evidence.decision_count != self.performance.step_count:
            raise ValueError("collapse evidence does not cover evaluated path")
        if self.collapse_evidence.trade_count != self.performance.trade_count:
            raise ValueError("collapse evidence trade count mismatch")
        if (
            self.collapse_evidence.executed_change_count
            != self.performance.traded_step_count
        ):
            raise ValueError("collapse evidence execution count mismatch")
        if self.step_economics is not None:
            if not isinstance(self.step_economics, ActionPathStepEconomics):
                raise TypeError("step_economics must be ActionPathStepEconomics")
            if len(self.step_economics.gross_returns) != self.performance.step_count:
                raise ValueError("step economics do not cover evaluated path")
        actions.setflags(write=False)
        object.__setattr__(self, "actions", actions)


def _metric(info: Mapping[str, object], name: str) -> float:
    value = info.get(name)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"evaluation info is missing numeric {name}")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"evaluation info {name} is non-finite")
    return result


def _liquidation_metric(info: Mapping[str, object], name: str) -> float:
    liquidation = info.get("hybrid_liquidation")
    if liquidation is None:
        return 0.0
    value = getattr(liquidation, name, None)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"liquidation is missing numeric {name}")
    return float(value)


def evaluate_action_path(
    environment: EvaluationEnvironment,
    *,
    evaluation_range: tuple[int, int],
    actions: object | None = None,
    model: object | None = None,
    deterministic: bool = True,
    action_change_tolerance: float = 1e-6,
) -> ActionPathEvaluation:
    """Execute either a declared target path or one causal deterministic policy."""

    start, stop = evaluation_range
    if (
        isinstance(start, bool)
        or isinstance(stop, bool)
        or not isinstance(start, int)
        or not isinstance(stop, int)
        or start < 0
        or stop <= start + 1
    ):
        raise ValueError("evaluation_range must contain at least one decision")
    if (actions is None) == (model is None):
        raise ValueError("provide exactly one of actions or model")
    if not isinstance(deterministic, bool):
        raise TypeError("deterministic must be a boolean")
    if (
        isinstance(action_change_tolerance, bool)
        or not np.isfinite(action_change_tolerance)
        or action_change_tolerance < 0.0
    ):
        raise ValueError("action_change_tolerance must be finite and non-negative")
    expected_count = stop - start - 1
    declared: np.ndarray | None = None
    if actions is not None:
        declared = np.asarray(actions, dtype=np.float32)
        if (
            declared.ndim != 2
            or len(declared) != expected_count
            or not np.isfinite(declared).all()
        ):
            raise ValueError("declared actions do not cover the evaluation range")
    predict = None if model is None else getattr(model, "predict", None)
    if model is not None and not callable(predict):
        raise TypeError("evaluation model must expose predict")

    observation, _ = environment.reset(
        options={
            "start_idx": start,
            "episode_bars": expected_count,
            "initial_state_mode": "cash",
        }
    )
    multipliers = environment.dataset.resolved_array("contract_multipliers")
    trades = ClosedTradeTracker(multipliers)
    initial_book = getattr(environment, "hybrid", None)
    if initial_book is not None:
        trades.seed_positions(
            quantities=initial_book.quantities,
            prices=initial_book.mark_prices,
        )
    evaluated_actions: list[np.ndarray] = []
    gross_returns: list[float] = []
    net_returns: list[float] = []
    rewards: list[float] = []
    turnover: list[float] = []
    costs: list[float] = []
    active_dimension_count = 0
    inactive_dimension_count = 0
    proposal_distance_count = 0
    submitted_change_count = 0
    downstream_no_trade_suppression_count = 0
    execution_rejection_count = 0
    execution_rejection_reasons: Counter[str] = Counter()
    risk_projection_reasons: Counter[str] = Counter()
    executed_change_count = 0
    previous_submitted: np.ndarray | None = None
    action_dimension_count: int | None = None
    for offset in range(expected_count):
        if environment.current_index != start + offset:
            raise ValueError("evaluation environment advanced outside the range")
        if declared is not None:
            action = declared[offset]
        else:
            assert callable(predict)
            raw_action, _ = predict(observation, deterministic=deterministic)
            action = np.asarray(raw_action, dtype=np.float32).reshape(-1)
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action_dimension_count is None:
            action_dimension_count = int(action.size)
        elif action.size != action_dimension_count:
            raise ValueError("evaluation action dimensions changed within path")
        if not np.isfinite(action).all():
            raise ValueError("evaluation action contains non-finite values")
        reference = (
            np.zeros_like(action) if previous_submitted is None else previous_submitted
        )
        active = np.ones(action.shape, dtype=np.bool_)
        if isinstance(observation, Mapping):
            if "current_weights" in observation:
                reference = np.asarray(
                    observation["current_weights"], dtype=np.float32
                ).reshape(-1)
                if reference.shape != action.shape or not np.isfinite(reference).all():
                    raise ValueError("current weights do not match evaluation action")
            if "active" in observation:
                active_values = np.asarray(observation["active"]).reshape(-1)
                if active_values.shape != action.shape:
                    raise ValueError("active mask does not match evaluation action")
                active = active_values > 0.5
        proposed = active & (np.abs(action - reference) > action_change_tolerance)
        active_dimension_count += int(np.count_nonzero(active))
        inactive_dimension_count += int(np.count_nonzero(~active))
        proposal_distance_count += int(np.count_nonzero(proposed))
        submitted_change = bool(np.any(proposed))
        submitted_change_count += int(submitted_change)
        evaluated_actions.append(action.copy())
        previous_submitted = action.copy()
        observation, reward, terminated, truncated, raw_info = environment.step(action)
        if not isinstance(raw_info, Mapping):
            raise ValueError("evaluation environment info must be a mapping")
        info = raw_info
        execution = info.get("hybrid_execution")
        if execution is None:
            raise ValueError("evaluation info is missing hybrid execution")
        trades.ingest_stateful(execution)
        liquidation = info.get("hybrid_liquidation")
        if liquidation is not None:
            trades.ingest_liquidation(liquidation)
        gross = _metric(info, "interval_gross_return")
        net = _metric(info, "interval_net_return")
        liquidation_gross = _liquidation_metric(info, "interval_gross_return")
        liquidation_net = _liquidation_metric(info, "interval_net_return")
        gross_returns.append((1.0 + gross) * (1.0 + liquidation_gross) - 1.0)
        net_returns.append((1.0 + net) * (1.0 + liquidation_net) - 1.0)
        rewards.append(float(reward))
        requested_turnover = getattr(execution, "requested_turnover", None)
        if isinstance(requested_turnover, bool) or not isinstance(
            requested_turnover, int | float
        ):
            raise ValueError("hybrid execution is missing requested_turnover")
        rejected_count = getattr(execution, "rejected_count", 0)
        if (
            isinstance(rejected_count, bool)
            or not isinstance(rejected_count, int)
            or rejected_count < 0
        ):
            raise ValueError("hybrid execution rejected_count is invalid")
        execution_rejection_count += rejected_count
        rejected_events = 0
        for event in tuple(getattr(execution, "order_events", ())):
            if getattr(event, "event_type", None) != "rejected":
                continue
            reason = getattr(event, "reason", None)
            if not isinstance(reason, str) or not reason.strip():
                raise ValueError("rejected order event is missing a reason")
            execution_rejection_reasons[reason] += 1
            rejected_events += 1
        if rejected_events != rejected_count:
            raise ValueError(
                "hybrid execution rejected_count does not match rejected order events"
            )
        risk = info.get("hybrid_risk")
        if risk is not None:
            for reason in tuple(getattr(risk, "reasons", ())):
                if not isinstance(reason, str) or not reason.strip():
                    raise ValueError("hybrid risk projection reason is invalid")
                risk_projection_reasons[reason] += 1
        if submitted_change and float(requested_turnover) <= action_change_tolerance:
            downstream_no_trade_suppression_count += 1
        filled_turnover = getattr(execution, "filled_turnover", None)
        if isinstance(filled_turnover, bool) or not isinstance(
            filled_turnover, int | float
        ):
            raise ValueError("hybrid execution is missing filled_turnover")
        total_filled_turnover = float(filled_turnover) + _liquidation_metric(
            info, "filled_turnover"
        )
        turnover.append(total_filled_turnover)
        executed_change_count += int(total_filled_turnover > action_change_tolerance)
        costs.append(
            _metric(info, "interval_cost") + _liquidation_metric(info, "interval_cost")
        )
        if (bool(terminated) or bool(truncated)) != (offset == expected_count - 1):
            raise ValueError("evaluation environment ended outside the range")
    diagnostics = trades.diagnostics()
    performance = evaluate_path_performance(
        gross_step_returns=gross_returns,
        net_step_returns=net_returns,
        rewards=rewards,
        turnover=turnover,
        costs=costs,
        closed_trade_count=diagnostics.closed_trades,
        winning_trade_count=diagnostics.winning_trades,
        trade_epsilon=action_change_tolerance,
    )
    if action_dimension_count is None:
        raise RuntimeError("evaluation produced no action dimensions")
    evidence = ActionPathCollapseEvidence(
        decision_count=expected_count,
        action_dimension_count=action_dimension_count,
        active_dimension_count=active_dimension_count,
        inactive_dimension_count=inactive_dimension_count,
        proposal_distance_count=proposal_distance_count,
        submitted_change_count=submitted_change_count,
        downstream_no_trade_suppression_count=(downstream_no_trade_suppression_count),
        execution_rejection_count=execution_rejection_count,
        executed_change_count=executed_change_count,
        trade_count=performance.trade_count,
        constant_submitted_actions=submitted_change_count == 0,
        execution_rejection_reason_counts=tuple(
            sorted(execution_rejection_reasons.items())
        ),
        risk_projection_reason_counts=tuple(sorted(risk_projection_reasons.items())),
        hard_risk_violation=False,
    )
    return ActionPathEvaluation(
        actions=np.stack(evaluated_actions, axis=0),
        performance=performance,
        collapse_evidence=evidence,
        step_economics=ActionPathStepEconomics(
            gross_returns=np.asarray(gross_returns, dtype=np.float64),
            net_returns=np.asarray(net_returns, dtype=np.float64),
            costs=np.asarray(costs, dtype=np.float64),
            turnover=np.asarray(turnover, dtype=np.float64),
        ),
    )


__all__ = [
    "ActionPathEvaluation",
    "ActionPathStepEconomics",
    "EvaluationEnvironment",
    "evaluate_action_path",
]
