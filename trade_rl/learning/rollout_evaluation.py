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
                raise ValueError(
                    "action path step economics must be non-empty and finite"
                )
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
class ActionPathExecutionTrace:
    """Decision-boundary execution state retained without reassigning interval PnL."""

    pre_action_weights: np.ndarray
    risk_constrained_weights: np.ndarray
    post_step_weights: np.ndarray
    applied_risk_scales: np.ndarray
    strategy_intent_changes: np.ndarray
    realized_state_follows: np.ndarray
    rebalance_reassertions: np.ndarray
    hard_risk_violations: np.ndarray

    def __post_init__(self) -> None:
        weight_arrays: dict[str, np.ndarray] = {}
        for name in (
            "pre_action_weights",
            "risk_constrained_weights",
            "post_step_weights",
        ):
            array = np.asarray(getattr(self, name), dtype=np.float64).copy(order="C")
            if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
                raise ValueError("action path execution weights must be rank-two")
            if not np.isfinite(array).all():
                raise ValueError("action path execution weights must be finite")
            weight_arrays[name] = array
        shapes = {array.shape for array in weight_arrays.values()}
        if len(shapes) != 1:
            raise ValueError("action path execution weight traces are not aligned")
        steps = next(iter(weight_arrays.values())).shape[0]
        risk_scales = (
            np.asarray(self.applied_risk_scales, dtype=np.float64).reshape(-1).copy()
        )
        if (
            risk_scales.shape != (steps,)
            or not np.isfinite(risk_scales).all()
            or np.any((risk_scales < 0.0) | (risk_scales > 1.0))
        ):
            raise ValueError("action path applied risk scales are invalid")
        boolean_arrays: dict[str, np.ndarray] = {}
        for name in (
            "strategy_intent_changes",
            "realized_state_follows",
            "rebalance_reassertions",
            "hard_risk_violations",
        ):
            raw_boolean = np.asarray(getattr(self, name))
            if raw_boolean.dtype.kind != "b":
                raise ValueError("action path execution event traces must be boolean")
            boolean_array = raw_boolean.reshape(-1).astype(np.bool_, copy=True)
            if boolean_array.shape != (steps,):
                raise ValueError("action path execution event traces are not aligned")
            boolean_arrays[name] = boolean_array
        risk_scales.setflags(write=False)
        object.__setattr__(self, "applied_risk_scales", risk_scales)
        for name, array in {**weight_arrays, **boolean_arrays}.items():
            array.setflags(write=False)
            object.__setattr__(self, name, array)


@dataclass(frozen=True, slots=True)
class ActionPathEvaluation:
    actions: np.ndarray
    performance: PathPerformanceMetrics
    collapse_evidence: ActionPathCollapseEvidence
    step_economics: ActionPathStepEconomics | None = None
    execution_trace: ActionPathExecutionTrace | None = None

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
        trace = self.execution_trace
        if trace is not None:
            if not isinstance(trace, ActionPathExecutionTrace):
                raise TypeError("execution_trace must be ActionPathExecutionTrace")
            if trace.pre_action_weights.shape != actions.shape:
                raise ValueError("execution trace does not match evaluated actions")
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


def _observation_weights(
    observation: object,
    *,
    shape: tuple[int, ...],
    field: str,
) -> np.ndarray | None:
    if not isinstance(observation, Mapping) or "current_weights" not in observation:
        return None
    weights = np.asarray(observation["current_weights"], dtype=np.float64).reshape(-1)
    if weights.shape != shape or not np.isfinite(weights).all():
        raise ValueError(f"{field} current_weights do not match evaluation action")
    return weights


def _book_weights(
    environment: EvaluationEnvironment,
    *,
    shape: tuple[int, ...],
    field: str,
) -> np.ndarray:
    book = getattr(environment, "hybrid", None)
    weights = np.asarray(getattr(book, "weights", None), dtype=np.float64).reshape(-1)
    if weights.shape != shape or not np.isfinite(weights).all():
        raise ValueError(f"{field} book weights do not match evaluation action")
    return weights


def _risk_constrained_weights(
    info: Mapping[str, object], *, shape: tuple[int, ...]
) -> np.ndarray:
    risk = info.get("hybrid_risk")
    if risk is None:
        raise ValueError("evaluation info is missing hybrid risk")
    weights = np.asarray(getattr(risk, "weights", None), dtype=np.float64).reshape(-1)
    if weights.shape != shape or not np.isfinite(weights).all():
        raise ValueError("hybrid risk weights do not match evaluation action")
    return weights


def _applied_risk_scale(info: Mapping[str, object]) -> float:
    risk = info.get("hybrid_risk")
    if risk is None:
        raise ValueError("evaluation info is missing hybrid risk")
    value = getattr(risk, "risk_scale", None)
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not np.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise ValueError("hybrid risk is missing a valid applied risk scale")
    return float(value)


def _hard_risk_projection_violation(
    environment: EvaluationEnvironment,
    projected_weights: np.ndarray,
    *,
    applied_risk_scale: float,
) -> bool:
    risk_engine = getattr(environment, "pre_trade_risk", None)
    config = getattr(risk_engine, "config", None)
    if config is None:
        raise ValueError(
            "evaluation environment is missing authoritative pre-trade risk"
        )
    max_abs_weight = getattr(config, "max_abs_weight", None)
    max_gross = getattr(config, "max_gross", None)
    tolerance = getattr(config, "fail_closed_tolerance", None)
    for value in (max_abs_weight, max_gross, tolerance):
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not np.isfinite(float(value))
        ):
            raise ValueError("evaluation pre-trade risk limits are invalid")
    assert isinstance(max_abs_weight, int | float) and not isinstance(
        max_abs_weight, bool
    )
    assert isinstance(max_gross, int | float) and not isinstance(max_gross, bool)
    assert isinstance(tolerance, int | float) and not isinstance(tolerance, bool)
    scale_value = float(applied_risk_scale)
    max_abs_value = float(max_abs_weight)
    max_gross_value = float(max_gross)
    tolerance_value = float(tolerance)
    if not 0.0 <= scale_value <= 1.0 or tolerance_value < 0.0:
        raise ValueError("evaluation pre-trade risk limits are invalid")
    absolute = np.abs(projected_weights)
    return bool(
        np.max(absolute) > max_abs_value * scale_value + tolerance_value
        or np.sum(absolute) > max_gross_value * scale_value + tolerance_value
        or (scale_value == 0.0 and np.any(absolute > tolerance_value))
    )


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
    pre_action_weights: list[np.ndarray] = []
    risk_constrained_weights: list[np.ndarray] = []
    post_step_weights: list[np.ndarray] = []
    applied_risk_scales: list[float] = []
    strategy_intent_changes: list[bool] = []
    realized_state_follows: list[bool] = []
    rebalance_reassertions: list[bool] = []
    hard_risk_violations: list[bool] = []
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
    previous_active: np.ndarray | None = None
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
        observed_reference = _observation_weights(
            observation,
            shape=action.shape,
            field="pre-action observation",
        )
        proposal_reference = observed_reference
        if proposal_reference is None:
            proposal_reference = (
                np.zeros(action.shape, dtype=np.float64)
                if previous_submitted is None
                else previous_submitted.astype(np.float64, copy=False)
            )
        trace_reference = (
            observed_reference
            if observed_reference is not None
            else _book_weights(
                environment,
                shape=action.shape,
                field="pre-action",
            )
        )
        active = np.ones(action.shape, dtype=np.bool_)
        if isinstance(observation, Mapping) and "active" in observation:
            active_values = np.asarray(observation["active"]).reshape(-1)
            if active_values.shape != action.shape:
                raise ValueError("active mask does not match evaluation action")
            active = active_values > 0.5
        proposed = active & (
            np.abs(action - proposal_reference) > action_change_tolerance
        )
        trace_proposed = active & (
            np.abs(action - trace_reference) > action_change_tolerance
        )
        if previous_submitted is None or previous_active is None:
            strategy_intent_change = bool(np.any(trace_proposed))
            realized_state_follow = False
            rebalance_reassertion = False
        else:
            continuously_active = active & previous_active
            newly_active = active & ~previous_active
            drifted = continuously_active & (
                np.abs(trace_reference - previous_submitted) > action_change_tolerance
            )
            matches_current = (
                np.abs(action - trace_reference) <= action_change_tolerance
            )
            matches_previous = (
                np.abs(action - previous_submitted) <= action_change_tolerance
            )
            changed_from_previous = (
                np.abs(action - previous_submitted) > action_change_tolerance
            )
            realized_state_follow = bool(np.any(drifted & matches_current))
            rebalance_reassertion = bool(
                np.any(drifted & matches_previous & trace_proposed)
            )
            strategy_intent_change = bool(
                np.any(
                    (continuously_active & changed_from_previous & ~matches_current)
                    | (newly_active & proposed)
                )
            )
        active_dimension_count += int(np.count_nonzero(active))
        inactive_dimension_count += int(np.count_nonzero(~active))
        proposal_distance_count += int(np.count_nonzero(proposed))
        submitted_change = bool(np.any(proposed))
        submitted_change_count += int(submitted_change)
        evaluated_actions.append(action.copy())
        pre_action_weights.append(trace_reference.copy())
        strategy_intent_changes.append(strategy_intent_change)
        realized_state_follows.append(realized_state_follow)
        rebalance_reassertions.append(rebalance_reassertion)
        previous_submitted = action.copy()
        previous_active = active.copy()
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
        constrained = _risk_constrained_weights(info, shape=action.shape)
        observed_post_weights = _observation_weights(
            observation,
            shape=action.shape,
            field="post-step observation",
        )
        post_weights = (
            observed_post_weights
            if observed_post_weights is not None
            else _book_weights(
                environment,
                shape=action.shape,
                field="post-step",
            )
        )
        applied_risk_scale = _applied_risk_scale(info)
        risk_constrained_weights.append(constrained.copy())
        post_step_weights.append(post_weights.copy())
        applied_risk_scales.append(applied_risk_scale)
        hard_risk_violations.append(
            _hard_risk_projection_violation(
                environment,
                constrained,
                applied_risk_scale=applied_risk_scale,
            )
        )
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
    trace = ActionPathExecutionTrace(
        pre_action_weights=np.stack(pre_action_weights, axis=0),
        risk_constrained_weights=np.stack(risk_constrained_weights, axis=0),
        post_step_weights=np.stack(post_step_weights, axis=0),
        applied_risk_scales=np.asarray(applied_risk_scales, dtype=np.float64),
        strategy_intent_changes=np.asarray(strategy_intent_changes, dtype=np.bool_),
        realized_state_follows=np.asarray(realized_state_follows, dtype=np.bool_),
        rebalance_reassertions=np.asarray(rebalance_reassertions, dtype=np.bool_),
        hard_risk_violations=np.asarray(hard_risk_violations, dtype=np.bool_),
    )
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
        hard_risk_violation=bool(np.any(trace.hard_risk_violations)),
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
        execution_trace=trace,
    )


__all__ = [
    "ActionPathEvaluation",
    "ActionPathExecutionTrace",
    "ActionPathStepEconomics",
    "EvaluationEnvironment",
    "evaluate_action_path",
]
