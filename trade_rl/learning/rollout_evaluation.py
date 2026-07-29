"""Exact simulator rollouts used by Oracle and behavior-cloning audits."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from trade_rl.evaluation.closed_trades import ClosedTradeTracker
from trade_rl.learning.evaluation import (
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
class ActionPathEvaluation:
    actions: np.ndarray
    performance: PathPerformanceMetrics

    def __post_init__(self) -> None:
        actions = np.asarray(self.actions, dtype=np.float32).copy(order="C")
        if (
            actions.ndim != 2
            or len(actions) != self.performance.step_count
            or not np.isfinite(actions).all()
        ):
            raise ValueError("evaluated actions do not match path metrics")
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
    evaluated_actions: list[np.ndarray] = []
    gross_returns: list[float] = []
    net_returns: list[float] = []
    rewards: list[float] = []
    turnover: list[float] = []
    costs: list[float] = []
    for offset in range(expected_count):
        if environment.current_index != start + offset:
            raise ValueError("evaluation environment advanced outside the range")
        if declared is not None:
            action = declared[offset]
        else:
            assert callable(predict)
            raw_action, _ = predict(observation, deterministic=True)
            action = np.asarray(raw_action, dtype=np.float32).reshape(-1)
        evaluated_actions.append(np.asarray(action, dtype=np.float32).copy())
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
        filled_turnover = getattr(execution, "filled_turnover", None)
        if isinstance(filled_turnover, bool) or not isinstance(
            filled_turnover, int | float
        ):
            raise ValueError("hybrid execution is missing filled_turnover")
        turnover.append(
            float(filled_turnover) + _liquidation_metric(info, "filled_turnover")
        )
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
    )
    return ActionPathEvaluation(
        actions=np.stack(evaluated_actions, axis=0),
        performance=performance,
    )


__all__ = ["ActionPathEvaluation", "EvaluationEnvironment", "evaluate_action_path"]
