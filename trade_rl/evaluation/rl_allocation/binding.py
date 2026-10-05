"""Bind declarations to actual independent-account, regular-bar execution."""

from __future__ import annotations

from datetime import UTC

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.objectives import BoundObjectiveClock
from trade_rl.risk import PreTradeRiskConfig
from trade_rl.simulation import MarketExecutor
from trade_rl.strategies.forecasts.training_trace import _timestamp


def validate_runtime_binding(
    dataset: MarketDataset,
    bound: BoundObjectiveClock,
    executor: MarketExecutor,
    risk: PreTradeRiskConfig,
    recipe_digest: str,
    *,
    start_index: int,
    stop_index: int,
) -> None:
    if not isinstance(bound, BoundObjectiveClock):
        raise ValueError("a bound objective clock is required")
    if (
        any(
            isinstance(i, bool) or not isinstance(i, int)
            for i in (start_index, stop_index)
        )
        or not 0 <= start_index < stop_index < dataset.n_bars
    ):
        raise ValueError("allocation episode interval is outside the Dataset")
    objective, clock = bound.objective, bound.clock
    if not clock.terminal_profit_aligned:
        raise ValueError("profit clock requires equity_delta reward and gamma=1")
    if clock.decision_interval_seconds != clock.execution_interval_seconds:
        raise ValueError("allocation runtime requires one processing bar per decision")
    if (
        objective.capital.account_mode != "independent_symbol"
        or len(objective.capital.initial_equities) != 1
        or objective.terminal_valuation != "marked_continuation"
    ):
        raise ValueError(
            "allocation runtime requires one independent marked-continuation account"
        )
    if objective.maximum_drawdown != risk.drawdown_stop:
        raise ValueError("risk profile drawdown differs from objective")
    if (
        objective.economics_digest != executor.execution_policy_digest
        or objective.risk_digest != content_digest(risk)
        or objective.deployment_recipe_digest != recipe_digest
    ):
        raise ValueError(
            "declared profile or action recipe differs from actual runtime"
        )
    times = tuple(
        _timestamp(t, field="episode_time")
        for t in dataset.timestamps[start_index : stop_index + 1]
    )
    for actual, declared in (
        (times[0], objective.evaluation_start),
        (times[-1], objective.evaluation_stop_exclusive),
    ):
        expected = _timestamp(
            np.datetime64(declared.astimezone(UTC).replace(tzinfo=None), "us"),
            field="declared UTC endpoint",
        )
        if actual != expected:
            raise ValueError(
                "Dataset episode endpoints differ from declared UTC horizon"
            )
    ticks = tuple(int(t.astype(np.int64)) for t in times)
    if any(
        b - a != clock.execution_interval_seconds * 1_000_000_000
        for a, b in zip(ticks, ticks[1:])
    ):
        raise ValueError("Dataset intervals differ from declared financial clock")
