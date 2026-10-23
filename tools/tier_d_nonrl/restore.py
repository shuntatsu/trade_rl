"""Strict frozen declaration decoding through existing public constructors."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np

from trade_rl.artifacts import canonical_json_bytes, content_digest
from trade_rl.evaluation.allocation_global_execution import (
    GlobalAllocationExecutionPlan,
)
from trade_rl.evaluation.allocation_settled_forecast_comparison import (
    SettledNonRLComparisonPlan,
    TerminalAllocationNonRLControls,
)
from trade_rl.evaluation.allocation_terminal_execution import (
    TerminalAllocationExecutionPlan,
)
from trade_rl.evaluation.forecast_allocation import HorizonCostEstimates
from trade_rl.evaluation.objectives import (
    BoundObjectiveClock,
    CapitalContract,
    FinancialClockContract,
    ObjectiveContract,
)

CostTables = dict[tuple[int, str], tuple[HorizonCostEstimates, ...]]


def closed(value: object, keys: set[str]) -> dict[str, Any]:
    if type(value) is not dict or set(value) != keys:
        raise ValueError("frozen artifact has missing or unexpected fields")
    return value


def _list(value: object) -> list[Any]:
    if type(value) is not list:
        raise ValueError("frozen roster must be a JSON list")
    return value


def _same(original: object, restored: object) -> None:
    if canonical_json_bytes(original) != canonical_json_bytes(restored):
        raise ValueError("frozen artifact changed during public-constructor roundtrip")


def _pin(value: object, expected_digest: str) -> None:
    if type(expected_digest) is not str or content_digest(value) != expected_digest:
        raise ValueError("frozen artifact digest differs from its original pin")


def _terminal(value: object) -> TerminalAllocationExecutionPlan:
    raw = closed(
        value, {"schema", "native_plan", "recipe", "objective", "clock", "bound_digest"}
    )
    native = GlobalAllocationExecutionPlan.from_payload(raw["native_plan"])
    objective_raw = closed(
        raw["objective"],
        {
            "schema",
            "objective_id",
            "tax_treatment",
            "infrastructure_cost_treatment",
            "external_cash_flow_convention",
            "capital",
            "evaluation_start",
            "evaluation_stop_exclusive",
            "terminal_valuation",
            "economics_digest",
            "risk_digest",
            "deployment_recipe_digest",
            "maximum_drawdown",
        },
    )
    capital_raw = closed(
        objective_raw["capital"], {"mode", "currency", "initial_equities"}
    )
    capital = CapitalContract(
        capital_raw["mode"],
        capital_raw["currency"],
        tuple(_list(capital_raw["initial_equities"])),
    )
    objective = ObjectiveContract(
        capital,
        datetime.fromisoformat(objective_raw["evaluation_start"]),
        datetime.fromisoformat(objective_raw["evaluation_stop_exclusive"]),
        objective_raw["terminal_valuation"],
        objective_raw["economics_digest"],
        objective_raw["risk_digest"],
        objective_raw["deployment_recipe_digest"],
        objective_raw["maximum_drawdown"],
    )
    clock_raw = closed(
        raw["clock"],
        {
            "schema",
            "decision_interval_seconds",
            "execution_interval_seconds",
            "reward_interval_seconds",
            "economic_horizon_seconds",
            "rollout_steps",
            "gamma",
            "gae_lambda",
            "reward_schema",
        },
    )
    clock = FinancialClockContract(
        **{key: item for key, item in clock_raw.items() if key != "schema"}
    )
    recipe = closed(
        raw["recipe"],
        {
            "schema",
            "native_plan_digest",
            "policy_range",
            "closing_range",
            "closing_clocks_ns",
            "closing_source_digest",
            "clock",
            "terminal_valuation",
            "target",
            "coverage",
        },
    )
    interval = _list(recipe["closing_range"])
    if len(interval) != 2:
        raise ValueError("terminal closing range must be an exact pair")
    plan = TerminalAllocationExecutionPlan(
        native,
        interval[1],
        tuple(_list(recipe["closing_clocks_ns"])),
        recipe["closing_source_digest"],
        BoundObjectiveClock(objective, clock),
    )
    _same(raw, plan.payload)
    return plan


def restore_matrix(
    value: object, *, expected_digest: str
) -> SettledNonRLComparisonPlan:
    """Restore the original full declaration; never execute or re-declare it."""
    try:
        _pin(value, expected_digest)
        raw = closed(value, {"schema", "accounts", "scenarios", "groups"})
        controls = []
        for group in _list(raw["groups"]):
            group = closed(group, {"schema", "plans"})
            roles = closed(group["plans"], {"nonrl", "cash"})
            controls.append(
                TerminalAllocationNonRLControls(
                    _terminal(roles["nonrl"]), _terminal(roles["cash"])
                )
            )
        result = SettledNonRLComparisonPlan(
            tuple(controls),
            tuple(_list(raw["accounts"])),
            tuple(_list(raw["scenarios"])),
        )
        _same(raw, result.payload)
        return result
    except (KeyError, TypeError, OverflowError, IndexError) as error:
        raise ValueError("invalid frozen matrix") from error


def costs_payload(tables: CostTables) -> dict[str, Any]:
    return {
        "schema": "tier_d_cost_tables_v1",
        "tables": [
            {
                "horizon_hours": horizon,
                "scenario": scenario,
                "rows": [row.payload() for row in tables[horizon, scenario]],
            }
            for horizon, scenario in sorted(tables)
        ],
    }


def restore_costs(value: object, *, expected_digest: str) -> CostTables:
    """Decode the original cost rows, without invoking any estimator."""
    try:
        _pin(value, expected_digest)
        raw = closed(value, {"schema", "tables"})
        result: CostTables = {}
        for group in _list(raw["tables"]):
            group = closed(group, {"horizon_hours", "scenario", "rows"})
            horizon, scenario = group["horizon_hours"], group["scenario"]
            if (
                type(horizon) is not int
                or horizon <= 0
                or scenario not in {"BASEL0", "COST2L0"}
            ):
                raise ValueError("invalid frozen cost-table key")
            key = horizon, scenario
            if key in result:
                raise ValueError("duplicate frozen cost-table key")
            estimates = []
            for row in _list(group["rows"]):
                row = closed(
                    row,
                    {
                        "schema",
                        "valuation_basis",
                        "cost_basis",
                        "symbol",
                        "decision_time",
                        "available_at",
                        "horizon_end",
                        "source_identity",
                        "buy_cost",
                        "sell_cost",
                        "exit_cost",
                        "funding_return",
                        "borrow_return",
                        "cash_return",
                    },
                )
                arguments = {
                    name: item
                    for name, item in row.items()
                    if name not in {"schema", "cost_basis"}
                }
                for name in ("decision_time", "available_at", "horizon_end"):
                    if type(row[name]) is not int:
                        raise ValueError(
                            "cost clocks must be native integer nanoseconds"
                        )
                    arguments[name] = np.datetime64(row[name], "ns")
                estimate = HorizonCostEstimates(**arguments)
                _same(row, estimate.payload())
                estimates.append(estimate)
            if not estimates:
                raise ValueError("empty frozen cost table")
            result[key] = tuple(estimates)
        if not result:
            raise ValueError("missing frozen cost tables")
        _same(raw, costs_payload(result))
        return result
    except (KeyError, TypeError, OverflowError, IndexError) as error:
        raise ValueError("invalid frozen cost tables") from error
