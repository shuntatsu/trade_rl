"""Bind training objective capital, economics and finite valuation dates."""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any

from trade_rl.strategies.rl.allocation_receipt_time import parse_objective_datetime


def _mapping(value: object, keys: set[str], field: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"{field} must contain exactly its declared fields")
    return value


def _finite(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def validate_allocation_objective(
    objective: object, recipe: dict[str, Any], recipe_digest: str, clock: dict[str, Any]
) -> tuple[datetime, datetime]:
    values = _mapping(
        objective,
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
        "objective",
    )
    fixed = {
        "schema": "net_profit_objective_v1",
        "objective_id": "expected_terminal_net_profit_v1",
        "tax_treatment": "pretax",
        "infrastructure_cost_treatment": "reported_separately",
        "external_cash_flow_convention": "net_deposits_positive_withdrawals_negative",
        "terminal_valuation": "marked_continuation",
    }
    if any(values[key] != expected for key, expected in fixed.items()):
        raise ValueError("unsupported allocation training objective")
    profile = recipe["runtime_profile"]
    capital = _mapping(
        values["capital"], {"mode", "currency", "initial_equities"}, "capital"
    )
    if capital != {
        "mode": "independent_symbol",
        "currency": profile["currency"],
        "initial_equities": [profile["initial_capital"]],
    }:
        raise ValueError("objective capital differs from the inference recipe")
    if values["deployment_recipe_digest"] != recipe_digest or any(
        values[key] != profile[key] for key in ("economics_digest", "risk_digest")
    ):
        raise ValueError("objective profiles differ from the inference recipe")
    if not 0 < _finite(values["maximum_drawdown"], "maximum_drawdown") <= 0.2:
        raise ValueError("drawdown must be within (0, 0.20]")
    try:
        start, stop = (
            parse_objective_datetime(values[key])
            for key in ("evaluation_start", "evaluation_stop_exclusive")
        )
    except (TypeError, ValueError) as error:
        raise ValueError("objective timestamps are malformed") from error
    if any(time.tzinfo is None or time.utcoffset() is None for time in (start, stop)):
        raise ValueError("objective timestamps must match the declared finite horizon")
    elapsed = stop - start
    microseconds = (
        elapsed.days * 86400 + elapsed.seconds
    ) * 1_000_000 + elapsed.microseconds
    if microseconds != clock["economic_horizon_seconds"] * 1_000_000:
        raise ValueError("objective timestamps must match the declared finite horizon")
    return start, stop
