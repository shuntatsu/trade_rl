"""Independent complete dictionaries for the supported frozen execution route."""

from __future__ import annotations

import hashlib
from typing import Any

from tools.market_replay_oracle.identity import canonical

_RISK = {
    "drawdown_start": 0.1,
    "drawdown_stop": 0.2,
    "emergency_turnover_override": True,
    "fail_closed_tolerance": 1e-10,
    "max_abs_weight": 1.0,
    "max_gross": 1.0,
    "max_turnover": None,
}


def _digest(value: object) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def effective_profile(
    scenario: str, actor_advances: object, risk: dict[str, Any]
) -> dict[str, Any]:
    """Materialize static assumptions; Dataset rows remain separately bound."""
    if scenario not in {"BASEL0", "COST2L0", "BASEL2"}:
        raise ValueError("unsupported frozen execution scenario")
    if type(actor_advances) is not int or actor_advances <= 0:
        raise ValueError("actor advances must be a positive integer")
    if canonical(risk) != canonical(_RISK):
        raise ValueError("unsupported frozen risk assumptions")
    policy = {
        "allow_short": True,
        "borrow_rate_multiplier": 1.0,
        "collateral_haircut": 1.0,
        "fee_rate": 0.0,
        "impact_rate": 0.0001,
        "limit_offset_rate": 0.0005,
        "lot_size": 0.0,
        "maintenance_margin_rate": 0.25,
        "maker_fee_rate": 0.0,
        "margin_mode": "cross",
        "max_leverage": 1.0,
        "max_participation_rate": 0.05,
        "minimum_notional": 0.0,
        "multiplier": 2.0 if scenario == "COST2L0" else 1.0,
        "order_latency_bars": 2 if scenario == "BASEL2" else 0,
        "order_type": "market",
        "partial_fill_carry": True,
        "path_mode": "conservative",
        "processing_bar_volume_capacity": False,
        "random_seed": 0,
        "schema_version": "execution_policy_v2",
        "slippage_std": 0.0,
        "spread_rate": 0.0,
        "tail_slippage_multiplier": 5.0,
        "tail_slippage_probability": 0.0,
        "taker_fee_rate": 0.0,
        "tick_size": 0.0,
        "trigger_volume_fractions": [1.0, 0.5, 0.25, 0.0],
    }
    cost_digest = _digest(policy)
    retained_digest = _digest(
        {
            "schema_version": "insolvency_execution_policy_v1",
            "base_policy_digest": cost_digest,
            "insolvency_valuation": "retain_debt",
        }
    )
    return {
        "cost_policy": policy,
        "cost_policy_digest": cost_digest,
        "execution_policy_digest": retained_digest,
        "market_order_profile": None,
        "rule_stress": {
            "adverse_tick_rounding": False,
            "lot_size_factor": 1.0,
            "minimum_notional_factor": 1.0,
            "name": "nominal",
            "schema_version": "execution_rule_stress_v1",
            "tick_size_factor": 1.0,
        },
        "runtime_profile": {
            "schema": "allocation_runtime_profile_v1",
            "economics_digest": retained_digest,
            "risk_digest": _digest(risk),
            "initial_capital": 100000.0,
            "currency": "USDT",
            "decision_interval_seconds": 3600,
            "economic_horizon_seconds": actor_advances * 3600,
            "insolvency_valuation": "retain_debt",
            "calendar_kind": "continuous_24_7",
            "execution_bar_hours": 1.0,
        },
    }
