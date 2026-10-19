"""Hand-declared scenario policies, independent of native constructors."""

from __future__ import annotations

import pytest

from tools.market_replay_oracle.profiles import effective_profile

RISK = {
    "drawdown_start": 0.1,
    "drawdown_stop": 0.2,
    "emergency_turnover_override": True,
    "fail_closed_tolerance": 1e-10,
    "max_abs_weight": 1.0,
    "max_gross": 1.0,
    "max_turnover": None,
}
BASE = {
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
    "multiplier": 1.0,
    "order_latency_bars": 0,
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


@pytest.mark.parametrize(
    ("scenario", "multiplier", "latency", "digest"),
    [
        (
            "BASEL0",
            1.0,
            0,
            "cb3f470f066f2cc3c895ea8f674ff2e4c9df812816833f071950139720463102",
        ),
        (
            "COST2L0",
            2.0,
            0,
            "18dba7754b3a85e35f292ed7fe0e6aaeb6fed32719f42439e9e9753a2898ecd9",
        ),
        (
            "BASEL2",
            1.0,
            2,
            "d8840369ce817c2c4a49132b551c883d8e944badb74f5015df86f8ac406ebaad",
        ),
    ],
)
def test_complete_policy_and_retained_debt_identity(
    scenario: str, multiplier: float, latency: int, digest: str
) -> None:
    profile = effective_profile(scenario, 6527, RISK)
    assert profile["cost_policy"] == {
        **BASE,
        "multiplier": multiplier,
        "order_latency_bars": latency,
    }
    assert profile["execution_policy_digest"] == digest
    assert profile["market_order_profile"] is None
    assert profile["rule_stress"] == {
        "adverse_tick_rounding": False,
        "lot_size_factor": 1.0,
        "minimum_notional_factor": 1.0,
        "name": "nominal",
        "schema_version": "execution_rule_stress_v1",
        "tick_size_factor": 1.0,
    }
    assert profile["runtime_profile"] == {
        "schema": "allocation_runtime_profile_v1",
        "economics_digest": digest,
        "risk_digest": "4946570ab9636ee27e02ce3a8068fae400078686cbd61c14769f04bc586354fe",
        "initial_capital": 100000.0,
        "currency": "USDT",
        "decision_interval_seconds": 3600,
        "economic_horizon_seconds": 23497200,
        "insolvency_valuation": "retain_debt",
        "calendar_kind": "continuous_24_7",
        "execution_bar_hours": 1.0,
    }


def test_fresh_nested_results_and_fixed_risk() -> None:
    first = effective_profile("BASEL0", 6527, RISK)
    first["cost_policy"]["trigger_volume_fractions"][0] = 7.0
    first["runtime_profile"]["economic_horizon_seconds"] = 86400
    again = effective_profile("BASEL0", 6527, RISK)
    assert again["cost_policy"] == BASE
    assert again["runtime_profile"]["economic_horizon_seconds"] == 23497200
    with pytest.raises(ValueError):
        effective_profile("BASEL0", 6527, {**RISK, "drawdown_stop": 0.3})


@pytest.mark.parametrize("count", [True, 0, -1, 6527.0])
def test_invalid_actor_count_rejects(count: object) -> None:
    with pytest.raises(ValueError):
        effective_profile("BASEL0", count, RISK)


def test_unknown_scenario_does_not_inherit_base_defaults() -> None:
    with pytest.raises(ValueError):
        effective_profile("COST2L2", 6527, RISK)
