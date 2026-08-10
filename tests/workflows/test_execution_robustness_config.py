from __future__ import annotations

from dataclasses import fields

from trade_rl.workflows.market_walk_forward_config import ExecutionSensitivityScenario


def test_execution_sensitivity_scenario_declares_environment_cost_stress_fields() -> None:
    field_names = {field.name for field in fields(ExecutionSensitivityScenario)}

    assert {
        "fee_multiplier",
        "spread_multiplier",
        "impact_multiplier",
        "slippage_std_multiplier",
        "participation_fraction",
        "minimum_order_latency_bars",
        "tail_slippage_probability_floor",
        "tail_slippage_multiplier_floor",
        "borrow_rate_multiplier",
    } <= field_names
