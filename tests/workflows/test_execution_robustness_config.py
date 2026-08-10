from __future__ import annotations

from dataclasses import fields

import pytest

from trade_rl.simulation.execution_stress import ExecutionEnvironmentStress
from trade_rl.workflows.market_walk_forward_config import (
    ExecutionSensitivityConfig,
    ExecutionSensitivityScenario,
)


def test_execution_sensitivity_scenario_declares_environment_cost_stress_fields() -> (
    None
):
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


def test_scenario_stress_and_digest_include_environment_dimensions() -> None:
    scenario = ExecutionSensitivityScenario(
        name="fee-spread-2x",
        adverse_tick_rounding=False,
        report_only=True,
        fee_multiplier=2.0,
        spread_multiplier=2.0,
    )

    stress = scenario.stress()
    payload = scenario.digest_payload()

    assert isinstance(stress, ExecutionEnvironmentStress)
    assert stress.fee_multiplier == 2.0
    assert stress.spread_multiplier == 2.0
    assert payload["fee_multiplier"] == 2.0
    assert payload["spread_multiplier"] == 2.0
    assert payload["report_only"] is True
    assert payload["schema_version"] == "execution_environment_stress_v1"


def test_standard_gate_scenarios_cannot_hide_environment_cost_stress() -> None:
    scenarios = (
        ExecutionSensitivityScenario(name="nominal", adverse_tick_rounding=False),
        ExecutionSensitivityScenario(
            name="tick_2x",
            tick_size_factor=2.0,
            adverse_tick_rounding=True,
        ),
        ExecutionSensitivityScenario(
            name="lot_2x",
            lot_size_factor=2.0,
            adverse_tick_rounding=True,
        ),
        ExecutionSensitivityScenario(
            name="minimum_notional_2x",
            minimum_notional_factor=2.0,
            adverse_tick_rounding=True,
        ),
        ExecutionSensitivityScenario(
            name="joint_2x",
            tick_size_factor=2.0,
            lot_size_factor=2.0,
            minimum_notional_factor=2.0,
            adverse_tick_rounding=True,
            fee_multiplier=2.0,
        ),
        ExecutionSensitivityScenario(
            name="joint_5x",
            tick_size_factor=5.0,
            lot_size_factor=5.0,
            minimum_notional_factor=5.0,
            adverse_tick_rounding=True,
            report_only=True,
        ),
    )

    with pytest.raises(
        ValueError,
        match="standard execution sensitivity scenarios",
    ):
        ExecutionSensitivityConfig(
            scenarios=scenarios,
            required_scenario="joint_2x",
        )
