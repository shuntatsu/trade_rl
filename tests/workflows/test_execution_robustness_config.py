from __future__ import annotations

from dataclasses import fields

import pytest

from trade_rl.simulation.execution import ExecutionCostConfig, ExecutionRuleStress
from trade_rl.workflows.market_walk_forward_config import ExecutionSensitivityScenario


def _scenario(**overrides: object) -> ExecutionSensitivityScenario:
    values: dict[str, object] = {
        "name": "joint_execution_adverse",
        "tick_size_factor": 2.0,
        "lot_size_factor": 3.0,
        "minimum_notional_factor": 4.0,
        "adverse_tick_rounding": True,
        "report_only": True,
        "fee_multiplier": 1.5,
        "spread_multiplier": 2.0,
        "impact_multiplier": 3.0,
        "slippage_std_multiplier": 2.0,
        "slippage_std_floor": 0.0003,
        "participation_fraction": 0.5,
        "minimum_order_latency_bars": 2,
        "tail_slippage_probability_floor": 0.01,
        "tail_slippage_multiplier_floor": 7.0,
        "borrow_rate_multiplier": 2.0,
    }
    values.update(overrides)
    return ExecutionSensitivityScenario(**values)  # type: ignore[arg-type]


def test_execution_sensitivity_scenario_declares_environment_cost_stress_fields() -> (
    None
):
    field_names = {field.name for field in fields(ExecutionSensitivityScenario)}

    assert {
        "fee_multiplier",
        "spread_multiplier",
        "impact_multiplier",
        "slippage_std_multiplier",
        "slippage_std_floor",
        "participation_fraction",
        "minimum_order_latency_bars",
        "tail_slippage_probability_floor",
        "tail_slippage_multiplier_floor",
        "borrow_rate_multiplier",
    } <= field_names


def test_execution_sensitivity_scenario_builds_identity_bound_cost_stress() -> None:
    stress = _scenario().stress()

    assert stress.digest_payload() == {
        "adverse_tick_rounding": True,
        "borrow_rate_multiplier": 2.0,
        "fee_multiplier": 1.5,
        "impact_multiplier": 3.0,
        "lot_size_factor": 3.0,
        "minimum_notional_factor": 4.0,
        "minimum_order_latency_bars": 2,
        "name": "joint_execution_adverse",
        "participation_fraction": 0.5,
        "schema_version": "execution_environment_stress_v1",
        "slippage_std_floor": 0.0003,
        "slippage_std_multiplier": 2.0,
        "spread_multiplier": 2.0,
        "tail_slippage_multiplier_floor": 7.0,
        "tail_slippage_probability_floor": 0.01,
        "tick_size_factor": 2.0,
    }


def test_execution_sensitivity_scenario_applies_cost_stress_immutably() -> None:
    base = ExecutionCostConfig(
        fee_rate=0.001,
        maker_fee_rate=0.0002,
        taker_fee_rate=0.0003,
        spread_rate=0.0004,
        impact_rate=0.0005,
        max_participation_rate=0.2,
        slippage_std=0.0001,
        tail_slippage_probability=0.002,
        tail_slippage_multiplier=3.0,
        borrow_rate_multiplier=1.5,
        order_latency_bars=0,
    )

    stressed = _scenario().stress().apply(base)

    assert stressed is not base
    assert stressed.fee_rate == pytest.approx(0.0015)
    assert stressed.maker_fee_rate == pytest.approx(0.0003)
    assert stressed.taker_fee_rate == pytest.approx(0.00045)
    assert stressed.spread_rate == pytest.approx(0.0008)
    assert stressed.impact_rate == pytest.approx(0.0015)
    assert stressed.slippage_std == pytest.approx(0.0003)
    assert stressed.max_participation_rate == pytest.approx(0.1)
    assert stressed.order_latency_bars == 2
    assert stressed.tail_slippage_probability == pytest.approx(0.01)
    assert stressed.tail_slippage_multiplier == pytest.approx(7.0)
    assert stressed.borrow_rate_multiplier == pytest.approx(3.0)
    assert base.fee_rate == pytest.approx(0.001)
    assert base.max_participation_rate == pytest.approx(0.2)
    assert base.order_latency_bars == 0


def test_identity_execution_stress_preserves_legacy_rule_contract() -> None:
    scenario = ExecutionSensitivityScenario(
        name="joint_3x",
        tick_size_factor=3.0,
        lot_size_factor=3.0,
        minimum_notional_factor=3.0,
        adverse_tick_rounding=True,
        report_only=True,
    )

    stress = scenario.stress()

    assert type(stress) is ExecutionRuleStress
    assert scenario.digest_payload()["schema_version"] == "execution_rule_stress_v1"


def test_required_standard_scenario_rejects_execution_cost_stress() -> None:
    with pytest.raises(
        ValueError,
        match="standard execution sensitivity scenario",
    ):
        ExecutionSensitivityScenario(
            name="joint_2x",
            tick_size_factor=2.0,
            lot_size_factor=2.0,
            minimum_notional_factor=2.0,
            adverse_tick_rounding=True,
            fee_multiplier=1.5,
        )


@pytest.mark.parametrize(
    ("field_name", "value"),
    (
        ("fee_multiplier", 0.99),
        ("spread_multiplier", 0.99),
        ("impact_multiplier", 0.99),
        ("slippage_std_multiplier", 0.99),
        ("slippage_std_floor", -0.0001),
        ("participation_fraction", 0.0),
        ("participation_fraction", 1.01),
        ("minimum_order_latency_bars", -1),
        ("minimum_order_latency_bars", True),
        ("tail_slippage_probability_floor", -0.01),
        ("tail_slippage_probability_floor", 1.01),
        ("tail_slippage_multiplier_floor", 0.5),
        ("borrow_rate_multiplier", 0.99),
    ),
)
def test_execution_sensitivity_scenario_rejects_invalid_cost_stress(
    field_name: str,
    value: object,
) -> None:
    with pytest.raises(ValueError, match=field_name):
        _scenario(**{field_name: value})
