from __future__ import annotations

from dataclasses import replace

import pytest

from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.strategies.rl.portfolio_runtime_scenario import (
    PortfolioRuntimeExecutionScenarioV2,
    verify_portfolio_runtime_execution_scenario_v2_bytes,
)


def _scenario(
    *,
    role: str = "base",
    base: ExecutionCostConfig | None = None,
    runtime: ExecutionCostConfig | None = None,
    changed_fields: tuple[str, ...] = (),
    **overrides: object,
) -> PortfolioRuntimeExecutionScenarioV2:
    resolved_base = ExecutionCostConfig.zero() if base is None else base
    resolved_runtime = resolved_base if runtime is None else runtime
    values: dict[str, object] = {
        "schema_version": "portfolio_runtime_execution_scenario_v2",
        "scenario_id": "base-v1" if role == "base" else "fees-latency-v1",
        "role": role,
        "scenario_plan_sha256": "a" * 64,
        "training_context_sha256": "b" * 64,
        "transform_digest": "c" * 64,
        "checkpoint_sha256": "d" * 64,
        "base_execution_cost": resolved_base,
        "runtime_execution_cost": resolved_runtime,
        "changed_fields": changed_fields,
    }
    values.update(overrides)
    return PortfolioRuntimeExecutionScenarioV2(**values)  # type: ignore[arg-type]


def test_base_and_explicit_fee_latency_scenarios_bind_exact_config_changes() -> None:
    base = ExecutionCostConfig.zero()
    base_scenario = _scenario(base=base)
    assert base_scenario.changed_fields == ()
    assert base_scenario.base_execution_cost_digest == base.execution_policy_digest
    assert base_scenario.runtime_execution_cost_digest == base.execution_policy_digest

    runtime = replace(base, fee_rate=0.01, order_latency_bars=2)
    stress = _scenario(
        role="required_stress",
        base=base,
        runtime=runtime,
        changed_fields=("fee_rate", "order_latency_bars"),
    )
    assert stress.actual_changed_fields == ("fee_rate", "order_latency_bars")
    assert stress.to_payload()["schema_version"] == (
        "portfolio_runtime_execution_scenario_v2"
    )
    assert stress.to_payload()["base_execution_cost_digest"] == (
        base.execution_policy_digest
    )
    assert stress.to_payload()["runtime_execution_cost_digest"] == (
        runtime.execution_policy_digest
    )


def test_serialized_scenario_requires_expected_typed_value_and_external_digest() -> (
    None
):
    scenario = _scenario()
    raw = scenario.to_bytes()
    assert (
        verify_portfolio_runtime_execution_scenario_v2_bytes(
            raw, expected=scenario, expected_digest=scenario.digest
        )
        is scenario
    )

    with pytest.raises(ValueError, match="expected digest"):
        verify_portfolio_runtime_execution_scenario_v2_bytes(
            raw, expected=scenario, expected_digest="0" * 64
        )
    tampered = raw.replace(b'"scenario_id":"base-v1"', b'"scenario_id":"other-v1"')
    with pytest.raises(ValueError, match="expected declaration"):
        verify_portfolio_runtime_execution_scenario_v2_bytes(
            tampered, expected=scenario, expected_digest=scenario.digest
        )


def test_unknown_schema_and_extra_payload_fields_are_rejected_by_exact_bytes() -> None:
    scenario = _scenario()
    unknown_schema = scenario.to_bytes().replace(
        b"portfolio_runtime_execution_scenario_v2",
        b"portfolio_runtime_execution_scenario_v9",
    )
    with pytest.raises(ValueError, match="expected declaration"):
        verify_portfolio_runtime_execution_scenario_v2_bytes(
            unknown_schema, expected=scenario, expected_digest=scenario.digest
        )
    extra_field = scenario.to_bytes()[:-1] + b',"unexpected":true}'
    with pytest.raises(ValueError, match="expected declaration"):
        verify_portfolio_runtime_execution_scenario_v2_bytes(
            extra_field, expected=scenario, expected_digest=scenario.digest
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("borrow_rate_multiplier", 2.0),
        ("max_participation_rate", 0.5),
        ("allow_short", False),
        ("random_seed", 7),
        ("max_leverage", 2.0),
        ("processing_bar_volume_capacity", False),
    ],
)
def test_scenario_rejects_execution_changes_outside_version_allowlist(
    field: str, value: object
) -> None:
    base = ExecutionCostConfig.zero()
    runtime = replace(base, **{field: value})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unsupported execution field"):
        _scenario(
            role="required_stress",
            base=base,
            runtime=runtime,
            changed_fields=(field,),
        )


@pytest.mark.parametrize(
    "changed_fields",
    [
        (),
        ("order_latency_bars", "fee_rate"),
        ("fee_rate", "fee_rate"),
        ("fee_rate",),
        ("fee_rate", "order_latency_bars", "spread_rate"),
    ],
)
def test_declared_changed_fields_must_equal_sorted_canonical_diff(
    changed_fields: tuple[str, ...],
) -> None:
    base = ExecutionCostConfig.zero()
    runtime = replace(base, fee_rate=0.01, order_latency_bars=2)
    with pytest.raises(ValueError, match="changed_fields"):
        _scenario(
            role="required_stress",
            base=base,
            runtime=runtime,
            changed_fields=changed_fields,
        )


def test_base_role_cannot_contain_changes_and_stress_cannot_be_a_noop() -> None:
    base = ExecutionCostConfig.zero()
    changed = replace(base, fee_rate=0.01)
    with pytest.raises(ValueError, match="base scenario"):
        _scenario(
            base=base,
            runtime=changed,
            changed_fields=("fee_rate",),
        )
    with pytest.raises(ValueError, match="stress scenario must change"):
        _scenario(
            role="diagnostic_stress",
            base=base,
            runtime=base,
        )


def test_json_numeric_types_remain_distinct_in_execution_policy_diff() -> None:
    base = replace(ExecutionCostConfig.zero(), fee_rate=0.0)
    runtime = replace(base, fee_rate=0)
    scenario = _scenario(
        role="required_stress",
        base=base,
        runtime=runtime,
        changed_fields=("fee_rate",),
    )
    assert scenario.actual_changed_fields == ("fee_rate",)


def test_execution_config_nested_policy_inputs_must_be_immutable() -> None:
    mutable_fractions = [1.0, 0.5, 0.25, 0.0]
    config = replace(
        ExecutionCostConfig.zero(),
        trigger_volume_fractions=mutable_fractions,  # type: ignore[arg-type]
    )
    with pytest.raises(TypeError, match="four-value tuple"):
        _scenario(base=config)


def test_invalid_anchors_roles_ids_and_boolean_numeric_cost_values_fail_closed() -> (
    None
):
    scenario = _scenario()
    with pytest.raises(ValueError, match="SHA-256"):
        replace(scenario, training_context_sha256=" ")
    with pytest.raises(ValueError, match="role"):
        replace(scenario, role="research_gate")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="scenario_id"):
        replace(scenario, scenario_id="  ")

    base = ExecutionCostConfig.zero()
    bool_fee = replace(base, fee_rate=True)
    with pytest.raises(TypeError, match="non-boolean number"):
        _scenario(base=base, runtime=bool_fee)
