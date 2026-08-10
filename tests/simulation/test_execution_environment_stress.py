from __future__ import annotations

import pytest

from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.simulation.execution_stress import ExecutionEnvironmentStress


def _base_cost() -> ExecutionCostConfig:
    return ExecutionCostConfig(
        fee_rate=0.001,
        maker_fee_rate=0.0002,
        taker_fee_rate=0.0003,
        spread_rate=0.0004,
        impact_rate=0.0005,
        max_participation_rate=0.2,
        slippage_std=0.0006,
        tail_slippage_probability=0.01,
        tail_slippage_multiplier=4.0,
        borrow_rate_multiplier=1.5,
        order_latency_bars=1,
    )


def test_environment_stress_applies_all_dimensions_without_mutating_base() -> None:
    base = _base_cost()
    stress = ExecutionEnvironmentStress(
        name="joint-adverse",
        fee_multiplier=2.0,
        spread_multiplier=3.0,
        impact_multiplier=4.0,
        slippage_std_multiplier=5.0,
        participation_fraction=0.25,
        minimum_order_latency_bars=3,
        tail_slippage_probability_floor=0.05,
        tail_slippage_multiplier_floor=8.0,
        borrow_rate_multiplier=2.0,
    )

    stressed = stress.apply(base)

    assert stressed is not base
    assert base.fee_rate == pytest.approx(0.001)
    assert stressed.fee_rate == pytest.approx(0.002)
    assert stressed.maker_fee_rate == pytest.approx(0.0004)
    assert stressed.taker_fee_rate == pytest.approx(0.0006)
    assert stressed.spread_rate == pytest.approx(0.0012)
    assert stressed.impact_rate == pytest.approx(0.002)
    assert stressed.slippage_std == pytest.approx(0.003)
    assert stressed.max_participation_rate == pytest.approx(0.05)
    assert stressed.order_latency_bars == 3
    assert stressed.tail_slippage_probability == pytest.approx(0.05)
    assert stressed.tail_slippage_multiplier == pytest.approx(8.0)
    assert stressed.borrow_rate_multiplier == pytest.approx(3.0)
    assert stress.digest_payload()["schema_version"] == (
        "execution_environment_stress_v1"
    )


def test_neutral_environment_stress_preserves_cost_identity() -> None:
    base = _base_cost()

    assert ExecutionEnvironmentStress(name="neutral").apply(base) is base


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("fee_multiplier", 0.5),
        ("spread_multiplier", float("inf")),
        ("impact_multiplier", -1.0),
        ("slippage_std_multiplier", 0.0),
        ("participation_fraction", 0.0),
        ("participation_fraction", 1.1),
        ("minimum_order_latency_bars", -1),
        ("tail_slippage_probability_floor", 1.1),
        ("tail_slippage_multiplier_floor", -1.0),
        ("borrow_rate_multiplier", 0.9),
    ],
)
def test_environment_stress_rejects_non_adverse_or_invalid_values(
    field: str,
    value: object,
) -> None:
    with pytest.raises(ValueError):
        ExecutionEnvironmentStress(name="invalid", **{field: value})
