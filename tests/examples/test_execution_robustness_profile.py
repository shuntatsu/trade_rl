from __future__ import annotations

from pathlib import Path

import pytest

from trade_rl.simulation.execution import ExecutionRuleStress
from trade_rl.simulation.execution_stress import ExecutionEnvironmentStress
from trade_rl.workflows.market_walk_forward_config import MarketWalkForwardConfig

ROOT = Path(__file__).resolve().parents[2]
PROFILE = (
    ROOT
    / "examples"
    / "binance-multitimeframe"
    / "walk-forward-target-weight-execution-robustness.json"
)
EXPECTED_CANDIDATES = (
    "target-weight-growth-gamma-one-ppo",
    "target-weight-constrained-growth-gamma-one",
)
EXPECTED_REPORT_ONLY_STRESSES = {
    "fee_spread_2x": {
        "fee_multiplier": 2.0,
        "spread_multiplier": 2.0,
    },
    "impact_2x": {
        "impact_multiplier": 2.0,
    },
    "capacity_half": {
        "participation_fraction": 0.5,
    },
    "latency_1bar": {
        "minimum_order_latency_bars": 1,
    },
    "tail_slippage_adverse": {
        "slippage_std_floor": 0.0005,
        "tail_slippage_probability_floor": 0.01,
        "tail_slippage_multiplier_floor": 5.0,
    },
    "borrow_2x": {
        "borrow_rate_multiplier": 2.0,
    },
    "joint_execution_adverse": {
        "tick_size_factor": 2.0,
        "lot_size_factor": 2.0,
        "minimum_notional_factor": 2.0,
        "adverse_tick_rounding": True,
        "fee_multiplier": 1.5,
        "spread_multiplier": 2.0,
        "impact_multiplier": 2.0,
        "slippage_std_multiplier": 2.0,
        "slippage_std_floor": 0.0005,
        "participation_fraction": 0.5,
        "minimum_order_latency_bars": 1,
        "tail_slippage_probability_floor": 0.01,
        "tail_slippage_multiplier_floor": 5.0,
        "borrow_rate_multiplier": 2.0,
    },
}


def test_execution_robustness_profile_loads_report_only_stresses() -> None:
    assert PROFILE.is_file(), f"missing maintained robustness profile: {PROFILE}"

    config = MarketWalkForwardConfig.from_json(PROFILE, n_bars=192_672)

    assert tuple(candidate.name for candidate in config.candidates) == (
        EXPECTED_CANDIDATES
    )
    sensitivity = config.execution_sensitivity
    assert sensitivity.required_scenario == "joint_2x"
    by_name = {scenario.name: scenario for scenario in sensitivity.scenarios}
    assert type(by_name["joint_2x"].stress()) is ExecutionRuleStress
    assert type(by_name["joint_3x"].stress()) is ExecutionRuleStress
    for name, expected in EXPECTED_REPORT_ONLY_STRESSES.items():
        scenario = by_name[name]
        assert scenario.report_only is True
        stress = scenario.stress()
        assert isinstance(stress, ExecutionEnvironmentStress)
        assert stress.adverse_tick_rounding is expected.get(
            "adverse_tick_rounding",
            False,
        )
        for field_name, value in expected.items():
            observed = getattr(stress, field_name)
            if isinstance(value, float):
                assert observed == pytest.approx(value)
            else:
                assert observed == value


def test_execution_robustness_profile_does_not_widen_required_gate() -> None:
    assert PROFILE.is_file(), f"missing maintained robustness profile: {PROFILE}"

    config = MarketWalkForwardConfig.from_json(PROFILE, n_bars=192_672)
    sensitivity = config.execution_sensitivity
    required = next(
        scenario
        for scenario in sensitivity.scenarios
        if scenario.name == sensitivity.required_scenario
    )

    assert required.name == "joint_2x"
    assert required.report_only is False
    assert type(required.stress()) is ExecutionRuleStress
