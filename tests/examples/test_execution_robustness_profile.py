from __future__ import annotations

from pathlib import Path

from trade_rl.workflows.market_walk_forward_config import MarketWalkForwardConfig

ROOT = Path(__file__).resolve().parents[2]
PROFILE = (
    ROOT
    / "examples"
    / "binance-multitimeframe"
    / "walk-forward-target-weight-execution-robustness.json"
)


def test_execution_robustness_profile_is_report_only_extension() -> None:
    config = MarketWalkForwardConfig.from_json(PROFILE, n_bars=40_000)
    scenarios = {
        scenario.name: scenario for scenario in config.execution_sensitivity.scenarios
    }

    assert config.execution_sensitivity.required_scenario == "joint_2x"
    assert {
        "fee_spread_2x",
        "impact_2x",
        "slippage_2x",
        "capacity_50pct",
        "latency_1bar",
        "tail_slippage",
        "borrow_2x",
        "joint_adverse",
    } <= scenarios.keys()
    for name, scenario in scenarios.items():
        if name not in {
            "nominal",
            "tick_2x",
            "lot_2x",
            "minimum_notional_2x",
            "joint_2x",
            "joint_5x",
        }:
            assert scenario.report_only is True

    joint = scenarios["joint_adverse"]
    stress = joint.stress()
    assert stress.fee_multiplier == 2.0
    assert stress.spread_multiplier == 2.0
    assert stress.impact_multiplier == 2.0
    assert stress.participation_fraction == 0.5
    assert stress.minimum_order_latency_bars == 1
    assert stress.tail_slippage_probability_floor == 0.01
    assert stress.tail_slippage_multiplier_floor == 10.0
    assert stress.borrow_rate_multiplier == 2.0

    payloads = {
        item["name"]: item
        for item in config.execution_sensitivity.digest_payload()["scenarios"]
    }
    assert payloads["joint_adverse"]["fee_multiplier"] == 2.0
    assert payloads["joint_adverse"]["report_only"] is True
