from __future__ import annotations

import inspect

from trade_rl.rl.environment import ResidualMarketEnv
from trade_rl.rl.environment_policy_schedule_contract import (
    EnvironmentPolicyScheduleContract,
    EnvironmentPolicyScheduleContractBuilder,
)


def test_policy_schedule_contract_module_owns_maintained_boundary() -> None:
    assert EnvironmentPolicyScheduleContract.__module__ == (
        "trade_rl.rl.environment_policy_schedule_contract"
    )
    assert EnvironmentPolicyScheduleContractBuilder.__module__ == (
        "trade_rl.rl.environment_policy_schedule_contract"
    )


def test_environment_constructor_delegates_policy_schedule_contract() -> None:
    source = inspect.getsource(ResidualMarketEnv.__init__)

    assert source.count("EnvironmentPolicyScheduleContractBuilder(") == 1
    for forbidden in (
        "CausalEmergencyRiskMonitor(",
        "ActionSpec(",
        "pre-trade max_gross cannot exceed execution max_leverage",
        "random_initial_gross cannot exceed pre-trade max_gross",
        "action_spec alpha mode does not match environment",
        "action_spec factor count does not match environment",
        "target weight count does not match dataset symbols",
        "decision interval cannot exceed episode duration",
        "episode_hour_choices cannot be shorter than the resolved",
    ):
        assert forbidden not in source
    assert len(source.splitlines()) <= 190


def test_policy_schedule_builder_preserves_validation_order() -> None:
    source = inspect.getsource(EnvironmentPolicyScheduleContractBuilder.build)
    markers = (
        "CausalEmergencyRiskMonitor(",
        "pre-trade max_gross cannot exceed execution max_leverage",
        "random_initial_gross cannot exceed pre-trade max_gross",
        "if action_spec is None:",
        "action_spec alpha mode does not match environment",
        "action_spec factor count does not match environment",
        "target weight count does not match dataset symbols",
        "resolve_nominal_episode_bars(",
        "resolve_nominal_decision_bars(",
        "decision interval cannot exceed episode duration",
        "resolved_reward_config(",
        "resolved_decision_hours =",
        "episode_hour_choices cannot be shorter than the resolved",
    )

    positions = tuple(source.index(marker) for marker in markers)
    assert positions == tuple(sorted(positions))
