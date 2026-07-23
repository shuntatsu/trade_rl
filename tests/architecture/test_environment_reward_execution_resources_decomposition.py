from __future__ import annotations

import inspect

from trade_rl.rl.environment import ResidualMarketEnv
from trade_rl.rl.environment_reward_execution_resources import (
    EnvironmentRewardExecutionResources,
    EnvironmentRewardExecutionResourcesBuilder,
)


def test_reward_execution_resource_module_owns_maintained_boundary() -> None:
    assert EnvironmentRewardExecutionResources.__module__ == (
        "trade_rl.rl.environment_reward_execution_resources"
    )
    assert EnvironmentRewardExecutionResourcesBuilder.__module__ == (
        "trade_rl.rl.environment_reward_execution_resources"
    )


def test_environment_constructor_delegates_reward_execution_resources() -> None:
    source = inspect.getsource(ResidualMarketEnv.__init__)

    assert source.count("EnvironmentRewardExecutionResourcesBuilder(") == 1
    assert source.count("self._install_reward_execution_resources(") == 1
    assert source.index("EnvironmentRewardExecutionResourcesBuilder(") < source.index(
        "EnvironmentObservationContractBuilder("
    )
    for forbidden in (
        "RewardTracker(",
        "minimum_reward_start_index(",
        "MarketExecutor(",
        "self._reward_history_cache: dict",
        "self._reward_history_cache = {}",
    ):
        assert forbidden not in source
    assert len(source.splitlines()) <= 152


def test_reward_execution_builder_preserves_construction_order() -> None:
    source = inspect.getsource(EnvironmentRewardExecutionResourcesBuilder.build)
    markers = (
        "reward_tracker = RewardTracker(",
        "minimum_start_index = self.minimum_start_index",
        "minimum_reward_start_index(",
        "hybrid_executor = MarketExecutor(",
        "shadow_executor = MarketExecutor(",
        "executor=hybrid_executor",
        "reward_history_cache={}",
    )

    positions = tuple(source.index(marker) for marker in markers)
    assert positions == tuple(sorted(positions))
    assert source.count("MarketExecutor(") == 2
