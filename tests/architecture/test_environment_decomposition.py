from __future__ import annotations

import inspect

from trade_rl.rl.environment import ResidualMarketEnv
from trade_rl.rl.environment_episode import EpisodeContractSampler
from trade_rl.rl.environment_execution import EnvironmentExecutionCoordinator
from trade_rl.rl.environment_observation import EnvironmentObservationAssembler
from trade_rl.rl.environment_transition import EnvironmentTerminationCoordinator


def test_residual_market_env_delegates_cross_cutting_runtime_responsibilities() -> None:
    source = inspect.getsource(ResidualMarketEnv)

    assert EpisodeContractSampler.__module__ == "trade_rl.rl.environment_episode"
    assert EnvironmentExecutionCoordinator.__module__ == "trade_rl.rl.environment_execution"
    assert EnvironmentObservationAssembler.__module__ == "trade_rl.rl.environment_observation"
    assert EnvironmentTerminationCoordinator.__module__ == "trade_rl.rl.environment_transition"

    assert "self._episode_sampler.sample" in source
    assert "self._execution_coordinator.execute_target" in source
    assert "self._observation_assembler.observation" in source
    assert "self._termination_coordinator.resolve" in source


def test_residual_market_env_remains_the_gymnasium_facade() -> None:
    public_methods = {
        name
        for name, member in inspect.getmembers(ResidualMarketEnv, inspect.isfunction)
        if not name.startswith("_")
    }

    assert {"reset", "step", "observation_snapshot"}.issubset(public_methods)
