from __future__ import annotations

import inspect

from trade_rl.rl.environment import ResidualMarketEnv
from trade_rl.rl.environment_runtime_services import (
    EnvironmentRuntimeServices,
    EnvironmentRuntimeServicesBuilder,
)


def test_runtime_services_module_owns_maintained_contract() -> None:
    assert EnvironmentRuntimeServices.__module__ == (
        "trade_rl.rl.environment_runtime_services"
    )
    assert EnvironmentRuntimeServicesBuilder.__module__ == (
        "trade_rl.rl.environment_runtime_services"
    )


def test_environment_constructor_delegates_runtime_service_wiring() -> None:
    source = inspect.getsource(ResidualMarketEnv.__init__)

    assert "EnvironmentRuntimeServicesBuilder(" in source
    for forbidden in (
        "EpisodeContractSampler(",
        "EnvironmentExecutionCoordinator(",
        "EnvironmentObservationAssembler(",
        "EnvironmentDecisionPlanner(",
        "EnvironmentRiskProjector(",
        "EnvironmentRewardCoordinator(",
        "EnvironmentInfoBuilder(",
        "EnvironmentTerminationCoordinator(",
    ):
        assert forbidden not in source
    assert len(source.splitlines()) <= 240


def test_runtime_service_builder_preserves_construction_order() -> None:
    source = inspect.getsource(EnvironmentRuntimeServicesBuilder.build)
    constructors = (
        "EpisodeContractSampler(",
        "EnvironmentExecutionCoordinator(",
        "EnvironmentObservationAssembler(",
        "EnvironmentDecisionPlanner(",
        "EnvironmentRiskProjector(",
        "EnvironmentRewardCoordinator(",
        "EnvironmentInfoBuilder(",
        "EnvironmentTerminationCoordinator(",
    )

    positions = tuple(source.index(name) for name in constructors)
    assert positions == tuple(sorted(positions))
