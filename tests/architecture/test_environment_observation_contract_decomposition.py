"""Architecture contract for static observation construction ownership."""

from __future__ import annotations

import importlib.util
import inspect

from trade_rl.rl.environment import ResidualMarketEnv


def test_environment_observation_contract_module_owns_static_construction() -> None:
    module_name = "trade_rl.rl.environment_observation_contract"
    assert importlib.util.find_spec(module_name) is not None, (
        "static observation construction must be owned by "
        f"{module_name}"
    )

    module = __import__(module_name, fromlist=["*"])
    assert hasattr(module, "EnvironmentObservationContract")
    assert hasattr(module, "EnvironmentObservationContractBuilder")


def test_environment_constructor_delegates_observation_contract() -> None:
    source = inspect.getsource(ResidualMarketEnv.__init__)

    assert "EnvironmentObservationContractBuilder" in source
    assert ".build(" in source
    for forbidden in (
        "spaces.Box",
        "spaces.Dict",
        "SequenceWindowSpec",
        "build_sequence_policy_plane",
        "observation_passthrough_indices",
    ):
        assert forbidden not in source, (
            f"ResidualMarketEnv.__init__ must not construct observation contracts "
            f"through {forbidden}"
        )


def test_environment_constructor_stays_within_reviewable_span() -> None:
    source_lines, _ = inspect.getsourcelines(ResidualMarketEnv.__init__)
    assert len(source_lines) <= 360, (
        "ResidualMarketEnv.__init__ must remain a bounded orchestration facade; "
        f"observed {len(source_lines)} lines"
    )
