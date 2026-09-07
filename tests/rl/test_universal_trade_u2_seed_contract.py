from __future__ import annotations

from dataclasses import replace

import pytest

from tests.rl.test_universal_trade_u2_environment import (
    U2EnvironmentFixture,
    _factory,
)
from trade_rl.artifacts.hashing import content_digest
from trade_rl.rl.universal_instrument_binding import InstrumentDatasetBinding
from trade_rl.workflows.universal_trade_rl_u2_environment import (
    build_universal_trade_rl_u2_environment,
)

pytest_plugins = ("tests.rl.test_universal_trade_u2_environment",)

_RUN_SEED = 17
_ENVIRONMENT_GENERATION_DIGEST = content_digest(
    {"fixture": "u2-seed-contract-generation"}
)


def _build_seed_environment(
    fixture: U2EnvironmentFixture,
    *,
    bindings: tuple[InstrumentDatasetBinding, ...] | None = None,
    environment_index: int = 0,
):
    return build_universal_trade_rl_u2_environment(
        closure=fixture.closure,
        u1_contract=fixture.u1_contract,
        policy_contract=fixture.policy_contract,
        normalizer=fixture.normalizer,
        bindings=fixture.bindings if bindings is None else bindings,
        environment_factory=_factory(fixture),
        run_seed=_RUN_SEED,
        environment_index=environment_index,
        environment_generation_digest=_ENVIRONMENT_GENERATION_DIGEST,
    )


def test_u2_episode_sampling_ignores_unrelated_binding_metadata(
    u2_environment_fixture: U2EnvironmentFixture,
) -> None:
    drifted_bindings = tuple(
        replace(
            binding,
            execution_metadata_digest=content_digest(
                {
                    "fixture": "seed-contract-execution-drift",
                    "symbol": binding.concrete_symbol,
                }
            ),
            instrument_descriptor_digest=content_digest(
                {
                    "fixture": "seed-contract-descriptor-drift",
                    "symbol": binding.concrete_symbol,
                }
            ),
        )
        for binding in u2_environment_fixture.bindings
    )
    reference = _build_seed_environment(u2_environment_fixture)
    drifted = _build_seed_environment(
        u2_environment_fixture,
        bindings=drifted_bindings,
    )
    try:
        reference.reset(seed=_RUN_SEED)
        drifted.reset(seed=_RUN_SEED)

        reference_episode = reference.active_episode_binding
        drifted_episode = drifted.active_episode_binding
        assert reference_episode.episode_seed == drifted_episode.episode_seed
        assert reference_episode.episode_start == drifted_episode.episode_start
        assert reference_episode.episode_stop == drifted_episode.episode_stop
    finally:
        reference.close()
        drifted.close()


def test_u2_worker_reset_seed_is_member_seed_plus_environment_index(
    u2_environment_fixture: U2EnvironmentFixture,
) -> None:
    environment = _build_seed_environment(
        u2_environment_fixture,
        environment_index=3,
    )
    try:
        assert environment.canonical_probe_seed == 20
        assert environment.run_seed == _RUN_SEED
        assert environment.environment_index == 3

        for invalid_seed in (_RUN_SEED, 19, 21):
            with pytest.raises(ValueError, match="seed|run_seed|environment_index"):
                environment.reset(seed=invalid_seed)

        environment.reset(seed=20)
    finally:
        environment.close()
