from __future__ import annotations

import pytest

from tests.rl.test_universal_trade_u2_environment import (
    U2EnvironmentFixture,
    _factory,
)
from trade_rl.artifacts.hashing import content_digest
from trade_rl.workflows.universal_trade_rl_u2_environment import (
    build_universal_trade_rl_u2_environment,
)

pytest_plugins = ("tests.rl.test_universal_trade_u2_environment",)


def _build_with_generation(
    fixture: U2EnvironmentFixture,
    *,
    environment_index: int,
    environment_generation_digest: str,
):
    return build_universal_trade_rl_u2_environment(
        closure=fixture.closure,
        u1_contract=fixture.u1_contract,
        policy_contract=fixture.policy_contract,
        normalizer=fixture.normalizer,
        bindings=fixture.bindings,
        environment_factory=_factory(fixture),
        run_seed=17,
        environment_index=environment_index,
        environment_generation_digest=environment_generation_digest,
    )


def test_u2_workers_expose_one_shared_environment_generation_digest_runtime(
    u2_environment_fixture: U2EnvironmentFixture,
) -> None:
    generation_digest = content_digest({"fixture": "u2-shared-generation"})
    worker0 = _build_with_generation(
        u2_environment_fixture,
        environment_index=0,
        environment_generation_digest=generation_digest,
    )
    worker3 = _build_with_generation(
        u2_environment_fixture,
        environment_index=3,
        environment_generation_digest=generation_digest,
    )
    try:
        assert worker0.environment_digest == generation_digest
        assert worker3.environment_digest == generation_digest
        assert worker0.router_digest != worker3.router_digest
        assert worker0.environment_index == 0
        assert worker3.environment_index == 3
    finally:
        worker0.close()
        worker3.close()


def test_u2_environment_rejects_invalid_generation_digest(
    u2_environment_fixture: U2EnvironmentFixture,
) -> None:
    with pytest.raises(ValueError, match="digest|SHA|sha"):
        _build_with_generation(
            u2_environment_fixture,
            environment_index=0,
            environment_generation_digest="not-a-sha256",
        )
