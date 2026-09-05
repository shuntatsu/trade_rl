from __future__ import annotations

import pytest

from tests.integrations.test_universal_trade_rl_u2_replay import (
    ReplayIntegrationFixture,
    _scope,
)
from tests.rl.universal_trade_test_support import make_u1_wrapper
from trade_rl.rl.universal_trade_environment import UniversalTradeEnvironment


@pytest.mark.parametrize("executor_name", ("hybrid_executor", "shadow_executor"))
def test_u2_replay_rejects_shared_executor_rng_before_stepping(
    replay_fixture: ReplayIntegrationFixture,
    executor_name: str,
) -> None:
    scope = _scope(replay_fixture, cell="B")
    dataset = replay_fixture.session.datasets[scope.concrete_symbol]
    first_environment = make_u1_wrapper(
        dataset=dataset,
        contract=replay_fixture.policy_contract,
        normalizer=replay_fixture.normalizer,
    )
    second_environment = make_u1_wrapper(
        dataset=dataset,
        contract=replay_fixture.policy_contract,
        normalizer=replay_fixture.normalizer,
    )
    first_executor = getattr(first_environment.base_env, executor_name)
    second_executor = getattr(second_environment.base_env, executor_name)
    second_executor._rng = first_executor._rng

    issued = iter((first_environment, second_environment))
    original_factory = replay_fixture.session.environment_factory
    replay_fixture.session.environment_factory = lambda _dataset: next(issued)
    try:
        first: UniversalTradeEnvironment = (
            replay_fixture.session._create_verified_environment(scope)
        )
        assert first is first_environment
        with pytest.raises(ValueError, match="reuse|shared|fresh|mutable|random|RNG"):
            replay_fixture.session._create_verified_environment(scope)
    finally:
        replay_fixture.session.environment_factory = original_factory
        first_environment.close()
        second_environment.close()
