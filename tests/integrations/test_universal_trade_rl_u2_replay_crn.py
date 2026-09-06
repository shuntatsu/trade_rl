from __future__ import annotations

import pytest

from tests.integrations.test_universal_trade_rl_u2_replay import (
    ReplayIntegrationFixture,
    _scope,
)
from trade_rl.artifacts.hashing import content_digest
from trade_rl.workflows.universal_trade_rl_u2_development_authority import (
    replay_universal_trade_rl_u2_development_scope,
)
from trade_rl.workflows.universal_trade_rl_u2_predevelopment import (
    universal_trade_rl_u2_evaluation_seed,
)
from trade_rl.workflows.universal_trade_rl_u2_replay import (
    UniversalTradeRLU2ReplayRequest,
    UniversalTradeRLU2ReplayVariant,
)

pytest_plugins = ("tests.integrations.test_universal_trade_rl_u2_replay",)


def test_u2_development_authority_rejects_non_scope_common_seed_before_stepping(
    replay_fixture: ReplayIntegrationFixture,
) -> None:
    scope = _scope(replay_fixture, cell="B")
    expected_seed = universal_trade_rl_u2_evaluation_seed(
        u2_contract_digest=replay_fixture.u2_contract.digest,
        scope_digest=scope.digest,
    )
    wrong_seed = next(seed for seed in (0, 1, 2) if seed != expected_seed)
    request = UniversalTradeRLU2ReplayRequest(
        scope_digest=scope.digest,
        policy_variant=UniversalTradeRLU2ReplayVariant.CASH,
        evaluation_seed=wrong_seed,
        paired_candidate_checkpoint_digest=content_digest(
            {"fixture": "scope-common-crn-checkpoint"}
        ),
    )

    with pytest.raises(ValueError, match="scope|common|evaluation|seed|RNG"):
        replay_universal_trade_rl_u2_development_scope(
            session=replay_fixture.session,
            request=request,
        )


def test_u2_development_authority_accepts_exact_scope_common_seed(
    replay_fixture: ReplayIntegrationFixture,
) -> None:
    scope = _scope(replay_fixture, cell="B")
    expected_seed = universal_trade_rl_u2_evaluation_seed(
        u2_contract_digest=replay_fixture.u2_contract.digest,
        scope_digest=scope.digest,
    )
    request = UniversalTradeRLU2ReplayRequest(
        scope_digest=scope.digest,
        policy_variant=UniversalTradeRLU2ReplayVariant.CASH,
        evaluation_seed=expected_seed,
        paired_candidate_checkpoint_digest=content_digest(
            {"fixture": "scope-common-crn-checkpoint"}
        ),
    )

    evidence = replay_universal_trade_rl_u2_development_scope(
        session=replay_fixture.session,
        request=request,
    )

    assert evidence.evaluation_seed == expected_seed
    assert evidence.scope_digest == scope.digest
