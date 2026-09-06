from __future__ import annotations

import math

import numpy as np
import pytest

from tests.integrations.test_universal_trade_rl_u2_replay import (
    ReplayIntegrationFixture,
    _build_replay_fixture,
    _scope,
)
from trade_rl.artifacts.hashing import content_digest
from trade_rl.workflows.universal_trade_rl_u2_predevelopment import (
    universal_trade_rl_u2_evaluation_seed,
)
from trade_rl.workflows.universal_trade_rl_u2_replay import (
    UniversalTradeRLU2ReplayRequest,
    UniversalTradeRLU2ReplayVariant,
)
from trade_rl.workflows.universal_trade_rl_u2_selection import (
    build_universal_trade_rl_u2_selection_leaf_metrics,
)

_CHECKPOINT_DIGEST = content_digest({"fixture": "u2-selection-metrics"})


class _ConstantCandidate:
    def predict(
        self,
        _observation: dict[str, object],
        *,
        deterministic: bool,
    ) -> tuple[np.ndarray, None]:
        assert deterministic is True
        return np.asarray([0.25], dtype=np.float32), None


@pytest.fixture(scope="module")
def selection_replay_fixture() -> ReplayIntegrationFixture:
    return _build_replay_fixture()


def test_u2_selection_leaf_metrics_are_derived_from_one_candidate_replay(
    selection_replay_fixture: ReplayIntegrationFixture,
) -> None:
    scope = _scope(selection_replay_fixture, cell="B")
    evaluation_seed = universal_trade_rl_u2_evaluation_seed(
        u2_contract_digest=selection_replay_fixture.u2_contract.digest,
        scope_digest=scope.digest,
    )
    evidence = selection_replay_fixture.session.replay(
        UniversalTradeRLU2ReplayRequest(
            scope_digest=scope.digest,
            policy_variant=UniversalTradeRLU2ReplayVariant.CANDIDATE,
            evaluation_seed=evaluation_seed,
            paired_candidate_checkpoint_digest=_CHECKPOINT_DIGEST,
        ),
        model=_ConstantCandidate(),
    )

    leaf = build_universal_trade_rl_u2_selection_leaf_metrics(
        training_seed=1,
        replay_evidence=evidence,
    )

    expected_net_log_growth = math.fsum(
        math.log1p(value) for value in evidence.net_simple_returns
    )
    expected_gross_log_growth = math.fsum(
        math.log1p(value) for value in evidence.gross_simple_returns
    )
    expected_turnover_per_day = evidence.turnover_total / (
        evidence.observed_decision_count * 0.25 / 24.0
    )

    assert leaf.training_seed == 1
    assert leaf.cell == evidence.cell
    assert leaf.concrete_symbol == evidence.concrete_symbol
    assert leaf.tile_identity == evidence.scope_digest
    assert leaf.replay_evidence_digest == evidence.digest
    assert leaf.leaf_net_log_growth == pytest.approx(expected_net_log_growth)
    assert leaf.leaf_gross_log_growth == pytest.approx(expected_gross_log_growth)
    assert leaf.leaf_net_wealth == pytest.approx(math.exp(expected_net_log_growth))
    assert leaf.leaf_gross_wealth == pytest.approx(math.exp(expected_gross_log_growth))
    assert leaf.turnover_per_day == pytest.approx(expected_turnover_per_day)
    assert leaf.meaningful_execution is (
        evidence.executed_change_count > 0 or evidence.turnover_total > 1e-6
    )
    assert leaf.hard_risk_violation_count == evidence.hard_risk_violation_count
    assert (
        leaf.unexplained_execution_rejection_count
        == evidence.execution_rejection_count
    )


def test_u2_selection_leaf_metrics_reject_baseline_replay(
    selection_replay_fixture: ReplayIntegrationFixture,
) -> None:
    scope = _scope(selection_replay_fixture, cell="B")
    evaluation_seed = universal_trade_rl_u2_evaluation_seed(
        u2_contract_digest=selection_replay_fixture.u2_contract.digest,
        scope_digest=scope.digest,
    )
    evidence = selection_replay_fixture.session.replay(
        UniversalTradeRLU2ReplayRequest(
            scope_digest=scope.digest,
            policy_variant=UniversalTradeRLU2ReplayVariant.CASH,
            evaluation_seed=evaluation_seed,
            paired_candidate_checkpoint_digest=_CHECKPOINT_DIGEST,
        )
    )

    with pytest.raises(ValueError, match="candidate|policy|variant|Selection"):
        build_universal_trade_rl_u2_selection_leaf_metrics(
            training_seed=1,
            replay_evidence=evidence,
        )
