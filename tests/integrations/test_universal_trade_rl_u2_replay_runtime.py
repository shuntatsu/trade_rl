from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from tests.integrations.test_universal_trade_rl_u2_replay import (
    ReplayIntegrationFixture,
    _scope,
    replay_fixture,
)
from trade_rl.artifacts.hashing import content_digest
from trade_rl.workflows.universal_trade_rl_u2_replay import (
    UniversalTradeRLU2ReplayRequest,
    UniversalTradeRLU2ReplayVariant,
)


class DeterministicModelSpy:
    def __init__(self, action: object = 0.0) -> None:
        self.action = action
        self.deterministic_flags: list[bool] = []

    def predict(
        self,
        observation: dict[str, np.ndarray],
        *,
        deterministic: bool,
    ) -> tuple[np.ndarray, None]:
        assert observation
        self.deterministic_flags.append(deterministic)
        return np.asarray(self.action, dtype=np.float32), None


class ForbiddenModelSpy:
    def predict(
        self,
        observation: dict[str, np.ndarray],
        *,
        deterministic: bool,
    ) -> tuple[np.ndarray, None]:
        raise AssertionError("baseline replay must never invoke a policy model")


def _request(
    fixture: ReplayIntegrationFixture,
    *,
    variant: UniversalTradeRLU2ReplayVariant,
    seed: int = 0,
) -> UniversalTradeRLU2ReplayRequest:
    scope = _scope(fixture, cell="B")
    return UniversalTradeRLU2ReplayRequest(
        scope_digest=scope.digest,
        policy_variant=variant,
        evaluation_seed=seed,
        paired_candidate_checkpoint_digest=content_digest(
            {"fixture": "candidate-checkpoint", "seed": seed}
        ),
    )


def test_u2_cash_replay_completes_exact_720h_external_truncation(
    replay_fixture: ReplayIntegrationFixture,
) -> None:
    scope = _scope(replay_fixture, cell="B")
    evidence = replay_fixture.session.replay(
        _request(replay_fixture, variant=UniversalTradeRLU2ReplayVariant.CASH)
    )

    assert evidence.scope_digest == scope.digest
    assert evidence.policy_variant == UniversalTradeRLU2ReplayVariant.CASH.value
    assert evidence.evaluation_seed == 0
    assert evidence.observed_decision_count == scope.decision_count == 2_880
    assert evidence.runtime_start_bar_index == scope.evaluation_start_bar_index
    assert evidence.runtime_end_bar_index == scope.outcome_stop_bar_index_exclusive - 1
    assert evidence.final_current_bar_index == evidence.runtime_end_bar_index
    assert evidence.terminated is False
    assert evidence.truncated is True
    assert evidence.normal_completion is True
    assert evidence.terminal_accounting_mode == "mark_to_market"
    assert evidence.terminal_liquidation_cost == pytest.approx(0.0)
    assert len(evidence.net_simple_returns) == scope.decision_count
    assert len(evidence.normalized_action_trace) == scope.decision_count
    assert set(evidence.normalized_action_trace) == {0.0}
    assert len(evidence.realized_exposure_trace) == scope.decision_count
    assert math.prod(
        1.0 + value for value in evidence.net_simple_returns
    ) == pytest.approx(
        evidence.net_wealth_ratio,
        rel=0.0,
        abs=1e-10,
    )


@pytest.mark.parametrize(
    ("variant", "expected_action"),
    (
        (UniversalTradeRLU2ReplayVariant.CASH, 0.0),
        (UniversalTradeRLU2ReplayVariant.CONSTANT_LONG, 1.0),
        (UniversalTradeRLU2ReplayVariant.CONSTANT_SHORT, -1.0),
    ),
)
def test_u2_baseline_replay_uses_exact_preregistered_normalized_action(
    replay_fixture: ReplayIntegrationFixture,
    variant: UniversalTradeRLU2ReplayVariant,
    expected_action: float,
) -> None:
    evidence = replay_fixture.session.replay(_request(replay_fixture, variant=variant))

    assert evidence.normal_completion is True
    assert set(evidence.normalized_action_trace) == {expected_action}


def test_u2_candidate_replay_uses_deterministic_inference_on_every_decision(
    replay_fixture: ReplayIntegrationFixture,
) -> None:
    scope = _scope(replay_fixture, cell="B")
    model = DeterministicModelSpy(action=[0.0])
    evidence = replay_fixture.session.replay(
        _request(replay_fixture, variant=UniversalTradeRLU2ReplayVariant.CANDIDATE),
        model=model,
    )

    assert evidence.normal_completion is True
    assert evidence.observed_decision_count == scope.decision_count
    assert len(model.deterministic_flags) == scope.decision_count
    assert set(model.deterministic_flags) == {True}
    assert set(evidence.normalized_action_trace) == {0.0}


def test_u2_candidate_replay_requires_model(
    replay_fixture: ReplayIntegrationFixture,
) -> None:
    with pytest.raises((TypeError, ValueError), match="candidate|model"):
        replay_fixture.session.replay(
            _request(replay_fixture, variant=UniversalTradeRLU2ReplayVariant.CANDIDATE)
        )


def test_u2_baseline_replay_rejects_model_before_inference(
    replay_fixture: ReplayIntegrationFixture,
) -> None:
    with pytest.raises((TypeError, ValueError), match="baseline|model|candidate"):
        replay_fixture.session.replay(
            _request(replay_fixture, variant=UniversalTradeRLU2ReplayVariant.CASH),
            model=ForbiddenModelSpy(),
        )


@pytest.mark.parametrize(
    "action",
    (
        [0.0, 0.0],
        [float("nan")],
        [1.5],
    ),
)
def test_u2_candidate_replay_rejects_malformed_action_through_strict_u1_surface(
    replay_fixture: ReplayIntegrationFixture,
    action: list[float],
) -> None:
    with pytest.raises((TypeError, ValueError), match="action|shape|finite|range|bound"):
        replay_fixture.session.replay(
            _request(replay_fixture, variant=UniversalTradeRLU2ReplayVariant.CANDIDATE),
            model=DeterministicModelSpy(action=action),
        )
