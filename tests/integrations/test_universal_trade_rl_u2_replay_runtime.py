from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from tests.integrations.test_universal_trade_rl_u2_replay import (
    ReplayIntegrationFixture,
    _scope,
    replay_fixture,
)
from tests.rl.universal_trade_test_support import make_u1_base_env
from trade_rl.artifacts.hashing import content_digest
from trade_rl.data.market import MarketDataset
from trade_rl.rl.universal_trade_environment import UniversalTradeEnvironment
from trade_rl.workflows.universal_trade_rl_u2_replay import (
    UniversalTradeRLU2ReplayEvidence,
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


class SeedRecordingEnvironment(UniversalTradeEnvironment):
    def __init__(
        self,
        dataset: MarketDataset,
        fixture: ReplayIntegrationFixture,
        reset_seeds: list[int | None],
    ) -> None:
        self._reset_seeds = reset_seeds
        super().__init__(
            make_u1_base_env(dataset=dataset),
            contract=fixture.policy_contract,
            normalizer=fixture.normalizer,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        self._reset_seeds.append(seed)
        return super().reset(seed=seed, options=options)


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


@pytest.fixture(scope="module")
def baseline_evidence(
    replay_fixture: ReplayIntegrationFixture,
) -> dict[UniversalTradeRLU2ReplayVariant, UniversalTradeRLU2ReplayEvidence]:
    variants = (
        UniversalTradeRLU2ReplayVariant.CASH,
        UniversalTradeRLU2ReplayVariant.CONSTANT_LONG,
        UniversalTradeRLU2ReplayVariant.CONSTANT_SHORT,
    )
    return {
        variant: replay_fixture.session.replay(
            _request(replay_fixture, variant=variant)
        )
        for variant in variants
    }


@pytest.fixture(scope="module")
def candidate_evidence(
    replay_fixture: ReplayIntegrationFixture,
) -> tuple[UniversalTradeRLU2ReplayEvidence, DeterministicModelSpy]:
    model = DeterministicModelSpy(action=[0.0])
    evidence = replay_fixture.session.replay(
        _request(replay_fixture, variant=UniversalTradeRLU2ReplayVariant.CANDIDATE),
        model=model,
    )
    return evidence, model


def test_u2_cash_replay_completes_exact_720h_external_truncation(
    replay_fixture: ReplayIntegrationFixture,
    baseline_evidence: dict[
        UniversalTradeRLU2ReplayVariant, UniversalTradeRLU2ReplayEvidence
    ],
) -> None:
    scope = _scope(replay_fixture, cell="B")
    evidence = baseline_evidence[UniversalTradeRLU2ReplayVariant.CASH]

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
    baseline_evidence: dict[
        UniversalTradeRLU2ReplayVariant, UniversalTradeRLU2ReplayEvidence
    ],
    variant: UniversalTradeRLU2ReplayVariant,
    expected_action: float,
) -> None:
    evidence = baseline_evidence[variant]

    assert evidence.normal_completion is True
    assert set(evidence.normalized_action_trace) == {expected_action}


def test_u2_candidate_replay_uses_deterministic_inference_on_every_decision(
    replay_fixture: ReplayIntegrationFixture,
    candidate_evidence: tuple[
        UniversalTradeRLU2ReplayEvidence, DeterministicModelSpy
    ],
) -> None:
    scope = _scope(replay_fixture, cell="B")
    evidence, model = candidate_evidence

    assert evidence.normal_completion is True
    assert evidence.observed_decision_count == scope.decision_count
    assert len(model.deterministic_flags) == scope.decision_count
    assert set(model.deterministic_flags) == {True}
    assert set(evidence.normalized_action_trace) == {0.0}


def test_u2_replay_pair_identity_is_equal_across_candidate_and_all_baselines(
    baseline_evidence: dict[
        UniversalTradeRLU2ReplayVariant, UniversalTradeRLU2ReplayEvidence
    ],
    candidate_evidence: tuple[
        UniversalTradeRLU2ReplayEvidence, DeterministicModelSpy
    ],
) -> None:
    candidate, _model = candidate_evidence
    evidences = (candidate, *baseline_evidence.values())
    identity_fields = (
        "scope_closure_digest",
        "scope_digest",
        "universe_manifest_digest",
        "u1_contract_digest",
        "u2_contract_digest",
        "source_dataset_digest",
        "evaluation_dataset_digest",
        "concrete_symbol",
        "symbol_role",
        "cell",
        "source_window",
        "tile_index",
        "evaluation_seed",
        "paired_candidate_checkpoint_digest",
        "runtime_start_bar_index",
        "runtime_end_bar_index",
    )
    reference = tuple(getattr(candidate, field) for field in identity_fields)

    for evidence in evidences:
        assert tuple(getattr(evidence, field) for field in identity_fields) == reference


def test_u2_replay_evidence_digest_binds_seed_checkpoint_and_dataset_identity(
    baseline_evidence: dict[
        UniversalTradeRLU2ReplayVariant, UniversalTradeRLU2ReplayEvidence
    ],
) -> None:
    evidence = baseline_evidence[UniversalTradeRLU2ReplayVariant.CASH]
    changed_checkpoint = content_digest({"fixture": "different-checkpoint"})
    changed_dataset = content_digest({"fixture": "different-development-view"})

    seed_identity = replace(evidence, evaluation_seed=1, digest="")
    checkpoint_identity = replace(
        evidence,
        paired_candidate_checkpoint_digest=changed_checkpoint,
        digest="",
    )
    dataset_identity = replace(
        evidence,
        evaluation_dataset_digest=changed_dataset,
        digest="",
    )

    assert len(
        {
            evidence.digest,
            seed_identity.digest,
            checkpoint_identity.digest,
            dataset_identity.digest,
        }
    ) == 4
    with pytest.raises(ValueError, match="digest"):
        replace(
            evidence,
            paired_candidate_checkpoint_digest=changed_checkpoint,
        )


def test_u2_replay_evidence_rejects_return_tampering_even_with_new_identity(
    baseline_evidence: dict[
        UniversalTradeRLU2ReplayVariant, UniversalTradeRLU2ReplayEvidence
    ],
) -> None:
    evidence = baseline_evidence[UniversalTradeRLU2ReplayVariant.CASH]
    tampered_returns = (
        evidence.net_simple_returns[0] + 1e-6,
        *evidence.net_simple_returns[1:],
    )

    with pytest.raises(ValueError, match="return|wealth|reconcil"):
        replace(evidence, net_simple_returns=tampered_returns, digest="")


def test_u2_replay_passes_request_seed_exactly_to_u1_reset(
    replay_fixture: ReplayIntegrationFixture,
) -> None:
    observed_reset_seeds: list[int | None] = []
    issued: list[SeedRecordingEnvironment] = []
    original_factory = replay_fixture.session.environment_factory

    def factory(dataset: MarketDataset) -> UniversalTradeEnvironment:
        environment = SeedRecordingEnvironment(
            dataset,
            replay_fixture,
            observed_reset_seeds,
        )
        issued.append(environment)
        return environment

    replay_fixture.session.environment_factory = factory
    try:
        evidence = replay_fixture.session.replay(
            _request(
                replay_fixture,
                variant=UniversalTradeRLU2ReplayVariant.CASH,
                seed=2,
            )
        )
    finally:
        replay_fixture.session.environment_factory = original_factory

    assert issued
    assert observed_reset_seeds == [2]
    assert evidence.evaluation_seed == 2


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
    with pytest.raises(
        (TypeError, ValueError), match="action|shape|finite|range|bound"
    ):
        replay_fixture.session.replay(
            _request(replay_fixture, variant=UniversalTradeRLU2ReplayVariant.CANDIDATE),
            model=DeterministicModelSpy(action=action),
        )
