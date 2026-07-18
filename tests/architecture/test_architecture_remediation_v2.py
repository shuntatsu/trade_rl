from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from trade_rl.evaluation.offline_confirmation import create_fresh_confirmation_evidence
from trade_rl.evaluation.series import ReturnKind, ReturnSeries
from trade_rl.evaluation.walk_forward.folds import IndexRange, WalkForwardFold
from trade_rl.integrations import binance as binance_module
from trade_rl.release.asymmetric import (
    PublicVerificationKey,
)
from trade_rl.release.attestation import ReleaseAttestation
from trade_rl.release.offline_approval import create_release_attestation
from trade_rl.release.offline_signing import generate_private_key, public_key_bytes
from trade_rl.workflows import market_walk_forward as market_walk_forward_module
from trade_rl.workflows.fold_runner import (
    CandidateConfiguration,
    CandidateEvaluation,
    CandidateEvaluationRequest,
    CandidateTrainingRequest,
    ConcreteFoldRunner,
    EvaluationPhase,
    FoldExecutionConfig,
    PolicyTrainingArtifact,
    SeedPolicyFinalist,
)

DATASET_ID = "a" * 64
SIGNAL_DIGEST = "b" * 64
NOW = datetime(2026, 7, 16, tzinfo=UTC)
CONFIRMATION_PRIVATE_KEY = generate_private_key()
CONFIRMATION_PUBLIC_KEY = PublicVerificationKey(
    key_id="confirmation-key",
    public_key=public_key_bytes(CONFIRMATION_PRIVATE_KEY),
    purpose="fresh-confirmation",
    valid_from=NOW - timedelta(days=1),
    valid_until=NOW + timedelta(days=365),
)


def _fold() -> WalkForwardFold:
    return WalkForwardFold(
        fold_index=0,
        train=IndexRange(0, 20),
        checkpoint_validation=IndexRange(22, 26),
        configuration_selection=IndexRange(28, 32),
        test=IndexRange(34, 38),
        purge_bars=2,
    )


def _evaluation(
    request: CandidateEvaluationRequest, *, score: float
) -> CandidateEvaluation:
    return CandidateEvaluation(
        score=score,
        returns=ReturnSeries(
            values=tuple(0.001 for _ in range(request.evaluation_range.size)),
            kind=ReturnKind.BASE_BAR,
            periods_per_year=8_760,
        ),
        evaluation_digest=f"{100 + int(score * 10_000):064x}",
    )


def test_walk_forward_selects_and_outer_tests_deployable_ensemble_identity() -> None:
    @dataclass
    class Trainer:
        def train(self, request: CandidateTrainingRequest) -> PolicyTrainingArtifact:
            return PolicyTrainingArtifact(
                configuration=request.configuration.name,
                seed_finalists=(
                    SeedPolicyFinalist(
                        seed=0,
                        policy_digest="1" * 64,
                        checkpoint_score=0.1,
                        checkpoint_evaluation_digest="3" * 64,
                    ),
                    SeedPolicyFinalist(
                        seed=1,
                        policy_digest="2" * 64,
                        checkpoint_score=0.2,
                        checkpoint_evaluation_digest="4" * 64,
                    ),
                ),
            )

    @dataclass
    class Evaluator:
        calls: list[CandidateEvaluationRequest] = field(default_factory=list)

        def evaluate(self, request: CandidateEvaluationRequest) -> CandidateEvaluation:
            self.calls.append(request)
            if request.configuration == "baseline":
                return _evaluation(request, score=0.0)
            if request.policy_digest == "1" * 64:
                return _evaluation(request, score=0.02)
            if request.policy_digest == "2" * 64:
                return _evaluation(request, score=0.03)
            return _evaluation(request, score=0.025)

    artifact = Trainer().train(
        CandidateTrainingRequest(
            dataset_id=DATASET_ID,
            fold_index=0,
            configuration=CandidateConfiguration("candidate"),
            train=_fold().train,
            checkpoint_validation=_fold().checkpoint_validation,
        )
    )
    assert artifact.ensemble_policy_digest not in {"1" * 64, "2" * 64}

    evaluator = Evaluator()
    result = ConcreteFoldRunner(
        config=FoldExecutionConfig(
            dataset_id=DATASET_ID,
            signal_digest=SIGNAL_DIGEST,
            candidates=(CandidateConfiguration("candidate"),),
            minimum_selection_uplift=0.0,
            selected_at=NOW,
            experiment_plan_digest="e" * 64,
        ),
        trainer=Trainer(),
        evaluator=evaluator,
    ).run_fold(_fold())

    assert result.selection.selected_policy_digest == artifact.ensemble_policy_digest
    assert result.selected_member_policy_digests == ("1" * 64, "2" * 64)
    assert result.selected_member_seeds == (0, 1)
    ensemble_calls = [
        call
        for call in evaluator.calls
        if call.policy_digest == artifact.ensemble_policy_digest
    ]
    assert [call.phase for call in ensemble_calls] == [
        EvaluationPhase.CONFIGURATION_SELECTION,
        EvaluationPhase.OUTER_TEST,
    ]


def test_maintained_training_environment_liquidates_at_episode_end() -> None:
    resolver = getattr(
        market_walk_forward_module,
        "_maintained_training_environment",
        None,
    )
    assert callable(resolver)
    from trade_rl.rl.environment_config import ResidualMarketEnvConfig

    resolved = resolver(
        ResidualMarketEnvConfig(
            initial_capital=100_000.0,
            liquidate_on_end=False,
        ),
        episode_bars=100,
    )
    assert resolved.liquidate_on_end is True
    assert resolved.terminal_accounting_mode == "liquidate_at_close"


def test_release_attestation_requires_authenticated_signature() -> None:
    private_key = generate_private_key()
    public_key = PublicVerificationKey(
        key_id="ci-release-key",
        public_key=public_key_bytes(private_key),
        purpose="release-verification",
        valid_from=NOW - timedelta(days=1),
        valid_until=NOW + timedelta(days=365),
    )
    attestation = create_release_attestation(
        bundle_digest="a" * 64,
        dataset_id="b" * 64,
        training_run_digest="c" * 64,
        run_kind="research_selected_final",
        selection_proposal_digest="d" * 64,
        selection_authorization_digest="e" * 64,
        walk_forward_run_digest="f" * 64,
        gate_evidence_digest="1" * 64,
        confirmation_evidence_digest="2" * 64,
        selected_policy_digest="3" * 64,
        git_commit="4" * 40,
        dependency_digest="5" * 64,
        approver="risk-committee",
        approved_at=NOW,
        expires_at=NOW + timedelta(days=30),
        key_id=public_key.key_id,
        private_key=private_key,
    )
    attestation.verify({public_key.key_id: public_key}, trusted_at=NOW)
    assert not hasattr(ReleaseAttestation, "create")
    wrong_key = generate_private_key()
    with pytest.raises(ValueError, match="signature"):
        attestation.verify(
            {
                public_key.key_id: PublicVerificationKey(
                    key_id=public_key.key_id,
                    public_key=public_key_bytes(wrong_key),
                    purpose="release-verification",
                    valid_from=NOW - timedelta(days=1),
                    valid_until=NOW + timedelta(days=365),
                )
            },
            trusted_at=NOW,
        )


def test_confirmation_evidence_recomputes_metrics_and_rejects_tampering() -> None:
    evidence = create_fresh_confirmation_evidence(
        dataset_id="a" * 64,
        environment_digest="b" * 64,
        policy_digest="c" * 64,
        training_run_digest="d" * 64,
        git_commit="e" * 40,
        dependency_digest="f" * 64,
        start_time=NOW,
        end_time=NOW + timedelta(days=30),
        returns=(0.01, -0.005, 0.02),
        return_period_hours=240.0,
        order_log_digest="1" * 64,
        fill_log_digest="2" * 64,
        reconciliation_digest="3" * 64,
        required_after=NOW,
        created_at=NOW + timedelta(days=30),
        key_id="confirmation-key",
        private_key=CONFIRMATION_PRIVATE_KEY,
    )
    evidence.verify(
        {"confirmation-key": CONFIRMATION_PUBLIC_KEY},
        expected_required_after=NOW,
        trusted_now=NOW + timedelta(days=30),
    )
    assert evidence.total_return == pytest.approx((1.01 * 0.995 * 1.02) - 1.0)
    assert evidence.days == pytest.approx(30.0)
    with pytest.raises(ValueError, match="total_return|signature|digest"):
        evidence.with_returns((0.50, 0.0, 0.0)).verify(
            {"confirmation-key": CONFIRMATION_PUBLIC_KEY},
            expected_required_after=NOW,
            trusted_now=NOW + timedelta(days=30),
        )


def test_paired_bootstrap_requires_positive_excess_not_only_positive_return() -> None:
    from trade_rl.evaluation.research_gate import (
        paired_block_bootstrap_excess_lower_bound,
    )

    same = paired_block_bootstrap_excess_lower_bound(
        (0.01, -0.005, 0.002, 0.004),
        (0.01, -0.005, 0.002, 0.004),
        samples=200,
        block_size=2,
        seed=0,
    )
    better = paired_block_bootstrap_excess_lower_bound(
        (0.02, 0.01, 0.015, 0.012),
        (0.0, -0.005, 0.0, 0.001),
        samples=200,
        block_size=2,
        seed=0,
    )
    assert same == pytest.approx(0.0)
    assert better > 0.0


def test_funding_events_are_aggregated_into_native_bar() -> None:
    timestamps = np.asarray(
        [
            np.datetime64("2026-07-02T00:00:00", "ms"),
            np.datetime64("2026-07-03T00:00:00", "ms"),
        ]
    )
    events = [
        (int(datetime(2026, 7, 1, 8, tzinfo=UTC).timestamp() * 1_000), 0.001),
        (int(datetime(2026, 7, 1, 16, tzinfo=UTC).timestamp() * 1_000), 0.002),
        (int(datetime(2026, 7, 2, 0, tzinfo=UTC).timestamp() * 1_000), -0.0005),
    ]
    result = binance_module._align_funding(timestamps, events)
    assert len(result) == 3
    funding, available, counts = result
    assert funding[0] == pytest.approx(0.0025)
    assert available.tolist() == [True, False]
    assert counts.tolist() == [3, 0]


def test_instrument_contract_supports_effective_dated_execution_rules() -> None:
    from trade_rl.data.contracts import InstrumentContract, InstrumentExecutionRule

    contract = InstrumentContract(
        symbol="BTCUSDT",
        listed_at=datetime(2020, 1, 1, tzinfo=UTC),
        execution_rules=(
            InstrumentExecutionRule(
                effective_at=datetime(2020, 1, 1, tzinfo=UTC),
                tick_size=0.1,
                lot_size=0.001,
                minimum_notional=5.0,
            ),
            InstrumentExecutionRule(
                effective_at=datetime(2026, 1, 1, tzinfo=UTC),
                tick_size=0.01,
                lot_size=0.0001,
                minimum_notional=10.0,
            ),
        ),
    )
    tick, lot, minimum = contract.execution_rule_arrays(
        np.asarray(
            [
                np.datetime64("2025-12-31T00:00:00", "ns"),
                np.datetime64("2026-01-02T00:00:00", "ns"),
            ]
        )
    )
    np.testing.assert_allclose(tick, [0.1, 0.01])
    np.testing.assert_allclose(lot, [0.001, 0.0001])
    np.testing.assert_allclose(minimum, [5.0, 10.0])


def test_serving_state_snapshot_rejects_stale_or_mismatched_state() -> None:
    from trade_rl.serving.state import ServingStateGuard, ServingStateSnapshot

    guard = ServingStateGuard()
    first = ServingStateSnapshot.create(
        dataset_id="a" * 64,
        decision_index=10,
        portfolio_state=np.asarray([1.0, 0.0]),
        pending_target=np.asarray([0.2, -0.2]),
        observation_digest="b" * 64,
    )
    guard.accept(first)
    with pytest.raises(ValueError, match="stale|monotonic"):
        guard.accept(first)
    with pytest.raises(ValueError, match="pending|state"):
        guard.require_matches(
            first,
            dataset_id="a" * 64,
            decision_index=11,
            pending_target=np.asarray([0.0, 0.0]),
        )


def test_execution_rule_history_only_needs_to_cover_requested_dataset_range() -> None:
    from trade_rl.data.contracts import InstrumentContract, InstrumentExecutionRule

    contract = InstrumentContract(
        symbol="BTCUSDT",
        listed_at=datetime(2020, 1, 1, tzinfo=UTC),
        execution_rules=(
            InstrumentExecutionRule(
                effective_at=datetime(2025, 1, 1, tzinfo=UTC),
                tick_size=0.1,
                lot_size=0.001,
                minimum_notional=5.0,
            ),
        ),
    )
    tick, _, _ = contract.execution_rule_arrays(
        np.asarray([np.datetime64("2025-12-31T00:00:00", "ns")])
    )
    np.testing.assert_allclose(tick, [0.1])


def test_deployable_ensemble_execution_limits_are_part_of_eligibility() -> None:
    @dataclass
    class Trainer:
        def train(self, request: CandidateTrainingRequest) -> PolicyTrainingArtifact:
            return PolicyTrainingArtifact(
                configuration=request.configuration.name,
                seed_finalists=(
                    SeedPolicyFinalist(
                        seed=0,
                        policy_digest="1" * 64,
                        checkpoint_score=0.1,
                        checkpoint_evaluation_digest="3" * 64,
                    ),
                    SeedPolicyFinalist(
                        seed=1,
                        policy_digest="2" * 64,
                        checkpoint_score=0.2,
                        checkpoint_evaluation_digest="4" * 64,
                    ),
                ),
            )

    artifact = Trainer().train(
        CandidateTrainingRequest(
            dataset_id=DATASET_ID,
            fold_index=0,
            configuration=CandidateConfiguration("candidate"),
            train=_fold().train,
            checkpoint_validation=_fold().checkpoint_validation,
        )
    )

    @dataclass
    class Evaluator:
        def evaluate(self, request: CandidateEvaluationRequest) -> CandidateEvaluation:
            evaluation = _evaluation(
                request,
                score=0.0 if request.configuration == "baseline" else 0.02,
            )
            if request.policy_digest == artifact.ensemble_policy_digest:
                return CandidateEvaluation(
                    score=evaluation.score,
                    returns=evaluation.returns,
                    evaluation_digest="9" * 64,
                    cost_fraction=0.50,
                )
            return evaluation

    result = ConcreteFoldRunner(
        config=FoldExecutionConfig(
            dataset_id=DATASET_ID,
            signal_digest=SIGNAL_DIGEST,
            candidates=(CandidateConfiguration("candidate"),),
            minimum_selection_uplift=0.0,
            maximum_selection_cost_fraction=0.10,
            selected_at=NOW,
            experiment_plan_digest="e" * 64,
        ),
        trainer=Trainer(),
        evaluator=Evaluator(),
    ).run_fold(_fold())

    assert result.selection.selected_configuration == "baseline"
    assert "selection_cost_above_limit" in result.candidate_aggregates[0].reasons


def test_serving_state_match_requires_explicit_portfolio_state() -> None:
    from trade_rl.serving.state import ServingStateGuard, ServingStateSnapshot

    snapshot = ServingStateSnapshot.create(
        dataset_id="a" * 64,
        decision_index=10,
        portfolio_state=np.asarray([1.0, 0.0]),
        pending_target=np.asarray([0.2, -0.2]),
        observation_digest=ServingStateSnapshot.observation_digest_for(
            np.asarray([1.0, 2.0])
        ),
    )
    with pytest.raises(ValueError, match="portfolio"):
        ServingStateGuard().require_matches(
            snapshot,
            dataset_id="a" * 64,
            decision_index=10,
            portfolio_state=None,
            pending_target=np.asarray([0.2, -0.2]),
            current_flat=np.asarray([1.0, 2.0]),
        )


def test_confirmation_return_cadence_must_cover_declared_interval() -> None:
    with pytest.raises(ValueError, match="cadence|interval|duration"):
        create_fresh_confirmation_evidence(
            dataset_id="a" * 64,
            environment_digest="b" * 64,
            policy_digest="c" * 64,
            training_run_digest="d" * 64,
            git_commit="e" * 40,
            dependency_digest="f" * 64,
            start_time=NOW,
            end_time=NOW + timedelta(days=2),
            returns=(0.01, 0.02),
            return_period_hours=1.0,
            order_log_digest="1" * 64,
            fill_log_digest="2" * 64,
            reconciliation_digest="3" * 64,
            required_after=NOW,
            created_at=NOW + timedelta(days=2),
            key_id="confirmation-key",
            private_key=CONFIRMATION_PRIVATE_KEY,
        )
