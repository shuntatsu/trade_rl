from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

from trade_rl.evaluation.series import ReturnKind, ReturnSeries
from trade_rl.evaluation.walk_forward.folds import IndexRange, WalkForwardFold
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
NOW = datetime(2026, 7, 13, 12, 0, tzinfo=UTC)


def fold() -> WalkForwardFold:
    return WalkForwardFold(
        fold_index=2,
        train=IndexRange(0, 80),
        checkpoint_validation=IndexRange(82, 92),
        configuration_selection=IndexRange(94, 104),
        test=IndexRange(106, 126),
        purge_bars=2,
    )


@dataclass
class FakeTrainer:
    calls: list[CandidateTrainingRequest] = field(default_factory=list)

    def train(self, request: CandidateTrainingRequest) -> PolicyTrainingArtifact:
        self.calls.append(request)
        digit = 1 + len(self.calls)
        return PolicyTrainingArtifact(
            configuration=request.configuration.name,
            seed_finalists=(
                SeedPolicyFinalist(
                    seed=0,
                    policy_digest=f"{digit:064x}",
                    checkpoint_score=0.1,
                    checkpoint_evaluation_digest=f"{50 + digit:064x}",
                ),
            ),
        )


@dataclass
class FakeEvaluator:
    selection_scores: dict[str, float]
    calls: list[CandidateEvaluationRequest] = field(default_factory=list)

    def evaluate(self, request: CandidateEvaluationRequest) -> CandidateEvaluation:
        self.calls.append(request)
        score = (
            self.selection_scores[request.configuration]
            if request.phase is EvaluationPhase.CONFIGURATION_SELECTION
            else 0.001
        )
        value = 0.0005 if request.configuration == "baseline" else 0.001
        return CandidateEvaluation(
            score=score,
            returns=ReturnSeries(
                values=tuple(value for _ in range(request.evaluation_range.size)),
                kind=ReturnKind.BASE_BAR,
                periods_per_year=8_760,
            ),
            evaluation_digest=f"{100 + len(self.calls):064x}",
        )


def config(*, minimum_uplift: float = 0.0) -> FoldExecutionConfig:
    return FoldExecutionConfig(
        dataset_id=DATASET_ID,
        signal_digest=SIGNAL_DIGEST,
        candidates=(
            CandidateConfiguration(name="candidate_b"),
            CandidateConfiguration(name="candidate_a"),
        ),
        minimum_selection_uplift=minimum_uplift,
        selected_at=NOW,
        experiment_plan_digest="e" * 64,
    )


def test_fold_runner_scopes_training_selection_and_outer_test_ranges() -> None:
    trainer = FakeTrainer()
    evaluator = FakeEvaluator(
        selection_scores={
            "baseline": 0.0,
            "candidate_a": 0.01,
            "candidate_b": 0.02,
        }
    )
    runner = ConcreteFoldRunner(config=config(), trainer=trainer, evaluator=evaluator)

    result = runner.run_fold(fold())

    assert {call.configuration.name for call in trainer.calls} == {
        "candidate_a",
        "candidate_b",
    }
    assert all(call.train == fold().train for call in trainer.calls)
    assert all(
        call.checkpoint_validation == fold().checkpoint_validation
        for call in trainer.calls
    )

    selection_calls = [
        call
        for call in evaluator.calls
        if call.phase is EvaluationPhase.CONFIGURATION_SELECTION
    ]
    outer_calls = [
        call for call in evaluator.calls if call.phase is EvaluationPhase.OUTER_TEST
    ]
    assert len(selection_calls) == 3
    assert all(
        call.evaluation_range == fold().configuration_selection
        for call in selection_calls
    )
    assert {(call.configuration, call.policy_digest) for call in outer_calls} == {
        ("baseline", None),
        ("candidate_b", f"{2:064x}"),
    }
    assert all(call.evaluation_range == fold().test for call in outer_calls)

    assert result.dataset_id == DATASET_ID
    assert result.selection.selected_configuration == "candidate_b"
    assert result.selection.selected_policy_digest == f"{2:064x}"
    assert result.selection.evaluation_digest == result.selection_evaluation_digest
    assert result.selected_oos.start == fold().test.start
    assert result.selected_oos.stop == fold().test.stop
    assert result.baseline_oos.start == fold().test.start
    assert result.baseline_oos.stop == fold().test.stop


def test_fold_runner_selects_across_seed_local_checkpoint_finalists() -> None:
    @dataclass
    class MultiSeedTrainer:
        def train(self, request: CandidateTrainingRequest) -> PolicyTrainingArtifact:
            return PolicyTrainingArtifact(
                configuration=request.configuration.name,
                seed_finalists=(
                    SeedPolicyFinalist(
                        seed=0,
                        policy_digest=f"{7:064x}",
                        checkpoint_score=0.3,
                        checkpoint_evaluation_digest=f"{70:064x}",
                    ),
                    SeedPolicyFinalist(
                        seed=1,
                        policy_digest=f"{8:064x}",
                        checkpoint_score=0.2,
                        checkpoint_evaluation_digest=f"{80:064x}",
                    ),
                ),
            )

    @dataclass
    class DigestEvaluator(FakeEvaluator):
        def evaluate(self, request: CandidateEvaluationRequest) -> CandidateEvaluation:
            if (
                request.phase is EvaluationPhase.CONFIGURATION_SELECTION
                and request.policy_digest is not None
            ):
                self.calls.append(request)
                score = {f"{7:064x}": 0.01, f"{8:064x}": 0.04}[request.policy_digest]
                return CandidateEvaluation(
                    score=score,
                    returns=ReturnSeries(
                        values=tuple(
                            0.001 for _ in range(request.evaluation_range.size)
                        ),
                        kind=ReturnKind.BASE_BAR,
                        periods_per_year=8_760,
                    ),
                    evaluation_digest=f"{300 + len(self.calls):064x}",
                )
            return super().evaluate(request)

    evaluator = DigestEvaluator(
        selection_scores={"baseline": 0.0, "candidate_a": 0.0, "candidate_b": 0.0}
    )
    one_candidate = FoldExecutionConfig(
        dataset_id=DATASET_ID,
        signal_digest=SIGNAL_DIGEST,
        candidates=(CandidateConfiguration(name="candidate_a"),),
        minimum_selection_uplift=0.0,
        selected_at=NOW,
        experiment_plan_digest="e" * 64,
    )

    result = ConcreteFoldRunner(
        config=one_candidate,
        trainer=MultiSeedTrainer(),
        evaluator=evaluator,
    ).run_fold(fold())

    assert result.selection.selected_policy_digest == f"{7:064x}"
    assert tuple(item.seed for item in result.seed_finalists) == (0, 1)
    assert tuple(item.selection_score for item in result.seed_finalists) == (0.01, 0.04)
    assert result.candidate_aggregates[0].median_score == 0.025
    assert result.candidate_aggregates[0].seed_count == 2
    assert all(
        call.evaluation_range == fold().configuration_selection
        for call in evaluator.calls
        if call.phase is EvaluationPhase.CONFIGURATION_SELECTION
    )


def test_fold_runner_falls_back_to_baseline_without_duplicate_outer_evaluation() -> (
    None
):
    trainer = FakeTrainer()
    evaluator = FakeEvaluator(
        selection_scores={
            "baseline": 0.01,
            "candidate_a": 0.02,
            "candidate_b": 0.03,
        }
    )
    runner = ConcreteFoldRunner(
        config=config(minimum_uplift=0.05),
        trainer=trainer,
        evaluator=evaluator,
    )

    result = runner.run_fold(fold())

    outer_calls = [
        call for call in evaluator.calls if call.phase is EvaluationPhase.OUTER_TEST
    ]
    assert [(call.configuration, call.policy_digest) for call in outer_calls] == [
        ("baseline", None)
    ]
    assert result.selection.mode.value == "baseline_only"
    assert result.selection.selected_configuration == "baseline"
    assert result.selection.selected_policy_digest is None
    assert result.selected_oos == result.baseline_oos


def test_fold_runner_seals_outer_test_and_preserves_execution_evidence() -> None:
    from trade_rl.evaluation.evidence import ExecutionDiagnostics

    trainer = FakeTrainer()

    @dataclass
    class EvidenceEvaluator(FakeEvaluator):
        def evaluate(self, request: CandidateEvaluationRequest) -> CandidateEvaluation:
            result = super().evaluate(request)
            return CandidateEvaluation(
                score=result.score,
                returns=result.returns,
                evaluation_digest=result.evaluation_digest,
                diagnostics=ExecutionDiagnostics(
                    total_cost=4.0, turnover_total=0.5, n_trades=3
                ),
            )

    evaluator = EvidenceEvaluator(
        selection_scores={"baseline": 0.0, "candidate_a": 0.01, "candidate_b": 0.02}
    )
    runner = ConcreteFoldRunner(config=config(), trainer=trainer, evaluator=evaluator)
    result = runner.run_fold(fold())

    assert result.selected_oos.diagnostics.total_cost == 4.0
    assert result.sealed_test_access.access_digest
    import pytest

    with pytest.raises(ValueError, match="already opened"):
        runner.run_fold(fold())


def test_fold_runner_rejects_unstable_high_cost_seed_distribution() -> None:
    from trade_rl.evaluation.evidence import ExecutionDiagnostics

    @dataclass
    class ThreeSeedTrainer:
        def train(self, request: CandidateTrainingRequest) -> PolicyTrainingArtifact:
            return PolicyTrainingArtifact(
                configuration=request.configuration.name,
                seed_finalists=tuple(
                    SeedPolicyFinalist(
                        seed=seed,
                        policy_digest=f"{20 + seed:064x}",
                        checkpoint_score=0.1,
                        checkpoint_evaluation_digest=f"{40 + seed:064x}",
                    )
                    for seed in range(3)
                ),
            )

    @dataclass
    class UnstableEvaluator(FakeEvaluator):
        def evaluate(self, request: CandidateEvaluationRequest) -> CandidateEvaluation:
            if (
                request.phase is EvaluationPhase.CONFIGURATION_SELECTION
                and request.policy_digest
            ):
                self.calls.append(request)
                seed = int(request.policy_digest, 16) - 20
                score = (0.06, -0.04, 0.05)[seed]
                return CandidateEvaluation(
                    score=score,
                    returns=ReturnSeries(
                        values=tuple(
                            0.001 for _ in range(request.evaluation_range.size)
                        ),
                        kind=ReturnKind.BASE_BAR,
                        periods_per_year=8_760,
                    ),
                    evaluation_digest=f"{500 + len(self.calls):064x}",
                    diagnostics=ExecutionDiagnostics(
                        turnover_total=9.0 if seed == 0 else 1.0,
                        total_cost=2_000.0 if seed == 0 else 100.0,
                    ),
                    turnover_per_day=1.5 if seed == 0 else 0.2,
                    cost_fraction=0.04 if seed == 0 else 0.002,
                    maximum_drawdown=0.25 if seed == 0 else 0.05,
                )
            return super().evaluate(request)

    evaluator = UnstableEvaluator(
        selection_scores={"baseline": 0.0, "candidate_a": 0.0}
    )
    strict = FoldExecutionConfig(
        dataset_id=DATASET_ID,
        signal_digest=SIGNAL_DIGEST,
        candidates=(CandidateConfiguration(name="candidate_a"),),
        minimum_selection_uplift=0.0,
        minimum_seed_success_fraction=2.0 / 3.0,
        minimum_worst_seed_uplift=-0.01,
        maximum_seed_score_std=0.03,
        maximum_selection_turnover_per_day=1.0,
        maximum_selection_cost_fraction=0.03,
        maximum_selection_drawdown=0.15,
        selected_at=NOW,
        experiment_plan_digest="e" * 64,
    )

    result = ConcreteFoldRunner(
        config=strict, trainer=ThreeSeedTrainer(), evaluator=evaluator
    ).run_fold(fold())

    assert result.selection.selected_configuration == "baseline"
    aggregate = result.candidate_aggregates[0]
    assert not aggregate.eligible
    assert "worst_seed_uplift_below_threshold" in aggregate.reasons
    assert "seed_score_dispersion_above_limit" in aggregate.reasons
    assert "selection_turnover_above_limit" in aggregate.reasons
    assert "selection_cost_above_limit" in aggregate.reasons
    assert "selection_drawdown_above_limit" in aggregate.reasons


def test_fold_runner_rejects_negative_candidate_even_when_baseline_is_worse() -> None:
    trainer = FakeTrainer()
    evaluator = FakeEvaluator(
        selection_scores={
            "baseline": -0.10,
            "candidate_a": -0.02,
            "candidate_b": -0.01,
        }
    )
    runner = ConcreteFoldRunner(
        config=config(minimum_uplift=0.0), trainer=trainer, evaluator=evaluator
    )

    result = runner.run_fold(fold())

    assert result.selection.selected_configuration == "baseline"
    assert all(not aggregate.eligible for aggregate in result.candidate_aggregates)
    assert all(
        "median_seed_score_below_threshold" in aggregate.reasons
        for aggregate in result.candidate_aggregates
    )
