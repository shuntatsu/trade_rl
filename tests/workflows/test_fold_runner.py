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
            policy_digest=f"{digit:064x}",
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
