from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest

from trade_rl.evaluation.stage_a_zero_shot_contracts import (
    StageACandidate,
    build_stage_a_zero_shot_evaluation_plan,
)
from trade_rl.evaluation.stage_a_zero_shot_gate import StageAValidationSelection
from trade_rl.evaluation.walk_forward.folds import IndexRange
from trade_rl.evaluation.walk_forward.sealed_test import SealedTestLedger
from trade_rl.workflows.stage_a_zero_shot_runner import (
    StageAZeroShotEvaluationOrchestrator,
)
from trade_rl.workflows.stage_a_zero_shot_runner_contracts import (
    StageAEvaluationCellRequest,
    StageAEvaluationCellResult,
    StageATestFoldRange,
    StageATestSchedule,
    StageAValidationRun,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _plan(*, passing_threshold: float = 0.05):
    seeds = (0, 1)
    candidates = tuple(
        StageACandidate.create(
            candidate_id=candidate_id,
            candidate_config_digest=_digest(f"{candidate_id}:config"),
            final_training_completion_digest=_digest(f"{candidate_id}:complete"),
            policy_identity=_digest(f"{candidate_id}:policy"),
            checkpoint_digests=tuple(
                (seed, _digest(f"{candidate_id}:checkpoint:{seed}")) for seed in seeds
            ),
        )
        for candidate_id in ("candidate-a", "candidate-b")
    )
    return build_stage_a_zero_shot_evaluation_plan(
        symbol_disjoint_manifest_digest=_digest("symbol-manifest"),
        symbol_disjoint_triplet_manifest_digest=_digest("triplet-manifest"),
        dataset_identity=_digest("dataset"),
        feature_identity=_digest("features"),
        execution_identity=_digest("execution"),
        evaluation_identity=_digest("evaluation"),
        candidates=candidates,
        seeds=seeds,
        folds=(0, 1),
        validation_triplet_ids=(
            _digest("validation-triplet-a"),
            _digest("validation-triplet-b"),
        ),
        test_triplet_ids=(_digest("test-triplet-a"), _digest("test-triplet-b")),
        bootstrap_confidence_level=0.95,
        bootstrap_resamples=1_000,
        bootstrap_seed=17,
        minimum_validation_lower_bound=passing_threshold,
        minimum_test_lower_bound=passing_threshold,
        minimum_validation_worst_triplet_excess=passing_threshold,
        minimum_test_worst_triplet_excess=passing_threshold,
        minimum_validation_worst_seed_excess=passing_threshold,
        minimum_test_worst_seed_excess=passing_threshold,
        minimum_validation_triplet_pass_fraction=1.0,
        minimum_test_triplet_pass_fraction=1.0,
    )


def _schedule(plan):
    return StageATestSchedule(
        plan_digest=plan.digest,
        evaluation_identity=plan.evaluation_identity,
        fold_ranges=tuple(
            StageATestFoldRange(fold, IndexRange(100 + fold * 20, 120 + fold * 20))
            for fold in plan.folds
        ),
    )


class RecordingEvaluator:
    def __init__(
        self,
        *,
        growth_by_candidate: dict[str | None, float] | None = None,
        events: list[tuple[object, ...]] | None = None,
        mismatch_once: bool = False,
    ) -> None:
        self.growth_by_candidate = growth_by_candidate or {
            None: 0.0,
            "candidate-a": 0.20,
            "candidate-b": 0.10,
        }
        self.events = events if events is not None else []
        self.requests: list[StageAEvaluationCellRequest] = []
        self.mismatch_once = mismatch_once

    def evaluate(
        self, request: StageAEvaluationCellRequest
    ) -> StageAEvaluationCellResult:
        self.requests.append(request)
        self.events.append(
            ("evaluate", request.split, request.fold, request.candidate_id)
        )
        request_digest = request.digest
        if self.mismatch_once:
            self.mismatch_once = False
            request_digest = _digest("mismatched-request")
        return StageAEvaluationCellResult(
            request_digest=request_digest,
            execution_evidence_digest=_digest(f"execution:{request.digest}"),
            log_growth=self.growth_by_candidate[request.candidate_id],
        )


class RecordingLedger:
    def __init__(self, events: list[tuple[object, ...]]) -> None:
        self.events = events
        self.delegate = SealedTestLedger()

    @property
    def records(self):
        return self.delegate.records

    def authorize_once(self, **kwargs):
        self.events.append(("authorize", kwargs["fold_index"]))
        return self.delegate.authorize_once(**kwargs)


def _orchestrator(*, plan=None, evaluator=None, events=None):
    resolved_plan = plan or _plan()
    resolved_events = events if events is not None else []
    resolved_evaluator = evaluator or RecordingEvaluator(events=resolved_events)
    ledger = RecordingLedger(resolved_events)
    return (
        StageAZeroShotEvaluationOrchestrator(
            plan=resolved_plan,
            evaluator=resolved_evaluator,
            test_schedule=_schedule(resolved_plan),
            sealed_test_ledger=ledger,
        ),
        resolved_evaluator,
        ledger,
        resolved_events,
    )


def test_validation_evaluates_complete_cartesian_product_with_shared_baselines() -> None:
    orchestrator, evaluator, ledger, events = _orchestrator()
    run = orchestrator.evaluate_validation()
    plan = orchestrator.plan
    cells = len(plan.validation_triplet_ids) * len(plan.folds) * len(plan.seeds)
    baseline_requests = [request for request in evaluator.requests if request.is_baseline]
    policy_requests = [request for request in evaluator.requests if not request.is_baseline]

    assert len(baseline_requests) == cells
    assert len(policy_requests) == cells * len(plan.candidate_ids)
    assert ledger.records == ()
    assert run.selection.passed
    assert run.selection.selected_candidate_id == "candidate-a"
    assert run.evidence.candidate_ids == plan.candidate_ids

    expected_prefix = [
        ("evaluate", "validation", 0, None),
        ("evaluate", "validation", 0, "candidate-a"),
        ("evaluate", "validation", 0, "candidate-b"),
    ]
    assert events[:3] == expected_prefix

    by_baseline_key: dict[tuple[str, int, int], set[tuple[str, float]]] = {}
    for observation in run.evidence.observations:
        by_baseline_key.setdefault(observation.baseline_key, set()).add(
            (
                observation.baseline_execution_evidence_digest,
                observation.baseline_log_growth,
            )
        )
    assert all(len(values) == 1 for values in by_baseline_key.values())


def test_validation_rejects_evaluator_result_for_another_request() -> None:
    evaluator = RecordingEvaluator(mismatch_once=True)
    orchestrator, _, ledger, _ = _orchestrator(evaluator=evaluator)
    with pytest.raises(ValueError, match="result request identity mismatch"):
        orchestrator.evaluate_validation()
    assert ledger.records == ()


def test_failed_validation_never_opens_or_evaluates_sealed_test() -> None:
    plan = _plan()
    events: list[tuple[object, ...]] = []
    evaluator = RecordingEvaluator(
        growth_by_candidate={None: 0.0, "candidate-a": 0.0, "candidate-b": 0.0},
        events=events,
    )
    orchestrator, _, ledger, _ = _orchestrator(
        plan=plan, evaluator=evaluator, events=events
    )
    validation_run = orchestrator.evaluate_validation()
    assert not validation_run.selection.passed
    evaluator.requests.clear()
    events.clear()

    with pytest.raises(ValueError, match="passed validation selection"):
        orchestrator.evaluate_sealed_test(validation_run)
    assert evaluator.requests == []
    assert ledger.records == ()
    assert events == []


def test_forged_validation_run_is_rejected_before_sealed_access() -> None:
    orchestrator, evaluator, ledger, events = _orchestrator()
    validation_run = orchestrator.evaluate_validation()
    summaries = tuple(
        replace(
            summary,
            lower_confidence_bound=(
                0.10 if summary.candidate_id == "candidate-a" else 0.30
            ),
            digest="",
        )
        for summary in validation_run.selection.candidate_summaries
    )
    selection = StageAValidationSelection(
        plan_digest=validation_run.selection.plan_digest,
        validation_evidence_digest=validation_run.evidence.digest,
        candidate_summaries=summaries,
        minimum_lower_bound=validation_run.selection.minimum_lower_bound,
        minimum_worst_triplet_excess=(
            validation_run.selection.minimum_worst_triplet_excess
        ),
        minimum_worst_seed_excess=validation_run.selection.minimum_worst_seed_excess,
        minimum_triplet_pass_fraction=(
            validation_run.selection.minimum_triplet_pass_fraction
        ),
        selected_candidate_id="candidate-b",
        passed=True,
        reason="candidate_selected_by_validation_gate",
    )
    forged = StageAValidationRun(evidence=validation_run.evidence, selection=selection)
    evaluator.requests.clear()
    events.clear()

    with pytest.raises(ValueError, match="does not match recomputation"):
        orchestrator.evaluate_sealed_test(forged)
    assert evaluator.requests == []
    assert ledger.records == ()
    assert events == []


def test_sealed_test_authorizes_every_fold_before_selected_only_evaluation() -> None:
    orchestrator, evaluator, ledger, events = _orchestrator()
    validation_run = orchestrator.evaluate_validation()
    evaluator.requests.clear()
    events.clear()

    sealed_run = orchestrator.evaluate_sealed_test(validation_run)
    plan = orchestrator.plan
    cells = len(plan.test_triplet_ids) * len(plan.folds) * len(plan.seeds)
    baseline_requests = [request for request in evaluator.requests if request.is_baseline]
    policy_requests = [request for request in evaluator.requests if not request.is_baseline]

    assert events[: len(plan.folds)] == [("authorize", fold) for fold in plan.folds]
    assert len(baseline_requests) == cells
    assert len(policy_requests) == cells
    assert {request.candidate_id for request in policy_requests} == {"candidate-a"}
    assert sealed_run.evidence.candidate_ids == ("candidate-a",)
    assert len(sealed_run.access_records) == len(plan.folds)
    assert sealed_run.decision.passed
    assert ledger.records == sealed_run.access_records


def test_sealed_test_cannot_be_opened_twice() -> None:
    orchestrator, _, _, _ = _orchestrator()
    validation_run = orchestrator.evaluate_validation()
    orchestrator.evaluate_sealed_test(validation_run)
    with pytest.raises(ValueError, match="already opened"):
        orchestrator.evaluate_sealed_test(validation_run)
