from __future__ import annotations

import hashlib

import pytest

from trade_rl.evaluation.stage_a_zero_shot_contracts import (
    StageACandidate,
    build_stage_a_zero_shot_evaluation_plan,
)
from trade_rl.evaluation.walk_forward.folds import IndexRange
from trade_rl.workflows.stage_a_zero_shot_runner_contracts import (
    StageAEvaluationCellRequest,
    StageAEvaluationCellResult,
    StageATestFoldRange,
    StageATestSchedule,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _plan():
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
        validation_triplet_ids=(_digest("validation-triplet"),),
        test_triplet_ids=(_digest("test-triplet"),),
        bootstrap_confidence_level=0.95,
        bootstrap_resamples=1_000,
        bootstrap_seed=17,
        minimum_validation_lower_bound=0.05,
        minimum_test_lower_bound=0.05,
        minimum_validation_worst_triplet_excess=0.05,
        minimum_test_worst_triplet_excess=0.05,
        minimum_validation_worst_seed_excess=0.05,
        minimum_test_worst_seed_excess=0.05,
        minimum_validation_triplet_pass_fraction=1.0,
        minimum_test_triplet_pass_fraction=1.0,
    )


def _request(*, candidate: bool = False) -> StageAEvaluationCellRequest:
    plan = _plan()
    candidate_id = "candidate-a" if candidate else None
    checkpoint = (
        plan.candidate(candidate_id).checkpoint_digest(0) if candidate_id else None
    )
    return StageAEvaluationCellRequest(
        plan_digest=plan.digest,
        split="validation",
        triplet_id=plan.validation_triplet_ids[0],
        fold=0,
        seed=0,
        candidate_id=candidate_id,
        checkpoint_digest=checkpoint,
        dataset_identity=plan.dataset_identity,
        feature_identity=plan.feature_identity,
        execution_identity=plan.execution_identity,
        evaluation_identity=plan.evaluation_identity,
    )


def test_policy_request_requires_candidate_and_checkpoint_together() -> None:
    request = _request()
    with pytest.raises(ValueError, match="requires candidate and checkpoint"):
        StageAEvaluationCellRequest(
            plan_digest=request.plan_digest,
            split=request.split,
            triplet_id=request.triplet_id,
            fold=request.fold,
            seed=request.seed,
            candidate_id="candidate-a",
            checkpoint_digest=None,
            dataset_identity=request.dataset_identity,
            feature_identity=request.feature_identity,
            execution_identity=request.execution_identity,
            evaluation_identity=request.evaluation_identity,
        )


def test_cell_request_digest_binds_policy_identity() -> None:
    baseline = _request()
    policy = _request(candidate=True)
    assert baseline.is_baseline
    assert not policy.is_baseline
    assert baseline.digest != policy.digest


def test_cell_result_rejects_non_finite_growth() -> None:
    with pytest.raises(ValueError, match="must be finite"):
        StageAEvaluationCellResult(
            request_digest=_digest("request"),
            execution_evidence_digest=_digest("execution-evidence"),
            log_growth=float("nan"),
        )


def test_test_schedule_requires_unique_folds() -> None:
    plan = _plan()
    with pytest.raises(ValueError, match="folds must be unique"):
        StageATestSchedule(
            plan_digest=plan.digest,
            evaluation_identity=plan.evaluation_identity,
            fold_ranges=(
                StageATestFoldRange(0, IndexRange(100, 120)),
                StageATestFoldRange(0, IndexRange(120, 140)),
            ),
        )


def test_test_schedule_validates_exact_plan_fold_closure() -> None:
    plan = _plan()
    schedule = StageATestSchedule(
        plan_digest=plan.digest,
        evaluation_identity=plan.evaluation_identity,
        fold_ranges=(
            StageATestFoldRange(0, IndexRange(100, 120)),
            StageATestFoldRange(1, IndexRange(120, 140)),
        ),
    )
    schedule.validate_plan(plan)
    assert schedule.range_for(1) == IndexRange(120, 140)

    incomplete = StageATestSchedule(
        plan_digest=plan.digest,
        evaluation_identity=plan.evaluation_identity,
        fold_ranges=(StageATestFoldRange(0, IndexRange(100, 120)),),
    )
    with pytest.raises(ValueError, match="fold closure mismatch"):
        incomplete.validate_plan(plan)
