from __future__ import annotations

import hashlib

import pytest

from tests.stage_a_helpers import stage_a_test_manifest
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


def _manifest():
    return stage_a_test_manifest(
        symbol_disjoint_manifest_digest=_digest("symbol-manifest"),
        symbol_disjoint_triplet_manifest_digest=_digest("triplet-manifest"),
        feature_identity=_digest("features"),
        validation_triplet_ids=(_digest("validation-triplet"),),
        test_triplet_ids=(_digest("test-triplet"),),
        folds=(0, 1),
    )


def _plan(manifest=None):
    manifest = manifest or _manifest()
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
        symbol_disjoint_manifest_digest=manifest.symbol_disjoint_manifest_digest,
        symbol_disjoint_triplet_manifest_digest=manifest.symbol_disjoint_triplet_manifest_digest,
        evaluation_dataset_manifest_digest=manifest.digest,
        feature_identity=manifest.feature_identity,
        execution_identity=_digest("execution"),
        evaluation_identity=_digest("evaluation"),
        candidates=candidates,
        seeds=seeds,
        folds=manifest.folds_declared,
        validation_triplet_ids=manifest.triplet_ids_for("validation"),
        test_triplet_ids=manifest.triplet_ids_for("test"),
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
    manifest = _manifest()
    plan = _plan(manifest)
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
        evaluation_dataset_manifest_digest=manifest.digest,
        dataset_id=manifest.dataset_id_for(
            "validation", plan.validation_triplet_ids[0]
        ),
        evaluation_range=manifest.range_for("validation", 0),
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
            evaluation_dataset_manifest_digest=request.evaluation_dataset_manifest_digest,
            dataset_id=request.dataset_id,
            evaluation_range=request.evaluation_range,
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
    manifest = _manifest()
    plan = _plan(manifest)
    with pytest.raises(ValueError, match="folds must be unique"):
        StageATestSchedule(
            plan_digest=plan.digest,
            evaluation_dataset_manifest_digest=manifest.digest,
            evaluation_identity=plan.evaluation_identity,
            fold_ranges=(
                StageATestFoldRange(0, IndexRange(100, 120)),
                StageATestFoldRange(0, IndexRange(120, 140)),
            ),
        )


def test_test_schedule_validates_exact_plan_fold_closure() -> None:
    manifest = _manifest()
    plan = _plan(manifest)
    schedule = StageATestSchedule(
        plan_digest=plan.digest,
        evaluation_dataset_manifest_digest=manifest.digest,
        evaluation_identity=plan.evaluation_identity,
        fold_ranges=tuple(
            StageATestFoldRange(fold, manifest.range_for("test", fold))
            for fold in plan.folds
        ),
    )
    schedule.validate_manifest(plan, manifest)
    assert schedule.range_for(1) == manifest.range_for("test", 1)

    incomplete = StageATestSchedule(
        plan_digest=plan.digest,
        evaluation_dataset_manifest_digest=manifest.digest,
        evaluation_identity=plan.evaluation_identity,
        fold_ranges=(StageATestFoldRange(0, manifest.range_for("test", 0)),),
    )
    with pytest.raises(ValueError, match="fold closure mismatch"):
        incomplete.validate_manifest(plan, manifest)


def test_cell_request_validates_exact_manifest_dataset_and_range() -> None:
    manifest = _manifest()
    plan = _plan(manifest)
    request = _request(candidate=True)
    request.validate_manifest(plan, manifest)

    forged_dataset = StageAEvaluationCellRequest(
        **{
            **request.constructor_payload(),
            "dataset_id": _digest("forged-dataset"),
        }
    )
    with pytest.raises(ValueError, match="dataset identity mismatch"):
        forged_dataset.validate_manifest(plan, manifest)

    forged_range = StageAEvaluationCellRequest(
        **{
            **request.constructor_payload(),
            "evaluation_range": IndexRange(
                request.evaluation_range.start + 1,
                request.evaluation_range.stop,
            ),
        }
    )
    with pytest.raises(ValueError, match="evaluation range mismatch"):
        forged_range.validate_manifest(plan, manifest)
