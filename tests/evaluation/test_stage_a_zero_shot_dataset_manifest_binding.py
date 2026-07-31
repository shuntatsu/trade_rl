from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from trade_rl.artifacts.hashing import content_digest
from trade_rl.evaluation.stage_a_zero_shot_contracts import (
    STAGE_A_EVALUATION_PLAN_SCHEMA,
    STAGE_A_EVIDENCE_SCHEMA,
    STAGE_A_OBSERVATION_SCHEMA,
    StageACandidate,
    StageAEvaluationObservation,
    build_stage_a_evaluation_evidence,
    build_stage_a_zero_shot_evaluation_plan,
    load_stage_a_zero_shot_evaluation_plan,
    write_stage_a_zero_shot_evaluation_plan,
)
from trade_rl.evaluation.walk_forward.folds import IndexRange
from trade_rl.workflows.stage_a_evaluation_dataset_manifest import (
    StageAEvaluationDatasetFold,
    StageAEvaluationDatasetManifest,
    StageAEvaluationDatasetTriplet,
)


def _digest(label: str) -> str:
    return content_digest({"label": label})


def _manifest() -> StageAEvaluationDatasetManifest:
    return StageAEvaluationDatasetManifest(
        symbol_disjoint_manifest_digest=_digest("symbols"),
        symbol_disjoint_triplet_manifest_digest=_digest("triplets"),
        source_closure_digest=_digest("source"),
        source_metadata_evidence_digest=_digest("metadata"),
        indicator_cache_id="cache",
        feature_identity=_digest("features"),
        timeline_start_time=datetime(2024, 1, 1, tzinfo=UTC),
        timeline_end_time=datetime(2024, 1, 2, tzinfo=UTC),
        triplets=(
            StageAEvaluationDatasetTriplet(
                split="validation",
                triplet_id=_digest("validation-triplet"),
                symbols=("ETHUSDT", "BNBUSDT", "SOLUSDT"),
                dataset_id=_digest("validation-dataset"),
            ),
            StageAEvaluationDatasetTriplet(
                split="test",
                triplet_id=_digest("test-triplet"),
                symbols=("XRPUSDT", "ADAUSDT", "DOGEUSDT"),
                dataset_id=_digest("test-dataset"),
            ),
        ),
        folds=(
            StageAEvaluationDatasetFold(
                fold=0,
                configuration_selection=IndexRange(20, 30),
                test=IndexRange(35, 45),
            ),
            StageAEvaluationDatasetFold(
                fold=1,
                configuration_selection=IndexRange(45, 55),
                test=IndexRange(60, 70),
            ),
        ),
    )


def _candidate() -> StageACandidate:
    return StageACandidate.create(
        candidate_id="candidate",
        candidate_config_digest=_digest("candidate-config"),
        final_training_completion_digest=_digest("training-completion"),
        policy_identity=_digest("policy"),
        checkpoint_digests=((0, _digest("checkpoint-0")), (1, _digest("checkpoint-1"))),
    )


def _plan(manifest: StageAEvaluationDatasetManifest):
    return build_stage_a_zero_shot_evaluation_plan(
        symbol_disjoint_manifest_digest=manifest.symbol_disjoint_manifest_digest,
        symbol_disjoint_triplet_manifest_digest=(
            manifest.symbol_disjoint_triplet_manifest_digest
        ),
        evaluation_dataset_manifest_digest=manifest.digest,
        feature_identity=manifest.feature_identity,
        execution_identity=_digest("execution"),
        evaluation_identity=_digest("evaluation"),
        candidates=(_candidate(),),
        seeds=(0, 1),
        folds=manifest.folds_declared,
        validation_triplet_ids=manifest.triplet_ids_for("validation"),
        test_triplet_ids=manifest.triplet_ids_for("test"),
        bootstrap_confidence_level=0.95,
        bootstrap_resamples=1_000,
        bootstrap_seed=7,
        minimum_validation_lower_bound=0.0,
        minimum_test_lower_bound=0.0,
        minimum_validation_worst_triplet_excess=0.0,
        minimum_test_worst_triplet_excess=0.0,
        minimum_validation_worst_seed_excess=0.0,
        minimum_test_worst_seed_excess=0.0,
        minimum_validation_triplet_pass_fraction=1.0,
        minimum_test_triplet_pass_fraction=1.0,
    )


def _observations(plan, manifest):
    triplet_id = manifest.triplet_ids_for("validation")[0]
    dataset_id = manifest.dataset_id_for("validation", triplet_id)
    return tuple(
        StageAEvaluationObservation.create(
            candidate_id="candidate",
            split="validation",
            triplet_id=triplet_id,
            fold=fold,
            seed=seed,
            checkpoint_digest=plan.candidate("candidate").checkpoint_digest(seed),
            evaluation_dataset_manifest_digest=manifest.digest,
            dataset_id=dataset_id,
            evaluation_range=manifest.range_for("validation", fold),
            feature_identity=plan.feature_identity,
            execution_identity=plan.execution_identity,
            evaluation_identity=plan.evaluation_identity,
            policy_execution_evidence_digest=_digest(f"policy-{fold}-{seed}"),
            baseline_execution_evidence_digest=_digest(f"baseline-{fold}-{seed}"),
            policy_log_growth=0.02,
            baseline_log_growth=0.01,
        )
        for fold in plan.folds
        for seed in plan.seeds
    )


def test_plan_observation_and_evidence_use_manifest_bound_schemas() -> None:
    manifest = _manifest()
    plan = _plan(manifest)
    evidence = build_stage_a_evaluation_evidence(
        plan=plan,
        manifest=manifest,
        split="validation",
        observations=_observations(plan, manifest),
    )

    assert plan.schema_version == STAGE_A_EVALUATION_PLAN_SCHEMA
    assert plan.evaluation_dataset_manifest_digest == manifest.digest
    assert evidence.schema_version == STAGE_A_EVIDENCE_SCHEMA
    assert evidence.observations[0].schema_version == STAGE_A_OBSERVATION_SCHEMA
    assert evidence.observations[0].dataset_id == manifest.dataset_id_for(
        "validation", manifest.triplet_ids_for("validation")[0]
    )
    assert evidence.observations[0].evaluation_range == IndexRange(20, 30)


def test_evidence_rejects_dataset_or_range_drift_from_manifest() -> None:
    manifest = _manifest()
    plan = _plan(manifest)
    observations = list(_observations(plan, manifest))
    first = observations[0]
    observations[0] = StageAEvaluationObservation.create(
        **{
            **first.constructor_payload(),
            "dataset_id": _digest("forged-dataset"),
        }
    )

    with pytest.raises(ValueError, match="dataset identity mismatch"):
        build_stage_a_evaluation_evidence(
            plan=plan,
            manifest=manifest,
            split="validation",
            observations=tuple(observations),
        )

    observations = list(_observations(plan, manifest))
    first = observations[0]
    observations[0] = StageAEvaluationObservation.create(
        **{
            **first.constructor_payload(),
            "evaluation_range": IndexRange(21, 30),
        }
    )
    with pytest.raises(ValueError, match="evaluation range mismatch"):
        build_stage_a_evaluation_evidence(
            plan=plan,
            manifest=manifest,
            split="validation",
            observations=tuple(observations),
        )


def test_plan_loader_rejects_pre_v3_dataset_identity_payload(tmp_path) -> None:
    manifest = _manifest()
    plan = _plan(manifest)
    path = write_stage_a_zero_shot_evaluation_plan(tmp_path / "plan.json", plan)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["dataset_identity"] = payload.pop("evaluation_dataset_manifest_digest")
    payload["schema_version"] = "stage_a_zero_shot_evaluation_plan_v2"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="field closure mismatch|unsupported"):
        load_stage_a_zero_shot_evaluation_plan(path)


def test_plan_rejects_manifest_identity_or_closure_drift() -> None:
    manifest = _manifest()
    plan = _plan(manifest)
    plan.validate_manifest(manifest)

    changed = StageAEvaluationDatasetManifest(
        **{
            **manifest.constructor_payload(),
            "source_closure_digest": _digest("changed-source"),
        }
    )
    with pytest.raises(ValueError, match="manifest digest mismatch"):
        plan.validate_manifest(changed)
