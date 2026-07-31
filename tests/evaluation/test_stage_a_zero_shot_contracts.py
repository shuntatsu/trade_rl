from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.stage_a_helpers import stage_a_test_manifest, stage_a_test_manifest_for_plan
from trade_rl.artifacts.hashing import content_digest
from trade_rl.evaluation.stage_a_zero_shot_contracts import (
    StageACandidate,
    StageAEvaluationObservation,
    build_stage_a_evaluation_evidence,
    build_stage_a_zero_shot_evaluation_plan,
    load_stage_a_evaluation_evidence,
    load_stage_a_zero_shot_evaluation_plan,
    write_stage_a_evaluation_evidence,
    write_stage_a_zero_shot_evaluation_plan,
)

_SEEDS = (0, 1, 2)
_FOLDS = (0, 1)
_VALIDATION_TRIPLETS = (
    content_digest({"triplet": "validation-0"}),
    content_digest({"triplet": "validation-1"}),
)
_TEST_TRIPLETS = (content_digest({"triplet": "test-0"}),)


def _digest(label: str) -> str:
    return content_digest({"label": label})


def _candidate(candidate_id: str) -> StageACandidate:
    return StageACandidate.create(
        candidate_id=candidate_id,
        candidate_config_digest=_digest(f"{candidate_id}-config"),
        final_training_completion_digest=_digest(f"{candidate_id}-completion"),
        policy_identity=_digest(f"{candidate_id}-policy"),
        checkpoint_digests=tuple(
            (seed, _digest(f"{candidate_id}-checkpoint-{seed}")) for seed in _SEEDS
        ),
    )


def _manifest():
    return stage_a_test_manifest(
        symbol_disjoint_manifest_digest=_digest("symbol-manifest"),
        symbol_disjoint_triplet_manifest_digest=_digest("triplet-manifest"),
        feature_identity=_digest("feature"),
        validation_triplet_ids=_VALIDATION_TRIPLETS,
        test_triplet_ids=_TEST_TRIPLETS,
        folds=_FOLDS,
    )


def _plan():
    manifest = _manifest()
    return build_stage_a_zero_shot_evaluation_plan(
        symbol_disjoint_manifest_digest=manifest.symbol_disjoint_manifest_digest,
        symbol_disjoint_triplet_manifest_digest=manifest.symbol_disjoint_triplet_manifest_digest,
        evaluation_dataset_manifest_digest=manifest.digest,
        feature_identity=manifest.feature_identity,
        execution_identity=_digest("execution"),
        evaluation_identity=_digest("evaluation"),
        candidates=(_candidate("candidate-a"), _candidate("candidate-b")),
        seeds=_SEEDS,
        folds=_FOLDS,
        validation_triplet_ids=_VALIDATION_TRIPLETS,
        test_triplet_ids=_TEST_TRIPLETS,
        bootstrap_confidence_level=0.95,
        bootstrap_resamples=2_000,
        bootstrap_seed=31,
        minimum_validation_lower_bound=0.0,
        minimum_test_lower_bound=0.0,
        minimum_validation_worst_triplet_excess=0.0,
        minimum_test_worst_triplet_excess=0.0,
        minimum_validation_worst_seed_excess=0.0,
        minimum_test_worst_seed_excess=0.0,
        minimum_validation_triplet_pass_fraction=1.0,
        minimum_test_triplet_pass_fraction=1.0,
    )


def _observations(
    *,
    split: str,
    candidate_ids: tuple[str, ...] = ("candidate-a", "candidate-b"),
) -> tuple[StageAEvaluationObservation, ...]:
    plan = _plan()
    manifest = stage_a_test_manifest_for_plan(plan)
    triplets = (
        plan.validation_triplet_ids if split == "validation" else plan.test_triplet_ids
    )
    values: list[StageAEvaluationObservation] = []
    for candidate_id in candidate_ids:
        candidate = plan.candidate(candidate_id)
        checkpoints = dict(candidate.checkpoint_digests)
        for triplet_index, triplet_id in enumerate(triplets):
            for fold in plan.folds:
                for seed in plan.seeds:
                    values.append(
                        StageAEvaluationObservation.create(
                            candidate_id=candidate_id,
                            split=split,
                            triplet_id=triplet_id,
                            fold=fold,
                            seed=seed,
                            checkpoint_digest=checkpoints[seed],
                            evaluation_dataset_manifest_digest=manifest.digest,
                            dataset_id=manifest.dataset_id_for(split, triplet_id),
                            evaluation_range=manifest.range_for(split, fold),
                            feature_identity=plan.feature_identity,
                            execution_identity=plan.execution_identity,
                            evaluation_identity=plan.evaluation_identity,
                            policy_execution_evidence_digest=_digest(
                                f"policy-execution-{split}-{candidate_id}-{triplet_index}-{fold}-{seed}"
                            ),
                            baseline_execution_evidence_digest=_digest(
                                f"baseline-execution-{split}-{triplet_index}-{fold}-{seed}"
                            ),
                            policy_log_growth=0.02 + 0.001 * fold,
                            baseline_log_growth=0.01,
                        )
                    )
    return tuple(values)


def test_plan_round_trips_and_binds_candidate_seed_checkpoints(tmp_path: Path) -> None:
    plan = _plan()
    path = write_stage_a_zero_shot_evaluation_plan(tmp_path / "plan.json", plan)

    assert load_stage_a_zero_shot_evaluation_plan(path) == plan
    assert tuple(item.candidate_id for item in plan.candidates) == (
        "candidate-a",
        "candidate-b",
    )
    assert (
        tuple(seed for seed, _ in plan.candidate("candidate-a").checkpoint_digests)
        == _SEEDS
    )

    wrong_seed_candidate = StageACandidate.create(
        candidate_id="candidate-c",
        candidate_config_digest=_digest("candidate-c-config"),
        final_training_completion_digest=_digest("candidate-c-completion"),
        policy_identity=_digest("candidate-c-policy"),
        checkpoint_digests=((0, _digest("candidate-c-checkpoint-0")),),
    )
    manifest = _manifest()
    with pytest.raises(ValueError, match="checkpoint seed closure"):
        build_stage_a_zero_shot_evaluation_plan(
            symbol_disjoint_manifest_digest=manifest.symbol_disjoint_manifest_digest,
            symbol_disjoint_triplet_manifest_digest=(
                manifest.symbol_disjoint_triplet_manifest_digest
            ),
            evaluation_dataset_manifest_digest=manifest.digest,
            feature_identity=manifest.feature_identity,
            execution_identity=_digest("execution"),
            evaluation_identity=_digest("evaluation"),
            candidates=(wrong_seed_candidate,),
            seeds=_SEEDS,
            folds=_FOLDS,
            validation_triplet_ids=_VALIDATION_TRIPLETS,
            test_triplet_ids=_TEST_TRIPLETS,
            bootstrap_confidence_level=0.95,
            bootstrap_resamples=2_000,
            bootstrap_seed=31,
            minimum_validation_lower_bound=0.0,
            minimum_test_lower_bound=0.0,
            minimum_validation_worst_triplet_excess=0.0,
            minimum_test_worst_triplet_excess=0.0,
            minimum_validation_worst_seed_excess=0.0,
            minimum_test_worst_seed_excess=0.0,
            minimum_validation_triplet_pass_fraction=1.0,
            minimum_test_triplet_pass_fraction=1.0,
        )


def test_evidence_requires_the_complete_candidate_fold_seed_triplet_product(
    tmp_path: Path,
) -> None:
    plan = _plan()
    observations = _observations(split="validation")
    evidence = build_stage_a_evaluation_evidence(
        plan=plan,
        manifest=stage_a_test_manifest_for_plan(plan),
        split="validation",
        observations=observations,
    )
    path = write_stage_a_evaluation_evidence(tmp_path / "evidence.json", evidence)

    assert load_stage_a_evaluation_evidence(
        path, plan=plan, manifest=stage_a_test_manifest_for_plan(plan)
    ) == evidence
    assert len(evidence.observations) == (
        len(plan.candidates)
        * len(plan.folds)
        * len(plan.seeds)
        * len(plan.validation_triplet_ids)
    )

    with pytest.raises(ValueError, match="observation closure"):
        build_stage_a_evaluation_evidence(
            plan=plan,
            manifest=stage_a_test_manifest_for_plan(plan),
            split="validation",
            observations=observations[:-1],
        )

    with pytest.raises(ValueError, match="duplicate observation"):
        build_stage_a_evaluation_evidence(
            plan=plan,
            manifest=stage_a_test_manifest_for_plan(plan),
            split="validation",
            observations=(*observations, observations[0]),
        )


def test_evidence_rejects_checkpoint_mismatch_and_payload_tampering(
    tmp_path: Path,
) -> None:
    plan = _plan()
    observations = list(_observations(split="validation"))
    first = observations[0]
    observations[0] = StageAEvaluationObservation.create(
        candidate_id=first.candidate_id,
        split=first.split,
        triplet_id=first.triplet_id,
        fold=first.fold,
        seed=first.seed,
        checkpoint_digest=_digest("wrong-checkpoint"),
        evaluation_dataset_manifest_digest=(
            first.evaluation_dataset_manifest_digest
        ),
        dataset_id=first.dataset_id,
        evaluation_range=first.evaluation_range,
        feature_identity=first.feature_identity,
        execution_identity=first.execution_identity,
        evaluation_identity=first.evaluation_identity,
        policy_execution_evidence_digest=first.policy_execution_evidence_digest,
        baseline_execution_evidence_digest=first.baseline_execution_evidence_digest,
        policy_log_growth=first.policy_log_growth,
        baseline_log_growth=first.baseline_log_growth,
    )
    with pytest.raises(ValueError, match="checkpoint digest mismatch"):
        build_stage_a_evaluation_evidence(
            plan=plan,
            manifest=stage_a_test_manifest_for_plan(plan),
            split="validation",
            observations=tuple(observations),
        )

    valid = build_stage_a_evaluation_evidence(
        plan=plan,
        manifest=stage_a_test_manifest_for_plan(plan),
        split="validation",
        observations=_observations(split="validation"),
    )
    path = write_stage_a_evaluation_evidence(tmp_path / "evidence.json", valid)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["observations"][0]["policy_log_growth"] += 0.5
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="observation digest mismatch"):
        load_stage_a_evaluation_evidence(
        path, plan=plan, manifest=stage_a_test_manifest_for_plan(plan)
    )


def test_test_evidence_can_be_scoped_to_one_declared_candidate() -> None:
    plan = _plan()
    evidence = build_stage_a_evaluation_evidence(
        plan=plan,
        manifest=stage_a_test_manifest_for_plan(plan),
        split="test",
        candidate_ids=("candidate-a",),
        observations=_observations(split="test", candidate_ids=("candidate-a",)),
    )

    assert evidence.candidate_ids == ("candidate-a",)
    assert evidence.triplet_ids == plan.test_triplet_ids
    evidence.validate_plan(plan)
