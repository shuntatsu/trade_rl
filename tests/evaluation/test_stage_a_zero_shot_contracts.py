from __future__ import annotations

import json
from pathlib import Path

import pytest

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


def _plan():
    return build_stage_a_zero_shot_evaluation_plan(
        symbol_disjoint_manifest_digest=_digest("symbol-manifest"),
        symbol_disjoint_triplet_manifest_digest=_digest("triplet-manifest"),
        dataset_identity=_digest("dataset"),
        feature_identity=_digest("feature"),
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
    )


def _observations(
    *,
    split: str,
    candidate_ids: tuple[str, ...] = ("candidate-a", "candidate-b"),
) -> tuple[StageAEvaluationObservation, ...]:
    plan = _plan()
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
                            dataset_identity=plan.dataset_identity,
                            execution_evidence_digest=_digest(
                                f"execution-{split}-{candidate_id}-{triplet_index}-{fold}-{seed}"
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
    assert tuple(seed for seed, _ in plan.candidate("candidate-a").checkpoint_digests) == _SEEDS

    wrong_seed_candidate = StageACandidate.create(
        candidate_id="candidate-c",
        candidate_config_digest=_digest("candidate-c-config"),
        final_training_completion_digest=_digest("candidate-c-completion"),
        policy_identity=_digest("candidate-c-policy"),
        checkpoint_digests=((0, _digest("candidate-c-checkpoint-0")),),
    )
    with pytest.raises(ValueError, match="checkpoint seed closure"):
        build_stage_a_zero_shot_evaluation_plan(
            symbol_disjoint_manifest_digest=_digest("symbol-manifest"),
            symbol_disjoint_triplet_manifest_digest=_digest("triplet-manifest"),
            dataset_identity=_digest("dataset"),
            feature_identity=_digest("feature"),
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
        )


def test_evidence_requires_the_complete_candidate_fold_seed_triplet_product(
    tmp_path: Path,
) -> None:
    plan = _plan()
    observations = _observations(split="validation")
    evidence = build_stage_a_evaluation_evidence(
        plan=plan,
        split="validation",
        observations=observations,
    )
    path = write_stage_a_evaluation_evidence(tmp_path / "evidence.json", evidence)

    assert load_stage_a_evaluation_evidence(path, plan=plan) == evidence
    assert len(evidence.observations) == (
        len(plan.candidates)
        * len(plan.folds)
        * len(plan.seeds)
        * len(plan.validation_triplet_ids)
    )

    with pytest.raises(ValueError, match="observation closure"):
        build_stage_a_evaluation_evidence(
            plan=plan,
            split="validation",
            observations=observations[:-1],
        )

    with pytest.raises(ValueError, match="duplicate observation"):
        build_stage_a_evaluation_evidence(
            plan=plan,
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
        dataset_identity=first.dataset_identity,
        execution_evidence_digest=first.execution_evidence_digest,
        policy_log_growth=first.policy_log_growth,
        baseline_log_growth=first.baseline_log_growth,
    )
    with pytest.raises(ValueError, match="checkpoint digest mismatch"):
        build_stage_a_evaluation_evidence(
            plan=plan,
            split="validation",
            observations=tuple(observations),
        )

    valid = build_stage_a_evaluation_evidence(
        plan=plan,
        split="validation",
        observations=_observations(split="validation"),
    )
    path = write_stage_a_evaluation_evidence(tmp_path / "evidence.json", valid)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["observations"][0]["policy_log_growth"] += 0.5
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="observation digest mismatch"):
        load_stage_a_evaluation_evidence(path, plan=plan)


def test_test_evidence_can_be_scoped_to_one_declared_candidate() -> None:
    plan = _plan()
    evidence = build_stage_a_evaluation_evidence(
        plan=plan,
        split="test",
        candidate_ids=("candidate-a",),
        observations=_observations(split="test", candidate_ids=("candidate-a",)),
    )

    assert evidence.candidate_ids == ("candidate-a",)
    assert evidence.triplet_ids == plan.test_triplet_ids
    evidence.validate_plan(plan)
