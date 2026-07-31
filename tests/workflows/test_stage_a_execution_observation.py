from __future__ import annotations

from pathlib import Path

import pytest

from tests.stage_a_helpers import stage_a_test_manifest
from tests.evaluation.replay_support import (
    CANDIDATE_CONFIG_DIGEST,
    COST,
    DATASET_ID,
    EVALUATION_RUN_DIGEST,
    FOLD,
    SEED,
    execution_episode,
)
from trade_rl.artifacts.hashing import content_digest
from trade_rl.evaluation.stage_a_zero_shot_contracts import (
    StageACandidate,
    build_stage_a_zero_shot_evaluation_plan,
)
from trade_rl.workflows.execution_promotion_artifacts import (
    ExecutionPromotionArtifacts,
    write_execution_promotion_artifacts,
)
from trade_rl.workflows.stage_a_execution_observation import (
    build_stage_a_observation_from_execution_artifacts,
)

_BASELINE_CONFIG_DIGEST = "b" * 64
_TRIPLET_ID = content_digest({"triplet": "validation"})


def _checkpoint(seed: int) -> str:
    return content_digest({"checkpoint_seed": seed})


def _plan():
    candidate = StageACandidate.create(
        candidate_id="candidate-a",
        candidate_config_digest=CANDIDATE_CONFIG_DIGEST,
        final_training_completion_digest=content_digest({"completion": "candidate-a"}),
        policy_identity=content_digest({"policy": "candidate-a"}),
        checkpoint_digests=(
            (SEED, _checkpoint(SEED)),
            (SEED + 1, _checkpoint(SEED + 1)),
        ),
    )
    symbol_manifest = content_digest({"manifest": "symbols"})
    triplet_manifest = content_digest({"manifest": "triplets"})
    feature_identity = content_digest({"feature": "stage-a"})
    test_triplet_id = content_digest({"triplet": "test"})
    manifest = stage_a_test_manifest(
        symbol_disjoint_manifest_digest=symbol_manifest,
        symbol_disjoint_triplet_manifest_digest=triplet_manifest,
        feature_identity=feature_identity,
        validation_triplet_ids=(_TRIPLET_ID,),
        test_triplet_ids=(test_triplet_id,),
        folds=(FOLD, FOLD + 1),
        dataset_ids_by_triplet={_TRIPLET_ID: DATASET_ID},
    )
    plan = build_stage_a_zero_shot_evaluation_plan(
        symbol_disjoint_manifest_digest=symbol_manifest,
        symbol_disjoint_triplet_manifest_digest=triplet_manifest,
        evaluation_dataset_manifest_digest=manifest.digest,
        feature_identity=feature_identity,
        execution_identity=COST.execution_policy_digest,
        evaluation_identity=EVALUATION_RUN_DIGEST,
        candidates=(candidate,),
        seeds=(SEED, SEED + 1),
        folds=(FOLD, FOLD + 1),
        validation_triplet_ids=(_TRIPLET_ID,),
        test_triplet_ids=(test_triplet_id,),
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
    return plan, candidate, manifest


def _artifacts(
    root: Path, candidate_config_digest: str, *, fold: int = FOLD
) -> ExecutionPromotionArtifacts:
    events, book, order_book = execution_episode()
    return write_execution_promotion_artifacts(
        root=root,
        candidate_config_digest=candidate_config_digest,
        evaluation_run_digest=EVALUATION_RUN_DIGEST,
        fold=fold,
        seed=SEED,
        dataset_id=DATASET_ID,
        cost=COST,
        actions=((0.4,),),
        observation_digests=("1" * 64, "2" * 64),
        equity_curve=(1_000.0, 1_000.0),
        order_events=events,
        terminal_book=book,
        terminal_order_book=order_book,
        sensitivity_path_modes=("conservative",),
    )


def test_builds_observation_from_two_verified_promotion_roots(tmp_path: Path) -> None:
    plan, candidate, manifest = _plan()
    policy = _artifacts(tmp_path / "policy", CANDIDATE_CONFIG_DIGEST)
    baseline = _artifacts(tmp_path / "baseline", _BASELINE_CONFIG_DIGEST)

    observation = build_stage_a_observation_from_execution_artifacts(
        plan=plan,
        manifest=manifest,
        candidate_id=candidate.candidate_id,
        split="validation",
        triplet_id=_TRIPLET_ID,
        checkpoint_digest=_checkpoint(SEED),
        policy_artifacts=policy,
        baseline_artifacts=baseline,
        baseline_candidate_config_digest=_BASELINE_CONFIG_DIGEST,
        policy_log_growth=0.02,
        baseline_log_growth=0.01,
    )

    assert observation.policy_execution_evidence_digest == policy.evidence_digest
    assert observation.baseline_execution_evidence_digest == baseline.evidence_digest
    assert (observation.fold, observation.seed) == (FOLD, SEED)


def test_rejects_policy_baseline_cell_mismatch(tmp_path: Path) -> None:
    plan, candidate, manifest = _plan()
    policy = _artifacts(tmp_path / "policy", CANDIDATE_CONFIG_DIGEST)
    baseline = _artifacts(
        tmp_path / "baseline",
        _BASELINE_CONFIG_DIGEST,
        fold=FOLD + 1,
    )

    with pytest.raises(ValueError, match="cell mismatch"):
        build_stage_a_observation_from_execution_artifacts(
            plan=plan,
            manifest=manifest,
            candidate_id=candidate.candidate_id,
            split="validation",
            triplet_id=_TRIPLET_ID,
            checkpoint_digest=_checkpoint(SEED),
            policy_artifacts=policy,
            baseline_artifacts=baseline,
            baseline_candidate_config_digest=_BASELINE_CONFIG_DIGEST,
            policy_log_growth=0.02,
            baseline_log_growth=0.01,
        )
