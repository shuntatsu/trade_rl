"""Replay-bound Stage A observation construction."""

from __future__ import annotations

from trade_rl.domain.common import require_sha256
from trade_rl.evaluation._stage_a_zero_shot_contract_helpers import (
    StageAEvaluationSplit,
)
from trade_rl.evaluation._stage_a_zero_shot_evidence import (
    StageAEvaluationObservation,
)
from trade_rl.evaluation._stage_a_zero_shot_plan import StageAZeroShotEvaluationPlan
from trade_rl.simulation.execution_promotion import validate_execution_promotion
from trade_rl.workflows.execution_promotion_artifacts import (
    ExecutionPromotionArtifacts,
)


def _validate_artifacts(
    *,
    artifacts: ExecutionPromotionArtifacts,
    plan: StageAZeroShotEvaluationPlan,
    expected_candidate_config_digest: str,
    label: str,
) -> tuple[int, int]:
    identity = artifacts.artifact.replay_identity
    if artifacts.artifact.dataset_id != plan.dataset_identity:
        raise ValueError(f"Stage A {label} artifact dataset identity mismatch")
    if artifacts.artifact.execution_policy_digest != plan.execution_identity:
        raise ValueError(f"Stage A {label} artifact execution identity mismatch")
    if identity.evaluation_run_digest != plan.evaluation_identity:
        raise ValueError(f"Stage A {label} artifact evaluation identity mismatch")
    if identity.candidate_config_digest != expected_candidate_config_digest:
        raise ValueError(f"Stage A {label} artifact candidate configuration mismatch")
    if identity.fold not in plan.folds:
        raise ValueError(f"Stage A {label} artifact fold is not declared")
    if identity.seed not in plan.seeds:
        raise ValueError(f"Stage A {label} artifact seed is not declared")
    validate_execution_promotion(
        artifacts.evidence,
        expected_policy_digest=plan.execution_identity,
        event_artifact_path=artifacts.replay_path,
        expected_candidate_config_digest=expected_candidate_config_digest,
        expected_evaluation_run_digest=plan.evaluation_identity,
        expected_fold=identity.fold,
        expected_seed=identity.seed,
    )
    return identity.fold, identity.seed


def build_stage_a_observation_from_execution_artifacts(
    *,
    plan: StageAZeroShotEvaluationPlan,
    candidate_id: str,
    split: StageAEvaluationSplit,
    triplet_id: str,
    checkpoint_digest: str,
    policy_artifacts: ExecutionPromotionArtifacts,
    baseline_artifacts: ExecutionPromotionArtifacts,
    baseline_candidate_config_digest: str,
    policy_log_growth: float,
    baseline_log_growth: float,
) -> StageAEvaluationObservation:
    """Build one observation only from two fully verified promotion roots."""

    baseline_config = require_sha256(
        baseline_candidate_config_digest,
        field="stage_a_baseline_candidate_config_digest",
    )
    candidate = plan.candidate(candidate_id)
    if triplet_id not in plan.triplet_ids_for(split):
        raise ValueError("Stage A execution artifact triplet is not declared for split")
    expected_checkpoint = candidate.checkpoint_digest(
        policy_artifacts.artifact.replay_identity.seed
    )
    if checkpoint_digest != expected_checkpoint:
        raise ValueError("Stage A execution artifact checkpoint digest mismatch")

    policy_cell = _validate_artifacts(
        artifacts=policy_artifacts,
        plan=plan,
        expected_candidate_config_digest=candidate.candidate_config_digest,
        label="policy",
    )
    baseline_cell = _validate_artifacts(
        artifacts=baseline_artifacts,
        plan=plan,
        expected_candidate_config_digest=baseline_config,
        label="baseline",
    )
    if policy_cell != baseline_cell:
        raise ValueError("Stage A policy and baseline artifact cell mismatch")
    fold, seed = policy_cell
    if checkpoint_digest != candidate.checkpoint_digest(seed):
        raise ValueError("Stage A execution artifact checkpoint digest mismatch")

    return StageAEvaluationObservation.create(
        candidate_id=candidate_id,
        split=split,
        triplet_id=triplet_id,
        fold=fold,
        seed=seed,
        checkpoint_digest=checkpoint_digest,
        dataset_identity=plan.dataset_identity,
        feature_identity=plan.feature_identity,
        execution_identity=plan.execution_identity,
        evaluation_identity=plan.evaluation_identity,
        policy_execution_evidence_digest=policy_artifacts.evidence_digest,
        baseline_execution_evidence_digest=baseline_artifacts.evidence_digest,
        policy_log_growth=policy_log_growth,
        baseline_log_growth=baseline_log_growth,
    )


__all__ = ["build_stage_a_observation_from_execution_artifacts"]
