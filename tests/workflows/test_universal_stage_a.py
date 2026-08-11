from __future__ import annotations

from dataclasses import replace

import pytest

from trade_rl.artifacts.hashing import content_digest
from trade_rl.evaluation.stage_a_zero_shot_contracts import (
    StageACandidate,
    StageAZeroShotEvaluationPlan,
)
from trade_rl.rl.training import ResidualTrainingConfig
from trade_rl.rl.universal_architecture import (
    UniversalArchitectureName,
    apply_architecture_to_training_config,
)


def _digest(label: str) -> str:
    return content_digest({"label": label})


def _base_config() -> ResidualTrainingConfig:
    return ResidualTrainingConfig(
        timesteps=128,
        gamma=1.0,
        seeds=(7, 11),
        n_steps=8,
        batch_size=8,
    )


def _stage_a_candidate(name: UniversalArchitectureName) -> StageACandidate:
    return StageACandidate.create(
        candidate_id=name.value,
        candidate_config_digest=_digest(f"config:{name.value}"),
        final_training_completion_digest=_digest(f"completion:{name.value}"),
        policy_identity=_digest(f"policy:{name.value}"),
        checkpoint_digests=((7, _digest(f"{name.value}:7")), (11, _digest(f"{name.value}:11"))),
    )


def _plan(candidates: tuple[StageACandidate, ...]) -> StageAZeroShotEvaluationPlan:
    return StageAZeroShotEvaluationPlan(
        symbol_disjoint_manifest_digest=_digest("symbol-disjoint"),
        symbol_disjoint_triplet_manifest_digest=_digest("triplet-manifest"),
        evaluation_dataset_manifest_digest=_digest("evaluation-dataset"),
        feature_identity=_digest("feature"),
        execution_identity=_digest("execution"),
        evaluation_identity=_digest("evaluation"),
        candidates=candidates,
        seeds=(7, 11),
        folds=(0, 1),
        validation_triplet_ids=(_digest("validation-triplet"),),
        test_triplet_ids=(_digest("test-triplet"),),
        bootstrap_confidence_level=0.95,
        bootstrap_resamples=1_000,
        bootstrap_seed=19,
        minimum_validation_lower_bound=0.0,
        minimum_test_lower_bound=0.0,
        minimum_validation_worst_triplet_excess=0.0,
        minimum_test_worst_triplet_excess=0.0,
        minimum_validation_worst_seed_excess=0.0,
        minimum_test_worst_seed_excess=0.0,
        minimum_validation_triplet_pass_fraction=0.5,
        minimum_test_triplet_pass_fraction=0.5,
    )


def _wrapped_candidates() -> tuple[object, ...]:
    from trade_rl.workflows.universal_stage_a import UniversalStageACandidate

    base = _base_config()
    return tuple(
        UniversalStageACandidate(
            architecture=name,
            stage_a_candidate=_stage_a_candidate(name),
            training_config=apply_architecture_to_training_config(base, name),
        )
        for name in UniversalArchitectureName
    )


def test_universal_stage_a_requires_exact_four_ablation_candidates() -> None:
    from trade_rl.workflows.universal_stage_a import UniversalStageAPlan

    wrapped = _wrapped_candidates()
    plan = _plan(tuple(item.stage_a_candidate for item in wrapped))
    universal = UniversalStageAPlan(ablation_candidates=wrapped, stage_a_plan=plan)

    assert universal.stage_a_plan is plan
    assert universal.candidate_ids == tuple(name.value for name in UniversalArchitectureName)
    assert len({item.fixed_condition_digest for item in wrapped}) == 1


def test_universal_stage_a_rejects_non_architecture_condition_drift() -> None:
    from trade_rl.workflows.universal_stage_a import UniversalStageAPlan

    wrapped = list(_wrapped_candidates())
    wrapped[1] = replace(
        wrapped[1],
        training_config=replace(wrapped[1].training_config, learning_rate=9e-4),
    )
    plan = _plan(tuple(item.stage_a_candidate for item in wrapped))

    with pytest.raises(ValueError, match="non-architecture conditions"):
        UniversalStageAPlan(ablation_candidates=tuple(wrapped), stage_a_plan=plan)


def test_universal_stage_a_rejects_architecture_projection_mismatch() -> None:
    from trade_rl.workflows.universal_stage_a import UniversalStageACandidate

    name = UniversalArchitectureName.U_MEDIUM_DIRECT
    config = apply_architecture_to_training_config(_base_config(), name)

    with pytest.raises(ValueError, match="architecture projection"):
        UniversalStageACandidate(
            architecture=name,
            stage_a_candidate=_stage_a_candidate(name),
            training_config=replace(config, sequence_d_model=192),
        )


def test_universal_stage_a_rejects_plan_candidate_identity_drift() -> None:
    from trade_rl.workflows.universal_stage_a import UniversalStageAPlan

    wrapped = _wrapped_candidates()
    candidates = [item.stage_a_candidate for item in wrapped]
    candidates[0] = StageACandidate.create(
        candidate_id=candidates[0].candidate_id,
        candidate_config_digest=_digest("unexpected-config"),
        final_training_completion_digest=candidates[0].final_training_completion_digest,
        policy_identity=candidates[0].policy_identity,
        checkpoint_digests=candidates[0].checkpoint_digests,
    )

    with pytest.raises(ValueError, match="Stage A candidate identity"):
        UniversalStageAPlan(
            ablation_candidates=wrapped,
            stage_a_plan=_plan(tuple(candidates)),
        )
