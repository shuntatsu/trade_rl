from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from tests.evaluation.replay_support import execution_episode
from tests.stage_a_helpers import stage_a_test_manifest_for_plan
from trade_rl.artifacts.codec import canonical_json_bytes
from trade_rl.evaluation.stage_a_zero_shot_contracts import (
    StageACandidate,
    build_stage_a_zero_shot_evaluation_plan,
)
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.simulation.execution_promotion import execution_evidence_from_cost
from trade_rl.simulation.execution_replay import (
    build_execution_event_artifact,
    write_execution_event_artifact,
)
from trade_rl.simulation.funding_evidence import build_funding_evidence_artifact
from trade_rl.workflows.stage_a_execution_replay import (
    build_stage_a_execution_replay_artifact,
    validate_stage_a_execution_replay_sources,
)
from trade_rl.workflows.stage_a_zero_shot_runner_contracts import (
    StageAEvaluationCellRequest,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _request() -> tuple[object, StageAEvaluationCellRequest]:
    execution_identity = ExecutionCostConfig(
        path_mode="conservative"
    ).execution_policy_digest
    candidate = StageACandidate.create(
        candidate_id="candidate-a",
        candidate_config_digest=_digest("candidate-config"),
        final_training_completion_digest=_digest("training-complete"),
        policy_identity=_digest("policy"),
        checkpoint_digests=((0, _digest("checkpoint")),),
    )
    seed_plan = build_stage_a_zero_shot_evaluation_plan(
        symbol_disjoint_manifest_digest=_digest("symbol-manifest"),
        symbol_disjoint_triplet_manifest_digest=_digest("triplet-manifest"),
        evaluation_dataset_manifest_digest=_digest("placeholder-manifest"),
        feature_identity=_digest("features"),
        execution_identity=execution_identity,
        evaluation_identity=_digest("evaluation"),
        candidates=(candidate,),
        seeds=(0,),
        folds=(0,),
        validation_triplet_ids=(_digest("validation-triplet"),),
        test_triplet_ids=(_digest("test-triplet"),),
        bootstrap_confidence_level=0.95,
        bootstrap_resamples=1_000,
        bootstrap_seed=17,
        minimum_validation_lower_bound=0.0,
        minimum_test_lower_bound=0.0,
        minimum_validation_worst_triplet_excess=0.0,
        minimum_test_worst_triplet_excess=0.0,
        minimum_validation_worst_seed_excess=0.0,
        minimum_test_worst_seed_excess=0.0,
        minimum_validation_triplet_pass_fraction=1.0,
        minimum_test_triplet_pass_fraction=1.0,
    )
    manifest = stage_a_test_manifest_for_plan(seed_plan)
    plan = build_stage_a_zero_shot_evaluation_plan(
        symbol_disjoint_manifest_digest=seed_plan.symbol_disjoint_manifest_digest,
        symbol_disjoint_triplet_manifest_digest=(
            seed_plan.symbol_disjoint_triplet_manifest_digest
        ),
        evaluation_dataset_manifest_digest=manifest.digest,
        feature_identity=seed_plan.feature_identity,
        execution_identity=seed_plan.execution_identity,
        evaluation_identity=seed_plan.evaluation_identity,
        candidates=seed_plan.candidates,
        seeds=seed_plan.seeds,
        folds=seed_plan.folds,
        validation_triplet_ids=seed_plan.validation_triplet_ids,
        test_triplet_ids=seed_plan.test_triplet_ids,
        bootstrap_confidence_level=seed_plan.bootstrap_confidence_level,
        bootstrap_resamples=seed_plan.bootstrap_resamples,
        bootstrap_seed=seed_plan.bootstrap_seed,
        minimum_validation_lower_bound=seed_plan.minimum_validation_lower_bound,
        minimum_test_lower_bound=seed_plan.minimum_test_lower_bound,
        minimum_validation_worst_triplet_excess=(
            seed_plan.minimum_validation_worst_triplet_excess
        ),
        minimum_test_worst_triplet_excess=seed_plan.minimum_test_worst_triplet_excess,
        minimum_validation_worst_seed_excess=(
            seed_plan.minimum_validation_worst_seed_excess
        ),
        minimum_test_worst_seed_excess=seed_plan.minimum_test_worst_seed_excess,
        minimum_validation_triplet_pass_fraction=(
            seed_plan.minimum_validation_triplet_pass_fraction
        ),
        minimum_test_triplet_pass_fraction=seed_plan.minimum_test_triplet_pass_fraction,
    )
    manifest = stage_a_test_manifest_for_plan(plan)
    request = StageAEvaluationCellRequest(
        plan_digest=plan.digest,
        split="validation",
        triplet_id=plan.validation_triplet_ids[0],
        fold=0,
        seed=0,
        candidate_id="candidate-a",
        checkpoint_digest=plan.candidate("candidate-a").checkpoint_digest(0),
        evaluation_dataset_manifest_digest=manifest.digest,
        dataset_id=manifest.dataset_id_for("validation", plan.validation_triplet_ids[0]),
        evaluation_range=manifest.range_for("validation", 0),
        feature_identity=plan.feature_identity,
        execution_identity=plan.execution_identity,
        evaluation_identity=plan.evaluation_identity,
    )
    return plan, request


def _source_bytes(
    tmp_path: Path,
    request: StageAEvaluationCellRequest,
    *,
    candidate_config_digest: str,
) -> tuple[bytes, bytes]:
    events, terminal_book, terminal_order_book = execution_episode(
        dataset_id=request.dataset_id,
        execution_policy_digest=request.execution_identity,
        cash=1_000.0,
    )
    event_artifact = build_execution_event_artifact(
        candidate_config_digest=candidate_config_digest,
        evaluation_run_digest=request.digest,
        fold=request.fold,
        seed=request.seed,
        dataset_id=request.dataset_id,
        execution_policy_digest=request.execution_identity,
        actions=((0.4,),),
        observation_digests=(_digest("obs-0"), _digest("obs-1")),
        equity_curve=(1_000.0, 1_100.0),
        order_events=events,
        terminal_book=terminal_book,
        terminal_order_book=terminal_order_book,
    )
    event_path = write_execution_event_artifact(
        tmp_path / "order-events.json", event_artifact
    )
    evidence = execution_evidence_from_cost(
        dataset_id=request.dataset_id,
        cost=ExecutionCostConfig(path_mode="conservative"),
        sensitivity_path_modes=("conservative",),
        order_event_artifact_path=event_path,
    )
    return event_path.read_bytes(), canonical_json_bytes(evidence.to_mapping()) + b"\n"


def test_v3_replay_binds_funding_sidecar_and_validates_sources(tmp_path: Path) -> None:
    plan, request = _request()
    candidate_config_digest = plan.candidate("candidate-a").candidate_config_digest
    event_bytes, evidence_bytes = _source_bytes(
        tmp_path,
        request,
        candidate_config_digest=candidate_config_digest,
    )
    funding = build_funding_evidence_artifact(
        dataset_id=request.dataset_id,
        execution_policy_digest=request.execution_identity,
        symbol_count=1,
        boundaries=(),
    )

    artifact = build_stage_a_execution_replay_artifact(
        request=request,
        candidate_config_digest=candidate_config_digest,
        actions=((0.4,),),
        observation_digests=(_digest("obs-0"), _digest("obs-1")),
        equity_curve=(1_000.0, 1_100.0),
        event_artifact_bytes=event_bytes,
        execution_evidence_bytes=evidence_bytes,
        funding_evidence_bytes=funding.raw_bytes,
    )

    assert artifact.schema_version == "stage_a_execution_replay_v3"
    assert artifact.funding_evidence_digest == funding.digest
    assert artifact.funding_evidence_sha256 == hashlib.sha256(
        funding.raw_bytes
    ).hexdigest()
    assert artifact.funding_evidence_size_bytes == len(funding.raw_bytes)
    validate_stage_a_execution_replay_sources(
        artifact,
        event_artifact_bytes=event_bytes,
        execution_evidence_bytes=evidence_bytes,
        funding_evidence_bytes=funding.raw_bytes,
    )


def test_v3_replay_rejects_missing_or_wrong_funding_sidecar(tmp_path: Path) -> None:
    plan, request = _request()
    candidate_config_digest = plan.candidate("candidate-a").candidate_config_digest
    event_bytes, evidence_bytes = _source_bytes(
        tmp_path,
        request,
        candidate_config_digest=candidate_config_digest,
    )
    funding = build_funding_evidence_artifact(
        dataset_id=request.dataset_id,
        execution_policy_digest=request.execution_identity,
        symbol_count=1,
        boundaries=(),
    )
    artifact = build_stage_a_execution_replay_artifact(
        request=request,
        candidate_config_digest=candidate_config_digest,
        actions=((0.4,),),
        observation_digests=(_digest("obs-0"), _digest("obs-1")),
        equity_curve=(1_000.0, 1_100.0),
        event_artifact_bytes=event_bytes,
        execution_evidence_bytes=evidence_bytes,
        funding_evidence_bytes=funding.raw_bytes,
    )

    with pytest.raises(ValueError, match="funding evidence is required"):
        validate_stage_a_execution_replay_sources(
            artifact,
            event_artifact_bytes=event_bytes,
            execution_evidence_bytes=evidence_bytes,
        )
    with pytest.raises(ValueError, match="funding evidence"):
        validate_stage_a_execution_replay_sources(
            artifact,
            event_artifact_bytes=event_bytes,
            execution_evidence_bytes=evidence_bytes,
            funding_evidence_bytes=funding.raw_bytes.rstrip(),
        )
