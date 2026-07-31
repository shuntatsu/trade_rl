from __future__ import annotations

import hashlib
import math
from pathlib import Path

import pytest

from tests.evaluation.replay_support import execution_episode
from tests.stage_a_helpers import stage_a_test_manifest, stage_a_test_manifest_for_plan
from trade_rl.artifacts.codec import canonical_json_bytes
from trade_rl.evaluation.stage_a_zero_shot_contracts import (
    StageACandidate,
    StageAZeroShotEvaluationPlan,
    build_stage_a_zero_shot_evaluation_plan,
)
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.simulation.execution_promotion import execution_evidence_from_cost
from trade_rl.simulation.execution_replay import (
    build_execution_event_artifact,
    write_execution_event_artifact,
)
from trade_rl.workflows.stage_a_execution_replay import (
    StageAExecutionCellIdentity,
    build_stage_a_execution_replay_artifact,
)
from trade_rl.workflows.stage_a_zero_shot_runner_contracts import (
    StageAEvaluationCellRequest,
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


def _plan() -> StageAZeroShotEvaluationPlan:
    manifest = _manifest()
    candidate = StageACandidate.create(
        candidate_id="candidate-a",
        candidate_config_digest=_digest("candidate-a:config"),
        final_training_completion_digest=_digest("candidate-a:complete"),
        policy_identity=_digest("candidate-a:policy"),
        checkpoint_digests=(
            (0, _digest("candidate-a:checkpoint:0")),
            (1, _digest("candidate-a:checkpoint:1")),
        ),
    )
    return build_stage_a_zero_shot_evaluation_plan(
        symbol_disjoint_manifest_digest=manifest.symbol_disjoint_manifest_digest,
        symbol_disjoint_triplet_manifest_digest=manifest.symbol_disjoint_triplet_manifest_digest,
        evaluation_dataset_manifest_digest=manifest.digest,
        feature_identity=manifest.feature_identity,
        execution_identity=ExecutionCostConfig(
            path_mode="conservative"
        ).execution_policy_digest,
        evaluation_identity=_digest("evaluation"),
        candidates=(candidate,),
        seeds=(0, 1),
        folds=(0, 1),
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


def _request(
    *, policy: bool = True
) -> tuple[StageAZeroShotEvaluationPlan, StageAEvaluationCellRequest]:
    plan = _plan()
    manifest = stage_a_test_manifest_for_plan(plan)
    candidate_id = "candidate-a" if policy else None
    checkpoint_digest = (
        plan.candidate(candidate_id).checkpoint_digest(0) if candidate_id else None
    )
    return plan, StageAEvaluationCellRequest(
        plan_digest=plan.digest,
        split="validation",
        triplet_id=plan.validation_triplet_ids[0],
        fold=0,
        seed=0,
        candidate_id=candidate_id,
        checkpoint_digest=checkpoint_digest,
        evaluation_dataset_manifest_digest=manifest.digest,
        dataset_id=manifest.dataset_id_for(
            "validation", plan.validation_triplet_ids[0]
        ),
        evaluation_range=manifest.range_for("validation", 0),
        feature_identity=plan.feature_identity,
        execution_identity=plan.execution_identity,
        evaluation_identity=plan.evaluation_identity,
    )


def _promotion_bytes(
    tmp_path: Path,
    request: StageAEvaluationCellRequest,
    *,
    candidate_config_digest: str,
    terminal_equity: float = 1_100.0,
) -> tuple[bytes, bytes, str]:
    actions = ((0.4,),)
    observations = (_digest("observation-0"), _digest("observation-1"))
    equity = (1_000.0, terminal_equity)
    events, terminal_book, terminal_order_book = execution_episode(
        dataset_id=request.dataset_id,
        execution_policy_digest=request.execution_identity,
        cash=terminal_equity - 100.0,
    )
    event_artifact = build_execution_event_artifact(
        candidate_config_digest=candidate_config_digest,
        evaluation_run_digest=request.digest,
        fold=request.fold,
        seed=request.seed,
        dataset_id=request.dataset_id,
        execution_policy_digest=request.execution_identity,
        actions=actions,
        observation_digests=observations,
        equity_curve=equity,
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
    return (
        event_path.read_bytes(),
        canonical_json_bytes(evidence.to_mapping()) + b"\n",
        evidence.digest,
    )


def test_builds_policy_replay_and_recomputes_log_growth(tmp_path: Path) -> None:
    plan, request = _request(policy=True)
    candidate_config_digest = plan.candidate("candidate-a").candidate_config_digest
    event_bytes, evidence_bytes, evidence_digest = _promotion_bytes(
        tmp_path,
        request,
        candidate_config_digest=candidate_config_digest,
    )

    artifact = build_stage_a_execution_replay_artifact(
        request=request,
        candidate_config_digest=candidate_config_digest,
        actions=((0.4,),),
        observation_digests=(_digest("observation-0"), _digest("observation-1")),
        equity_curve=(1_000.0, 1_100.0),
        event_artifact_bytes=event_bytes,
        execution_evidence_bytes=evidence_bytes,
    )

    assert artifact.cell_identity.request_digest == request.digest
    assert artifact.log_growth == pytest.approx(math.log(1.1))
    assert artifact.execution_evidence_digest == evidence_digest
    assert (
        StageAExecutionCellIdentity.from_request(
            request,
            candidate_config_digest=candidate_config_digest,
        )
        == artifact.cell_identity
    )


def test_rejects_equity_curve_that_disagrees_with_terminal_book(
    tmp_path: Path,
) -> None:
    plan, request = _request(policy=True)
    candidate_config_digest = plan.candidate("candidate-a").candidate_config_digest
    event_bytes, evidence_bytes, _ = _promotion_bytes(
        tmp_path,
        request,
        candidate_config_digest=candidate_config_digest,
        terminal_equity=1_100.0,
    )

    with pytest.raises(ValueError, match="terminal value mismatch"):
        build_stage_a_execution_replay_artifact(
            request=request,
            candidate_config_digest=candidate_config_digest,
            actions=((0.4,),),
            observation_digests=(
                _digest("observation-0"),
                _digest("observation-1"),
            ),
            equity_curve=(1_000.0, 1_200.0),
            event_artifact_bytes=event_bytes,
            execution_evidence_bytes=evidence_bytes,
        )


def test_rejects_non_positive_equity(tmp_path: Path) -> None:
    plan, request = _request(policy=True)
    candidate_config_digest = plan.candidate("candidate-a").candidate_config_digest
    event_bytes, evidence_bytes, _ = _promotion_bytes(
        tmp_path,
        request,
        candidate_config_digest=candidate_config_digest,
    )

    with pytest.raises(ValueError, match="equity curve must be positive"):
        build_stage_a_execution_replay_artifact(
            request=request,
            candidate_config_digest=candidate_config_digest,
            actions=((0.4,),),
            observation_digests=(_digest("observation-0"), _digest("observation-1")),
            equity_curve=(1_000.0, 0.0),
            event_artifact_bytes=event_bytes,
            execution_evidence_bytes=evidence_bytes,
        )


def test_rejects_noncanonical_execution_evidence(tmp_path: Path) -> None:
    plan, request = _request(policy=True)
    candidate_config_digest = plan.candidate("candidate-a").candidate_config_digest
    event_bytes, evidence_bytes, _ = _promotion_bytes(
        tmp_path,
        request,
        candidate_config_digest=candidate_config_digest,
    )

    with pytest.raises(ValueError, match="canonical encoding"):
        build_stage_a_execution_replay_artifact(
            request=request,
            candidate_config_digest=candidate_config_digest,
            actions=((0.4,),),
            observation_digests=(_digest("observation-0"), _digest("observation-1")),
            equity_curve=(1_000.0, 1_100.0),
            event_artifact_bytes=event_bytes,
            execution_evidence_bytes=evidence_bytes.rstrip(),
        )
