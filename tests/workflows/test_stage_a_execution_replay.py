from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
import pytest

from trade_rl.artifacts.codec import canonical_json_bytes
from trade_rl.evaluation.stage_a_zero_shot_contracts import (
    StageACandidate,
    StageAZeroShotEvaluationPlan,
    build_stage_a_zero_shot_evaluation_plan,
)
from trade_rl.simulation.accounting import BookState
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.simulation.execution_promotion import execution_evidence_from_cost
from trade_rl.simulation.execution_replay import (
    build_execution_event_artifact,
    write_execution_event_artifact,
)
from trade_rl.simulation.orders import OrderBookState, OrderEvent, OrderStatus
from trade_rl.workflows.stage_a_execution_replay import (
    StageAExecutionCellIdentity,
    build_stage_a_execution_replay_artifact,
)
from trade_rl.workflows.stage_a_zero_shot_runner_contracts import (
    StageAEvaluationCellRequest,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _plan() -> StageAZeroShotEvaluationPlan:
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
        symbol_disjoint_manifest_digest=_digest("symbol-manifest"),
        symbol_disjoint_triplet_manifest_digest=_digest("triplet-manifest"),
        dataset_identity=_digest("dataset"),
        feature_identity=_digest("features"),
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
        dataset_identity=plan.dataset_identity,
        feature_identity=plan.feature_identity,
        execution_identity=plan.execution_identity,
        evaluation_identity=plan.evaluation_identity,
    )


def _event(dataset_id: str, execution_policy_digest: str) -> OrderEvent:
    return OrderEvent(
        schema_version="order_event_v1",
        sequence=0,
        order_id="a" * 64,
        replaced_order_id=None,
        dataset_id=dataset_id,
        execution_policy_digest=execution_policy_digest,
        symbol_index=0,
        event_type="filled",
        processing_index=1,
        timestamp_ns=1,
        previous_status=OrderStatus.ELIGIBLE,
        new_status=OrderStatus.FILLED,
        requested_quantity=1.0,
        remaining_quantity=0.0,
        filled_quantity=1.0,
        execution_price=100.0,
        filled_notional=100.0,
        capacity_before=10.0,
        capacity_after=9.0,
        participation_rate=0.1,
        trigger_segment=None,
        available_volume_fraction=1.0,
        reason=None,
        path_mode="conservative",
        path_points=(100.0, 101.0, 99.0, 100.5),
    )


def _promotion_bytes(
    tmp_path: Path,
    request: StageAEvaluationCellRequest,
    *,
    terminal_equity: float = 1_100.0,
) -> tuple[bytes, bytes, str]:
    event = _event(request.dataset_identity, request.execution_identity)
    event_artifact = build_execution_event_artifact(
        dataset_id=request.dataset_identity,
        execution_policy_digest=request.execution_identity,
        order_events=(event,),
        terminal_book=BookState(
            quantities=np.array((1.0,), dtype=np.float64),
            cash=terminal_equity - 100.0,
            mark_prices=np.array((100.0,), dtype=np.float64),
            peak_value=max(1_000.0, terminal_equity),
        ),
        terminal_order_book=OrderBookState.empty(),
    )
    event_path = write_execution_event_artifact(
        tmp_path / "order-events.json", event_artifact
    )
    evidence = execution_evidence_from_cost(
        dataset_id=request.dataset_identity,
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
    event_bytes, evidence_bytes, evidence_digest = _promotion_bytes(tmp_path, request)

    artifact = build_stage_a_execution_replay_artifact(
        request=request,
        candidate_config_digest=plan.candidate("candidate-a").candidate_config_digest,
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
            candidate_config_digest=plan.candidate(
                "candidate-a"
            ).candidate_config_digest,
        )
        == artifact.cell_identity
    )


def test_rejects_equity_curve_that_disagrees_with_terminal_book(
    tmp_path: Path,
) -> None:
    plan, request = _request(policy=True)
    event_bytes, evidence_bytes, _ = _promotion_bytes(
        tmp_path, request, terminal_equity=1_100.0
    )

    with pytest.raises(ValueError, match="terminal value mismatch"):
        build_stage_a_execution_replay_artifact(
            request=request,
            candidate_config_digest=(
                plan.candidate("candidate-a").candidate_config_digest
            ),
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
    event_bytes, evidence_bytes, _ = _promotion_bytes(tmp_path, request)

    with pytest.raises(ValueError, match="equity curve must be positive"):
        build_stage_a_execution_replay_artifact(
            request=request,
            candidate_config_digest=plan.candidate(
                "candidate-a"
            ).candidate_config_digest,
            actions=((0.4,),),
            observation_digests=(_digest("observation-0"), _digest("observation-1")),
            equity_curve=(1_000.0, 0.0),
            event_artifact_bytes=event_bytes,
            execution_evidence_bytes=evidence_bytes,
        )


def test_rejects_noncanonical_execution_evidence(tmp_path: Path) -> None:
    plan, request = _request(policy=True)
    event_bytes, evidence_bytes, _ = _promotion_bytes(tmp_path, request)

    with pytest.raises(ValueError, match="canonical encoding"):
        build_stage_a_execution_replay_artifact(
            request=request,
            candidate_config_digest=plan.candidate(
                "candidate-a"
            ).candidate_config_digest,
            actions=((0.4,),),
            observation_digests=(_digest("observation-0"), _digest("observation-1")),
            equity_curve=(1_000.0, 1_100.0),
            event_artifact_bytes=event_bytes,
            execution_evidence_bytes=evidence_bytes.rstrip(),
        )
