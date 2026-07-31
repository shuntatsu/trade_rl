from __future__ import annotations

import hashlib
from dataclasses import replace
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
from trade_rl.workflows.stage_a_execution_store import StageAExecutionPromotionStore
from trade_rl.workflows.stage_a_production_evaluator import (
    ArtifactBackedStageAEvaluationCellEvaluator,
)
from trade_rl.workflows.stage_a_zero_shot_runner_contracts import (
    StageAEvaluationCellRequest,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


_BASELINE_CONFIG = _digest("baseline:config")


def _plan() -> StageAZeroShotEvaluationPlan:
    cost = ExecutionCostConfig(path_mode="conservative")
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
        execution_identity=cost.execution_policy_digest,
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
    plan: StageAZeroShotEvaluationPlan, *, policy: bool
) -> StageAEvaluationCellRequest:
    candidate_id = "candidate-a" if policy else None
    checkpoint = (
        plan.candidate("candidate-a").checkpoint_digest(0) if policy else None
    )
    return StageAEvaluationCellRequest(
        plan_digest=plan.digest,
        split="validation",
        triplet_id=plan.validation_triplet_ids[0],
        fold=0,
        seed=0,
        candidate_id=candidate_id,
        checkpoint_digest=checkpoint,
        dataset_identity=plan.dataset_identity,
        feature_identity=plan.feature_identity,
        execution_identity=plan.execution_identity,
        evaluation_identity=plan.evaluation_identity,
    )


def _event(request: StageAEvaluationCellRequest) -> OrderEvent:
    return OrderEvent(
        schema_version="order_event_v1",
        sequence=0,
        order_id="a" * 64,
        replaced_order_id=None,
        dataset_id=request.dataset_identity,
        execution_policy_digest=request.execution_identity,
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


def _publish(
    store: StageAExecutionPromotionStore,
    request: StageAEvaluationCellRequest,
    *,
    candidate_config_digest: str,
    root: Path,
    final_equity: float,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    event_artifact = build_execution_event_artifact(
        dataset_id=request.dataset_identity,
        execution_policy_digest=request.execution_identity,
        order_events=(_event(request),),
        terminal_book=BookState(
            quantities=np.array((1.0,), dtype=np.float64),
            cash=final_equity - 100.0,
            mark_prices=np.array((100.0,), dtype=np.float64),
            peak_value=max(1_000.0, final_equity),
        ),
        terminal_order_book=OrderBookState.empty(),
    )
    event_path = write_execution_event_artifact(root / "events.json", event_artifact)
    evidence = execution_evidence_from_cost(
        dataset_id=request.dataset_identity,
        cost=ExecutionCostConfig(path_mode="conservative"),
        sensitivity_path_modes=("conservative",),
        order_event_artifact_path=event_path,
    )
    evidence_path = root / "evidence.json"
    evidence_path.write_bytes(canonical_json_bytes(evidence.to_mapping()) + b"\n")
    store.publish(
        request=request,
        candidate_config_digest=candidate_config_digest,
        actions=((0.4,),),
        observation_digests=(_digest("observation-0"), _digest("observation-1")),
        equity_curve=(1_000.0, final_equity),
        event_artifact_path=event_path,
        execution_evidence_path=evidence_path,
    )


def _evaluator(
    tmp_path: Path,
) -> tuple[
    StageAZeroShotEvaluationPlan,
    StageAExecutionPromotionStore,
    ArtifactBackedStageAEvaluationCellEvaluator,
]:
    plan = _plan()
    store = StageAExecutionPromotionStore(tmp_path / "store")
    evaluator = ArtifactBackedStageAEvaluationCellEvaluator(
        plan=plan,
        store=store,
        baseline_candidate_config_digest=_BASELINE_CONFIG,
    )
    return plan, store, evaluator


def test_returns_verified_policy_and_baseline_growth(tmp_path: Path) -> None:
    plan, store, evaluator = _evaluator(tmp_path)
    policy = _request(plan, policy=True)
    baseline = _request(plan, policy=False)
    _publish(
        store,
        policy,
        candidate_config_digest=plan.candidate("candidate-a").candidate_config_digest,
        root=tmp_path / "policy",
        final_equity=1_100.0,
    )
    _publish(
        store,
        baseline,
        candidate_config_digest=_BASELINE_CONFIG,
        root=tmp_path / "baseline",
        final_equity=1_020.0,
    )

    policy_result = evaluator.evaluate(policy)
    baseline_result = evaluator.evaluate(baseline)

    assert policy_result.request_digest == policy.digest
    assert policy_result.log_growth == pytest.approx(0.09531017980432493)
    assert baseline_result.request_digest == baseline.digest
    assert baseline_result.log_growth == pytest.approx(0.01980262729617973)
    assert policy_result.execution_evidence_digest != (
        baseline_result.execution_evidence_digest
    )
    assert policy_result.execution_evidence_digest == store.load(policy.digest).artifact.digest
    assert baseline_result.execution_evidence_digest == (
        store.load(baseline.digest).artifact.digest
    )


def test_rejects_policy_candidate_configuration_substitution(tmp_path: Path) -> None:
    plan, store, evaluator = _evaluator(tmp_path)
    request = _request(plan, policy=True)
    _publish(
        store,
        request,
        candidate_config_digest=_digest("forged-candidate-config"),
        root=tmp_path / "policy",
        final_equity=1_100.0,
    )

    with pytest.raises(ValueError, match="candidate configuration mismatch"):
        evaluator.evaluate(request)


def test_rejects_baseline_configuration_substitution(tmp_path: Path) -> None:
    plan, store, evaluator = _evaluator(tmp_path)
    request = _request(plan, policy=False)
    _publish(
        store,
        request,
        candidate_config_digest=_digest("candidate-dependent-baseline"),
        root=tmp_path / "baseline",
        final_equity=1_020.0,
    )

    with pytest.raises(ValueError, match="baseline configuration mismatch"):
        evaluator.evaluate(request)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("feature_identity", _digest("other-features"), "feature identity mismatch"),
        ("evaluation_identity", _digest("other-evaluation"), "evaluation identity mismatch"),
        ("triplet_id", _digest("other-triplet"), "triplet is not declared"),
        ("fold", 9, "fold is not declared"),
        ("seed", 9, "seed is not declared"),
    ),
)
def test_rejects_request_outside_the_plan(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    plan, _, evaluator = _evaluator(tmp_path)
    request = replace(_request(plan, policy=True), **{field: value}, digest="")

    with pytest.raises(ValueError, match=message):
        evaluator.evaluate(request)


def test_rejects_checkpoint_substitution_before_store_access(tmp_path: Path) -> None:
    plan, _, evaluator = _evaluator(tmp_path)
    request = replace(
        _request(plan, policy=True),
        checkpoint_digest=_digest("other-checkpoint"),
        digest="",
    )

    with pytest.raises(ValueError, match="checkpoint identity mismatch"):
        evaluator.evaluate(request)
