from __future__ import annotations

import hashlib
import math
from dataclasses import replace

import numpy as np
import pytest

from trade_rl.evaluation.stage_a_zero_shot_contracts import (
    StageACandidate,
    StageAZeroShotEvaluationPlan,
    build_stage_a_zero_shot_evaluation_plan,
)
from trade_rl.simulation.accounting import BookState
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.simulation.orders import OrderBookState, OrderEvent, OrderStatus
from trade_rl.workflows.stage_a_execution_producer import (
    StageAEvaluationEpisodeExecutor,
    StageAEvaluationEpisodeResult,
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


def _request(*, policy: bool) -> StageAEvaluationCellRequest:
    plan = _plan()
    candidate_id = "candidate-a" if policy else None
    checkpoint_digest = (
        plan.candidate("candidate-a").checkpoint_digest(0) if policy else None
    )
    return StageAEvaluationCellRequest(
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


def _result(*, policy: bool = True) -> StageAEvaluationEpisodeResult:
    request = _request(policy=policy)
    candidate_config_digest = (
        _plan().candidate("candidate-a").candidate_config_digest
        if policy
        else _digest("baseline-config")
    )
    return StageAEvaluationEpisodeResult(
        request_digest=request.digest,
        policy_source_digest=_digest("policy-source") if policy else None,
        candidate_config_digest=candidate_config_digest,
        actions=((0.4,),),
        observation_digests=(_digest("observation-0"), _digest("observation-1")),
        equity_curve=(1_000.0, 1_100.0),
        order_events=(_event(request),),
        terminal_book=BookState(
            quantities=np.array((1.0,), dtype=np.float64),
            cash=1_000.0,
            mark_prices=np.array((100.0,), dtype=np.float64),
            peak_value=1_100.0,
        ),
        terminal_order_book=OrderBookState.empty(),
    )


def test_policy_result_validates_complete_request_identity() -> None:
    request = _request(policy=True)
    result = _result(policy=True)

    validated = result.validate_against(
        request,
        expected_policy_source_digest=_digest("policy-source"),
        expected_candidate_config_digest=(
            _plan().candidate("candidate-a").candidate_config_digest
        ),
    )

    assert validated is result


def test_baseline_result_requires_null_policy_source() -> None:
    request = _request(policy=False)
    result = _result(policy=False)

    assert (
        result.validate_against(
            request,
            expected_policy_source_digest=None,
            expected_candidate_config_digest=_digest("baseline-config"),
        )
        is result
    )


def test_executor_protocol_exposes_request_bound_execution() -> None:
    assert hasattr(StageAEvaluationEpisodeExecutor, "execute")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("request_digest", _digest("other-request"), "request digest mismatch"),
        (
            "policy_source_digest",
            _digest("other-policy-source"),
            "policy source digest mismatch",
        ),
        (
            "candidate_config_digest",
            _digest("other-config"),
            "candidate config digest mismatch",
        ),
    ),
)
def test_validate_rejects_identity_substitution(
    field: str,
    value: str,
    message: str,
) -> None:
    request = _request(policy=True)
    result = replace(_result(policy=True), **{field: value})

    with pytest.raises(ValueError, match=message):
        result.validate_against(
            request,
            expected_policy_source_digest=_digest("policy-source"),
            expected_candidate_config_digest=(
                _plan().candidate("candidate-a").candidate_config_digest
            ),
        )


def test_validate_rejects_policy_result_with_null_source() -> None:
    request = _request(policy=True)
    result = replace(_result(policy=True), policy_source_digest=None)

    with pytest.raises(ValueError, match="requires a policy source"):
        result.validate_against(
            request,
            expected_policy_source_digest=None,
            expected_candidate_config_digest=(
                _plan().candidate("candidate-a").candidate_config_digest
            ),
        )


def test_validate_rejects_baseline_result_with_policy_source() -> None:
    request = _request(policy=False)
    result = replace(
        _result(policy=False),
        policy_source_digest=_digest("unexpected-policy-source"),
    )

    with pytest.raises(ValueError, match="must not define a policy source"):
        result.validate_against(
            request,
            expected_policy_source_digest=_digest("unexpected-policy-source"),
            expected_candidate_config_digest=_digest("baseline-config"),
        )


@pytest.mark.parametrize(
    ("actions", "message"),
    (
        ((), "actions must not be empty"),
        (((),), "actions must not be empty"),
        (((math.nan,),), "actions.*finite"),
        (((math.inf,),), "actions.*finite"),
    ),
)
def test_rejects_invalid_actions(
    actions: tuple[tuple[float, ...], ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        replace(_result(), actions=actions)


@pytest.mark.parametrize(
    ("observation_digests", "message"),
    (
        ((), "observations must not be empty"),
        ((_digest("observation-0"),), "observation closure mismatch"),
        (
            (
                _digest("observation-0"),
                _digest("observation-1"),
                _digest("observation-2"),
            ),
            "observation closure mismatch",
        ),
    ),
)
def test_rejects_invalid_observation_closure(
    observation_digests: tuple[str, ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        replace(_result(), observation_digests=observation_digests)


@pytest.mark.parametrize(
    ("equity_curve", "message"),
    (
        ((1_000.0,), "equity closure mismatch"),
        ((1_000.0, 0.0), "equity curve must be positive"),
        ((1_000.0, -1.0), "equity curve must be positive"),
        ((1_000.0, math.nan), "equity curve must be finite"),
        ((1_000.0, math.inf), "equity curve must be finite"),
    ),
)
def test_rejects_invalid_equity_curve(
    equity_curve: tuple[float, ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        replace(_result(), equity_curve=equity_curve)


def test_rejects_equity_that_disagrees_with_terminal_book() -> None:
    with pytest.raises(ValueError, match="terminal equity mismatch"):
        replace(_result(), equity_curve=(1_000.0, 1_200.0))


def test_rejects_missing_order_events() -> None:
    with pytest.raises(ValueError, match="order events must not be empty"):
        replace(_result(), order_events=())


def test_validate_rejects_order_event_dataset_substitution() -> None:
    request = _request(policy=True)
    event = replace(_event(request), dataset_id=_digest("other-dataset"))
    result = replace(_result(), order_events=(event,))

    with pytest.raises(ValueError, match="order event dataset mismatch"):
        result.validate_against(
            request,
            expected_policy_source_digest=_digest("policy-source"),
            expected_candidate_config_digest=(
                _plan().candidate("candidate-a").candidate_config_digest
            ),
        )


def test_validate_rejects_order_event_execution_substitution() -> None:
    request = _request(policy=True)
    event = replace(
        _event(request),
        execution_policy_digest=_digest("other-execution"),
    )
    result = replace(_result(), order_events=(event,))

    with pytest.raises(ValueError, match="order event execution mismatch"):
        result.validate_against(
            request,
            expected_policy_source_digest=_digest("policy-source"),
            expected_candidate_config_digest=(
                _plan().candidate("candidate-a").candidate_config_digest
            ),
        )


def test_rejects_invalid_terminal_state_types() -> None:
    with pytest.raises(ValueError, match="terminal book"):
        replace(_result(), terminal_book=object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="terminal order book"):
        replace(_result(), terminal_order_book=object())  # type: ignore[arg-type]
