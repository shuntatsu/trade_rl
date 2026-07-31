from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from trade_rl.simulation.accounting import BookState
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.simulation.execution_promotion import (
    ExecutionPromotionError,
    execution_evidence_from_cost,
    validate_execution_promotion,
)
from trade_rl.simulation.execution_replay import (
    ExecutionReplayIdentity,
    build_execution_event_artifact,
    load_execution_event_artifact,
    write_execution_event_artifact_content_addressed,
)
from trade_rl.simulation.orders import (
    OrderBookState,
    OrderEvent,
    OrderIntent,
    OrderStatus,
    OrderType,
    PendingOrder,
    TimeInForce,
)

_DATASET = "d" * 64
_COST = ExecutionCostConfig(path_mode="conservative")
_POLICY = _COST.execution_policy_digest
_CANDIDATE = "c" * 64
_EVALUATION_RUN = "f" * 64


def _episode() -> tuple[tuple[OrderEvent, ...], BookState, OrderBookState]:
    intent = OrderIntent.create(
        dataset_id=_DATASET,
        target_identity="target",
        execution_policy_digest=_POLICY,
        symbol_index=0,
        requested_quantity=1.0,
        order_type=OrderType.MARKET,
        time_in_force=TimeInForce.GTC,
        limit_price=None,
        stop_price=None,
        submit_index=0,
        eligible_index=1,
        expiry_index=None,
        submission_reference_price=100.0,
        decision_equity=1_000.0,
    )
    submitted = PendingOrder.from_intent(intent)
    eligible = submitted.mark_eligible(processing_index=1)
    filled = eligible.apply_fill(quantity=1.0, notional=100.0, processing_index=1)

    common = dict(
        schema_version="order_event_v1",
        order_id=intent.order_id,
        replaced_order_id=None,
        dataset_id=_DATASET,
        execution_policy_digest=_POLICY,
        symbol_index=0,
        requested_quantity=1.0,
        capacity_before=0.0,
        capacity_after=0.0,
        participation_rate=0.0,
        trigger_segment=None,
        available_volume_fraction=0.0,
        path_mode="conservative",
        path_points=(),
    )
    events = (
        OrderEvent(
            **common,
            sequence=0,
            event_type="submitted",
            processing_index=0,
            timestamp_ns=1,
            previous_status=OrderStatus.SUBMITTED,
            new_status=OrderStatus.SUBMITTED,
            remaining_quantity=1.0,
            filled_quantity=0.0,
            execution_price=None,
            filled_notional=0.0,
            reason=None,
        ),
        OrderEvent(
            **common,
            sequence=1,
            event_type="eligible",
            processing_index=1,
            timestamp_ns=2,
            previous_status=OrderStatus.SUBMITTED,
            new_status=OrderStatus.ELIGIBLE,
            remaining_quantity=1.0,
            filled_quantity=0.0,
            execution_price=None,
            filled_notional=0.0,
            reason=None,
        ),
        OrderEvent(
            **{
                **common,
                "capacity_before": 1_000.0,
                "capacity_after": 900.0,
                "participation_rate": 0.1,
                "trigger_segment": "open",
                "available_volume_fraction": 1.0,
                "path_points": (100.0, 101.0, 99.0, 100.5),
            },
            sequence=2,
            event_type="filled",
            processing_index=1,
            timestamp_ns=2,
            previous_status=OrderStatus.ELIGIBLE,
            new_status=OrderStatus.FILLED,
            remaining_quantity=0.0,
            filled_quantity=1.0,
            execution_price=100.0,
            filled_notional=100.0,
            reason="filled",
        ),
    )
    book = BookState(
        quantities=np.array((1.0,), dtype=np.float64),
        cash=900.0,
        mark_prices=np.array((100.0,), dtype=np.float64),
        peak_value=1_000.0,
        fill_count=1,
        rebalance_events=1,
    )
    order_book = OrderBookState(active_orders=(), terminal_orders=(filled,))
    return events, book, order_book


def _artifact(**changes: object):
    events, book, order_book = _episode()
    values = {
        "candidate_config_digest": _CANDIDATE,
        "evaluation_run_digest": _EVALUATION_RUN,
        "fold": 2,
        "seed": 7,
        "dataset_id": _DATASET,
        "execution_policy_digest": _POLICY,
        "actions": ((0.4,),),
        "observation_digests": ("1" * 64, "2" * 64),
        "equity_curve": (1_000.0, 1_000.0),
        "order_events": events,
        "terminal_book": book,
        "terminal_order_book": order_book,
    }
    values.update(changes)
    return build_execution_event_artifact(**values)


def test_replay_artifact_binds_candidate_run_fold_seed_and_traces() -> None:
    artifact = _artifact()

    assert artifact.replay_identity == ExecutionReplayIdentity(
        candidate_config_digest=_CANDIDATE,
        evaluation_run_digest=_EVALUATION_RUN,
        fold=2,
        seed=7,
    )
    assert artifact.replay_evidence.seed == 7
    assert artifact.replay_evidence.step_count == 1
    assert artifact.replay_evidence.order_event_count == 3

    assert _artifact(candidate_config_digest="a" * 64).digest != artifact.digest
    assert _artifact(evaluation_run_digest="b" * 64).digest != artifact.digest
    assert _artifact(fold=3).digest != artifact.digest
    assert _artifact(seed=8).digest != artifact.digest
    assert _artifact(actions=((0.5,),)).digest != artifact.digest
    assert _artifact(equity_curve=(1_000.0, 999.0)).digest != artifact.digest
    assert _artifact(observation_digests=("1" * 64, "3" * 64)).digest != artifact.digest


def test_terminal_order_book_must_match_event_stream() -> None:
    events, book, _ = _episode()

    with pytest.raises(ValueError, match="terminal order book"):
        build_execution_event_artifact(
            candidate_config_digest=_CANDIDATE,
            evaluation_run_digest=_EVALUATION_RUN,
            fold=2,
            seed=7,
            dataset_id=_DATASET,
            execution_policy_digest=_POLICY,
            actions=((0.4,),),
            observation_digests=("1" * 64, "2" * 64),
            equity_curve=(1_000.0, 1_000.0),
            order_events=events,
            terminal_book=book,
            terminal_order_book=OrderBookState.empty(),
        )


def test_terminal_fill_count_must_match_fill_events() -> None:
    events, book, order_book = _episode()
    book.fill_count = 2

    with pytest.raises(ValueError, match="fill count"):
        build_execution_event_artifact(
            candidate_config_digest=_CANDIDATE,
            evaluation_run_digest=_EVALUATION_RUN,
            fold=2,
            seed=7,
            dataset_id=_DATASET,
            execution_policy_digest=_POLICY,
            actions=((0.4,),),
            observation_digests=("1" * 64, "2" * 64),
            equity_curve=(1_000.0, 1_000.0),
            order_events=events,
            terminal_book=book,
            terminal_order_book=order_book,
        )


def test_content_addressed_replay_write_is_idempotent_for_identical_bytes(
    tmp_path: Path,
) -> None:
    artifact = _artifact()

    first = write_execution_event_artifact_content_addressed(tmp_path, artifact)
    second = write_execution_event_artifact_content_addressed(tmp_path, artifact)

    assert first == second
    assert first.name == f"{artifact.digest}.execution-replay.json"
    assert load_execution_event_artifact(first) == artifact


def test_promotion_rejects_replay_identity_substitution(tmp_path: Path) -> None:
    artifact = _artifact()
    path = write_execution_event_artifact_content_addressed(tmp_path, artifact)
    evidence = execution_evidence_from_cost(
        dataset_id=_DATASET,
        cost=_COST,
        order_event_artifact_path=path,
        sensitivity_path_modes=("conservative",),
    )

    validate_execution_promotion(
        evidence,
        expected_policy_digest=_POLICY,
        event_artifact_path=path,
        expected_candidate_config_digest=_CANDIDATE,
        expected_evaluation_run_digest=_EVALUATION_RUN,
        expected_fold=2,
        expected_seed=7,
    )

    with pytest.raises(ExecutionPromotionError, match="candidate"):
        validate_execution_promotion(
            evidence,
            expected_policy_digest=_POLICY,
            event_artifact_path=path,
            expected_candidate_config_digest="0" * 64,
            expected_evaluation_run_digest=_EVALUATION_RUN,
            expected_fold=2,
            expected_seed=7,
        )

    forged = replace(evidence, replay_evidence_digest="0" * 64)
    with pytest.raises(ExecutionPromotionError, match="replay evidence"):
        validate_execution_promotion(
            forged,
            expected_policy_digest=_POLICY,
            event_artifact_path=path,
            expected_candidate_config_digest=_CANDIDATE,
            expected_evaluation_run_digest=_EVALUATION_RUN,
            expected_fold=2,
            expected_seed=7,
        )
