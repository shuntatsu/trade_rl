from __future__ import annotations

from pathlib import Path

import numpy as np

from trade_rl.simulation.accounting import BookState
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.simulation.execution_replay import (
    ExecutionEventArtifact,
    build_execution_event_artifact,
    write_execution_event_artifact,
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

DATASET_ID = "d" * 64
COST = ExecutionCostConfig(path_mode="conservative")
POLICY_DIGEST = COST.execution_policy_digest
CANDIDATE_CONFIG_DIGEST = "c" * 64
EVALUATION_RUN_DIGEST = "e" * 64
FOLD = 2
SEED = 7


def execution_episode(
    *,
    dataset_id: str = DATASET_ID,
    execution_policy_digest: str = POLICY_DIGEST,
    reason: str = "filled",
    cash: float = 900.0,
    peak_value: float | None = None,
) -> tuple[tuple[OrderEvent, ...], BookState, OrderBookState]:
    intent = OrderIntent.create(
        dataset_id=dataset_id,
        target_identity="target",
        execution_policy_digest=execution_policy_digest,
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
    common = {
        "schema_version": "order_event_v1",
        "order_id": intent.order_id,
        "replaced_order_id": None,
        "dataset_id": dataset_id,
        "execution_policy_digest": execution_policy_digest,
        "symbol_index": 0,
        "requested_quantity": 1.0,
        "capacity_before": 0.0,
        "capacity_after": 0.0,
        "participation_rate": 0.0,
        "trigger_segment": None,
        "available_volume_fraction": 0.0,
        "path_mode": "conservative",
        "path_points": (),
    }
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
            reason=reason,
        ),
    )
    terminal_value = cash + 100.0
    book = BookState(
        quantities=np.array((1.0,), dtype=np.float64),
        cash=cash,
        mark_prices=np.array((100.0,), dtype=np.float64),
        peak_value=(max(1_000.0, terminal_value) if peak_value is None else peak_value),
        fill_count=1,
        rebalance_events=1,
    )
    terminal_order = PendingOrder(
        intent=filled.intent,
        remaining_quantity=filled.remaining_quantity,
        cumulative_filled_quantity=filled.cumulative_filled_quantity,
        cumulative_filled_notional=filled.cumulative_filled_notional,
        status=filled.status,
        trigger_index=filled.trigger_index,
        last_processed_index=filled.last_processed_index,
        terminal_reason=reason,
        evidence_version=filled.evidence_version,
    )
    order_book = OrderBookState(active_orders=(), terminal_orders=(terminal_order,))
    return events, book, order_book


def execution_artifact(
    *,
    dataset_id: str = DATASET_ID,
    cost: ExecutionCostConfig = COST,
    candidate_config_digest: str = CANDIDATE_CONFIG_DIGEST,
    evaluation_run_digest: str = EVALUATION_RUN_DIGEST,
    fold: int = FOLD,
    seed: int = SEED,
    reason: str = "filled",
    cash: float = 900.0,
) -> ExecutionEventArtifact:
    events, book, order_book = execution_episode(
        dataset_id=dataset_id,
        execution_policy_digest=cost.execution_policy_digest,
        reason=reason,
        cash=cash,
    )
    return build_execution_event_artifact(
        candidate_config_digest=candidate_config_digest,
        evaluation_run_digest=evaluation_run_digest,
        fold=fold,
        seed=seed,
        dataset_id=dataset_id,
        execution_policy_digest=cost.execution_policy_digest,
        actions=((0.4,),),
        observation_digests=("1" * 64, "2" * 64),
        equity_curve=(1_000.0, 1_000.0),
        order_events=events,
        terminal_book=book,
        terminal_order_book=order_book,
    )


def write_execution_artifact(
    path: Path,
    *,
    dataset_id: str = DATASET_ID,
    cost: ExecutionCostConfig = COST,
) -> tuple[ExecutionEventArtifact, Path]:
    artifact = execution_artifact(dataset_id=dataset_id, cost=cost)
    return artifact, write_execution_event_artifact(path, artifact)
