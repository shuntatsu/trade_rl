from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from trade_rl.simulation.accounting import BookState
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.simulation.execution_promotion import (
    ExecutionEvidence,
    ExecutionPromotionError,
    execution_evidence_from_cost,
    validate_execution_event_artifact,
    validate_execution_promotion,
)
from trade_rl.simulation.execution_replay import (
    build_execution_event_artifact,
    write_execution_event_artifact,
)
from trade_rl.simulation.orders import (
    OrderBookState,
    OrderEvent,
    OrderStatus,
)

_DATASET_ID = "d" * 64
_POLICY_DIGEST = "e" * 64


def _event(*, reason: str | None = None) -> OrderEvent:
    return OrderEvent(
        schema_version="order_event_v1",
        sequence=0,
        order_id="a" * 64,
        replaced_order_id=None,
        dataset_id=_DATASET_ID,
        execution_policy_digest=_POLICY_DIGEST,
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
        reason=reason,
        path_mode="conservative",
        path_points=(100.0, 101.0, 99.0, 100.5),
    )


def _terminal_book(*, cash: float = 900.0) -> BookState:
    return BookState(
        quantities=np.array((1.0,), dtype=np.float64),
        cash=cash,
        mark_prices=np.array((100.0,), dtype=np.float64),
        peak_value=1_000.0,
    )


def _evidence(tmp_path: Path) -> tuple[ExecutionEvidence, Path]:
    artifact = build_execution_event_artifact(
        dataset_id=_DATASET_ID,
        execution_policy_digest=_POLICY_DIGEST,
        order_events=(_event(),),
        terminal_book=_terminal_book(),
        terminal_order_book=OrderBookState.empty(),
    )
    path = write_execution_event_artifact(tmp_path / "order-events.json", artifact)
    evidence = execution_evidence_from_cost(
        dataset_id=_DATASET_ID,
        cost=ExecutionCostConfig(path_mode="conservative"),
        order_event_artifact_path=path,
        sensitivity_path_modes=("conservative",),
    )
    return evidence, path


def test_zero_order_events_cannot_be_promoted_as_complete_evidence() -> None:
    evidence = ExecutionEvidence(
        dataset_id=_DATASET_ID,
        execution_policy_digest=_POLICY_DIGEST,
        path_mode="conservative",
        processing_bar_volume_capacity=True,
        partial_fill_carry=True,
        trigger_volume_fractions=(1.0, 0.5, 0.25, 0.0),
        order_event_count=0,
        complete_order_evidence=True,
        sensitivity_path_modes=("conservative",),
    )

    with pytest.raises(ExecutionPromotionError, match="order event"):
        validate_execution_promotion(
            evidence,
            expected_policy_digest=_POLICY_DIGEST,
        )


def test_event_artifact_rejects_forged_count_and_terminal_digests(
    tmp_path: Path,
) -> None:
    evidence, path = _evidence(tmp_path)
    validate_execution_event_artifact(evidence, path)

    with pytest.raises(ExecutionPromotionError, match="event count"):
        validate_execution_event_artifact(
            replace(evidence, order_event_count=evidence.order_event_count + 1),
            path,
        )
    with pytest.raises(ExecutionPromotionError, match="terminal book"):
        validate_execution_event_artifact(
            replace(evidence, terminal_book_digest="f" * 64),
            path,
        )
    with pytest.raises(ExecutionPromotionError, match="terminal order book"):
        validate_execution_event_artifact(
            replace(evidence, terminal_order_book_digest="1" * 64),
            path,
        )


def test_event_artifact_substitution_is_rejected(tmp_path: Path) -> None:
    evidence, path = _evidence(tmp_path)
    replacement_artifact = build_execution_event_artifact(
        dataset_id=_DATASET_ID,
        execution_policy_digest=_POLICY_DIGEST,
        order_events=(_event(reason="substituted"),),
        terminal_book=_terminal_book(cash=899.0),
        terminal_order_book=OrderBookState.empty(),
    )
    path.unlink()
    write_execution_event_artifact(path, replacement_artifact)

    with pytest.raises(ExecutionPromotionError, match="artifact digest"):
        validate_execution_event_artifact(evidence, path)


def test_event_artifact_symlink_is_rejected(tmp_path: Path) -> None:
    evidence, path = _evidence(tmp_path)
    link = tmp_path / "order-events-link.json"
    try:
        link.symlink_to(path.name)
    except OSError:
        pytest.skip("symlinks are unavailable on this platform")

    with pytest.raises(ExecutionPromotionError, match="regular non-symlink"):
        validate_execution_event_artifact(evidence, link)
