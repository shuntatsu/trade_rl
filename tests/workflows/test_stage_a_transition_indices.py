from __future__ import annotations

import numpy as np
import pytest

from trade_rl.simulation.accounting import BookState
from trade_rl.simulation.orders import (
    ORDER_EVENT_SCHEMA,
    OrderBookState,
    OrderEvent,
    OrderStatus,
)
from trade_rl.workflows.stage_a_execution_producer import StageAEvaluationEpisodeResult

_DIGEST = "a" * 64


def _book() -> BookState:
    return BookState(
        quantities=np.zeros(1, dtype=np.float64),
        cash=1_000.0,
        mark_prices=np.ones(1, dtype=np.float64),
        peak_value=1_000.0,
    )


def _event() -> OrderEvent:
    return OrderEvent(
        schema_version=ORDER_EVENT_SCHEMA,
        sequence=0,
        order_id=_DIGEST,
        replaced_order_id=None,
        dataset_id=_DIGEST,
        execution_policy_digest=_DIGEST,
        symbol_index=0,
        event_type="submitted",
        processing_index=11,
        timestamp_ns=11,
        previous_status=OrderStatus.SUBMITTED,
        new_status=OrderStatus.SUBMITTED,
        requested_quantity=1.0,
        remaining_quantity=1.0,
        filled_quantity=0.0,
        execution_price=None,
        filled_notional=0.0,
        capacity_before=1.0,
        capacity_after=1.0,
        participation_rate=0.0,
        trigger_segment=None,
        available_volume_fraction=1.0,
        reason=None,
        path_mode="conservative",
        path_points=(1.0, 1.0, 1.0, 1.0),
    )


def _result(
    *, transition_end_indices: tuple[int, ...]
) -> StageAEvaluationEpisodeResult:
    return StageAEvaluationEpisodeResult(
        request_digest=_DIGEST,
        policy_source_digest=None,
        candidate_config_digest=_DIGEST,
        actions=((0.0,), (0.0,)),
        observation_digests=(_DIGEST, _DIGEST, _DIGEST),
        equity_curve=(1_000.0, 1_000.0, 1_000.0),
        transition_end_indices=transition_end_indices,
        order_events=(_event(),),
        terminal_book=_book(),
        terminal_order_book=OrderBookState.empty(),
    )


def test_stage_a_episode_records_one_strictly_increasing_end_index_per_step() -> None:
    result = _result(transition_end_indices=(12, 15))

    assert result.transition_end_indices == (12, 15)


@pytest.mark.parametrize(
    "indices",
    [
        (12,),
        (12, 12),
        (15, 12),
        (12, -1),
    ],
)
def test_stage_a_episode_rejects_invalid_transition_end_indices(
    indices: tuple[int, ...],
) -> None:
    with pytest.raises(ValueError, match="transition end indices"):
        _result(transition_end_indices=indices)
