from __future__ import annotations

import trade_rl.simulation.legacy_trace_adapter as legacy_trace_adapter
from trade_rl.simulation.funding_evidence import FundingBoundaryEvidence
from trade_rl.simulation.legacy_trace_adapter import canonicalize_legacy_fill_events
from trade_rl.simulation.orders import ORDER_EVENT_SCHEMA, OrderEvent, OrderStatus

_DIGEST = "a" * 64


def _event(
    *,
    sequence: int,
    event_type: str,
    filled_quantity: float,
    execution_price: float | None,
    timestamp_ns: int,
) -> OrderEvent:
    return OrderEvent(
        schema_version=ORDER_EVENT_SCHEMA,
        sequence=sequence,
        order_id=_DIGEST,
        replaced_order_id=None,
        dataset_id="dataset",
        execution_policy_digest=_DIGEST,
        symbol_index=0,
        event_type=event_type,
        processing_index=sequence,
        timestamp_ns=timestamp_ns,
        previous_status=OrderStatus.ELIGIBLE,
        new_status=(
            OrderStatus.FILLED if event_type == "filled" else OrderStatus.ELIGIBLE
        ),
        requested_quantity=1.0 if filled_quantity >= 0.0 else -1.0,
        remaining_quantity=0.0
        if event_type == "filled"
        else (1.0 if filled_quantity >= 0.0 else -1.0),
        filled_quantity=filled_quantity,
        execution_price=execution_price,
        filled_notional=0.0
        if execution_price is None
        else abs(filled_quantity * execution_price),
        capacity_before=10.0,
        capacity_after=9.0,
        participation_rate=0.1 if filled_quantity else 0.0,
        trigger_segment=None,
        available_volume_fraction=1.0,
        reason=None,
        path_mode="neutral",
        path_points=(100.0, 99.0, 101.0, 100.0),
    )


def test_legacy_adapter_keeps_only_fills_and_tracks_signed_position() -> None:
    events = (
        _event(
            sequence=0,
            event_type="no_fill",
            filled_quantity=0.0,
            execution_price=None,
            timestamp_ns=5,
        ),
        _event(
            sequence=1,
            event_type="filled",
            filled_quantity=1.0,
            execution_price=100.1,
            timestamp_ns=10,
        ),
        _event(
            sequence=2,
            event_type="filled",
            filled_quantity=-1.0,
            execution_price=104.9,
            timestamp_ns=200,
        ),
    )

    fills = canonicalize_legacy_fill_events(
        events,
        price_tick=0.1,
        lot_size=0.001,
    )

    assert [fill.price_ticks for fill in fills] == [1001, 1049]
    assert [fill.quantity_lots for fill in fills] == [1000, -1000]
    assert [fill.position_lots for fill in fills] == [1000, 0]
    assert [fill.timestamp_ns for fill in fills] == [10, 200]


def test_legacy_funding_boundary_becomes_single_instrument_canonical_record() -> None:
    boundary = FundingBoundaryEvidence(
        processing_index=1,
        timestamp_ns=3_600_000_000_000,
        funding_due=(True,),
        signed_quantities=(10.0,),
        mark_prices=(120.0,),
        contract_multipliers=(1.0,),
        funding_rates=(0.001,),
        funding_amount=-1.2,
        equity_before_funding=1_200.0,
        equity_after_funding=1_198.8,
    )

    record = legacy_trace_adapter.canonicalize_legacy_funding_boundary_record(
        boundary,
        sequence=2,
        price_tick=0.1,
        lot_size=0.001,
        currency_precision=8,
        equity_before_minor=120_000_000_000,
    )

    assert record.sequence == 2
    assert record.event_type == "funding"
    assert record.timestamp_ns == boundary.timestamp_ns
    assert record.price_ticks == 1_200
    assert record.quantity_lots == 0
    assert record.fee_minor == 0
    assert record.funding_minor == -120_000_000
    assert record.position_lots == 10_000
    assert record.equity_minor == 119_880_000_000
    assert record.terminal_reason is None
