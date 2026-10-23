"""Integration of source-derived H contexts with the pinned event reader."""

from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from typing import Any

import pytest

from tools.market_replay_oracle import EventBatchPin, read_pinned_order_events

from .test_events import _event, _legacy
from .test_exact_fill_binding import _binding


@pytest.fixture(scope="module")
def binding() -> Any:
    return _binding()


def _contextualize(event: dict[str, Any], context: Any) -> dict[str, Any]:
    event.update(
        dataset_id=context.dataset_id,
        execution_policy_digest=context.execution_policy_digest,
        symbol_index=context.symbol_index,
        processing_index=context.processing_index,
        timestamp_ns=context.processing_time_ns,
    )
    return event


def _read(event: dict[str, Any], context: Any) -> tuple[Any, ...]:
    raw = json.dumps([event], allow_nan=False, separators=(",", ":")).encode("utf-8")
    pin = EventBatchPin(raw_bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    return read_pinned_order_events(raw, pin, context)


def test_bound_positive_lot_reconstructs_v2_fill_exactly(binding: Any) -> None:
    context = binding.event_context("H24_RIDGE_A1_BTCUSDT", "BASEL0", "nonrl", 1)
    assert context.effective_lot_size == 0.125

    event = _contextualize(
        _event(
            filled_lot_count=3,
            lot_size=0.125,
            filled_quantity=0.375,
            remaining_quantity=0.625,
        ),
        context,
    )
    parsed = _read(event, context)

    assert len(parsed) == 1
    assert parsed[0].exact_filled_quantity == Fraction(3, 8)


def test_bound_positive_lot_rejects_legacy_v1_fill(binding: Any) -> None:
    context = binding.event_context("H24_RIDGE_A1_BTCUSDT", "BASEL0", "nonrl", 1)
    event = _contextualize(_legacy(), context)

    with pytest.raises(ValueError, match="positive lot fill requires v2 witness"):
        _read(event, context)


def test_bound_positive_lot_rejects_self_consistent_but_wrong_step(
    binding: Any,
) -> None:
    context = binding.event_context("H24_RIDGE_A1_BTCUSDT", "BASEL0", "nonrl", 1)
    event = _contextualize(
        _event(
            filled_lot_count=3,
            lot_size=0.25,
            filled_quantity=0.75,
            remaining_quantity=0.25,
        ),
        context,
    )

    with pytest.raises(
        ValueError, match="event lot differs from expected effective lot"
    ):
        _read(event, context)


def test_verified_zero_lot_context_accepts_legacy_v1_fill(binding: Any) -> None:
    context = binding.event_context("H24_RIDGE_A1_BTCUSDT", "BASEL0", "nonrl", 6528)
    assert context.processing_index == 9431
    assert context.effective_lot_size == 0.0

    event = _contextualize(_legacy(), context)
    parsed = _read(event, context)

    assert parsed[0].exact_filled_quantity == Fraction(3, 10)


@pytest.mark.parametrize("sign", [1, -1])
def test_bound_lot_context_accepts_signed_v2_witnesses(binding: Any, sign: int) -> None:
    context = binding.event_context("H24_RIDGE_A1_BTCUSDT", "BASEL0", "nonrl", 1)
    quantity = sign * 0.375
    event = _contextualize(
        _event(
            requested_quantity=sign * 1.0,
            remaining_quantity=sign * 0.625,
            filled_quantity=quantity,
            filled_lot_count=sign * 3,
            lot_size=0.125,
        ),
        context,
    )

    assert _read(event, context)[0].exact_filled_quantity == sign * Fraction(3, 8)


@pytest.mark.parametrize("mismatch", ["symbol", "policy", "decision_pair"])
def test_bound_context_rejects_wrong_event_identity_or_clock(
    binding: Any, mismatch: str
) -> None:
    context = binding.event_context("H24_RIDGE_A1_BTCUSDT", "BASEL0", "nonrl", 1)
    event = _contextualize(
        _legacy(
            fill=False,
            event_type="submitted",
            previous_status="submitted",
            new_status="submitted",
            processing_index=context.decision_index,
            timestamp_ns=context.decision_time_ns,
        ),
        context,
    )
    event["processing_index"] = context.decision_index
    event["timestamp_ns"] = context.decision_time_ns

    if mismatch == "symbol":
        event["symbol_index"] = 1
    elif mismatch == "policy":
        event["execution_policy_digest"] = (
            "0" * 64 if context.execution_policy_digest != "0" * 64 else "1" * 64
        )
    else:
        event["timestamp_ns"] = context.processing_time_ns

    with pytest.raises(ValueError):
        _read(event, context)
