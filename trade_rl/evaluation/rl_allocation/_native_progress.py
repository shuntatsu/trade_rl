"""Passive advance notifications; no execution, accounting, IO or admission."""

from __future__ import annotations

from collections.abc import Callable
from copy import copy
from dataclasses import fields
from hashlib import sha256
from typing import Literal

import numpy as np

from trade_rl.artifacts import canonical_json_bytes
from trade_rl.evaluation.rl_allocation.transition_facts import (
    _pending,
    native_allocation_account_facts,
)
from trade_rl.simulation import BookState, MarketExecutor
from trade_rl.simulation.orders.model import OrderBookState

AllocationAdvanceObserver = Callable[[Literal["actor", "reserve"], int, int], None]


def validate_advance_observer(observer: AllocationAdvanceObserver | None) -> None:
    if observer is not None and not callable(observer):
        raise ValueError("on_advance must be callable or None")


def _state(executor: MarketExecutor, book: BookState, orders: OrderBookState) -> bytes:
    # Reuse the native visible-state reader without deep-copying growing history.
    visible = copy(book)
    visible.returns_history = []
    archive = sha256()
    for order in orders.terminal_orders:
        archive.update(canonical_json_bytes(_pending(order)))
        archive.update(b"\n")
    history = np.asarray(book.returns_history, dtype=np.float64)
    return canonical_json_bytes(
        {
            "account": native_allocation_account_facts(visible, orders),
            "array_metadata": {
                field.name: [value.dtype.str, list(value.shape), list(value.strides)]
                for field in fields(book)
                if isinstance(value := getattr(book, field.name), np.ndarray)
            },
            "history_length": len(book.returns_history),
            "history_sha256": sha256(history.tobytes()).hexdigest(),
            "terminal_orders_sha256": archive.hexdigest(),
            "rng": executor._rng.bit_generator.state,
        }
    )


def notify_advance(
    observer: AllocationAdvanceObserver,
    phase: Literal["actor", "reserve"],
    start_index: int,
    processing_index: int,
    *,
    executor: MarketExecutor,
    book: BookState,
    orders: OrderBookState,
    validate_binding: Callable[[], None],
) -> None:
    before = _state(executor, book, orders)
    rng, history, archive = executor._rng, book.returns_history, orders.terminal_orders
    try:
        observer(phase, start_index, processing_index)
        if (
            executor._rng is not rng
            or book.returns_history is not history
            or orders.terminal_orders is not archive
            or _state(executor, book, orders) != before
        ):
            raise ValueError(
                "on_advance observer changed native account/order/RNG state"
            )
        validate_binding()
    except Exception as error:
        # Keep the primary exception and native frontier while distinguishing an
        # observer fault from the returned account's economic termination.
        try:
            setattr(error, "_allocation_advance_observer_failure", True)
        except Exception:
            pass  # Diagnostic annotation must never replace the primary failure.
        raise
