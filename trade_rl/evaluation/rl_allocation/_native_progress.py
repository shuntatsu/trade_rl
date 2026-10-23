"""Passive advance notifications; no execution, accounting, IO or admission."""

from __future__ import annotations

import weakref
from collections.abc import Callable, Sequence
from copy import copy
from dataclasses import fields
from hashlib import sha256
from typing import Any, Literal

import numpy as np

from trade_rl.artifacts import canonical_json_bytes
from trade_rl.evaluation.rl_allocation.transition_facts import (
    _pending,
    native_allocation_account_facts,
)
from trade_rl.simulation import BookState, MarketExecutor
from trade_rl.simulation.orders.model import (
    OrderBookState,
    PendingOrder,
    TerminalOrderArchive,
)

AllocationAdvanceObserver = Callable[[Literal["actor", "reserve"], int, int], None]

_ARCHIVE_NODE_DIGESTS: weakref.WeakKeyDictionary[Any, tuple[Any, str]] = (
    weakref.WeakKeyDictionary()
)


def _terminal_orders_sha256(orders: Sequence[PendingOrder]) -> str:
    if isinstance(orders, TerminalOrderArchive):
        tail = orders._tail
        if tail is None:
            return sha256().hexdigest()
        cached = _ARCHIVE_NODE_DIGESTS.get(tail)
        if cached is not None:
            return cached[1]
        to_hash = []
        curr: Any = tail
        while curr is not None and curr not in _ARCHIVE_NODE_DIGESTS:
            to_hash.append(curr)
            curr = curr.previous
        hasher = sha256() if curr is None else _ARCHIVE_NODE_DIGESTS[curr][0].copy()
        for node in reversed(to_hash):
            hasher.update(canonical_json_bytes(_pending(node.order)))
            hasher.update(b"\n")
            _ARCHIVE_NODE_DIGESTS[node] = (hasher.copy(), hasher.hexdigest())
        return _ARCHIVE_NODE_DIGESTS[tail][1]
    archive = sha256()
    for order in orders:
        archive.update(canonical_json_bytes(_pending(order)))
        archive.update(b"\n")
    return archive.hexdigest()


def validate_advance_observer(observer: AllocationAdvanceObserver | None) -> None:
    if observer is not None and not callable(observer):
        raise ValueError("on_advance must be callable or None")


def _state(executor: MarketExecutor, book: BookState, orders: OrderBookState) -> bytes:
    # Reuse the native visible-state reader without deep-copying growing history.
    visible = copy(book)
    visible.returns_history = []
    terminal_sha = _terminal_orders_sha256(orders.terminal_orders)
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
            "terminal_orders_sha256": terminal_sha,
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
