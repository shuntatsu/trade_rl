"""Regression tests for terminal order archive fastpaths and digest parity."""

from __future__ import annotations

import hashlib
from dataclasses import asdict

import pytest

from trade_rl.artifacts import canonical_json_bytes
from trade_rl.evaluation.rl_allocation._native_progress import (
    _terminal_orders_sha256,
)
from trade_rl.evaluation.rl_allocation.transition_facts import _pending
from trade_rl.simulation.orders.model import (
    OrderDomainError,
    OrderIntent,
    OrderType,
    PendingOrder,
    TerminalOrderArchive,
    TimeInForce,
)


def _make_intent(
    *,
    submit_index: int = 0,
    eligible_index: int = 1,
    quantity: float = 1.0,
    symbol_index: int = 0,
) -> OrderIntent:
    return OrderIntent.create(
        dataset_id="d" * 64,
        target_identity="test-intent",
        execution_policy_digest="e" * 64,
        symbol_index=symbol_index,
        requested_quantity=quantity,
        order_type=OrderType.MARKET,
        time_in_force=TimeInForce.GTC,
        limit_price=None,
        stop_price=None,
        submit_index=submit_index,
        eligible_index=eligible_index,
        expiry_index=None,
        submission_reference_price=100.0,
        decision_equity=1000.0,
    )


def _make_terminal_order(
    index: int,
    *,
    submit_index: int | None = None,
    last_processed_index: int | None = None,
    trigger_index: int | None = None,
) -> PendingOrder:
    sub = index if submit_index is None else submit_index
    intent = _make_intent(submit_index=sub, eligible_index=sub + 1)
    p = PendingOrder.from_intent(intent)
    p = p.mark_eligible(processing_index=sub + 1)
    if trigger_index is not None:
        p = p.mark_triggered(processing_index=trigger_index)
    proc = sub + 1 if last_processed_index is None else last_processed_index
    return p.cancel(processing_index=proc, reason="test_done")


def test_terminal_order_archive_metadata_and_asdict_parity() -> None:
    orders = [
        _make_terminal_order(0, submit_index=2, last_processed_index=5),
        _make_terminal_order(1, submit_index=1, last_processed_index=8),
        _make_terminal_order(2, submit_index=10, last_processed_index=12),
    ]

    archive = TerminalOrderArchive()
    assert archive.max_clock_index == -1
    assert archive.asdict_tuple == ()

    for i, order in enumerate(orders):
        archive = archive.append(order)
        expected_asdicts = tuple(asdict(o) for o in archive)
        assert archive.asdict_tuple == expected_asdicts
        assert len(archive) == i + 1

    assert archive.max_submit_index == 10
    assert archive.max_clock_index == 12

    # Verify non-terminal active order is rejected
    active_intent = _make_intent(submit_index=0, eligible_index=1)
    pending_active = PendingOrder.from_intent(active_intent)
    with pytest.raises(
        OrderDomainError, match="terminal archive requires terminal orders"
    ):
        archive.append(pending_active)


def test_native_progress_terminal_orders_sha256_incremental_parity() -> None:
    orders = [_make_terminal_order(i) for i in range(15)]
    archive = TerminalOrderArchive()

    for order in orders:
        archive = archive.append(order)

        # Baseline reference: recompute from scratch using plain list
        ref_hasher = hashlib.sha256()
        for o in archive:
            ref_hasher.update(canonical_json_bytes(_pending(o)))
            ref_hasher.update(b"\n")
        ref_digest = ref_hasher.hexdigest()

        # Archive fastpath
        archive_digest = _terminal_orders_sha256(archive)
        assert archive_digest == ref_digest

        # Generic sequence fallback
        tuple_digest = _terminal_orders_sha256(tuple(archive))
        assert tuple_digest == ref_digest


def test_native_progress_terminal_orders_sha256_branching_parity() -> None:
    root = TerminalOrderArchive([_make_terminal_order(0), _make_terminal_order(1)])

    branch_a = root.append(_make_terminal_order(2, submit_index=5))
    branch_b = root.append(_make_terminal_order(3, submit_index=8))

    ref_a = hashlib.sha256()
    for o in branch_a:
        ref_a.update(canonical_json_bytes(_pending(o)))
        ref_a.update(b"\n")

    ref_b = hashlib.sha256()
    for o in branch_b:
        ref_b.update(canonical_json_bytes(_pending(o)))
        ref_b.update(b"\n")

    assert _terminal_orders_sha256(branch_a) == ref_a.hexdigest()
    assert _terminal_orders_sha256(branch_b) == ref_b.hexdigest()


def test_terminal_order_clock_ahead_rejection_parity() -> None:
    from tests.evaluation.test_after_cost_allocation import market
    from trade_rl.evaluation.allocation import _context
    from trade_rl.risk.pretrade import PreTradeRisk, PreTradeRiskConfig
    from trade_rl.simulation.accounting import BookState
    from trade_rl.simulation.execution import ExecutionCostConfig, MarketExecutor
    from trade_rl.simulation.orders.model import OrderBookState

    dataset = market()
    ex = MarketExecutor(dataset, ExecutionCostConfig.zero())
    quantities = [0.0] * dataset.n_symbols
    book = BookState(
        quantities,
        1000.0,
        dataset.resolved_array("mark_price")[0],
        1000.0,
    )
    risk = PreTradeRisk(PreTradeRiskConfig(drawdown_start=0.1, drawdown_stop=0.2))

    # Order with clock index 4
    future_order = _make_terminal_order(0, submit_index=2, last_processed_index=4)

    # 1. TerminalOrderArchive path at start_index=2 -> should raise
    archive_orders = OrderBookState(active_orders=(), terminal_orders=(future_order,))
    with pytest.raises(ValueError, match="terminal order clock is ahead"):
        _context(
            ex,
            book,
            archive_orders,
            account_id="test",
            pretrade_risk=risk,
            symbol_index=0,
            start_index=2,
        )

    # At start_index=4 -> should succeed
    ctx_archive = _context(
        ex,
        book,
        archive_orders,
        account_id="test",
        pretrade_risk=risk,
        symbol_index=0,
        start_index=4,
    )
    assert ctx_archive.state_digest is not None

    # 2. Raw tuple fallback path at start_index=2 -> should raise
    class MockOrderBook:
        active_orders = ()
        terminal_orders = (future_order,)

    with pytest.raises(ValueError, match="terminal order clock is ahead"):
        _context(
            ex,
            book,
            MockOrderBook(),  # type: ignore[arg-type]
            account_id="test",
            pretrade_risk=risk,
            symbol_index=0,
            start_index=2,
        )

    # At start_index=4 -> should produce identical state_digest
    ctx_tuple = _context(
        ex,
        book,
        MockOrderBook(),  # type: ignore[arg-type]
        account_id="test",
        pretrade_risk=risk,
        symbol_index=0,
        start_index=4,
    )
    assert ctx_archive.state_digest == ctx_tuple.state_digest
