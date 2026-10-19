"""Regression tests for terminal order archive fastpaths and digest parity."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, fields, replace

import pytest

from trade_rl.artifacts import canonical_json_bytes
from trade_rl.evaluation.rl_allocation._native_progress import (
    _terminal_orders_sha256,
)
from trade_rl.evaluation.rl_allocation.transition_facts import _pending
from trade_rl.simulation.orders.model import (
    OrderBookState,
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


def test_archive_retains_only_linear_immutable_snapshots() -> None:
    """Appending may not retain a copied all-history tuple per node."""
    archive = TerminalOrderArchive()
    originals = []
    past = [archive]
    for index in range(128):
        order = _make_terminal_order(index)
        originals.append(order)
        archive = archive.append(order)
        past.append(archive)

    node = archive._tail
    seen = []
    while node is not None:
        names = {field.name for field in fields(node)}
        assert "_order_bytes" in names
        assert "asdict_tuple" not in names
        assert "asdict_payload" not in names
        assert type(node._order_bytes) is bytes
        seen.append(node)
        node = node.previous
    assert len(seen) == 128
    assert sum(len(node._order_bytes) for node in seen) == sum(
        len(canonical_json_bytes(asdict(order))) for order in originals
    )
    assert past[0].asdict_tuple == ()
    for index in (1, 2, 64, 128):
        assert past[index].asdict_tuple == tuple(
            asdict(order) for order in originals[:index]
        )


def test_archive_seals_original_order_and_detaches_public_order_views() -> None:
    original = _make_terminal_order(0)
    expected = asdict(original)
    archive = TerminalOrderArchive((original,))
    cached = _terminal_orders_sha256(archive)

    # A caller retaining the original PendingOrder is not allowed to alter
    # native archive facts after a successful append.
    object.__setattr__(original, "terminal_reason", "changed_after_append")
    assert asdict(archive[-1]) == expected

    # The public iterator/index accessors expose detached objects, not the
    # sealed order owned by the archive or its cached progress digest.
    exposed = archive[-1]
    object.__setattr__(exposed, "terminal_reason", "changed_in_view")
    object.__setattr__(exposed.intent, "target_identity", "changed_in_view")
    assert asdict(next(iter(archive))) == expected

    tail = archive._tail
    assert tail is not None
    internal_view = tail.order
    object.__setattr__(internal_view, "terminal_reason", "changed_via_tail")
    assert asdict(archive[-1]) == expected

    assert _terminal_orders_sha256(archive) == cached
    assert _terminal_orders_sha256(tuple(archive)) == cached


def test_archive_asdict_materialization_does_not_alias_future_receipts() -> None:
    orders = tuple(_make_terminal_order(i) for i in range(4))
    archive = TerminalOrderArchive(orders)
    before = canonical_json_bytes(tuple(asdict(order) for order in orders))
    earlier = archive.asdict_tuple
    earlier[0]["intent"]["target_identity"] = "forged"
    earlier[-1]["terminal_reason"] = "forged"
    assert canonical_json_bytes(archive.asdict_tuple) == before

    branched = archive.append(_make_terminal_order(10))
    assert canonical_json_bytes(archive.asdict_tuple) == before
    assert canonical_json_bytes(branched.asdict_tuple[:-1]) == before
    assert _terminal_orders_sha256(archive) == _terminal_orders_sha256(tuple(orders))


@pytest.mark.parametrize(
    "change", ["order_id", "target_identity", "execution_policy_digest"]
)
def test_terminal_archive_rejects_noncanonical_intent_at_trust_boundary(
    change: str,
) -> None:
    original = _make_terminal_order(0)
    invalid_value = "f" * 64 if change != "target_identity" else "forged-identity"
    forged = replace(
        original,
        intent=replace(original.intent, **{change: invalid_value}),
    )
    with pytest.raises(OrderDomainError, match="order intent identity digest mismatch"):
        TerminalOrderArchive((forged,))
    with pytest.raises(OrderDomainError, match="order intent identity digest mismatch"):
        OrderBookState(active_orders=(), terminal_orders=(forged,))


def test_context_streams_immutable_terminal_bytes_without_materializing_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.evaluation.test_after_cost_allocation import market
    from trade_rl.evaluation.allocation import _context
    from trade_rl.risk.pretrade import PreTradeRisk, PreTradeRiskConfig
    from trade_rl.simulation.accounting import BookState
    from trade_rl.simulation.execution import ExecutionCostConfig, MarketExecutor

    dataset = market()
    executor = MarketExecutor(dataset, ExecutionCostConfig.zero())
    book = BookState(
        [0.0] * dataset.n_symbols,
        1000.0,
        dataset.resolved_array("mark_price")[0],
        1000.0,
    )
    risk = PreTradeRisk(PreTradeRiskConfig(drawdown_start=0.1, drawdown_stop=0.2))
    archived = (
        _make_terminal_order(0),
        replace(_make_terminal_order(1), terminal_reason="正常終了"),
        _make_terminal_order(2),
    )
    order_book = OrderBookState(active_orders=(), terminal_orders=archived)

    class LegacyOrderBook:
        active_orders = ()
        terminal_orders = archived

    expected = _context(
        executor,
        book,
        LegacyOrderBook(),  # type: ignore[arg-type]
        account_id="test",
        pretrade_risk=risk,
        symbol_index=0,
        start_index=4,
    )

    def no_materialization(_archive: TerminalOrderArchive) -> object:
        pytest.fail("hotpath materialized terminal history as dicts")

    with monkeypatch.context() as patch:
        patch.setattr(
            TerminalOrderArchive, "asdict_tuple", property(no_materialization)
        )
        actual = _context(
            executor,
            book,
            order_book,
            account_id="test",
            pretrade_risk=risk,
            symbol_index=0,
            start_index=4,
        )

    assert actual.state_digest == expected.state_digest


def test_archived_digest_falls_back_if_json_encoder_is_patched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from trade_rl.artifacts import canonical
    from trade_rl.artifacts.hashing import content_digest
    from trade_rl.evaluation.allocation import _archived_context_digest

    archive = TerminalOrderArchive((_make_terminal_order(0),))
    payload = {
        "schema": "independent_allocation_context_v1",
        "account_id": "test",
        "book": {"cash": 1000.0},
    }
    original_bytes = content_digest(
        {**payload, "terminal_orders": archive.asdict_tuple}
    )
    assert _archived_context_digest(payload, archive) == original_bytes

    original = canonical.json.JSONEncoder.iterencode

    def changed_encoder(self, value, _one_shot=False):
        if isinstance(value, dict) and value.get("terminal_orders"):
            value = {
                **value,
                "terminal_orders": [
                    {**row, "terminal_reason": "changed-codec"}
                    for row in value["terminal_orders"]
                ],
            }
        yield from original(self, value, _one_shot)

    with monkeypatch.context() as patch:
        patch.setattr(canonical.json.JSONEncoder, "iterencode", changed_encoder)
        expected = content_digest({**payload, "terminal_orders": archive.asdict_tuple})
        assert expected != original_bytes
        assert _archived_context_digest(payload, archive) == expected
