"""Regression tests for terminal order archive fastpaths and digest parity."""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import asdict, fields, replace
from types import SimpleNamespace

import pytest

from trade_rl.artifacts import canonical_json_bytes
from trade_rl.evaluation.rl_allocation._native_progress import (
    _terminal_orders_sha256,
)
from trade_rl.evaluation.rl_allocation.transition_facts import _pending
from trade_rl.simulation.orders.model import (
    OrderBookState,
    OrderDomainError,
    OrderEvent,
    OrderIntent,
    OrderStatus,
    OrderType,
    PendingOrder,
    TerminalOrderArchive,
    TimeInForce,
)
from trade_rl.simulation.stateful.execution import StatefulExecutionResult


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
    quantity: float = 1.0,
) -> PendingOrder:
    sub = index if submit_index is None else submit_index
    intent = _make_intent(submit_index=sub, eligible_index=sub + 1, quantity=quantity)
    p = PendingOrder.from_intent(intent)
    p = p.mark_eligible(processing_index=sub + 1)
    if trigger_index is not None:
        p = p.mark_triggered(processing_index=trigger_index)
    proc = sub + 1 if last_processed_index is None else last_processed_index
    return p.cancel(processing_index=proc, reason="test_done")


def _terminal_event(order: PendingOrder, sequence: int) -> OrderEvent:
    return OrderEvent(
        schema_version="order_event_v1",
        sequence=sequence,
        order_id=order.order_id,
        replaced_order_id=None,
        dataset_id=order.intent.dataset_id,
        execution_policy_digest=order.intent.execution_policy_digest,
        symbol_index=order.intent.symbol_index,
        event_type="cancelled",
        processing_index=order.last_processed_index or 0,
        timestamp_ns=sequence,
        previous_status=OrderStatus.ELIGIBLE,
        new_status=OrderStatus.CANCELLED,
        requested_quantity=order.intent.requested_quantity,
        remaining_quantity=order.remaining_quantity,
        filled_quantity=order.cumulative_filled_quantity,
        execution_price=None,
        filled_notional=0.0,
        capacity_before=0.0,
        capacity_after=0.0,
        participation_rate=0.0,
        trigger_segment=None,
        available_volume_fraction=1.0,
        reason=order.terminal_reason,
        path_mode="conservative",
        path_points=(),
    )


def _captured_execution_result() -> StatefulExecutionResult:
    from tests.evaluation.test_allocation_transition_trace import literal_args
    from trade_rl.evaluation.rl_allocation.env import AllocationTradingEnv

    env = AllocationTradingEnv(**literal_args())
    env.reset()
    captured = []
    env._transition_recorder = SimpleNamespace(
        freeze_execution=lambda result: captured.append(result.execution) or b""
    )
    env.step(3)
    assert len(captured) == 1
    result = captured[0]
    assert isinstance(result, StatefulExecutionResult)
    return result


def _facts_execution(
    archive: TerminalOrderArchive, events: tuple[OrderEvent, ...]
) -> StatefulExecutionResult:
    base = _captured_execution_result()
    return replace(
        base,
        order_book=OrderBookState(active_orders=(), terminal_orders=archive),
        order_events=tuple(events),
    )


def _legacy_terminal_order_reasons(
    execution: StatefulExecutionResult,
) -> list[dict[str, str]]:
    terminal_ids = {
        event.order_id for event in execution.order_events if event.new_status.terminal
    }
    return [
        {"order_id": order.order_id, "reason": order.terminal_reason}
        for order in execution.order_book.terminal_orders
        if order.order_id in terminal_ids
    ]


def _assert_native_facts_match_legacy(execution: StatefulExecutionResult) -> None:
    from trade_rl.evaluation.rl_allocation.transition_facts import (
        native_allocation_execution_facts,
    )

    observed = native_allocation_execution_facts(execution)
    patcher = pytest.MonkeyPatch()
    try:
        patcher.setattr(
            TerminalOrderArchive,
            "_select_terminal_orders",
            lambda _archive, _order_ids: None,
        )
        expected = native_allocation_execution_facts(execution)
    finally:
        patcher.undo()
    assert canonical_json_bytes(observed) == canonical_json_bytes(expected)


@pytest.mark.parametrize(
    "selected_indexes",
    [(), (2,), (0, 2), (0, 1, 2, 3), (None,), (2, 0, 2)],
)
def test_native_execution_facts_terminal_reasons_match_legacy_archive_order(
    selected_indexes: tuple[int | None, ...],
) -> None:
    reasons = ("first", "二番目", "third", "fourth")
    orders = tuple(
        replace(
            _make_terminal_order(index, quantity=(index + 1) / 4),
            terminal_reason=reasons[index],
        )
        for index in range(len(reasons))
    )
    archive = TerminalOrderArchive(orders)
    events: list[OrderEvent] = []
    for sequence, index in enumerate(selected_indexes):
        order_index = 0 if index is None else index
        event = _terminal_event(orders[order_index], sequence)
        if index is None:
            event = replace(event, order_id="f" * 64)
        events.append(event)
    execution = _facts_execution(archive, tuple(events))

    _assert_native_facts_match_legacy(execution)


def test_native_execution_facts_branching_and_detached_order_views_match_legacy():
    root_orders = tuple(
        replace(_make_terminal_order(i, quantity=i + 1), terminal_reason=f"root-{i}")
        for i in range(2)
    )
    root = TerminalOrderArchive(root_orders)
    branch_a = root.append(
        replace(_make_terminal_order(2, quantity=3), terminal_reason="branch-a")
    )
    branch_b = root.append(
        replace(_make_terminal_order(3, quantity=4), terminal_reason="branch-b")
    )

    # Mutating caller-owned and public views must not alter the archived projection.
    object.__setattr__(root_orders[0], "terminal_reason", "caller-change")
    exposed = branch_a[1]
    object.__setattr__(exposed, "terminal_reason", "view-change")
    selected = (branch_a, branch_b)
    for archive in selected:
        execution = _facts_execution(
            archive,
            tuple(_terminal_event(order, i) for i, order in enumerate(archive)),
        )
        _assert_native_facts_match_legacy(execution)


def test_native_execution_facts_reconstructs_only_selected_terminal_orders():
    from trade_rl.evaluation.rl_allocation.transition_facts import (
        native_allocation_execution_facts,
    )

    orders = tuple(_make_terminal_order(i) for i in range(12))
    archive = TerminalOrderArchive(orders)
    execution = _facts_execution(
        archive, (_terminal_event(orders[2], 0), _terminal_event(orders[9], 1))
    )
    decoder_code = PendingOrder.from_mapping.__func__.__code__
    calls = 0
    prior_profile = sys.getprofile()

    def count_decoder_calls(frame, event, arg):
        nonlocal calls
        if event == "call" and frame.f_code is decoder_code:
            calls += 1
        if prior_profile is not None:
            prior_profile(frame, event, arg)

    sys.setprofile(count_decoder_calls)
    try:
        native_allocation_execution_facts(execution)
    finally:
        sys.setprofile(prior_profile)

    assert calls == 2


def test_native_execution_facts_falls_back_when_pending_decoder_is_patched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from trade_rl.evaluation.rl_allocation.transition_facts import (
        native_allocation_execution_facts,
    )

    orders = tuple(_make_terminal_order(i) for i in range(6))
    archive = TerminalOrderArchive(orders)
    execution = _facts_execution(
        archive, (_terminal_event(orders[1], 0), _terminal_event(orders[4], 1))
    )
    expected = native_allocation_execution_facts(execution)
    original_decoder = PendingOrder.from_mapping.__func__
    decoder_calls = 0

    def wrapped_decoder(cls, value):
        nonlocal decoder_calls
        decoder_calls += 1
        return original_decoder(cls, value)

    with monkeypatch.context() as patch:
        patch.setattr(PendingOrder, "from_mapping", classmethod(wrapped_decoder))
        observed = native_allocation_execution_facts(execution)

    assert canonical_json_bytes(observed) == canonical_json_bytes(expected)
    assert decoder_calls == len(orders)


@pytest.mark.parametrize(
    "decoder_binding",
    (
        "default_decoder",
        "instance_decode",
        "class_decode",
        "class_raw_decode",
        "scan_once",
    ),
)
def test_native_execution_facts_falls_back_when_json_decoder_binding_changes(
    monkeypatch: pytest.MonkeyPatch, decoder_binding: str
) -> None:
    from trade_rl.evaluation.rl_allocation.transition_facts import (
        native_allocation_execution_facts,
    )
    from trade_rl.simulation.orders.model import (
        _terminal_reason_selection_bindings_unchanged,
    )

    orders = tuple(_make_terminal_order(i) for i in range(6))
    archive = TerminalOrderArchive(orders)
    selected_indexes = (1, 4)
    execution = _facts_execution(
        archive,
        tuple(
            _terminal_event(orders[index], sequence)
            for sequence, index in enumerate(selected_indexes)
        ),
    )
    selected_order_ids = {orders[index].order_id for index in selected_indexes}
    original_loads = json.loads
    original_loads_code = json.loads.__code__
    default_decoder = json._default_decoder
    assert _terminal_reason_selection_bindings_unchanged(archive)

    def reject_unselected_order(value):
        intent = value.get("intent") if isinstance(value, dict) else None
        order_id = intent.get("order_id") if isinstance(intent, dict) else None
        if order_id is not None and order_id not in selected_order_ids:
            raise ValueError("patched JSON decoder rejected an unselected order")
        return value

    with monkeypatch.context() as patch:
        if decoder_binding == "default_decoder":
            original_decode = default_decoder.decode
            replacement = SimpleNamespace(
                decode=lambda source: reject_unselected_order(original_decode(source))
            )
            patch.setattr(json, "_default_decoder", replacement)
        elif decoder_binding == "instance_decode":
            original_decode = default_decoder.decode
            patch.setitem(
                default_decoder.__dict__,
                "decode",
                lambda source: reject_unselected_order(original_decode(source)),
            )
        elif decoder_binding == "class_decode":
            original_decode = json.JSONDecoder.decode

            def patched_decode(self, source, *args, **kwargs):
                value = original_decode(self, source, *args, **kwargs)
                return reject_unselected_order(value)

            patch.setattr(json.JSONDecoder, "decode", patched_decode)
        elif decoder_binding == "class_raw_decode":
            original_raw_decode = json.JSONDecoder.raw_decode

            def patched_raw_decode(self, source, *args, **kwargs):
                value, end = original_raw_decode(self, source, *args, **kwargs)
                return reject_unselected_order(value), end

            patch.setattr(json.JSONDecoder, "raw_decode", patched_raw_decode)
        else:
            original_scan_once = default_decoder.scan_once

            def patched_scan_once(source, index):
                value, end = original_scan_once(source, index)
                return reject_unselected_order(value), end

            patch.setitem(default_decoder.__dict__, "scan_once", patched_scan_once)

        assert json.loads is original_loads
        assert json.loads.__code__ is original_loads_code
        with pytest.raises(ValueError) as expected:
            _legacy_terminal_order_reasons(execution)
        with pytest.raises(type(expected.value)) as observed:
            native_allocation_execution_facts(execution)
        assert not _terminal_reason_selection_bindings_unchanged(archive)

    assert str(observed.value) == str(expected.value)
    assert _terminal_reason_selection_bindings_unchanged(archive)


def test_native_execution_facts_falls_back_for_archive_subclass_and_patched_iterator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from trade_rl.evaluation.rl_allocation.transition_facts import (
        native_allocation_execution_facts,
    )

    orders = tuple(
        replace(_make_terminal_order(i), terminal_reason=f"reason-{i}")
        for i in range(3)
    )

    class ReverseArchive(TerminalOrderArchive):
        def __iter__(self):
            yield from reversed(tuple(super().__iter__()))

    subclass_archive = ReverseArchive(orders)
    subclass_execution = _facts_execution(
        subclass_archive,
        tuple(_terminal_event(order, i) for i, order in enumerate(orders)),
    )
    subclass_facts = native_allocation_execution_facts(subclass_execution)
    assert subclass_facts["terminal_order_reasons"] == _legacy_terminal_order_reasons(
        subclass_execution
    )
    assert [row["order_id"] for row in subclass_facts["terminal_order_reasons"]] == [
        order.order_id for order in reversed(orders)
    ]

    archive = TerminalOrderArchive(orders)
    execution = _facts_execution(
        archive, tuple(_terminal_event(order, i) for i, order in enumerate(orders))
    )
    original_iter = TerminalOrderArchive.__iter__

    def reverse_native_iterator(self):
        yield from reversed(tuple(original_iter(self)))

    with monkeypatch.context() as patch:
        patch.setattr(TerminalOrderArchive, "__iter__", reverse_native_iterator)
        facts = native_allocation_execution_facts(execution)
        assert facts["terminal_order_reasons"] == _legacy_terminal_order_reasons(
            execution
        )


@pytest.mark.parametrize("replacement_kind", ["valid", "malformed"])
def test_native_execution_facts_falls_back_after_late_ancestor_byte_mutation(
    replacement_kind: str,
) -> None:
    from trade_rl.evaluation.rl_allocation.transition_facts import (
        native_allocation_execution_facts,
    )

    first = replace(_make_terminal_order(0), terminal_reason="ancestor")
    second = replace(_make_terminal_order(1), terminal_reason="descendant")
    ancestor = TerminalOrderArchive((first,))
    archive = ancestor.append(second)
    execution = _facts_execution(
        archive, (_terminal_event(first, 0), _terminal_event(second, 1))
    )
    node = ancestor._tail
    assert node is not None
    if replacement_kind == "valid":
        payload = json.loads(node._order_bytes)
        payload["terminal_reason"] = "mutated-valid-bytes"
        replacement_bytes = canonical_json_bytes(payload)
    else:
        replacement_bytes = b"{"
    object.__setattr__(node, "_order_bytes", replacement_bytes)

    if replacement_bytes == b"{":
        with pytest.raises(json.JSONDecodeError) as expected:
            _legacy_terminal_order_reasons(execution)
        with pytest.raises(type(expected.value)) as actual:
            native_allocation_execution_facts(execution)
        assert str(actual.value) == str(expected.value)
    else:
        _assert_native_facts_match_legacy(execution)


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


def test_archived_digest_falls_back_if_json_mapping_converter_is_patched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from trade_rl.artifacts import canonical
    from trade_rl.artifacts.hashing import content_digest
    from trade_rl.evaluation.allocation import _archived_context_digest

    archive = TerminalOrderArchive((_make_terminal_order(0),))
    payload = {
        "schema": "independent_allocation_context_v1",
        "book": {"cash": 1000.0},
    }
    standard = content_digest({**payload, "terminal_orders": archive.asdict_tuple})
    assert _archived_context_digest(payload, archive) == standard
    original = canonical._mapping_value

    def changed_mapping(value):
        converted = original(value)
        if "terminal_reason" in converted:
            converted["terminal_reason"] = "changed-mapping"
        return converted

    with monkeypatch.context() as patch:
        patch.setattr(canonical, "_mapping_value", changed_mapping)
        expected = content_digest({**payload, "terminal_orders": archive.asdict_tuple})
        assert expected != standard
        assert _archived_context_digest(payload, archive) == expected


@pytest.mark.parametrize("change", ["replacement", "in_place_code"])
def test_archived_digest_falls_back_if_json_encoder_initializer_changes(
    monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    from types import FunctionType

    from trade_rl.artifacts import canonical
    from trade_rl.artifacts.hashing import content_digest
    from trade_rl.evaluation.allocation import _archived_context_digest

    archive = TerminalOrderArchive((_make_terminal_order(0),))
    payload = {"schema": "independent_allocation_context_v1", "book": {"cash": 10.0}}
    standard = content_digest({**payload, "terminal_orders": archive.asdict_tuple})
    assert _archived_context_digest(payload, archive) == standard
    initializer = canonical.json.JSONEncoder.__init__
    original = FunctionType(
        initializer.__code__,
        initializer.__globals__,
        argdefs=initializer.__defaults__,
        closure=initializer.__closure__,
    )
    original.__kwdefaults__ = initializer.__kwdefaults__

    def changed_initializer(self, *args, **kwargs):
        self._context_guard_original_initializer(*args, **kwargs)
        self.item_separator = ", "

    with monkeypatch.context() as patch:
        patch.setattr(
            canonical.json.JSONEncoder,
            "_context_guard_original_initializer",
            original,
            raising=False,
        )
        if change == "replacement":
            patch.setattr(canonical.json.JSONEncoder, "__init__", changed_initializer)
        else:
            patch.setattr(initializer, "__code__", changed_initializer.__code__)
        expected = content_digest({**payload, "terminal_orders": archive.asdict_tuple})
        assert expected != standard
        assert _archived_context_digest(payload, archive) == expected


def test_archived_digest_rejects_custom_archive_serializer() -> None:
    from trade_rl.artifacts.hashing import content_digest
    from trade_rl.evaluation.allocation import _archived_context_digest

    class CustomArchive(TerminalOrderArchive):
        def _canonical_order_json_chunks(self):
            yield b'{"forged":true}'

    archive = CustomArchive((_make_terminal_order(0),))
    payload = {"schema": "independent_allocation_context_v1", "book": {"cash": 10.0}}
    expected = content_digest({**payload, "terminal_orders": archive.asdict_tuple})
    assert _archived_context_digest(payload, archive) == expected


def test_archived_digest_rejects_modified_native_archive_serializer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from trade_rl.artifacts.hashing import content_digest
    from trade_rl.evaluation.allocation import _archived_context_digest

    archive = TerminalOrderArchive((_make_terminal_order(0),))
    payload = {"schema": "independent_allocation_context_v1", "book": {"cash": 10.0}}
    expected = content_digest({**payload, "terminal_orders": archive.asdict_tuple})

    def changed_chunks(_archive: TerminalOrderArchive):
        yield b'{"forged":true}'

    with monkeypatch.context() as patch:
        patch.setattr(
            TerminalOrderArchive, "_canonical_order_json_chunks", changed_chunks
        )
        assert _archived_context_digest(payload, archive) == expected


def test_large_archive_pickle_roundtrip_is_iterative_and_receipt_exact() -> None:
    import pickle

    orders = tuple(_make_terminal_order(i) for i in range(6599))
    archive = TerminalOrderArchive(orders)
    serialized = pickle.dumps(archive, protocol=pickle.HIGHEST_PROTOCOL)
    restored = pickle.loads(serialized)
    assert isinstance(restored, TerminalOrderArchive)
    assert len(restored) == len(orders)
    assert restored.asdict_tuple == archive.asdict_tuple
    assert _terminal_orders_sha256(restored) == _terminal_orders_sha256(archive)
    assert restored.max_clock_index == archive.max_clock_index

    # A later append may not affect any earlier native execution snapshot.
    next_order = _make_terminal_order(6599)
    appended = restored.append(next_order)
    assert len(restored) == len(orders)
    assert len(appended) == len(orders) + 1
    assert appended[-1].order_id == next_order.order_id
    assert restored[-1].order_id == orders[-1].order_id

    order_book = OrderBookState(active_orders=(), terminal_orders=archive)
    copied_book = pickle.loads(pickle.dumps(order_book))
    assert copied_book.terminal_orders.asdict_tuple == archive.asdict_tuple
