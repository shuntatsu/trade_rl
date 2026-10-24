"""Same-HEAD full terminal-byte oracle for the stream digest memo."""

import sys
from collections.abc import Iterator
from contextlib import contextmanager
from importlib import import_module
from types import CodeType

import pytest

from tests.evaluation.test_allocation_terminal_execution import carrier, plans
from trade_rl.artifacts import canonical_json_bytes
from trade_rl.strategies.forecasts.simple_stream import FrozenSimpleReturnStream


@contextmanager
def _digest_reads_inside(code: CodeType) -> Iterator[list[int]]:
    """Count public stream.digest calls made from one exact getter function."""
    digest_code = FrozenSimpleReturnStream.digest.fget.__code__
    reads: list[int] = []
    active: list[int] = []
    previous = sys.getprofile()

    def profile(frame, event, _arg):
        if event == "call" and frame.f_code is code:
            active.append(0)
        elif event == "call" and frame.f_code is digest_code and active:
            active[-1] += 1
        elif event == "return" and frame.f_code is code and active:
            reads.append(active.pop())

    sys.setprofile(profile)
    try:
        yield reads
    finally:
        sys.setprofile(previous)


@contextmanager
def _public_stream_digest_reads() -> Iterator[list[int]]:
    """Count public serializer calls during one bounded operation."""
    digest_code = FrozenSimpleReturnStream.digest.fget.__code__
    reads = [0]
    previous = sys.getprofile()

    def profile(frame, event, _arg):
        if event == "call" and frame.f_code is digest_code:
            reads[0] += 1

    sys.setprofile(profile)
    try:
        yield reads
    finally:
        sys.setprofile(previous)


def _run_route(route, monkeypatch, operation):
    memo_module = import_module(
        "trade_rl.evaluation.rl_allocation._simple_stream_digest"
    )
    memo_type = memo_module.SimpleStreamDigestMemo
    original_get = memo_type.get
    getter_calls = []

    if route == "reference":

        def public_get(_self, stream):
            getter_calls.append(None)
            return stream.digest

        replacement = public_get
        profiled_code = public_get.__code__
    elif route == "optimized":

        def original_get_with_count(self, stream):
            getter_calls.append(None)
            return original_get(self, stream)

        replacement = original_get_with_count
        profiled_code = original_get.__code__
    else:
        raise AssertionError(f"unknown digest route: {route}")

    with monkeypatch.context() as scoped:
        scoped.setattr(memo_type, "get", replacement)
        with _digest_reads_inside(profiled_code) as public_reads:
            result = operation()

    assert getter_calls
    assert len(public_reads) == len(getter_calls)
    if route == "reference":
        assert public_reads == [1] * len(getter_calls)
    else:
        assert any(count > 0 for count in public_reads)  # cold/fallback path
        assert any(count == 0 for count in public_reads)  # actual memo hit
    return result


def test_terminal_preflight_reuses_only_its_live_stream_memo(monkeypatch):
    execution_api = import_module("trade_rl.evaluation.allocation_terminal_execution")
    memo_type = import_module(
        "trade_rl.evaluation.rl_allocation._simple_stream_digest"
    ).SimpleStreamDigestMemo

    class StopBeforeNativeRun(Exception):
        pass

    counts = []
    for route in ("reference", "optimized"):
        folds, env = carrier()
        plan = plans(execution_api, env, folds)

        def stop_before_native(*_args, **_kwargs):
            raise StopBeforeNativeRun

        with monkeypatch.context() as scoped:
            if route == "reference":
                scoped.setattr(memo_type, "get", lambda _self, stream: stream.digest)
            scoped.setattr(
                execution_api,
                "run_declared_global_allocation_execution",
                stop_before_native,
            )
            with _public_stream_digest_reads() as reads:
                with pytest.raises(StopBeforeNativeRun):
                    execution_api.run_terminal_allocation_execution(folds, env, plan)
        counts.append(reads[0])

    assert counts[1] < counts[0]


def test_terminal_run_owns_a_fresh_memo_for_each_call(monkeypatch):
    execution_api = import_module("trade_rl.evaluation.allocation_terminal_execution")
    memo_type = import_module(
        "trade_rl.evaluation.rl_allocation._simple_stream_digest"
    ).SimpleStreamDigestMemo

    class StopBeforeNativeRun(Exception):
        pass

    folds, env = carrier()
    plan = plans(execution_api, env, folds)
    memos = []

    def stop_before_native(*_args, **kwargs):
        memo = kwargs["_stream_digest_memo"]
        assert type(memo) is memo_type
        memos.append(memo)
        raise StopBeforeNativeRun

    monkeypatch.setattr(
        execution_api,
        "run_declared_global_allocation_execution",
        stop_before_native,
    )
    for _ in range(2):
        with pytest.raises(StopBeforeNativeRun):
            execution_api.run_terminal_allocation_execution(folds, env, plan)

    assert len(memos) == 2
    assert memos[0] is not memos[1]


@pytest.mark.parametrize("mutation", ["packet", "serializer"])
def test_terminal_live_memo_rechecks_changes_before_native_execution(
    mutation, monkeypatch
):
    execution_api = import_module("trade_rl.evaluation.allocation_terminal_execution")
    memo_module = import_module(
        "trade_rl.evaluation.rl_allocation._simple_stream_digest"
    )
    canonical = import_module("trade_rl.artifacts.canonical")
    folds, env = carrier()
    plan = plans(execution_api, env, folds)
    original_run = execution_api.run_declared_global_allocation_execution
    observed_fallback = []

    def mutate_then_run(*args, **kwargs):
        memo = kwargs["_stream_digest_memo"]
        assert type(memo) is memo_module.SimpleStreamDigestMemo
        before = env.stream.digest
        if mutation == "packet":
            packet = env.stream.packets[-1]
            object.__setattr__(
                packet, "expected_simple_return", packet.expected_simple_return + 0.01
            )
        else:
            original_encode = canonical.json.JSONEncoder.encode

            def altered_encode(encoder, value):
                return original_encode(encoder, value) + " "

            monkeypatch.setattr(canonical.json.JSONEncoder, "encode", altered_encode)

        current = memo.get(env.stream)
        assert current == env.stream.digest
        assert current != before
        observed_fallback.append(True)
        return original_run(*args, **kwargs)

    monkeypatch.setattr(
        execution_api,
        "run_declared_global_allocation_execution",
        mutate_then_run,
    )
    monkeypatch.setattr(env, "reset", lambda **_: pytest.fail("reset before refusal"))

    with pytest.raises(ValueError):
        execution_api.run_terminal_allocation_execution(folds, env, plan)

    assert observed_fallback == [True]
    assert not hasattr(env, "book")


def test_standalone_terminal_declaration_keeps_the_direct_public_path(monkeypatch):
    execution_api = import_module("trade_rl.evaluation.allocation_terminal_execution")
    memo_type = import_module(
        "trade_rl.evaluation.rl_allocation._simple_stream_digest"
    ).SimpleStreamDigestMemo
    folds, env = carrier()
    plan = plans(execution_api, env, folds)

    def reject_private_memo(*_args, **_kwargs):
        pytest.fail("standalone declaration unexpectedly used the private memo")

    monkeypatch.setattr(memo_type, "get", reject_private_memo)
    declared = execution_api.declare_terminal_allocation_execution(
        folds,
        env,
        plan.native_plan,
        settlement_stop_index=plan.settlement_stop_index,
    )

    assert declared == plan


@pytest.mark.parametrize(
    ("case", "carrier_kwargs", "kind", "stop"),
    [
        pytest.param("long", {"side": 1}, "nonrl", 8, id="nonrl-long"),
        pytest.param("short", {"side": -1}, "nonrl", 8, id="nonrl-short"),
        pytest.param(
            "partial_close",
            {"side": 1, "partial": True},
            "nonrl",
            9,
            id="nonrl-partial-close-then-flat",
        ),
        pytest.param(
            "cash_carry",
            {"kind": "cash", "cash_rate": 0.876},
            "cash",
            9,
            id="cash-nonzero-carry",
        ),
    ],
)
def test_same_head_full_terminal_bytes_match_reference_and_memo(
    case, carrier_kwargs, kind, stop, monkeypatch
):
    execution_api = import_module("trade_rl.evaluation.allocation_terminal_execution")
    evidence_api = import_module("trade_rl.evaluation.allocation_terminal_evidence")
    observations = []

    for route in ("reference", "optimized"):
        folds, env = carrier(**carrier_kwargs)
        plan = plans(execution_api, env, folds, kind=kind, stop=stop)
        plan_bytes = canonical_json_bytes(plan.payload)
        progress = []

        def observe(phase, start_index, processing_index):
            progress.append((phase, start_index, processing_index))

        result = _run_route(
            route,
            monkeypatch,
            lambda: execution_api.run_terminal_allocation_execution(
                folds, env, plan, on_advance=observe
            ),
        )
        payload = result.payload
        receipt_bytes = result._bytes
        native_receipt = result.native_execution.receipt
        native_payload = native_receipt.payload
        actor_event_bytes = canonical_json_bytes(
            [
                row["native"]["order_events"]
                for row in native_payload["rows"]
                if row["native"] is not None
            ]
        )
        closing_event_bytes = canonical_json_bytes(
            [row["native"]["order_events"] for row in payload["closing_rows"]]
        )

        assert canonical_json_bytes(payload) == receipt_bytes
        assert payload["status"] == "completed"
        assert payload["settlement_complete"]
        assert payload["final_book"]["exact_quantities"] == ["0"]
        assert payload["final_active_orders"] == []
        assert payload["native_receipt"] == native_payload
        assert payload["native_receipt"]["plan"] == plan.native_plan.payload

        readback = evidence_api.TerminalAllocationExecutionReceipt.from_payload(
            payload, expected_plan=plan
        )
        assert canonical_json_bytes(readback.payload) == receipt_bytes

        assert {phase for phase, _start, _processing in progress} == {
            "actor",
            "reserve",
        }
        if case == "partial_close":
            assert len(payload["closing_rows"]) == 2
            assert payload["closing_rows"][0]["native"]["active_orders"]
            assert any(
                event["event_type"] == "filled"
                for event in payload["closing_rows"][1]["native"]["order_events"]
            )
        elif case == "cash_carry":
            assert any(
                row["native"]["execution"]["interval_cash_interest"] != 0
                for row in payload["closing_rows"]
            )

        observations.append(
            (
                plan_bytes,
                canonical_json_bytes(plan.native_plan.payload),
                native_receipt._bytes,
                native_receipt.digest,
                receipt_bytes,
                actor_event_bytes,
                closing_event_bytes,
                tuple(progress),
            )
        )

    assert observations[0] == observations[1]
