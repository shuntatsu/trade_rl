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

        assert canonical_json_bytes(payload) == receipt_bytes
        assert payload["status"] == "completed"
        assert payload["settlement_complete"]
        assert payload["final_book"]["exact_quantities"] == ["0"]
        assert payload["final_active_orders"] == []
        assert payload["native_receipt"] == result.native_execution.receipt.payload
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
                receipt_bytes,
                tuple(progress),
            )
        )

    assert observations[0] == observations[1]
