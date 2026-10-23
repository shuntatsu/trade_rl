"""Same-HEAD terminal-byte oracle for forecast-trace vector validation."""

import sys
from collections.abc import Iterator
from contextlib import contextmanager
from importlib import import_module
from types import CodeType

import numpy as np
import pytest

from tests.evaluation.test_allocation_terminal_execution import carrier, plans
from trade_rl.artifacts import canonical_json_bytes
from trade_rl.strategies.forecasts import training_trace
from trade_rl.strategies.forecasts.simple_stream import FrozenSimpleReturnStream


def _legacy_scalar_vector(value: np.ndarray, *, dtype: str, count: int) -> np.ndarray:
    """F02 scalar reference; independent integer-clock oracles live separately."""
    original = np.asarray(value)
    if original.shape != (count,):
        raise ValueError("trace arrays must match row_symbols")
    if dtype == "datetime64[ns]":
        if original.dtype.kind != "M":
            raise ValueError("trace clocks must contain timestamps")
        for timestamp in original:
            training_trace._timestamp(timestamp, field="trace clock")
    normalized = np.asarray(original, dtype=dtype)
    return np.frombuffer(normalized.tobytes(), dtype=normalized.dtype)


@contextmanager
def _profile_route(
    *, vector_code: CodeType, public_get_code: CodeType
) -> Iterator[dict[str, int]]:
    """Count scalar timestamp calls only while the selected vector runs."""
    timestamp_code = training_trace._timestamp.__code__
    digest_code = FrozenSimpleReturnStream.digest.fget.__code__
    counts = {
        "vector_calls": 0,
        "scalar_timestamp_calls": 0,
        "public_get_calls": 0,
        "public_digest_calls": 0,
    }
    active_vectors = 0
    active_getters = 0
    previous = sys.getprofile()

    def profile(frame, event, _arg):
        nonlocal active_vectors, active_getters
        if event == "call":
            if frame.f_code is vector_code:
                counts["vector_calls"] += 1
                active_vectors += 1
            elif frame.f_code is timestamp_code and active_vectors:
                counts["scalar_timestamp_calls"] += 1
            if frame.f_code is public_get_code:
                counts["public_get_calls"] += 1
                active_getters += 1
            elif frame.f_code is digest_code and active_getters:
                counts["public_digest_calls"] += 1
        elif event == "return":
            if frame.f_code is vector_code:
                active_vectors -= 1
            elif frame.f_code is public_get_code:
                active_getters -= 1

    sys.setprofile(profile)
    try:
        yield counts
    finally:
        sys.setprofile(previous)


def _run_route(route, monkeypatch, operation):
    memo_type = import_module(
        "trade_rl.evaluation.rl_allocation._simple_stream_digest"
    ).SimpleStreamDigestMemo

    if route == "reference":
        vector = _legacy_scalar_vector
    elif route == "optimized":
        vector = training_trace._vector
    else:
        raise AssertionError(f"unknown trace route: {route}")

    with monkeypatch.context() as scoped:
        if route == "reference":
            scoped.setattr(training_trace, "_vector", vector)
        scoped.setattr(memo_type, "get", lambda _self, stream: stream.digest)
        public_get_code = memo_type.get.__code__
        with _profile_route(
            vector_code=vector.__code__, public_get_code=public_get_code
        ) as counts:
            observation = operation()

    assert counts["public_get_calls"] > 0
    assert counts["public_digest_calls"] == counts["public_get_calls"]
    return observation, counts


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
def test_same_head_terminal_bytes_match_scalar_and_vector_trace_validation(
    case, carrier_kwargs, kind, stop, monkeypatch
):
    execution_api = import_module("trade_rl.evaluation.allocation_terminal_execution")
    evidence_api = import_module("trade_rl.evaluation.allocation_terminal_evidence")

    def execute_route():
        folds, env = carrier(**carrier_kwargs)
        plan = plans(execution_api, env, folds, kind=kind, stop=stop)
        plan_bytes = canonical_json_bytes(plan.payload)
        progress = []

        def observe(phase, start_index, processing_index):
            progress.append((phase, start_index, processing_index))

        result = execution_api.run_terminal_allocation_execution(
            folds, env, plan, on_advance=observe
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
        readback_bytes = canonical_json_bytes(readback.payload)
        assert readback_bytes == receipt_bytes
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

        return plan_bytes, receipt_bytes, readback_bytes, tuple(progress)

    reference, reference_counts = _run_route("reference", monkeypatch, execute_route)
    optimized, optimized_counts = _run_route("optimized", monkeypatch, execute_route)

    assert reference == optimized
    assert reference_counts["vector_calls"] > 0
    assert reference_counts["scalar_timestamp_calls"] > 0
    assert optimized_counts["vector_calls"] > 0
    assert (
        optimized_counts["scalar_timestamp_calls"]
        < reference_counts["scalar_timestamp_calls"]
    )
