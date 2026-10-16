"""Same-HEAD terminal receipt oracle for allocation episode clock binding."""

from __future__ import annotations

import sys
from collections.abc import Iterator
from contextlib import contextmanager
from importlib import import_module
from types import CodeType

import numpy as np
import pytest

from tests.evaluation.test_allocation_terminal_execution import carrier, plans
from trade_rl.artifacts import canonical_json_bytes
from trade_rl.evaluation.rl_allocation import binding as runtime_binding
from trade_rl.evaluation.runs import build_candidate_run_provenance


def _reference_episode_times(values: np.ndarray) -> tuple[np.datetime64, ...]:
    """Retain the pre-vectorization scalar clock contract as an oracle."""
    return tuple(
        runtime_binding._timestamp(value, field="episode_time") for value in values
    )


@contextmanager
def _profile_binding() -> Iterator[dict[str, int]]:
    """Count scalar timestamp dispatches made inside the real runtime binder."""
    binding_code: CodeType = runtime_binding.validate_runtime_binding.__code__
    timestamp_code: CodeType = runtime_binding._timestamp.__code__
    counts = {"binding_calls": 0, "scalar_timestamp_calls": 0}
    active_bindings = 0
    previous = sys.getprofile()

    def profile(frame, event, _arg):
        nonlocal active_bindings
        if event == "call":
            if frame.f_code is binding_code:
                counts["binding_calls"] += 1
                active_bindings += 1
            elif frame.f_code is timestamp_code and active_bindings:
                counts["scalar_timestamp_calls"] += 1
        elif event == "return" and frame.f_code is binding_code:
            active_bindings -= 1

    sys.setprofile(profile)
    try:
        yield counts
    finally:
        sys.setprofile(previous)


def _capture_terminal_route(route, case, carrier_kwargs, kind, stop, monkeypatch):
    execution_api = import_module("trade_rl.evaluation.allocation_terminal_execution")
    evidence_api = import_module("trade_rl.evaluation.allocation_terminal_evidence")

    with monkeypatch.context() as scoped:
        if route == "reference":
            # On the current scalar implementation the binder already performs
            # this exact operation. After extraction, this patches only the
            # helper call for the legacy route; optimized keeps the real helper.
            scoped.setattr(
                runtime_binding,
                "_episode_times",
                _reference_episode_times,
                raising=False,
            )
        elif route != "optimized":
            raise AssertionError(f"unknown binding route: {route}")

        with _profile_binding() as counts:
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

            # Compare the complete canonical schemas: no financial/provenance
            # fields are projected out of either the plan or runtime receipt.
            assert canonical_json_bytes(plan.payload) == plan_bytes
            assert canonical_json_bytes(payload) == receipt_bytes
            assert payload["status"] == "completed"
            assert payload["settlement_complete"]
            assert payload["final_book"]["exact_quantities"] == ["0"]
            assert payload["final_active_orders"] == []
            assert payload["native_receipt"] == result.native_execution.receipt.payload
            assert payload["native_receipt"]["plan"] == plan.native_plan.payload

            current_provenance = build_candidate_run_provenance()
            common = plan.native_plan.payload["common"]
            assert (
                common["implementation_digest"]
                == current_provenance["implementation_digest"]
            )
            assert (
                common["runtime_environment_digest"]
                == current_provenance["runtime_environment_digest"]
            )

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

            full_bytes = (
                plan_bytes,
                receipt_bytes,
                readback_bytes,
                canonical_json_bytes(payload["native_receipt"]),
                tuple(progress),
            )

    return full_bytes, counts


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
def test_terminal_binding_bulk_clock_path_preserves_complete_receipt_bytes(
    case, carrier_kwargs, kind, stop, monkeypatch
):
    reference, reference_counts = _capture_terminal_route(
        "reference", case, carrier_kwargs, kind, stop, monkeypatch
    )
    optimized, optimized_counts = _capture_terminal_route(
        "optimized", case, carrier_kwargs, kind, stop, monkeypatch
    )

    assert reference == optimized
    assert reference_counts["binding_calls"] > 0
    assert reference_counts["scalar_timestamp_calls"] > 0
    # The reference helper must be selected by the actual binder. If it is not,
    # this route would look bulk-validated too and the byte oracle would be vacuous.
    assert reference_counts["scalar_timestamp_calls"] > (
        2 * reference_counts["binding_calls"]
    )
    assert optimized_counts["binding_calls"] > 0
    assert optimized_counts["binding_calls"] == reference_counts["binding_calls"]
    # The two scalar calls are the separately declared UTC endpoints. Interior
    # Dataset clocks must be bulk-validated, independent of episode row count.
    assert optimized_counts["scalar_timestamp_calls"] <= (
        2 * optimized_counts["binding_calls"]
    )
