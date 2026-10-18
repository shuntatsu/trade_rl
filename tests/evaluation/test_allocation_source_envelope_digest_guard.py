"""Global allocation guards and same-HEAD terminal parity for source-envelope memo."""

from __future__ import annotations

import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from importlib import import_module
from types import CodeType

import pytest

from tests.evaluation.test_allocation_global_execution import capability, declare
from tests.evaluation.test_allocation_nonrl_walk_forward import (
    global_control_fixture,
    global_control_folds,
)
from tests.evaluation.test_allocation_terminal_execution import carrier, plans
from trade_rl.artifacts import canonical_json_bytes
from trade_rl.evaluation.forecast_allocation import HorizonCostEstimates
from trade_rl.evaluation.rl_allocation import global_execution_context as context


@contextmanager
def _envelope_getter_route(monkeypatch, route):
    if route == "reference":
        module = import_module(
            "trade_rl.evaluation.rl_allocation._source_envelope_digest"
        )
        with monkeypatch.context() as scoped:
            scoped.setattr(
                module.SourceEnvelopeDigestMemo,
                "get_from_env",
                lambda _self, env, stream_memo, *, pin=module.allocation_array_pin, digest=module.hashing.content_digest: (
                    context._source_envelope_digest(env, stream_memo)
                ),
            )
            yield
        return
    if route == "optimized":
        yield
        return
    raise AssertionError(f"unknown envelope digest route: {route}")


def _mutate_cost_roster(env, mutation: str) -> None:
    keys = sorted(env._estimates)
    last_key = keys[-1]
    target = env._estimates[last_key]
    if mutation == "buy_cost":
        object.__setattr__(target, "buy_cost", target.buy_cost + 0.01)
    elif mutation == "sell_cost":
        object.__setattr__(target, "sell_cost", target.sell_cost + 0.01)
    elif mutation == "exit_cost":
        object.__setattr__(target, "exit_cost", target.exit_cost + 0.01)
    elif mutation == "funding_return":
        object.__setattr__(target, "funding_return", target.funding_return + 0.01)
    elif mutation == "borrow_return":
        object.__setattr__(target, "borrow_return", target.borrow_return + 0.01)
    elif mutation == "cash_return":
        object.__setattr__(target, "cash_return", target.cash_return + 0.01)
    elif mutation == "source_identity":
        object.__setattr__(target, "source_identity", target.source_identity + "-mut")
    elif mutation == "add_entry":
        env._estimates[last_key + 1] = replace(target, buy_cost=target.buy_cost + 0.01)
    elif mutation == "remove_entry":
        del env._estimates[last_key]
    elif mutation == "replace_entry":
        env._estimates[last_key] = replace(target, sell_cost=target.sell_cost + 0.01)
    else:
        raise AssertionError(f"unknown mutation: {mutation}")


@pytest.mark.parametrize(
    "mutation",
    [
        "buy_cost",
        "sell_cost",
        "exit_cost",
        "funding_return",
        "borrow_return",
        "cash_return",
        "source_identity",
        "add_entry",
        "remove_entry",
        "replace_entry",
    ],
)
def test_cost_mutation_after_declaration_matches_reference_before_reset(
    monkeypatch, mutation
):
    api = capability()
    outcomes = []
    for route in ("reference", "optimized"):
        env, folds = global_control_fixture(), global_control_folds(7)
        plan = declare(api, env, folds)
        _mutate_cost_roster(env, mutation)
        monkeypatch.setattr(
            env, "reset", lambda **_: pytest.fail("reset before refusal")
        )
        with _envelope_getter_route(monkeypatch, route):
            with pytest.raises(ValueError) as raised:
                api.run_declared_global_allocation_execution(folds, env, plan)
        assert not hasattr(env, "book")
        outcomes.append((type(raised.value), str(raised.value)))

    assert outcomes[0] == outcomes[1]


@pytest.mark.parametrize("mutation", ["buy_cost", "add_entry", "remove_entry"])
def test_cost_mutation_during_action_matches_reference_and_prevents_native_fill(
    monkeypatch, mutation
):
    import trade_rl.evaluation.allocation_nonrl_walk_forward as nonrl
    from trade_rl.evaluation.rl_allocation.continuous_walk_forward import (
        AllocationFoldPolicy,
    )

    api = capability()
    native_policy = AllocationFoldPolicy
    outcomes = []

    for route in ("reference", "optimized"):
        env, folds = global_control_fixture(), global_control_folds(7)
        plan = declare(api, env, folds)
        calls = []

        def policy_with_mutating_action(policy_digest, recipe_digest, action):
            def mutate_during_action(observation, recipe):
                result = action(observation, recipe)
                if not calls:
                    _mutate_cost_roster(env, mutation)
                calls.append(True)
                return result

            return native_policy(policy_digest, recipe_digest, mutate_during_action)

        with monkeypatch.context() as scoped:
            scoped.setattr(nonrl, "AllocationFoldPolicy", policy_with_mutating_action)
            with _envelope_getter_route(scoped, route):
                with pytest.raises(
                    ValueError, match="source/runtime contents changed"
                ) as raised:
                    api.run_declared_global_allocation_execution(folds, env, plan)

        receipt = raised.value.global_execution_receipt.payload
        assert calls == [True]
        assert receipt["status"] == "integrity_failure"
        assert len(receipt["rows"]) == 1
        assert receipt["rows"][0]["native"] is None
        assert env.index == 6 and env.book.fill_count == 0
        outcomes.append(
            (
                type(raised.value),
                str(raised.value),
                canonical_json_bytes(receipt),
            )
        )

    assert outcomes[0] == outcomes[1]


@pytest.mark.parametrize("mutation", ["buy_cost", "replace_entry", "remove_entry"])
def test_post_native_cost_mutation_matches_reference_integrity_failure(
    monkeypatch, mutation
):
    api = capability()
    outcomes = []

    for route in ("reference", "optimized"):
        env, folds = global_control_fixture(), global_control_folds(7)
        plan = declare(api, env, folds)
        native_step = env.step

        def mutate_after_native_step(action):
            result = native_step(action)
            _mutate_cost_roster(env, mutation)
            return result

        with monkeypatch.context() as scoped:
            scoped.setattr(env, "step", mutate_after_native_step)
            with _envelope_getter_route(scoped, route):
                with pytest.raises(
                    ValueError, match="source/runtime contents changed"
                ) as raised:
                    api.run_declared_global_allocation_execution(folds, env, plan)

        receipt = raised.value.global_execution_receipt.payload
        row = receipt["rows"][0]
        assert receipt["status"] == "integrity_failure"
        assert len(receipt["rows"]) == 1
        assert row["call_outcome"] == {"status": "returned", "action": 2}
        assert row["native"]["processing_index"] == env.index == 7
        assert env.book.fill_count == 1
        assert env._transition_recorder is None
        outcomes.append(
            (
                type(raised.value),
                str(raised.value),
                canonical_json_bytes(receipt),
            )
        )

    assert outcomes[0] == outcomes[1]


def test_observer_boundary_cost_mutation_matches_reference(monkeypatch):
    execution_api = import_module("trade_rl.evaluation.allocation_terminal_execution")
    outcomes = []

    for route in ("reference", "optimized"):
        folds, env = carrier(side=1)
        plan = plans(execution_api, env, folds, kind="nonrl", stop=8)

        def mutating_observer(phase, _start_index, _processing_index):
            if phase == "actor":
                _mutate_cost_roster(env, "buy_cost")

        with _envelope_getter_route(monkeypatch, route):
            with pytest.raises(
                ValueError, match="source/runtime contents changed"
            ) as raised:
                execution_api.run_terminal_allocation_execution(
                    folds, env, plan, on_advance=mutating_observer
                )

        receipt = raised.value.terminal_execution_receipt
        global_receipt = raised.value.global_execution_receipt.payload
        assert receipt["status"] == "integrity_failure"
        assert global_receipt["status"] == "integrity_failure"
        outcomes.append(
            (
                type(raised.value),
                str(raised.value),
                canonical_json_bytes(receipt),
                canonical_json_bytes(global_receipt),
            )
        )

    assert outcomes[0] == outcomes[1]


@contextmanager
def _payload_reads_inside(code: CodeType) -> Iterator[list[int]]:
    """Count public HorizonCostEstimates.payload calls inside one getter."""
    payload_code = HorizonCostEstimates.payload.__code__
    reads: list[int] = []
    active: list[int] = []
    previous = sys.getprofile()

    def profile(frame, event, _arg):
        if event == "call" and frame.f_code is code:
            active.append(0)
        elif event == "call" and frame.f_code is payload_code and active:
            active[-1] += 1
        elif event == "return" and frame.f_code is code and active:
            reads.append(active.pop())

    sys.setprofile(profile)
    try:
        yield reads
    finally:
        sys.setprofile(previous)


def _run_terminal_route(route, monkeypatch, operation):
    memo_module = import_module(
        "trade_rl.evaluation.rl_allocation._source_envelope_digest"
    )
    memo_type = memo_module.SourceEnvelopeDigestMemo
    original_get = memo_type.get_from_env
    getter_calls = []

    if route == "reference":

        def public_get(
            _self,
            env,
            stream_memo,
            *,
            pin=memo_module.allocation_array_pin,
            digest=memo_module.hashing.content_digest,
        ):
            getter_calls.append(None)
            return context._source_envelope_digest(env, stream_memo)

        replacement = public_get
        profiled_code = public_get.__code__
    elif route == "optimized":

        def original_get_with_count(
            self,
            env,
            stream_memo,
            *,
            pin=memo_module.allocation_array_pin,
            digest=memo_module.hashing.content_digest,
        ):
            getter_calls.append(None)
            return original_get(self, env, stream_memo, pin=pin, digest=digest)

        replacement = original_get_with_count
        profiled_code = original_get.__code__
    else:
        raise AssertionError(f"unknown envelope route: {route}")

    with monkeypatch.context() as scoped:
        scoped.setattr(memo_type, "get_from_env", replacement)
        with _payload_reads_inside(profiled_code) as payload_reads:
            result = operation()

    assert getter_calls
    assert len(payload_reads) == len(getter_calls)
    if route == "reference":
        assert all(count > 0 for count in payload_reads)
    else:
        assert payload_reads[0] > 0  # first baseline
        assert all(count == 0 for count in payload_reads[1:])
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
def test_same_head_full_terminal_bytes_match_reference_and_source_envelope_memo(
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

        result = _run_terminal_route(
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
        observations.append((plan_bytes, receipt_bytes, tuple(progress)))

    assert observations[0] == observations[1]
