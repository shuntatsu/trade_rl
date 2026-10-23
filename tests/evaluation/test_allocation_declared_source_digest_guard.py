"""Global allocation guards and same-HEAD terminal parity for declared-source memo."""

from __future__ import annotations

import sys
from contextlib import contextmanager
from importlib import import_module

import numpy as np
import pytest

from tests.evaluation.test_allocation_global_execution import capability, declare
from tests.evaluation.test_allocation_nonrl_walk_forward import (
    global_control_fixture,
    global_control_folds,
)
from tests.evaluation.test_allocation_terminal_execution import carrier, plans
from trade_rl.artifacts import canonical_json_bytes


@contextmanager
def _declared_source_getter_route(monkeypatch, route):
    if route == "reference":
        module = import_module(
            "trade_rl.evaluation.rl_allocation._declared_source_digest"
        )
        with monkeypatch.context() as scoped:
            scoped.setattr(
                module.DeclaredSourceDigestMemo,
                "get",
                lambda _self, env, *, pin=None, digest=None: (
                    module._public_declared_source_digest(env, pin=pin, digest=digest)
                ),
            )
            yield
        return
    if route == "optimized":
        yield
        return
    raise AssertionError(f"unknown declared source digest route: {route}")


def _mutate_dataset(env, mutation: str) -> None:
    if mutation == "close_price":
        new_close = env.dataset.close.copy()
        new_close[env.start_index, 0] += 0.5
        object.__setattr__(env.dataset, "close", new_close)
    elif mutation == "funding_rate":
        new_funding = env.dataset.funding_rate.copy()
        new_funding[env.start_index, 0] = -0.05
        object.__setattr__(env.dataset, "funding_rate", new_funding)
    elif mutation == "timestamp":
        new_ts = env.dataset.timestamps.copy()
        new_ts[env.start_index] += np.timedelta64(1, "ns")
        object.__setattr__(env.dataset, "timestamps", new_ts)
    elif mutation == "dataset_id":
        object.__setattr__(env.dataset, "dataset_id", "0" * 64)
    elif mutation == "feature_config":
        object.__setattr__(env.dataset, "feature_config_digest", "1" * 64)
    elif mutation == "start_index":
        env.start_index += 1
    elif mutation == "stop_index":
        env.stop_index -= 1
    else:
        raise AssertionError(f"unknown mutation: {mutation}")


@pytest.mark.parametrize(
    "mutation",
    [
        "close_price",
        "funding_rate",
        "timestamp",
        "dataset_id",
        "feature_config",
        "start_index",
        "stop_index",
    ],
)
def test_declared_source_mutation_after_declaration_matches_reference_before_reset(
    monkeypatch, mutation
):
    api = capability()
    outcomes = []
    for route in ("reference", "optimized"):
        env, folds = global_control_fixture(), global_control_folds(7)
        plan = declare(api, env, folds)
        _mutate_dataset(env, mutation)
        monkeypatch.setattr(
            env, "reset", lambda **_: pytest.fail("reset before refusal")
        )
        monkeypatch.setattr(
            env, "step", lambda _action: pytest.fail("step before refusal")
        )
        with _declared_source_getter_route(monkeypatch, route):
            with pytest.raises(ValueError) as raised:
                api.run_declared_global_allocation_execution(folds, env, plan)
        assert not hasattr(env, "book")
        outcomes.append((type(raised.value), str(raised.value)))

    assert outcomes[0] == outcomes[1]


@pytest.mark.parametrize("mutation", ["close_price", "funding_rate", "dataset_id"])
def test_declared_source_mutation_during_action_matches_reference_and_prevents_native_fill(
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
                    _mutate_dataset(env, mutation)
                calls.append(True)
                return result

            return native_policy(policy_digest, recipe_digest, mutate_during_action)

        with monkeypatch.context() as scoped:
            scoped.setattr(nonrl, "AllocationFoldPolicy", policy_with_mutating_action)
            with _declared_source_getter_route(scoped, route):
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


@pytest.mark.parametrize("mutation", ["close_price", "funding_rate"])
def test_post_native_declared_source_mutation_matches_reference_integrity_failure(
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
            _mutate_dataset(env, mutation)
            return result

        with monkeypatch.context() as scoped:
            scoped.setattr(env, "step", mutate_after_native_step)
            with _declared_source_getter_route(scoped, route):
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


def test_observer_boundary_declared_source_mutation_matches_reference(monkeypatch):
    execution_api = import_module("trade_rl.evaluation.allocation_terminal_execution")
    outcomes = []

    for route in ("reference", "optimized"):
        folds, env = carrier(side=1)
        plan = plans(execution_api, env, folds, kind="nonrl", stop=8)

        def mutating_observer(phase, _start_index, _processing_index):
            if phase == "actor":
                _mutate_dataset(env, "close_price")

        with _declared_source_getter_route(monkeypatch, route):
            with pytest.raises(
                ValueError, match="source/runtime contents changed"
            ) as raised:
                execution_api.run_terminal_allocation_execution(
                    folds, env, plan, on_advance=mutating_observer
                )

        receipt = raised.value.global_execution_receipt.payload
        assert receipt["status"] == "integrity_failure"
        outcomes.append(
            (
                type(raised.value),
                str(raised.value),
                canonical_json_bytes(receipt),
            )
        )

    assert outcomes[0] == outcomes[1]


@contextmanager
def _digest_calls_inside(target_code):
    memo_module = import_module(
        "trade_rl.evaluation.rl_allocation._declared_source_digest"
    )
    digest_code = memo_module.hashing.content_digest.__code__
    active = []
    calls = []
    previous = sys.getprofile()

    def profile(frame, event, _arg):
        if event == "call" and frame.f_code is target_code:
            active.append(0)
        elif event == "call" and frame.f_code is digest_code and active:
            active[-1] += 1
        elif event == "return" and frame.f_code is target_code and active:
            calls.append(active.pop())

    sys.setprofile(profile)
    try:
        yield calls
    finally:
        sys.setprofile(previous)


def _run_terminal_route(route, monkeypatch, operation):
    memo_module = import_module(
        "trade_rl.evaluation.rl_allocation._declared_source_digest"
    )
    memo_type = memo_module.DeclaredSourceDigestMemo
    original_get = memo_type.get
    getter_calls = []

    if route == "reference":

        def public_get(
            _self,
            env,
            *,
            pin=None,
            digest=None,
        ):
            getter_calls.append(None)
            return memo_module._public_declared_source_digest(
                env, pin=pin, digest=digest
            )

        replacement = public_get
        profiled_code = public_get.__code__
    elif route == "optimized":

        def original_get_with_count(
            self,
            env,
            *,
            pin=None,
            digest=None,
        ):
            getter_calls.append(None)
            return original_get(self, env, pin=pin, digest=digest)

        replacement = original_get_with_count
        profiled_code = original_get.__code__
    else:
        raise AssertionError(f"unknown declared source route: {route}")

    with monkeypatch.context() as scoped:
        scoped.setattr(memo_type, "get", replacement)
        with _digest_calls_inside(profiled_code) as digest_calls:
            result = operation()

    assert getter_calls
    assert len(digest_calls) == len(getter_calls)
    if route == "reference":
        assert all(count > 0 for count in digest_calls)
    else:
        assert digest_calls[0] > 0  # initial baseline arm
        assert all(count == 0 for count in digest_calls[1:])  # warm hits
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
def test_same_head_full_terminal_bytes_match_reference_and_declared_source_memo(
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
