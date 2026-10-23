"""Global allocation guards must observe live simple-stream contents."""

from contextlib import contextmanager
from importlib import import_module

import pytest

from tests.evaluation.test_allocation_global_execution import capability, declare
from tests.evaluation.test_allocation_nonrl_walk_forward import (
    global_control_fixture,
    global_control_folds,
)


@contextmanager
def _digest_getter_route(monkeypatch, route):
    if route == "reference":
        memo = import_module(
            "trade_rl.evaluation.rl_allocation._simple_stream_digest"
        ).SimpleStreamDigestMemo
        with monkeypatch.context() as scoped:
            scoped.setattr(memo, "get", lambda _self, stream: stream.digest)
            yield
        return
    if route == "optimized":
        yield
        return
    raise AssertionError(f"unknown digest route: {route}")


def _mutate_future_packet(env):
    packet = env.stream.packets[-1]
    assert packet.as_of > env.dataset.timestamps[env.start_index]
    object.__setattr__(
        packet, "expected_simple_return", packet.expected_simple_return + 0.01
    )


@pytest.mark.parametrize("route", ["reference", "optimized"])
def test_future_packet_mutation_after_declaration_rejects_before_reset(
    monkeypatch, route
):
    api = capability()
    env, folds = global_control_fixture(), global_control_folds(7)
    plan = declare(api, env, folds)
    _mutate_future_packet(env)
    monkeypatch.setattr(env, "reset", lambda **_: pytest.fail("reset before refusal"))

    with _digest_getter_route(monkeypatch, route):
        with pytest.raises(ValueError, match="current declaration|source/runtime"):
            api.run_declared_global_allocation_execution(folds, env, plan)

    assert not hasattr(env, "book")


@pytest.mark.parametrize("route", ["reference", "optimized"])
def test_future_packet_mutation_during_action_prevents_native_fill(monkeypatch, route):
    import trade_rl.evaluation.allocation_nonrl_walk_forward as nonrl
    from trade_rl.evaluation.rl_allocation.continuous_walk_forward import (
        AllocationFoldPolicy,
    )

    api = capability()
    env, folds = global_control_fixture(), global_control_folds(7)
    plan = declare(api, env, folds)
    native_policy = AllocationFoldPolicy
    calls = []

    def policy_with_mutating_action(policy_digest, recipe_digest, action):
        def mutate_during_action(observation, recipe):
            result = action(observation, recipe)
            if not calls:
                _mutate_future_packet(env)
            calls.append(True)
            return result

        return native_policy(policy_digest, recipe_digest, mutate_during_action)

    monkeypatch.setattr(nonrl, "AllocationFoldPolicy", policy_with_mutating_action)

    with _digest_getter_route(monkeypatch, route):
        with pytest.raises(
            ValueError, match="source/runtime contents changed"
        ) as raised:
            api.run_declared_global_allocation_execution(folds, env, plan)

    receipt = raised.value.global_execution_receipt.payload
    assert calls == [True]
    assert receipt["status"] == "integrity_failure"
    assert len(receipt["rows"]) == 1
    assert receipt["rows"][0]["native"] is None
    assert receipt["rows"][0]["call_outcome"] == {"status": "returned", "action": 2}
    assert env.index == 6 and env.book.fill_count == 0


@pytest.mark.parametrize("route", ["reference", "optimized"])
def test_post_native_future_packet_mutation_retains_event_as_integrity_failure(
    monkeypatch, route
):
    api = capability()
    env, folds = global_control_fixture(), global_control_folds(7)
    plan = declare(api, env, folds)
    native_step = env.step

    def mutate_after_native_step(action):
        result = native_step(action)
        _mutate_future_packet(env)
        return result

    monkeypatch.setattr(env, "step", mutate_after_native_step)

    with _digest_getter_route(monkeypatch, route):
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


def test_stream_digest_covers_public_subclass_scope_override():
    from trade_rl.artifacts import content_digest
    from trade_rl.strategies.forecasts.simple_stream import FrozenSimpleReturnStream

    base = global_control_fixture().stream

    class ScopedStream(FrozenSimpleReturnStream):
        @property
        def causal_scope_digest(self):
            return "f" * 64

    stream = ScopedStream(base.dataset_id, base.vintages, base.packets)
    payload = stream.payload()

    assert payload["causal_scope_digest"] == "f" * 64
    assert stream.digest == content_digest(payload)


def _complete_global_run(api, *, kind, boundary):
    from trade_rl.artifacts import canonical_json_bytes

    env = global_control_fixture(
        mode="direct" if kind == "cash" else "residual",
        cash_rate=0.876 if kind == "cash" else None,
    )
    folds = global_control_folds(boundary)
    plan = declare(api, env, folds, kind=kind)
    plan_bytes = canonical_json_bytes(plan.payload)
    result = api.run_declared_global_allocation_execution(folds, env, plan)
    assert result.receipt.payload["status"] == "completed"
    assert env._transition_recorder is None
    return plan_bytes, canonical_json_bytes(result.receipt.payload)


@pytest.mark.parametrize("kind", ["nonrl", "cash"])
@pytest.mark.parametrize("boundary", [7, 9])
def test_guarded_stream_digest_preserves_declared_and_receipt_bytes(
    monkeypatch, kind, boundary
):
    from importlib import import_module

    api = capability()
    memo = import_module(
        "trade_rl.evaluation.rl_allocation._simple_stream_digest"
    ).SimpleStreamDigestMemo

    with monkeypatch.context() as canonical_path:
        canonical_path.setattr(memo, "get", lambda self, stream: stream.digest)
        canonical_plan, canonical_receipt = _complete_global_run(
            api, kind=kind, boundary=boundary
        )

    guarded_plan, guarded_receipt = _complete_global_run(
        api, kind=kind, boundary=boundary
    )

    assert guarded_plan == canonical_plan
    assert guarded_receipt == canonical_receipt
