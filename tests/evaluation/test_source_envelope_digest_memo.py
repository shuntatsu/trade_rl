"""Live identity and compatibility contracts for the source-envelope memo."""

from __future__ import annotations

import importlib
import sys
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import replace
from types import FrameType, MappingProxyType

import numpy as np
import pytest

from tests.evaluation.test_allocation_nonrl_walk_forward import global_control_fixture
from trade_rl.artifacts import canonical, hashing
from trade_rl.evaluation import forecast_allocation
from trade_rl.evaluation.rl_allocation import global_execution_context as context


def memo_module():
    return importlib.import_module(
        "trade_rl.evaluation.rl_allocation._source_envelope_digest"
    )


def api():
    return memo_module().SourceEnvelopeDigestMemo


@pytest.fixture
def env():
    return global_control_fixture()


def _inputs(env) -> tuple[dict[str, object], dict[str, object]]:
    dataset = env.dataset
    arrays = {}
    for name, values in dataset.identity_arrays().items():
        if name == "global_features":
            continue
        if name in ("features", "feature_available"):
            selected = values[env.start_index : env.stop_index, env.symbol_index][
                :, list(env.feature_indices)
            ]
        elif values.ndim > 0 and values.shape[0] == dataset.n_bars:
            selected = values[env.start_index : env.stop_index + 1]
        else:
            selected = values
        arrays[name] = context.allocation_array_pin(selected)
    prefix = {
        "schema": "allocation_fee_global_source_envelope_v1",
        "dataset_id": dataset.dataset_id,
        "dataset_contract": dataset.identity_contract_payload(),
        "symbol_index": env.symbol_index,
        "feature_indices": list(env.feature_indices),
        "feature_names": list(dataset.feature_names),
        "range": [env.start_index, env.stop_index],
        "account_id": env.account_id,
        "forecast_context_digest": env.stream.digest,
    }
    return prefix, arrays


def measured_calls(operation: Callable[[], object]) -> tuple[object, int, int]:
    digest_calls = 0
    payload_calls = 0
    digest_code = hashing.content_digest.__code__
    payload_code = forecast_allocation.HorizonCostEstimates.payload.__code__

    def profile(frame: FrameType, event: str, _arg: object) -> None:
        nonlocal digest_calls, payload_calls
        if event == "call":
            if frame.f_code is digest_code:
                digest_calls += 1
            elif frame.f_code is payload_code:
                payload_calls += 1

    previous = sys.getprofile()
    try:
        sys.setprofile(profile)
        result = operation()
    finally:
        sys.setprofile(previous)
    return result, digest_calls, payload_calls


def test_lazy_first_digest_and_warm_access_preserve_canonical_result(env):
    Memo = api()
    memo, digest_calls, payload_calls = measured_calls(Memo)
    assert digest_calls == 0 and payload_calls == 0
    expected = context.allocation_source_envelope_digest(env)
    prefix, arrays = _inputs(env)

    result, digest_calls, payload_calls = measured_calls(
        lambda: memo.get(env, prefix, arrays)
    )
    assert result == expected
    assert digest_calls == 1 and payload_calls == len(env._estimates)

    prefix2, arrays2 = _inputs(env)
    result, digest_calls, payload_calls = measured_calls(
        lambda: memo.get(env, prefix2, arrays2)
    )
    assert result == expected
    assert digest_calls == 0 and payload_calls == 0


@pytest.mark.parametrize(
    "change",
    [
        "symbol",
        "decision_time",
        "available_at",
        "horizon_end",
        "source_identity",
        "buy_cost",
        "sell_cost",
        "exit_cost",
        "funding_return",
        "borrow_return",
        "cash_return",
        "valuation_basis",
        "signed_zero",
        "int_for_float",
        "bool_for_float",
        "non_ns_timestamp",
        "nat_timestamp",
        "add_entry",
        "remove_entry",
        "replace_entry",
        "swap_keys_order",
        "empty_roster",
        "large_int_key",
        "negative_int_key",
        "prefix_account",
        "prefix_contract",
        "prefix_signed_zero",
        "arrays_sha",
    ],
)
def test_live_cost_and_envelope_changes_match_the_current_public_digest(env, change):
    memo = api()()
    prefix, arrays = _inputs(env)
    original = memo.get(env, prefix, arrays)
    keys = sorted(env._estimates)
    last_key = keys[-1]
    target = env._estimates[last_key]

    if change == "symbol":
        object.__setattr__(target, "symbol", "ALTUSDT")
    elif change == "decision_time":
        object.__setattr__(
            target, "decision_time", target.decision_time + np.timedelta64(1, "ns")
        )
    elif change == "available_at":
        object.__setattr__(
            target, "available_at", target.available_at - np.timedelta64(1, "ns")
        )
    elif change == "horizon_end":
        object.__setattr__(
            target, "horizon_end", target.horizon_end + np.timedelta64(1, "ns")
        )
    elif change == "source_identity":
        object.__setattr__(target, "source_identity", target.source_identity + "-alt")
    elif change in (
        "buy_cost",
        "sell_cost",
        "exit_cost",
        "funding_return",
        "borrow_return",
        "cash_return",
    ):
        object.__setattr__(target, change, getattr(target, change) + 0.0025)
    elif change == "valuation_basis":
        object.__setattr__(target, "valuation_basis", "altered_basis")
    elif change == "signed_zero":
        object.__setattr__(target, "funding_return", -0.0)
    elif change == "int_for_float":
        object.__setattr__(target, "buy_cost", 0)
    elif change == "bool_for_float":
        object.__setattr__(target, "buy_cost", True)
    elif change == "non_ns_timestamp":
        object.__setattr__(
            target, "decision_time", np.datetime64("2025-01-01T00:00:00", "s")
        )
    elif change == "nat_timestamp":
        object.__setattr__(target, "decision_time", np.datetime64("NaT", "ns"))
    elif change == "add_entry":
        env._estimates[last_key + 1] = replace(target, buy_cost=target.buy_cost + 0.01)
    elif change == "remove_entry":
        del env._estimates[last_key]
    elif change == "replace_entry":
        env._estimates[last_key] = replace(target, sell_cost=target.sell_cost + 0.01)
    elif change == "swap_keys_order":
        first_key = keys[0]
        env._estimates[first_key], env._estimates[last_key] = (
            env._estimates[last_key],
            env._estimates[first_key],
        )
    elif change == "empty_roster":
        env._estimates = {}
    elif change == "large_int_key":
        env._estimates[2**80] = replace(target, exit_cost=target.exit_cost + 0.01)
    elif change == "negative_int_key":
        env._estimates[-1] = replace(target, cash_return=target.cash_return + 0.01)
    elif change == "prefix_account":
        env.account_id = "altered-account"
    elif change == "prefix_contract":
        prefix, arrays = _inputs(env)
        prefix["dataset_contract"] = {
            **prefix["dataset_contract"],
            "calendar_kind": "altered",
        }
        expected = hashing.content_digest(
            {
                **prefix,
                "cost_beliefs": [
                    env._estimates[k].payload() for k in sorted(env._estimates)
                ],
                "arrays": arrays,
            }
        )
        assert expected != original
        assert memo.get(env, prefix, arrays) == expected
        return
    elif change == "prefix_signed_zero":
        prefix, arrays = _inputs(env)
        prefix["dataset_contract"] = {
            **prefix["dataset_contract"],
            "probe_zero": -0.0,
        }
        expected = hashing.content_digest(
            {
                **prefix,
                "cost_beliefs": [
                    env._estimates[k].payload() for k in sorted(env._estimates)
                ],
                "arrays": arrays,
            }
        )
        assert expected != original
        assert memo.get(env, prefix, arrays) == expected
        return
    else:
        prefix, arrays = _inputs(env)
        arrays["close"] = {**arrays["close"], "sha256": "0" * 64}
        expected = hashing.content_digest(
            {
                **prefix,
                "cost_beliefs": [
                    env._estimates[k].payload() for k in sorted(env._estimates)
                ],
                "arrays": arrays,
            }
        )
        assert expected != original
        assert memo.get(env, prefix, arrays) == expected
        return

    expected = context.allocation_source_envelope_digest(env)
    assert expected != original
    prefix, arrays = _inputs(env)
    assert memo.get(env, prefix, arrays) == expected


def test_miss_does_not_replace_the_initial_baseline_and_bounds_memory(env):
    memo = api()()
    prefix, arrays = _inputs(env)
    original = memo.get(env, prefix, arrays)
    baseline_snapshot = memo._baseline
    assert baseline_snapshot is not None and len(baseline_snapshot) == 2

    last_key = max(env._estimates)
    target = env._estimates[last_key]
    old_buy = target.buy_cost
    object.__setattr__(target, "buy_cost", old_buy + 0.01)

    prefix_miss, arrays_miss = _inputs(env)
    miss_result, digest_calls, payload_calls = measured_calls(
        lambda: memo.get(env, prefix_miss, arrays_miss)
    )
    assert miss_result == context.allocation_source_envelope_digest(env) != original
    assert digest_calls == 1 and payload_calls == len(env._estimates)
    assert memo._baseline is baseline_snapshot

    object.__setattr__(target, "buy_cost", old_buy)
    prefix_hit, arrays_hit = _inputs(env)
    hit_result, digest_calls, payload_calls = measured_calls(
        lambda: memo.get(env, prefix_hit, arrays_hit)
    )
    assert hit_result == original
    assert digest_calls == 0 and payload_calls == 0
    assert vars(memo).keys() == {"_attempted", "_baseline"}


def test_equal_value_replacements_and_insertion_reordering_stay_warm(env):
    memo = api()()
    prefix, arrays = _inputs(env)
    original = memo.get(env, prefix, arrays)

    last_key = max(env._estimates)
    target = env._estimates[last_key]
    detached = target.payload()
    detached["buy_cost"] = 99.0
    detached["decision_time"] = 0

    object.__setattr__(target, "buy_cost", float.fromhex(target.buy_cost.hex()))
    env._estimates[last_key] = replace(target)
    env._estimates = dict(reversed(list(env._estimates.items())))

    prefix2, arrays2 = _inputs(env)
    result, digest_calls, payload_calls = measured_calls(
        lambda: memo.get(env, prefix2, arrays2)
    )
    assert result == original == context.allocation_source_envelope_digest(env)
    assert digest_calls == 0 and payload_calls == 0


def test_unsupported_mappings_and_callbacks_preserve_reference_evaluation_order(env):
    memo = api()()
    prefix, arrays = _inputs(env)
    memo.get(env, prefix, arrays)

    keys = sorted(env._estimates)
    first_key, last_key = keys[0], keys[-1]

    # 1. Subclass with __slots__ = () whose payload() mutates a later entry
    events: list[str] = []

    class MutatingEstimate(forecast_allocation.HorizonCostEstimates):
        __slots__ = ()

        def payload(self) -> dict[str, object]:
            events.append("first_payload")
            object.__setattr__(env._estimates[last_key], "buy_cost", 0.123)
            return super().payload()

    base_first = env._estimates[first_key]
    base_last_buy = env._estimates[last_key].buy_cost
    env._estimates[first_key] = MutatingEstimate(**base_first._arguments())

    object.__setattr__(env._estimates[last_key], "buy_cost", base_last_buy)
    expected = context.allocation_source_envelope_digest(env)
    assert events == ["first_payload"]

    events.clear()
    object.__setattr__(env._estimates[last_key], "buy_cost", base_last_buy)
    prefix2, arrays2 = _inputs(env)
    actual, digest_calls, payload_calls = measured_calls(
        lambda: memo.get(env, prefix2, arrays2)
    )
    assert actual == expected
    assert events == ["first_payload"]
    assert digest_calls == 1 and payload_calls == len(env._estimates)

    # 2. Custom key class whose __lt__ mutates a cost field during sorted()
    env._estimates[first_key] = base_first
    object.__setattr__(env._estimates[last_key], "buy_cost", base_last_buy)
    lt_calls: list[int] = []

    class MutatingKey:
        def __init__(self, value: int) -> None:
            self.value = value

        def __hash__(self) -> int:
            return hash(self.value)

        def __eq__(self, other: object) -> bool:
            return isinstance(other, MutatingKey) and self.value == other.value

        def __lt__(self, other: MutatingKey) -> bool:
            lt_calls.append(self.value)
            object.__setattr__(
                env._estimates[MutatingKey(last_key)], "sell_cost", 0.077
            )
            return self.value < other.value

    env._estimates = {MutatingKey(k): replace(v) for k, v in env._estimates.items()}
    expected_custom_key = context.allocation_source_envelope_digest(env)
    ref_lt_count = len(lt_calls)
    assert ref_lt_count > 0

    lt_calls.clear()
    object.__setattr__(
        env._estimates[MutatingKey(last_key)], "sell_cost", base_first.sell_cost
    )
    prefix3, arrays3 = _inputs(env)
    actual_custom_key = memo.get(env, prefix3, arrays3)
    assert actual_custom_key == expected_custom_key
    assert len(lt_calls) == ref_lt_count

    # 3. OrderedDict and MappingProxyType fallback cleanly
    env._estimates = OrderedDict((k.value, v) for k, v in env._estimates.items())
    prefix4, arrays4 = _inputs(env)
    res_od, digest_calls, _ = measured_calls(lambda: memo.get(env, prefix4, arrays4))
    assert (
        res_od == context.allocation_source_envelope_digest(env) and digest_calls == 1
    )

    env._estimates = MappingProxyType(dict(env._estimates))
    prefix5, arrays5 = _inputs(env)
    res_mp, digest_calls, _ = measured_calls(lambda: memo.get(env, prefix5, arrays5))
    assert (
        res_mp == context.allocation_source_envelope_digest(env) and digest_calls == 1
    )


def test_sequential_non_cost_callback_executes_once_and_updates_cost(env, monkeypatch):
    from trade_rl.evaluation.rl_allocation._simple_stream_digest import (
        SimpleStreamDigestMemo,
    )

    stream_memo = SimpleStreamDigestMemo()
    envelope_memo = api()()

    first = context._source_envelope_digest(env, stream_memo, envelope_memo)
    assert first == context.allocation_source_envelope_digest(env)

    last_key = max(env._estimates)
    target = env._estimates[last_key]
    original_contract = env.dataset.identity_contract_payload
    callback_calls = 0
    stream_calls = 0
    original_stream_get = stream_memo.get

    def mutating_contract():
        nonlocal callback_calls
        callback_calls += 1
        object.__setattr__(target, "buy_cost", target.buy_cost + 0.005)
        return original_contract()

    def counted_stream_get(stream):
        nonlocal stream_calls
        stream_calls += 1
        return original_stream_get(stream)

    monkeypatch.setattr(
        type(env.dataset),
        "identity_contract_payload",
        lambda _self: mutating_contract(),
    )
    monkeypatch.setattr(stream_memo, "get", counted_stream_get)

    second = context._source_envelope_digest(env, stream_memo, envelope_memo)
    assert callback_calls == 1
    assert stream_calls == 1
    assert second != first
    monkeypatch.undo()
    assert second == context.allocation_source_envelope_digest(env)


@pytest.mark.parametrize(
    "change",
    [
        "payload_method",
        "arguments_method",
        "serializer",
        "code",
        "defaults",
        "descriptor",
        "passed_digest",
    ],
)
def test_changes_before_memo_construction_do_not_become_trusted(
    env, monkeypatch, change
):
    Memo = api()
    custom_digest = None
    if change == "payload_method":
        original = forecast_allocation.HorizonCostEstimates.payload
        monkeypatch.setattr(
            forecast_allocation.HorizonCostEstimates,
            "payload",
            lambda self: {**original(self), "probe": True},
        )
    elif change == "arguments_method":
        original = forecast_allocation.HorizonCostEstimates._arguments
        monkeypatch.setattr(
            forecast_allocation.HorizonCostEstimates,
            "_arguments",
            lambda self: {**original(self), "buy_cost": 0.12345},
        )
    elif change == "serializer":
        original = canonical.to_json_value
        monkeypatch.setattr(canonical, "to_json_value", lambda value: original(value))
    elif change == "code":

        def altered(self) -> dict[str, object]:
            return {"schema": "altered"}

        monkeypatch.setattr(
            forecast_allocation.HorizonCostEstimates.payload,
            "__code__",
            altered.__code__,
        )
    elif change == "defaults":
        monkeypatch.setitem(canonical.json.dumps.__kwdefaults__, "sort_keys", True)
    elif change == "descriptor":
        monkeypatch.setattr(
            forecast_allocation.HorizonCostEstimates,
            "symbol",
            property(lambda _: "PATCHED"),
        )
    else:

        def _custom_digest(value: object) -> str:
            return hashing.content_digest(value)

        custom_digest = _custom_digest

    memo = Memo()
    prefix, arrays = _inputs(env)
    kwargs = {"digest": custom_digest} if custom_digest is not None else {}
    assert memo.get(
        env, prefix, arrays, **kwargs
    ) == context.allocation_source_envelope_digest(env)
    prefix2, arrays2 = _inputs(env)
    result, digest_calls, _ = measured_calls(
        lambda: memo.get(env, prefix2, arrays2, **kwargs)
    )
    assert result == context.allocation_source_envelope_digest(env)
    assert digest_calls > 0


def test_unstable_first_evaluation_is_not_cached(env):
    memo = api()()
    code, changed = hashing.content_digest.__code__, False
    last_key = max(env._estimates)
    target = env._estimates[last_key]
    prefix, arrays = _inputs(env)

    def mutate(frame, event, _arg):
        nonlocal changed
        if event == "call" and frame.f_code is code and not changed:
            changed = True
            object.__setattr__(target, "buy_cost", target.buy_cost + 0.05)

    previous = sys.getprofile()
    try:
        sys.setprofile(mutate)
        memo.get(env, prefix, arrays)
    finally:
        sys.setprofile(previous)

    assert memo._baseline is None
    prefix2, arrays2 = _inputs(env)
    result, digest_calls, payload_calls = measured_calls(
        lambda: memo.get(env, prefix2, arrays2)
    )
    assert result == context.allocation_source_envelope_digest(env)
    assert digest_calls == 1 and payload_calls == len(env._estimates)


def test_public_digest_errors_are_preserved(env):
    memo = api()()
    prefix, arrays = _inputs(env)
    memo.get(env, prefix, arrays)
    last_key = max(env._estimates)
    object.__setattr__(env._estimates[last_key], "buy_cost", float("nan"))
    prefix2, arrays2 = _inputs(env)
    with pytest.raises(ValueError, match="canonical JSON floats must be finite"):
        memo.get(env, prefix2, arrays2)


@pytest.mark.parametrize("method", ["encode", "iterencode"])
def test_json_encoder_method_changes_after_warming_use_public_digest(
    env, monkeypatch, method
):
    memo = api()()
    prefix, arrays = _inputs(env)
    initial = memo.get(env, prefix, arrays)
    encoder = canonical.json.JSONEncoder
    original = getattr(encoder, method)
    if method == "encode":

        def altered(self, value):
            return original(self, value) + " "
    else:

        def altered(self, value, _one_shot=False):
            return (*original(self, value, _one_shot), " ")

    monkeypatch.setattr(encoder, method, altered)
    expected = context.allocation_source_envelope_digest(env)
    assert expected != initial
    prefix2, arrays2 = _inputs(env)
    result, digest_calls, _ = measured_calls(lambda: memo.get(env, prefix2, arrays2))
    assert result == expected and digest_calls > 0


def test_get_from_env_warm_skips_array_pins_and_matches_canonical(env):
    memo = api()()
    stream_memo = context.SimpleStreamDigestMemo()
    expected = context.allocation_source_envelope_digest(env)

    # First call: baseline arming
    first = memo.get_from_env(env, stream_memo)
    assert first == expected
    assert memo._baseline is not None

    # Track calls to allocation_array_pin, content_digest, and payload during second call
    pin_calls = 0
    digest_calls = 0
    payload_calls = 0
    pin_code = context.allocation_array_pin.__code__
    digest_code = hashing.content_digest.__code__
    payload_code = forecast_allocation.HorizonCostEstimates.payload.__code__

    def profile(frame: FrameType, event: str, _arg: object) -> None:
        nonlocal pin_calls, digest_calls, payload_calls
        if event == "call":
            if frame.f_code is pin_code:
                pin_calls += 1
            elif frame.f_code is digest_code:
                digest_calls += 1
            elif frame.f_code is payload_code:
                payload_calls += 1

    previous = sys.getprofile()
    try:
        sys.setprofile(profile)
        second = memo.get_from_env(env, stream_memo)
    finally:
        sys.setprofile(previous)

    assert second == expected
    assert pin_calls == 0
    assert digest_calls == 0
    assert payload_calls == 0


def test_get_from_env_inplace_array_mutation_refuses_stale_cache(env):
    memo = api()()
    first = memo.get_from_env(env, None)

    original_close = env.dataset.close
    new_close = np.array(original_close, copy=True)
    new_close[env.start_index, 0] += 100.0
    object.__setattr__(env.dataset, "close", new_close)

    try:
        second = memo.get_from_env(env, None)
        expected = context.allocation_source_envelope_digest(env)
        assert second == expected != first
    finally:
        object.__setattr__(env.dataset, "close", original_close)


def test_get_from_env_unstable_arm_refuses_baseline(env):
    memo = api()()
    code = hashing.content_digest.__code__
    changed = False
    original_close = env.dataset.close

    def mutate_during_digest(frame: FrameType, event: str, _arg: object) -> None:
        nonlocal changed
        if event == "call" and frame.f_code is code and not changed:
            changed = True
            new_close = np.array(original_close, copy=True)
            new_close[env.start_index, 0] += 50.0
            object.__setattr__(env.dataset, "close", new_close)

    previous = sys.getprofile()
    try:
        sys.setprofile(mutate_during_digest)
        memo.get_from_env(env, None)
    finally:
        sys.setprofile(previous)
        object.__setattr__(env.dataset, "close", original_close)

    # Because array mutated during first evaluation, baseline arming must be refused
    assert memo._baseline is None


def test_get_from_env_fallback_when_pin_overridden(env):
    memo = api()()
    custom_pin_calls = 0

    def custom_pin(arr: np.ndarray) -> dict[str, object]:
        nonlocal custom_pin_calls
        custom_pin_calls += 1
        return context.allocation_array_pin(arr)

    result = memo.get_from_env(env, None, pin=custom_pin)
    assert result == context.allocation_source_envelope_digest(env)
    assert custom_pin_calls > 0
    # Because custom pin was provided, baseline should not be armed
    assert memo._baseline is None


@pytest.mark.parametrize(
    ("field", "initial", "mutated"),
    [
        ("funding_return", 0.0, -0.0),
        ("buy_cost", 0.0, 0),
        ("funding_return", 1.0, True),
    ],
)
def test_get_from_env_warm_hit_mutation_rejected_for_coercible_types(
    env, field, initial, mutated
):
    stream_memo = context.SimpleStreamDigestMemo()
    memo = api()()
    target = env._estimates[sorted(env._estimates)[-1]]
    orig_val = getattr(target, field)
    try:
        object.__setattr__(target, field, initial)
        baseline = memo.get_from_env(env, stream_memo)
        assert memo._baseline is not None

        object.__setattr__(target, field, mutated)
        result = memo.get_from_env(env, stream_memo)
        expected = context.allocation_source_envelope_digest(env)
        assert baseline != expected
        assert result == expected
    finally:
        object.__setattr__(target, field, orig_val)
