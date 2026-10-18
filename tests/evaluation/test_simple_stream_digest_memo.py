"""Live identity and compatibility contracts for the private stream memo."""

from __future__ import annotations

import importlib
import sys
from collections.abc import Callable
from types import FrameType

import numpy as np
import pytest

from tests.evaluation.test_allocation_nonrl_walk_forward import fixture
from trade_rl.artifacts import canonical, hashing
from trade_rl.strategies.forecasts import simple_stream


def api():
    return importlib.import_module(
        "trade_rl.evaluation.rl_allocation._simple_stream_digest"
    ).SimpleStreamDigestMemo


@pytest.fixture
def stream():
    return fixture(ranges=((6, 10),))[1][0].stream


def measured_calls(operation: Callable[[], object]) -> tuple[object, int]:
    calls = 0
    code = hashing.content_digest.__code__

    def profile(frame: FrameType, event: str, arg: object) -> None:
        nonlocal calls
        if event == "call" and frame.f_code is code:
            calls += 1

    previous = sys.getprofile()
    try:
        sys.setprofile(profile)
        result = operation()
    finally:
        sys.setprofile(previous)
    return result, calls


def test_lazy_first_digest_and_warm_access_preserve_canonical_result(stream):
    Memo = api()
    memo, calls = measured_calls(Memo)
    assert calls == 0
    expected = stream.digest
    result, calls = measured_calls(lambda: memo.get(stream))
    assert result == expected and calls > 0
    result, calls = measured_calls(lambda: memo.get(stream))
    assert result == expected and calls == 0


@pytest.mark.parametrize(
    "change",
    [
        "future_packet",
        "model",
        "trace",
        "vintage_digest",
        "simple_labels",
        "variance",
        "shape",
        "dtype",
        "data",
        "field_type",
        "signed_zero",
        "overflow",
    ],
)
def test_live_nested_changes_match_the_current_public_digest(stream, change):
    if change == "data":
        object.__setattr__(
            stream.vintages[0].model,
            "coefficients",
            stream.vintages[0].model.coefficients.copy(),
        )
    memo = api()()
    original = memo.get(stream)
    vintage, packet = stream.vintages[0], stream.packets[-1]
    if change == "future_packet":
        object.__setattr__(packet, "expected_simple_return", 0.2)
    elif change == "model":
        object.__setattr__(vintage.model, "intercept", 0.1)
    elif change == "trace":
        object.__setattr__(vintage.training.trace, "end_close", np.array([101.0]))
    elif change == "vintage_digest":
        object.__setattr__(vintage, "_digest", "a" * 64)
    elif change == "simple_labels":
        object.__setattr__(vintage.training, "labels", np.array([0.25]))
    elif change == "variance":
        object.__setattr__(vintage.training, "fit_prefix_marginal_variance", 0.25)
    elif change == "shape":
        vintage.model.coefficients.shape = (1, 1)
    elif change == "dtype":
        vintage.model.coefficients.dtype = np.int64
    elif change == "data":
        vintage.model.coefficients[0] = 0.2
    elif change == "field_type":
        object.__setattr__(packet, "horizon_seconds", True)
    elif change == "signed_zero":
        object.__setattr__(vintage.model, "intercept", -0.0)
    else:
        object.__setattr__(packet, "horizon_seconds", 2**80)
    expected = stream.digest
    assert expected != original
    assert memo.get(stream) == expected


def test_miss_does_not_replace_the_initial_baseline(stream):
    memo = api()()
    original = memo.get(stream)
    packet, value = stream.packets[-1], stream.packets[-1].expected_simple_return
    object.__setattr__(packet, "expected_simple_return", value + 0.01)
    assert memo.get(stream) == stream.digest != original
    object.__setattr__(packet, "expected_simple_return", value)
    result, calls = measured_calls(lambda: memo.get(stream))
    assert result == original and calls == 0


def test_payload_is_detached_and_equal_value_replacements_are_safe(stream):
    memo = api()()
    original = memo.get(stream)
    payload = stream.payload()
    payload["packets"][0]["feature_values"][0] = 99.0
    packet = stream.packets[-1]
    object.__setattr__(
        packet,
        "expected_simple_return",
        float.fromhex(packet.expected_simple_return.hex()),
    )
    assert memo.get(stream) == original == stream.digest
    object.__setattr__(packet, "feature_values", list(packet.feature_values))
    result, calls = measured_calls(lambda: memo.get(stream))
    assert result == original and calls > 0


def test_same_object_stride_and_unknown_dtype_metadata_force_fallback(stream):
    memo = api()()
    original = memo.get(stream)
    array = stream.vintages[0].model.coefficients
    array.strides = (0,)
    result, calls = measured_calls(lambda: memo.get(stream))
    assert result == original and calls > 0
    array.dtype = np.dtype("float64", metadata={"unknown": True})
    result, calls = measured_calls(lambda: memo.get(stream))
    assert result == original and calls > 0


def test_slots_only_subclass_uses_its_current_public_digest(stream):
    calls = []

    class Child(simple_stream.FrozenSimpleReturnStream):
        __slots__ = ()

        @property
        def digest(self):
            calls.append(None)
            return f"{len(calls):064x}"

    child = Child(stream.dataset_id, stream.vintages, stream.packets)
    memo = api()()
    assert memo.get(child) == "0" * 63 + "1"
    assert memo.get(child) == "0" * 63 + "2"


@pytest.mark.parametrize(
    "change", ["helper", "serializer", "code", "defaults", "descriptor"]
)
def test_changes_before_memo_construction_do_not_become_trusted(
    stream, monkeypatch, change
):
    Memo = api()  # The expected implementation is fixed before the patch.
    if change == "helper":
        original = simple_stream._model_payload
        monkeypatch.setattr(
            simple_stream,
            "_model_payload",
            lambda model: {**original(model), "probe": True},
        )
    elif change == "serializer":
        original = canonical.to_json_value
        monkeypatch.setattr(canonical, "to_json_value", lambda value: original(value))
    elif change == "code":

        def altered(self):
            return {"vintages": [], "packets": []}

        monkeypatch.setattr(
            simple_stream.FrozenSimpleReturnStream._causal_content,
            "__code__",
            altered.__code__,
        )
    elif change == "defaults":
        monkeypatch.setitem(canonical.json.dumps.__kwdefaults__, "sort_keys", True)
    else:
        monkeypatch.setattr(
            simple_stream.SimpleReturnPacket, "symbol", property(lambda _: "NEW")
        )
    memo = Memo()
    assert memo.get(stream) == stream.digest
    result, calls = measured_calls(lambda: memo.get(stream))
    assert result == stream.digest and calls > 0


def test_unstable_first_evaluation_is_not_cached(stream):
    memo = api()()
    code, changed = hashing.content_digest.__code__, False

    def mutate(frame, event, arg):
        nonlocal changed
        if event == "call" and frame.f_code is code and not changed:
            changed = True
            object.__setattr__(stream.packets[-1], "expected_simple_return", 0.2)

    previous = sys.getprofile()
    try:
        sys.setprofile(mutate)
        memo.get(stream)
    finally:
        sys.setprofile(previous)
    result, calls = measured_calls(lambda: memo.get(stream))
    assert result == stream.digest and calls > 0


def test_public_digest_errors_are_preserved(stream):
    memo = api()()
    memo.get(stream)
    object.__setattr__(stream.packets[-1], "expected_simple_return", float("nan"))
    with pytest.raises(ValueError):
        memo.get(stream)


@pytest.mark.parametrize("method", ["encode", "iterencode"])
def test_json_encoder_method_changes_after_warming_use_public_digest(
    stream, monkeypatch, method
):
    memo = api()()
    initial = memo.get(stream)
    encoder = canonical.json.JSONEncoder
    original = getattr(encoder, method)
    if method == "encode":

        def altered(self, value):
            return original(self, value) + " "
    else:

        def altered(self, value, _one_shot=False):
            return (*original(self, value, _one_shot), " ")

    monkeypatch.setattr(encoder, method, altered)
    expected = stream.digest
    assert expected != initial
    result, calls = measured_calls(lambda: memo.get(stream))
    assert result == expected and calls > 0


@pytest.mark.parametrize(
    ("field", "initial", "mutated"),
    [
        ("expected_simple_return", 0.0, -0.0),
        ("feature_values", (0.0,), (-0.0,)),
        ("horizon_seconds", 86400, 86400.0),
        ("horizon_seconds", 1, True),
    ],
)
def test_warm_hit_mutation_rejected_for_coercible_types(
    stream, field, initial, mutated
):
    packet = stream.packets[-1]
    orig_val = getattr(packet, field)
    try:
        object.__setattr__(packet, field, initial)
        memo = api()()
        baseline = memo.get(stream)
        assert memo._baseline is not None

        object.__setattr__(packet, field, mutated)
        result = memo.get(stream)
        expected = stream.digest
        assert baseline != expected
        assert result == expected
    finally:
        object.__setattr__(packet, field, orig_val)
