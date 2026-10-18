"""Live identity and compatibility contracts for the declared-source memo."""

from __future__ import annotations

import hashlib
import importlib
import sys
from collections import OrderedDict

import numpy as np
import pytest

from tests.evaluation.test_allocation_nonrl_walk_forward import global_control_fixture
from trade_rl.artifacts import canonical, hashing
from trade_rl.evaluation.rl_allocation import global_execution_context as context


def memo_module():
    return importlib.import_module(
        "trade_rl.evaluation.rl_allocation._declared_source_digest"
    )


def api():
    return memo_module().DeclaredSourceDigestMemo


@pytest.fixture
def env():
    return global_control_fixture()


def measured_calls(fn):
    code = hashing.content_digest.__code__
    digest_calls = 0

    def profile(frame, event, _arg):
        nonlocal digest_calls
        if event == "call" and frame.f_code is code:
            digest_calls += 1

    previous = sys.getprofile()
    sys.setprofile(profile)
    try:
        result = fn()
    finally:
        sys.setprofile(previous)
    return result, digest_calls


def test_lazy_baseline_and_warm_hit_skips_digest_and_array_pins(env):
    memo = api()()
    assert not memo._attempted
    assert memo._baseline is None

    expected = context.global_declared_source_digest(env)
    result, digest_calls = measured_calls(lambda: memo.get(env))
    assert result == expected
    assert digest_calls == 1
    assert memo._attempted
    assert memo._baseline is not None

    result2, digest_calls2 = measured_calls(lambda: memo.get(env))
    assert result2 == expected
    assert digest_calls2 == 0


def test_equivalent_array_replacement_hits_memo(env):
    memo = api()()
    expected = context.global_declared_source_digest(env)
    assert memo.get(env) == expected

    object.__setattr__(env.dataset, "close", env.dataset.close.copy())
    result, digest_calls = measured_calls(lambda: memo.get(env))
    assert result == expected
    assert digest_calls == 0


def test_miss_recovery_and_bounded_memory(env):
    memo = api()()
    original_expected = context.global_declared_source_digest(env)
    assert memo.get(env) == original_expected
    baseline = memo._baseline
    assert baseline is not None

    orig_arr = env.dataset.close
    new_arr = orig_arr.copy()
    new_arr[env.start_index, 0] += 0.5
    object.__setattr__(env.dataset, "close", new_arr)

    mutated_expected = context.global_declared_source_digest(env)
    assert mutated_expected != original_expected

    mutated_result, digest_calls = measured_calls(lambda: memo.get(env))
    assert mutated_result == mutated_expected
    assert digest_calls == 1
    # Miss must not overwrite baseline
    assert memo._baseline == baseline
    assert vars(memo).keys() == {"_attempted", "_baseline"}

    # Restoring original array restores warm hit
    object.__setattr__(env.dataset, "close", orig_arr)
    recovered_result, digest_calls2 = measured_calls(lambda: memo.get(env))
    assert recovered_result == original_expected
    assert digest_calls2 == 0
    assert vars(memo).keys() == {"_attempted", "_baseline"}


def test_live_array_mutations_match_public_digest(env):
    memo = api()()
    original = memo.get(env)

    identity_arrays = env.dataset.identity_arrays()
    for name, arr in identity_arrays.items():
        if name == "global_features":
            continue
        # Verify warm hit before mutation
        assert memo.get(env) == original

        # Mutate array element via new copy
        new_arr = arr.copy()
        idx = (
            (env.start_index, 0)
            if new_arr.ndim > 1
            else (env.start_index if new_arr.shape[0] > env.start_index else 0)
        )
        if np.issubdtype(arr.dtype, np.floating):
            new_arr[idx] += 0.5
        elif np.issubdtype(arr.dtype, np.datetime64):
            new_arr[idx] += np.timedelta64(1, "ns")
        elif np.issubdtype(arr.dtype, np.bool_):
            new_arr[idx] = not new_arr[idx]
        elif np.issubdtype(arr.dtype, np.integer):
            new_arr[idx] += 1
        else:
            continue

        if name == "symbol_active":
            object.__setattr__(env.dataset, "symbol_active", new_arr)
            object.__setattr__(env.dataset, "asset_active", new_arr)
        else:
            object.__setattr__(env.dataset, name, new_arr)
        expected = context.global_declared_source_digest(env)
        assert expected != original, f"mutation on {name} had no effect on digest"
        # Cache miss on mutated array must match reference digest exactly
        assert memo.get(env) == expected

        # Restore array and verify warm hit recovery
        if name == "symbol_active":
            object.__setattr__(env.dataset, "symbol_active", arr)
            object.__setattr__(env.dataset, "asset_active", arr)
        else:
            object.__setattr__(env.dataset, name, arr)
        assert memo.get(env) == original


def test_signed_zero_distinguished_and_matches_reference(env):
    memo = api()()
    original = memo.get(env)

    # In funding_rate, 0.0 -> -0.0
    new_funding = env.dataset.funding_rate.copy()
    new_funding[env.start_index, 0] = -0.0
    object.__setattr__(env.dataset, "funding_rate", new_funding)
    expected = context.global_declared_source_digest(env)
    assert expected != original
    assert memo.get(env) == expected


def test_shape_and_dtype_changes_match_reference(env):
    memo = api()()
    original = memo.get(env)

    # Dtype change
    object.__setattr__(env.dataset, "close", env.dataset.close.astype(np.float32))
    expected = context.global_declared_source_digest(env)
    assert expected != original
    assert memo.get(env) == expected

    # Range / shape change
    memo2 = api()()
    assert memo2.get(env) == expected
    env.stop_index -= 1
    expected2 = context.global_declared_source_digest(env)
    assert expected2 != expected
    assert memo2.get(env) == expected2


def test_nan_and_inf_rejection_parity(env):
    memo = api()()
    memo.get(env)  # arm baseline

    # NaN in float array
    orig_close = env.dataset.close
    nan_close = orig_close.copy()
    nan_close[env.start_index, 0] = np.nan
    object.__setattr__(env.dataset, "close", nan_close)
    with pytest.raises(
        ValueError, match="^declared global source arrays must be finite$"
    ):
        memo.get(env)

    # Inf in float array
    inf_close = orig_close.copy()
    inf_close[env.start_index, 0] = np.inf
    object.__setattr__(env.dataset, "close", inf_close)
    with pytest.raises(
        ValueError, match="^declared global source arrays must be finite$"
    ):
        memo.get(env)

    # NaT in timestamps array
    object.__setattr__(env.dataset, "close", orig_close)
    orig_ts = env.dataset.timestamps
    nat_ts = orig_ts.copy()
    nat_ts[env.start_index] = np.datetime64("NaT", "ns")
    object.__setattr__(env.dataset, "timestamps", nat_ts)
    with pytest.raises(
        ValueError, match="^declared global source arrays must be finite$"
    ):
        memo.get(env)
    object.__setattr__(env.dataset, "timestamps", orig_ts)


def test_contract_and_dataset_id_changes_match_reference(env):
    memo = api()()
    original = memo.get(env)

    # dataset_id change
    object.__setattr__(env.dataset, "dataset_id", "0" * 64)
    expected = context.global_declared_source_digest(env)
    assert expected != original
    assert memo.get(env) == expected

    # contract payload change
    object.__setattr__(env.dataset, "feature_config_digest", "1" * 64)
    expected2 = context.global_declared_source_digest(env)
    assert expected2 != expected
    assert memo.get(env) == expected2


def test_array_addition_and_removal_match_reference(env):
    memo = api()()
    original = memo.get(env)
    orig_ds = env.dataset

    class AddedDataset:
        def __init__(self, ds):
            self._ds = ds
            self.dataset_id = ds.dataset_id
            self.n_bars = ds.n_bars

        def identity_arrays(self):
            d = self._ds.identity_arrays()
            d["extra_array"] = np.ones((self._ds.n_bars, 1), dtype=np.float64)
            return d

        def identity_contract_payload(self):
            return self._ds.identity_contract_payload()

    env.dataset = AddedDataset(orig_ds)
    expected = context.global_declared_source_digest(env)
    assert expected != original
    assert memo.get(env) == expected

    class RemovedDataset:
        def __init__(self, ds):
            self._ds = ds
            self.dataset_id = ds.dataset_id
            self.n_bars = ds.n_bars

        def identity_arrays(self):
            d = self._ds.identity_arrays()
            del d["volume"]
            return d

        def identity_contract_payload(self):
            return self._ds.identity_contract_payload()

    env.dataset = RemovedDataset(orig_ds)
    expected2 = context.global_declared_source_digest(env)
    assert expected2 != original
    assert expected2 != expected
    assert memo.get(env) == expected2
    env.dataset = orig_ds


def test_unsupported_mapping_and_subclass_fallback(env):
    memo = api()()
    orig_ds = env.dataset

    class OrderedDataset:
        def __init__(self, ds):
            self._ds = ds
            self.dataset_id = ds.dataset_id
            self.n_bars = ds.n_bars

        def identity_arrays(self):
            return OrderedDict(self._ds.identity_arrays())

        def identity_contract_payload(self):
            return self._ds.identity_contract_payload()

    env.dataset = OrderedDataset(orig_ds)
    expected = context.global_declared_source_digest(env)
    assert memo.get(env) == expected

    class CustomArray(np.ndarray):
        pass

    class SubclassDataset:
        def __init__(self, ds):
            self._ds = ds
            self.dataset_id = ds.dataset_id
            self.n_bars = ds.n_bars

        def identity_arrays(self):
            d = self._ds.identity_arrays()
            d["close"] = d["close"].view(CustomArray)
            return d

        def identity_contract_payload(self):
            return self._ds.identity_contract_payload()

    env.dataset = SubclassDataset(orig_ds)
    expected2 = context.global_declared_source_digest(env)
    assert memo.get(env) == expected2
    env.dataset = orig_ds


def test_counting_dataset_single_pass_on_cold_warm_and_miss(env):
    memo = api()()
    orig_ds = env.dataset

    class CountingDataset:
        def __init__(self, ds):
            self._ds = ds
            self.dataset_id = ds.dataset_id
            self.n_bars = ds.n_bars
            self.arrays_calls = 0
            self.contract_calls = 0

        def identity_arrays(self):
            self.arrays_calls += 1
            return self._ds.identity_arrays()

        def identity_contract_payload(self):
            self.contract_calls += 1
            return self._ds.identity_contract_payload()

    counting_ds = CountingDataset(orig_ds)
    env.dataset = counting_ds

    # Reference call (outside counting)
    ref_cold = context.global_declared_source_digest(env)
    counting_ds.arrays_calls = 0
    counting_ds.contract_calls = 0

    # 1. Cold call increments calls by exactly 1 (never 2)
    memo_cold = memo.get(env)
    assert memo_cold == ref_cold
    assert counting_ds.arrays_calls == 1
    assert counting_ds.contract_calls == 1

    # 2. Warm hit increments calls by exactly 1
    memo_warm = memo.get(env)
    assert memo_warm == ref_cold
    assert counting_ds.arrays_calls == 2
    assert counting_ds.contract_calls == 2

    # 3. Cache miss increments calls by exactly 1
    counting_ds.dataset_id = "mutated_id_for_miss"
    ref_miss = context.global_declared_source_digest(env)
    counting_ds.arrays_calls = 2
    counting_ds.contract_calls = 2
    memo_miss = memo.get(env)
    assert memo_miss == ref_miss
    assert counting_ds.arrays_calls == 3
    assert counting_ds.contract_calls == 3

    env.dataset = orig_ds


def test_contract_payload_in_place_array_mutation_matches_reference_and_calls_once(
    env,
):
    memo = api()()
    orig_ds = env.dataset
    writable_close = orig_ds.close.copy()
    object.__setattr__(orig_ds, "close", writable_close)
    orig_close_bytes = writable_close.tobytes()
    arrays_called = 0
    contract_called = 0

    class MutatingDataset:
        def __init__(self, ds):
            self._ds = ds
            self.dataset_id = ds.dataset_id
            self.n_bars = ds.n_bars

        def identity_arrays(self):
            nonlocal arrays_called
            arrays_called += 1
            return self._ds.identity_arrays()

        def identity_contract_payload(self):
            nonlocal contract_called
            contract_called += 1
            self._ds.close[env.start_index, 0] += 0.5
            return self._ds.identity_contract_payload()

    env.dataset = MutatingDataset(orig_ds)

    # Reference calculation:
    ref_digest = context.global_declared_source_digest(env)
    assert arrays_called == 1 and contract_called == 1

    # Restore array, reset counters, run memo:
    writable_close[:] = np.frombuffer(
        orig_close_bytes, dtype=writable_close.dtype
    ).reshape(writable_close.shape)
    arrays_called = 0
    contract_called = 0
    memo_digest = memo.get(env)
    assert arrays_called == 1 and contract_called == 1
    assert memo_digest == ref_digest
    # Because identity_contract_payload mutated array in-place, arming is rejected
    assert memo._baseline is None

    env.dataset = orig_ds


def test_custom_mro_and_descriptors_on_env_and_dataset_fall_back_safely(env):
    orig_ds = env.dataset

    class DescriptorDataset:
        def __init__(self, ds):
            self._ds = ds
            self.n_bars = ds.n_bars

        @property
        def dataset_id(self):
            return self._ds.dataset_id

        def identity_arrays(self):
            return self._ds.identity_arrays()

        def identity_contract_payload(self):
            return self._ds.identity_contract_payload()

    memo = api()()
    env.dataset = DescriptorDataset(orig_ds)
    expected = context.global_declared_source_digest(env)
    assert memo.get(env) == expected

    # Custom __getattr__ on env
    class CustomEnv:
        def __init__(self, inner):
            self._inner = inner
            self.dataset = inner.dataset
            self.start_index = inner.start_index
            self.stop_index = inner.stop_index

        def __getattr__(self, name):
            return getattr(self._inner, name)

    memo2 = api()()
    c_env = CustomEnv(env)
    expected2 = context.global_declared_source_digest(c_env)
    assert memo2.get(c_env) == expected2
    env.dataset = orig_ds


@pytest.mark.parametrize(
    "monkey_target",
    [
        "custom_pin",
        "custom_digest",
        "pin_binding",
        "digest_binding",
        "sha256_binding",
        "json_encoder",
        "isfinite",
        "ascontiguousarray",
    ],
)
def test_monkeypatch_detection_falls_back_to_public_path(
    env, monkey_target, monkeypatch
):
    memo = api()()
    expected = context.global_declared_source_digest(env)
    kwargs = {}

    if monkey_target == "custom_pin":
        custom_called = False

        def custom_pin(values):
            nonlocal custom_called
            custom_called = True
            return context.allocation_array_pin(values)

        kwargs["pin"] = custom_pin
        result = memo.get(env, **kwargs)
        assert result == expected
        assert custom_called
        return

    if monkey_target == "custom_digest":
        custom_called = False

        def custom_digest(payload):
            nonlocal custom_called
            custom_called = True
            return hashing.content_digest(payload)

        kwargs["digest"] = custom_digest
        result = memo.get(env, **kwargs)
        assert result == expected
        assert custom_called
        return

    if monkey_target == "pin_binding":
        mod = memo_module()
        monkeypatch.setattr(mod, "allocation_array_pin", lambda v: {"pinned": True})
        assert memo.get(env) == mod._public_declared_source_digest(env)
        return

    if monkey_target == "digest_binding":
        monkeypatch.setattr(
            hashing, "content_digest", lambda payload: "monkeypatched-digest"
        )
        assert memo.get(env) == "monkeypatched-digest"
        return

    if monkey_target == "sha256_binding":
        orig_sha = hashlib.sha256
        monkeypatch.setattr(
            hashlib, "sha256", lambda *args, **kw: orig_sha(*args, **kw)
        )
        assert memo.get(env) == expected
        return

    if monkey_target == "json_encoder":
        monkeypatch.setattr(canonical.json.JSONEncoder, "default", lambda self, o: None)
        assert memo.get(env) == expected
        return

    if monkey_target == "isfinite":
        monkeypatch.setattr(np, "isfinite", lambda x: np.ones_like(x, dtype=bool))
        assert memo.get(env) == expected
        return

    if monkey_target == "ascontiguousarray":
        monkeypatch.setattr(np, "ascontiguousarray", lambda x: np.array(x, copy=True))
        assert memo.get(env) == expected
        return


def test_unstable_first_evaluation_is_not_cached(env):
    memo = api()()
    code = hashing.content_digest.__code__
    changed = False

    def mutate(frame, event, _arg):
        nonlocal changed
        if event == "call" and frame.f_code is code and not changed:
            changed = True
            new_close = env.dataset.close.copy()
            new_close[env.start_index, 0] += 0.5
            object.__setattr__(env.dataset, "close", new_close)

    previous = sys.getprofile()
    try:
        sys.setprofile(mutate)
        memo.get(env)
    finally:
        sys.setprofile(previous)

    assert memo._baseline is None
    result, digest_calls = measured_calls(lambda: memo.get(env))
    assert result == context.global_declared_source_digest(env)
    assert digest_calls == 1
