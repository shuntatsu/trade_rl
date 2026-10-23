"""Run-local optimization of declared source digest, never an authority.

Every hit rereads all declared source arrays, contract payload, dataset ID,
and range. Unsupported graphs or changed implementations use the current
public digest. This is not protection against arbitrary Python process
tampering or concurrent graph mutation.
"""

from __future__ import annotations

import hashlib
import struct
from collections import defaultdict
from collections.abc import Callable
from types import (
    BuiltinFunctionType,
    FunctionType,
    MethodDescriptorType,
    WrapperDescriptorType,
)
from typing import Any, cast

import numpy as np

from trade_rl.artifacts import canonical, hashing


def allocation_array_pin(values: np.ndarray) -> dict[str, Any]:
    return {
        "dtype": values.dtype.str,
        "shape": list(values.shape),
        "sha256": hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest(),
    }


def _defaults(value: object) -> object:
    kind = type(value)
    if kind in (str, int, bool, type(None)):
        return kind, value
    if kind is float:
        return kind, struct.pack("<d", value)
    if kind in (tuple, list):
        return kind, tuple(_defaults(v) for v in cast(tuple[object, ...], value))
    if kind is dict:
        return kind, tuple(
            (_defaults(k), _defaults(v))
            for k, v in cast(dict[object, object], value).items()
        )
    if isinstance(
        value,
        (
            type,
            FunctionType,
            BuiltinFunctionType,
            MethodDescriptorType,
            WrapperDescriptorType,
        ),
    ):
        return kind, value
    raise TypeError("unknown function default")


def _metadata(value: object) -> object:
    if isinstance(value, FunctionType):
        return (
            value.__code__,
            _defaults(value.__defaults__),
            _defaults(value.__kwdefaults__),
        )
    if isinstance(value, property):
        return tuple(_metadata(v) for v in (value.fget, value.fset, value.fdel))
    if isinstance(value, (classmethod, staticmethod)):
        return _metadata(value.__func__)
    return None


_UNKNOWN = object()


def _entry(value: object) -> tuple[object, object]:
    try:
        return value, _metadata(value)
    except TypeError:
        return value, _UNKNOWN


_CLASS_CONTRACT = tuple(
    (
        cls,
        None,
        tuple((name, *_entry(value)) for name, value in vars(cls).items()),
    )
    for cls in (canonical.json.JSONEncoder,)
)

_MODULES = (
    hashing,
    canonical,
    canonical.json,
    canonical.json.encoder,
)

_BINDINGS = tuple(
    (module, name, *_entry(value))
    for module in _MODULES
    for name, value in vars(module).items()
    if not name.startswith("__")
) + tuple(
    (module, name, *_entry(getattr(module, name)))
    for module, name in (
        (canonical.math, "isfinite"),
        (hashlib, "sha256"),
        (np, "ndarray"),
        (np, "dtype"),
        (np, "asarray"),
        (np, "isfinite"),
        (np, "ascontiguousarray"),
    )
)

_PIN_ENTRY = _entry(allocation_array_pin)
_DIGEST_ENTRY = _entry(hashing.content_digest)
_ORIGINAL_PIN = allocation_array_pin
_ORIGINAL_DIGEST = hashing.content_digest

_ComplexFuncCheck = tuple[
    str,
    FunctionType,
    Any,
    tuple[object, ...] | None,
    dict[str, object] | None,
    object,
    object,
]
_ClassCheck = tuple[
    type,
    tuple[str, ...] | None,
    int,
    tuple[tuple[str, object], ...],
    tuple[tuple[str, FunctionType, Any], ...],
    tuple[_ComplexFuncCheck, ...],
    tuple[tuple[str, object, object], ...],
]
_ModuleCheck = tuple[
    Any,
    tuple[tuple[str, object], ...],
    tuple[tuple[str, FunctionType, Any], ...],
    tuple[_ComplexFuncCheck, ...],
    tuple[tuple[str, object, object], ...],
]


def _build_optimized_class_checks() -> tuple[_ClassCheck, ...]:
    checks: list[_ClassCheck] = []
    for cls, schema, entries in _CLASS_CONTRACT:
        identities: list[tuple[str, object]] = []
        simple_funcs: list[tuple[str, FunctionType, Any]] = []
        complex_funcs: list[_ComplexFuncCheck] = []
        others: list[tuple[str, object, object]] = []
        for name, orig, meta in entries:
            if meta is None:
                identities.append((name, orig))
            elif type(orig) is FunctionType and meta is not _UNKNOWN:
                orig_def = orig.__defaults__
                orig_kw = (
                    dict(orig.__kwdefaults__)
                    if orig.__kwdefaults__ is not None
                    else None
                )
                meta_tuple = (
                    meta
                    if isinstance(meta, tuple) and len(meta) == 3
                    else (None, None, None)
                )
                if orig_def is None and orig_kw is None:
                    simple_funcs.append((name, orig, orig.__code__))
                else:
                    complex_funcs.append(
                        (
                            name,
                            orig,
                            orig.__code__,
                            orig_def,
                            orig_kw,
                            meta_tuple[1],
                            meta_tuple[2],
                        )
                    )
            else:
                others.append((name, orig, meta))
        checks.append(
            (
                cls,
                schema,
                len(entries),
                tuple(identities),
                tuple(simple_funcs),
                tuple(complex_funcs),
                tuple(others),
            )
        )
    return tuple(checks)


def _build_optimized_module_checks() -> tuple[_ModuleCheck, ...]:
    grouped: dict[Any, list[tuple[str, object, object]]] = defaultdict(list)
    for module, name, orig, meta in _BINDINGS:
        grouped[module].append((name, orig, meta))
    checks: list[_ModuleCheck] = []
    for module, entries in grouped.items():
        identities: list[tuple[str, object]] = []
        simple_funcs: list[tuple[str, FunctionType, Any]] = []
        complex_funcs: list[_ComplexFuncCheck] = []
        others: list[tuple[str, object, object]] = []
        for name, orig, meta in entries:
            if meta is None:
                identities.append((name, orig))
            elif type(orig) is FunctionType and meta is not _UNKNOWN:
                orig_def = orig.__defaults__
                orig_kw = (
                    dict(orig.__kwdefaults__)
                    if orig.__kwdefaults__ is not None
                    else None
                )
                meta_tuple = (
                    meta
                    if isinstance(meta, tuple) and len(meta) == 3
                    else (None, None, None)
                )
                if orig_def is None and orig_kw is None:
                    simple_funcs.append((name, orig, orig.__code__))
                else:
                    complex_funcs.append(
                        (
                            name,
                            orig,
                            orig.__code__,
                            orig_def,
                            orig_kw,
                            meta_tuple[1],
                            meta_tuple[2],
                        )
                    )
            else:
                others.append((name, orig, meta))
        checks.append(
            (
                module,
                tuple(identities),
                tuple(simple_funcs),
                tuple(complex_funcs),
                tuple(others),
            )
        )
    return tuple(checks)


_OPTIMIZED_CLASS_CHECKS = _build_optimized_class_checks()
_OPTIMIZED_MODULE_CHECKS = _build_optimized_module_checks()


def _same(value: object, original: object, metadata: object) -> bool:
    if value is not original or metadata is _UNKNOWN:
        return False
    if metadata is None:
        return True
    return _metadata(value) == metadata


def _implementation_matches(
    pin: Callable[..., Any], digest: Callable[..., Any]
) -> bool:
    if pin is not _ORIGINAL_PIN or not _same(pin, *_PIN_ENTRY):
        return False
    if digest is not _ORIGINAL_DIGEST or not _same(digest, *_DIGEST_ENTRY):
        return False
    if allocation_array_pin is not _ORIGINAL_PIN or not _same(
        allocation_array_pin, *_PIN_ENTRY
    ):
        return False

    for (
        cls,
        _schema,
        expected_len,
        identities,
        simple_funcs,
        complex_funcs,
        others,
    ) in _OPTIMIZED_CLASS_CHECKS:
        c_dict = vars(cls)
        if len(c_dict) != expected_len:
            return False
        for name, expected in identities:
            if c_dict.get(name) is not expected:
                return False
        for name, orig, code in simple_funcs:
            val = c_dict.get(name)
            if (
                val is not orig
                or getattr(val, "__code__", None) is not code
                or getattr(val, "__defaults__", None) is not None
                or getattr(val, "__kwdefaults__", None) is not None
            ):
                return False
        for (
            name,
            orig,
            code,
            orig_def,
            orig_kw,
            d_frozen,
            kw_frozen,
        ) in complex_funcs:
            val = c_dict.get(name)
            if val is not orig or getattr(val, "__code__", None) is not code:
                return False
            val_def = getattr(val, "__defaults__", None)
            if orig_def is None:
                if val_def is not None:
                    return False
            elif val_def is not orig_def and (
                type(val_def) is not tuple or _defaults(val_def) != d_frozen
            ):
                return False
            val_kw = getattr(val, "__kwdefaults__", None)
            if orig_kw is None:
                if val_kw is not None:
                    return False
            elif val_kw != orig_kw:
                return False
        for other_name, other_orig, meta in others:
            if not _same(c_dict.get(other_name), other_orig, meta):
                return False

    for (
        module,
        identities,
        simple_funcs,
        complex_funcs,
        others,
    ) in _OPTIMIZED_MODULE_CHECKS:
        m_dict = vars(module)
        for name, expected in identities:
            if m_dict.get(name) is not expected:
                return False
        for name, orig, code in simple_funcs:
            val = m_dict.get(name)
            if (
                val is not orig
                or getattr(val, "__code__", None) is not code
                or getattr(val, "__defaults__", None) is not None
                or getattr(val, "__kwdefaults__", None) is not None
            ):
                return False
        for (
            name,
            orig,
            code,
            orig_def,
            orig_kw,
            d_frozen,
            kw_frozen,
        ) in complex_funcs:
            val = m_dict.get(name)
            if val is not orig or getattr(val, "__code__", None) is not code:
                return False
            val_def = getattr(val, "__defaults__", None)
            if orig_def is None:
                if val_def is not None:
                    return False
            elif val_def is not orig_def and (
                type(val_def) is not tuple or _defaults(val_def) != d_frozen
            ):
                return False
            val_kw = getattr(val, "__kwdefaults__", None)
            if orig_kw is None:
                if val_kw is not None:
                    return False
            elif val_kw != orig_kw:
                return False
        for other_name, other_orig, meta in others:
            if not _same(m_dict.get(other_name), other_orig, meta):
                return False
    return True


_OBJECT_GETATTRIBUTE = vars(object)["__getattribute__"]
_ENV_ATTRS = ("dataset", "start_index", "stop_index")
_DATASET_ATTRS = (
    "identity_arrays",
    "identity_contract_payload",
    "dataset_id",
    "n_bars",
)


def _is_standard_env(env: Any) -> bool:
    if vars(object).get("__getattribute__") is not _OBJECT_GETATTRIBUTE:
        return False
    env_type = type(env)
    for base in env_type.__mro__:
        if base is not object:
            b_dict = vars(base)
            if (
                "__getattribute__" in b_dict
                or "__getattr__" in b_dict
                or any(attr in b_dict for attr in _ENV_ATTRS)
            ):
                return False
    env_dict = object.__getattribute__(env, "__dict__")
    if type(env_dict) is not dict:
        return False
    if any(attr not in env_dict for attr in _ENV_ATTRS):
        return False
    if (
        type(dict.__getitem__(env_dict, "start_index")) is not int
        or type(dict.__getitem__(env_dict, "stop_index")) is not int
    ):
        return False
    return True


def _is_standard_dataset(dataset: Any) -> bool:
    if vars(object).get("__getattribute__") is not _OBJECT_GETATTRIBUTE:
        return False
    ds_type = type(dataset)
    for base in ds_type.__mro__:
        if base is not object:
            b_dict = vars(base)
            if "__getattribute__" in b_dict or "__getattr__" in b_dict:
                return False
            for attr in ("identity_arrays", "identity_contract_payload"):
                if attr in b_dict and not isinstance(
                    b_dict[attr],
                    (
                        FunctionType,
                        BuiltinFunctionType,
                        MethodDescriptorType,
                        WrapperDescriptorType,
                    ),
                ):
                    return False
            if "dataset_id" in b_dict:
                val = b_dict["dataset_id"]
                if isinstance(val, property) or (
                    hasattr(val, "__get__")
                    and type(val).__name__ != "member_descriptor"
                ):
                    return False
            if "n_bars" in b_dict:
                val = b_dict["n_bars"]
                if (
                    hasattr(val, "__get__")
                    and not isinstance(val, (property, FunctionType))
                    and type(val).__name__ != "member_descriptor"
                ):
                    return False
    try:
        ds_dict = object.__getattribute__(dataset, "__dict__")
    except AttributeError:
        ds_dict = None
    if ds_dict is not None:
        if type(ds_dict) is not dict:
            return False
        if any(
            attr in ds_dict
            for attr in (
                "identity_arrays",
                "identity_contract_payload",
                "dataset_id",
                "n_bars",
            )
        ):
            return False
    dataset_id = getattr(dataset, "dataset_id", None)
    n_bars = getattr(dataset, "n_bars", None)
    if type(dataset_id) is not str:
        return False
    if type(n_bars) is not int:
        return False
    return True


def _freeze_json(value: Any) -> object:
    kind = type(value)
    if kind is str:
        return str, value
    if kind is int:
        return int, value
    if kind is dict:
        items = []
        for k in sorted(dict.keys(value)):
            if type(k) is not str:
                raise TypeError("non-string dict key")
            items.append((k, _freeze_json(dict.__getitem__(value, k))))
        return dict, tuple(items)
    if kind is list or kind is tuple:
        return kind, tuple(_freeze_json(v) for v in value)
    if kind is bool or kind is type(None):
        return kind, value
    if kind is float:
        if not canonical.math.isfinite(value):
            raise TypeError("nonfinite float")
        return float, struct.pack("<d", value)
    raise TypeError("unsupported json value")


def _public_declared_source_digest(
    env: Any,
    *,
    pin: Callable[[np.ndarray], dict[str, Any]] | None = None,
    digest: Callable[[object], str] | None = None,
) -> str:
    actual_pin = allocation_array_pin if pin is None else pin
    actual_digest = hashing.content_digest if digest is None else digest
    dataset = env.dataset
    arrays = {}
    for name, values in dataset.identity_arrays().items():
        if name == "global_features":
            continue
        selected = values
        if values.ndim > 0 and values.shape[0] == dataset.n_bars:
            selected = values[env.start_index : env.stop_index + 1]
        if not np.isfinite(selected).all():
            raise ValueError("declared global source arrays must be finite")
        arrays[name] = actual_pin(selected)
    return actual_digest(
        {
            "schema": "allocation_global_declared_source_v1",
            "dataset_contract": dataset.identity_contract_payload(),
            "dataset_id": dataset.dataset_id,
            "range": [env.start_index, env.stop_index],
            "arrays": arrays,
        }
    )


class DeclaredSourceDigestMemo:
    """A lazy fixed baseline; misses always use the current public serializer."""

    def __init__(self) -> None:
        self._attempted = False
        self._baseline: tuple[object, str] | None = None

    def get(
        self,
        env: Any,
        *,
        pin: Callable[[np.ndarray], dict[str, Any]] | None = None,
        digest: Callable[[object], str] | None = None,
    ) -> str:
        actual_pin = allocation_array_pin if pin is None else pin
        actual_digest = hashing.content_digest if digest is None else digest
        dataset = env.dataset
        fast_path = (
            _implementation_matches(actual_pin, actual_digest)
            and _is_standard_env(env)
            and _is_standard_dataset(dataset)
        )

        raw_identity = dataset.identity_arrays()
        if fast_path and type(raw_identity) is not dict:
            fast_path = False

        arrays: dict[str, Any] = {}
        array_snapshots: list[tuple[str, str, tuple[int, ...], bytes]] = []
        selected_refs: list[np.ndarray] = []

        for name, values in raw_identity.items():
            if name == "global_features":
                continue
            selected = values
            if values.ndim > 0 and values.shape[0] == dataset.n_bars:
                selected = values[env.start_index : env.stop_index + 1]
            if not np.isfinite(selected).all():
                raise ValueError("declared global source arrays must be finite")

            if fast_path:
                if (
                    type(name) is not str
                    or type(values) is not np.ndarray
                    or type(selected) is not np.ndarray
                ):
                    fast_path = False
                    for s_name, s_dtype, s_shape, s_bytes in array_snapshots:
                        arrays[s_name] = {
                            "dtype": s_dtype,
                            "shape": list(s_shape),
                            "sha256": hashlib.sha256(s_bytes).hexdigest(),
                        }
                    arrays[name] = actual_pin(selected)
                else:
                    raw_bytes = np.ascontiguousarray(selected).tobytes()
                    array_snapshots.append(
                        (name, selected.dtype.str, tuple(selected.shape), raw_bytes)
                    )
                    selected_refs.append(selected)
            else:
                arrays[name] = actual_pin(selected)

        contract = dataset.identity_contract_payload()
        dataset_id = dataset.dataset_id
        range_val = [env.start_index, env.stop_index]

        if fast_path:
            try:
                state = (
                    _freeze_json(contract),
                    _freeze_json(dataset_id),
                    _freeze_json(range_val),
                    tuple(array_snapshots),
                )
            except TypeError:
                fast_path = False
                for s_name, s_dtype, s_shape, s_bytes in array_snapshots:
                    arrays[s_name] = {
                        "dtype": s_dtype,
                        "shape": list(s_shape),
                        "sha256": hashlib.sha256(s_bytes).hexdigest(),
                    }

        if fast_path:
            if self._baseline is not None and state == self._baseline[0]:
                return self._baseline[1]

            computed_arrays = {
                name: {
                    "dtype": s_dtype,
                    "shape": list(s_shape),
                    "sha256": hashlib.sha256(s_bytes).hexdigest(),
                }
                for name, s_dtype, s_shape, s_bytes in array_snapshots
            }
            value = actual_digest(
                {
                    "schema": "allocation_global_declared_source_v1",
                    "dataset_contract": contract,
                    "dataset_id": dataset_id,
                    "range": range_val,
                    "arrays": computed_arrays,
                }
            )
            if not self._attempted:
                self._attempted = True
                is_stable = True
                env_dict = getattr(env, "__dict__", None)
                if type(env_dict) is dict:
                    if (
                        env_dict.get("start_index") != range_val[0]
                        or env_dict.get("stop_index") != range_val[1]
                        or env_dict.get("dataset") is not dataset
                    ):
                        is_stable = False
                else:
                    if [
                        getattr(env, "start_index", None),
                        getattr(env, "stop_index", None),
                    ] != range_val:
                        is_stable = False

                ds_dict = getattr(dataset, "__dict__", None)
                if type(ds_dict) is dict:
                    if (
                        ds_dict.get("dataset_id") != dataset_id
                        or ds_dict.get("n_bars") != dataset.n_bars
                    ):
                        is_stable = False
                else:
                    if getattr(dataset, "dataset_id", None) != dataset_id:
                        is_stable = False

                if is_stable:
                    for sel, (s_name, s_dtype, s_shape, s_bytes) in zip(
                        selected_refs, array_snapshots, strict=True
                    ):
                        if (
                            sel.dtype.str != s_dtype
                            or tuple(sel.shape) != s_shape
                            or np.ascontiguousarray(sel).tobytes() != s_bytes
                        ):
                            is_stable = False
                            break
                        cur_arr = getattr(dataset, s_name, None)
                        if isinstance(cur_arr, np.ndarray) and cur_arr is not sel:
                            cur_sel = (
                                cur_arr[env.start_index : env.stop_index + 1]
                                if cur_arr.ndim > 0
                                and cur_arr.shape[0] == dataset.n_bars
                                else cur_arr
                            )
                            if (
                                cur_sel.dtype.str != s_dtype
                                or tuple(cur_sel.shape) != s_shape
                                or np.ascontiguousarray(cur_sel).tobytes() != s_bytes
                            ):
                                is_stable = False
                                break
                if is_stable:
                    self._baseline = state, value
            return value

        value = actual_digest(
            {
                "schema": "allocation_global_declared_source_v1",
                "dataset_contract": contract,
                "dataset_id": dataset_id,
                "range": range_val,
                "arrays": arrays,
            }
        )
        if not self._attempted:
            self._attempted = True
        return value


def _declared_source_digest(
    env: Any,
    memo: DeclaredSourceDigestMemo | None = None,
) -> str:
    if memo is None:
        return _public_declared_source_digest(env)
    return memo.get(env)
