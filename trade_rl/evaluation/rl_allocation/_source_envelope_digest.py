"""Run-local optimization of the source-envelope digest, never an authority.

Every hit rereads all non-cost envelope fields, array pins, and native cost
estimate fields. Unsupported graphs or changed implementations use the current
public digest. This is not protection against arbitrary Python process
tampering or concurrent graph mutation.
"""

from __future__ import annotations

import operator
import struct
from collections import defaultdict
from collections.abc import Callable
from dataclasses import MISSING, fields
from types import (
    BuiltinFunctionType,
    FunctionType,
    MethodDescriptorType,
    WrapperDescriptorType,
)
from typing import Any, cast

import numpy as np

from trade_rl.artifacts import canonical, hashing
from trade_rl.evaluation import forecast_allocation
from trade_rl.strategies import allocation as allocation_strategy

_CLASSES = (
    forecast_allocation.HorizonCostEstimates,
    allocation_strategy.AllocationInputs,
)
_ESTIMATE_FIELDS = (
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
)
_TEXT = ("symbol", "source_identity", "valuation_basis")
_CLOCK = ("decision_time", "available_at", "horizon_end")
_FLOAT = (
    "buy_cost",
    "sell_cost",
    "exit_cost",
    "funding_return",
    "borrow_return",
    "cash_return",
)
_PREFIX_KEYS = (
    "schema",
    "dataset_id",
    "dataset_contract",
    "symbol_index",
    "feature_indices",
    "feature_names",
    "range",
    "account_id",
    "forecast_context_digest",
)
_ENV_ATTRS = (
    "symbol_index",
    "feature_indices",
    "start_index",
    "stop_index",
    "account_id",
)
_GET = operator.attrgetter(*_ESTIMATE_FIELDS)
_DTYPE = operator.attrgetter("dtype")
_NS = np.dtype("datetime64[ns]")
_UNKNOWN = object()
_OBJECT_GETATTRIBUTE = vars(object)["__getattribute__"]


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
    if value is MISSING or isinstance(
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


def _entry(value: object) -> tuple[object, object]:
    try:
        return value, _metadata(value)
    except TypeError:
        return value, _UNKNOWN


_CLASS_CONTRACT = tuple(
    (
        cls,
        tuple(f.name for f in fields(cls)) if cls in _CLASSES else None,
        tuple((name, *_entry(value)) for name, value in vars(cls).items()),
    )
    for cls in (*_CLASSES, canonical.json.JSONEncoder)
)
_MODULES = (
    forecast_allocation,
    allocation_strategy,
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
        (np, "int64"),
        (np, "datetime64"),
        (np, "ndarray"),
        (np, "dtype"),
        (np, "asarray"),
        (np, "isnat"),
        (np, "isfinite"),
    )
)
_DIGEST_ENTRY = _entry(hashing.content_digest)
_DIGEST_FUNC = hashing.content_digest

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


def _implementation_matches() -> bool:
    for (
        cls,
        schema,
        expected_len,
        identities,
        simple_funcs,
        complex_funcs,
        others,
    ) in _OPTIMIZED_CLASS_CHECKS:
        c_dict = vars(cls)
        if len(c_dict) != expected_len:
            return False
        if (
            schema is not None
            and tuple(getattr(cls, "__dataclass_fields__", ())) != schema
        ):
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


def _exact(values: tuple[Any, ...], kind: type) -> bool:
    return set(map(type, values)) == {kind}


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


def _estimates_state(estimates: object) -> object:
    if type(estimates) is not dict or not estimates:
        raise TypeError("unsupported cost mapping")
    keys = tuple(dict.keys(estimates))
    if not _exact(keys, int):
        raise TypeError("unsupported cost keys")
    values = tuple(dict.values(estimates))
    if not _exact(values, forecast_allocation.HorizonCostEstimates):
        raise TypeError("unsupported cost values")
    if (
        tuple(
            getattr(
                forecast_allocation.HorizonCostEstimates, "__dataclass_fields__", ()
            )
        )
        != _ESTIMATE_FIELDS
    ):
        raise TypeError("unsupported cost estimate schema")
    is_sorted = all(keys[i] < keys[i + 1] for i in range(len(keys) - 1))
    ordered = (
        values
        if is_sorted
        else tuple(dict.__getitem__(estimates, key) for key in sorted(keys))
    )
    cols = tuple(zip(*map(_GET, ordered)))
    text = (cols[0], cols[4], cols[11])
    for col in text:
        if not _exact(col, str):
            raise TypeError("unsupported text")

    clock_cols = (cols[1], cols[2], cols[3])
    for col in clock_cols:
        if not _exact(col, np.datetime64):
            raise TypeError("unsupported clock")
        for x in col:
            if x.dtype != _NS:
                raise TypeError("unsupported clock")
    arr_c = np.asarray(clock_cols, dtype=_NS)
    if np.isnat(arr_c).any():
        raise TypeError("nat clock")
    raw_c = arr_c.view("<i8")
    clocks = (raw_c[0].tobytes(), raw_c[1].tobytes(), raw_c[2].tobytes())

    float_cols = (cols[5], cols[6], cols[7], cols[8], cols[9], cols[10])
    for col in float_cols:
        if not _exact(col, float):
            raise TypeError("unsupported float")
    arr_f = np.asarray(float_cols, dtype="<f8")
    if not np.isfinite(arr_f).all():
        raise TypeError("nonfinite float")
    floats = tuple(arr_f[i].tobytes() for i in range(6))

    return clocks, floats, text


def _state(
    env: Any,
    prefix: dict[str, Any],
    arrays: dict[str, Any],
    digest: Callable[[object], str],
) -> object | None:
    try:
        if (
            digest is not _DIGEST_FUNC
            or not _same(digest, *_DIGEST_ENTRY)
            or not _implementation_matches()
        ):
            return None
        if (
            type(prefix) is not dict
            or tuple(dict.keys(prefix)) != _PREFIX_KEYS
            or type(arrays) is not dict
        ):
            return None
        env_type = type(env)
        if vars(object).get("__getattribute__") is not _OBJECT_GETATTRIBUTE or any(
            (base is not object and "__getattribute__" in vars(base))
            or "_estimates" in vars(base)
            or "__getattr__" in vars(base)
            or any(attr in vars(base) for attr in _ENV_ATTRS)
            for base in env_type.__mro__
        ):
            return None
        env_dict = object.__getattribute__(env, "__dict__")
        if type(env_dict) is not dict or "_estimates" not in env_dict:
            return None
        if any(attr not in env_dict for attr in _ENV_ATTRS):
            return None
        feature_indices = dict.__getitem__(env_dict, "feature_indices")
        if type(feature_indices) is not tuple:
            return None
        symbol_index = dict.__getitem__(env_dict, "symbol_index")
        start_index = dict.__getitem__(env_dict, "start_index")
        stop_index = dict.__getitem__(env_dict, "stop_index")
        account_id = dict.__getitem__(env_dict, "account_id")

        prefix_range = dict.__getitem__(prefix, "range")
        prefix_fi = dict.__getitem__(prefix, "feature_indices")

        if (
            symbol_index != dict.__getitem__(prefix, "symbol_index")
            or type(prefix_fi) is not list
            or tuple(prefix_fi) != feature_indices
            or type(prefix_range) is not list
            or len(prefix_range) != 2
            or prefix_range[0] != start_index
            or prefix_range[1] != stop_index
            or account_id != dict.__getitem__(prefix, "account_id")
        ):
            return None
        estimates = dict.__getitem__(env_dict, "_estimates")
        return (
            _freeze_json(prefix),
            _freeze_json(arrays),
            _estimates_state(estimates),
        )
    except (AttributeError, TypeError, ValueError, OverflowError):
        return None


def _public_digest(
    env: Any,
    prefix: dict[str, Any],
    arrays: dict[str, Any],
    *,
    digest: Callable[[object], str] = hashing.content_digest,
) -> str:
    return digest(
        {
            **prefix,
            "cost_beliefs": [
                env._estimates[key].payload() for key in sorted(env._estimates)
            ],
            "arrays": arrays,
        }
    )


class SourceEnvelopeDigestMemo:
    """A lazy fixed baseline; misses always use the current public serializer."""

    def __init__(self) -> None:
        self._attempted = False
        self._baseline: tuple[object, str] | None = None

    def get(
        self,
        env: Any,
        prefix: dict[str, Any],
        arrays: dict[str, Any],
        *,
        digest: Callable[[object], str] = hashing.content_digest,
    ) -> str:
        before = _state(env, prefix, arrays, digest)
        if self._attempted:
            if (
                self._baseline is not None
                and before is not None
                and before == self._baseline[0]
            ):
                return self._baseline[1]
            return _public_digest(env, prefix, arrays, digest=digest)
        self._attempted = True
        value = _public_digest(env, prefix, arrays, digest=digest)
        if before is not None and _state(env, prefix, arrays, digest) == before:
            self._baseline = before, value
        return value
