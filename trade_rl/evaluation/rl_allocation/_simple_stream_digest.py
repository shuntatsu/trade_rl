"""Run-local optimization of the known native serializer, never an authority.

Every hit rereads all fields and logical array bytes. Unsupported graphs or
changed implementations use the current public digest. This is not protection
against arbitrary Python process tampering or concurrent graph mutation.
"""

from __future__ import annotations

import operator
import struct
from collections import defaultdict
from dataclasses import MISSING, fields
from itertools import chain
from types import (
    BuiltinFunctionType,
    FunctionType,
    MethodDescriptorType,
    WrapperDescriptorType,
)
from typing import Any, cast

import numpy as np

from trade_rl.artifacts import canonical, hashing
from trade_rl.strategies.forecasts import (
    simple_return,
    simple_stream,
    supervised,
    training_trace,
)
from trade_rl.strategies.forecasts import (
    stream as forecast_stream,
)

_CLASSES = (
    simple_stream.FrozenSimpleReturnStream,
    simple_stream.SimpleReturnPacket,
    simple_stream.SimpleReturnVintage,
    forecast_stream.ForecastBlock,
    simple_return.SimpleReturnTrainingSet,
    simple_return.SimpleReturnRidgeModel,
    supervised.CausalForecastTrainingSet,
    training_trace.ForecastTrainingTrace,
)
_PACKET_FIELDS = (
    "symbol",
    "as_of",
    "source_available_at",
    "forecast_available_at",
    "horizon_end",
    "horizon_seconds",
    "expected_simple_return",
    "fit_prefix_marginal_variance",
    "vintage_digest",
    "feature_values",
    "decision_close",
    "return_unit",
    "valuation_basis",
    "variance_kind",
    "forecast_kind",
)
_TEXT = (
    "symbol",
    "vintage_digest",
    "return_unit",
    "valuation_basis",
    "variance_kind",
    "forecast_kind",
)
_CLOCK = ("as_of", "source_available_at", "forecast_available_at", "horizon_end")
_FLOAT = ("expected_simple_return", "fit_prefix_marginal_variance", "decision_close")
_GET = operator.attrgetter(*_PACKET_FIELDS)
_DTYPE = operator.attrgetter("dtype")
_NS = np.dtype("datetime64[ns]")
_UNKNOWN = object()


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


# Retain the original descriptors/functions strongly and freeze metadata now,
# not when a later memo is constructed after an unreviewed monkeypatch.
_CLASS_CONTRACT = tuple(
    (
        cls,
        tuple(f.name for f in fields(cls)) if cls in _CLASSES else None,
        tuple((name, *_entry(value)) for name, value in vars(cls).items()),
    )
    for cls in (*_CLASSES, canonical.json.JSONEncoder)
)
_MODULES = (
    simple_stream,
    simple_return,
    forecast_stream,
    supervised,
    training_trace,
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
        (np, "fromiter"),
    )
)


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


def _freeze(value: Any) -> object:
    kind = type(value)
    if kind in (str, int, bool, type(None)):
        return kind, value
    if kind is float:
        return kind, struct.pack("<d", value)
    if kind is np.datetime64:
        return kind, value.dtype.str, value.tobytes()
    if kind is np.ndarray:
        dtype = value.dtype
        if (
            dtype.hasobject
            or dtype.fields is not None
            or dtype.subdtype is not None
            or dtype.metadata is not None
            or dtype.kind not in "biufMm"
        ):
            raise TypeError("unsupported array dtype")
        return kind, dtype.str, value.shape, value.strides, value.tobytes(order="C")
    if kind is tuple:
        if value and _exact(value, str):
            return kind, str, value
        return kind, tuple(_freeze(v) for v in value)
    if kind in _CLASSES:
        return kind, tuple(
            (f.name, _freeze(getattr(value, f.name))) for f in fields(value)
        )
    raise TypeError("unsupported type or subclass")


def _packets(values: object) -> object:
    if type(values) is not tuple or not _exact(
        values, simple_stream.SimpleReturnPacket
    ):
        raise TypeError("unsupported packets")
    if (
        tuple(f.name for f in fields(simple_stream.SimpleReturnPacket))
        != _PACKET_FIELDS
    ):
        raise TypeError("unsupported packet schema")
    columns = dict(zip(_PACKET_FIELDS, zip(*map(_GET, values)), strict=True))
    text = tuple(columns[name] for name in _TEXT)
    if any(not _exact(column, str) for column in text):
        raise TypeError("unsupported text")
    numeric = []
    for name in _CLOCK:
        column = columns[name]
        if not _exact(column, np.datetime64):
            raise TypeError("unsupported clock")
        for x in column:
            if x.dtype != _NS:
                raise TypeError("unsupported clock")
        numeric.append(np.asarray(column, dtype=_NS).view("<i8").tobytes())
    if not _exact(columns["horizon_seconds"], int):
        raise TypeError("unsupported integer")
    numeric.append(np.asarray(columns["horizon_seconds"], dtype="<i8").tobytes())
    for name in _FLOAT:
        if not _exact(columns[name], float):
            raise TypeError("unsupported float")
        numeric.append(np.asarray(columns[name], dtype="<f8").tobytes())
    features = columns["feature_values"]
    if not _exact(features, tuple) or set(map(type, chain.from_iterable(features))) != {
        float
    }:
        raise TypeError("unsupported feature values")
    lengths = np.fromiter(map(len, features), dtype="<i8", count=len(features))
    if lengths.min() <= 0:
        raise TypeError("empty features")
    flat = np.fromiter(
        chain.from_iterable(features), dtype="<f8", count=int(lengths.sum())
    )
    flat_fields = tuple(chain.from_iterable(map(_GET, values)))
    return tuple(numeric), lengths.tobytes(), flat.tobytes(), text, flat_fields


def _state(stream: simple_stream.FrozenSimpleReturnStream) -> object | None:
    try:
        if (
            type(stream) is not simple_stream.FrozenSimpleReturnStream
            or not _implementation_matches()
        ):
            return None
        if tuple(f.name for f in fields(stream)) != (
            "dataset_id",
            "vintages",
            "packets",
        ):
            return None
        return (
            _freeze(stream.dataset_id),
            _freeze(stream.vintages),
            _packets(stream.packets),
        )
    except (AttributeError, TypeError, ValueError, OverflowError):
        return None


class SimpleStreamDigestMemo:
    """A lazy fixed baseline; misses always use the current public serializer."""

    def __init__(self) -> None:
        self._attempted = False
        self._baseline: tuple[object, str] | None = None

    def get(self, stream: simple_stream.FrozenSimpleReturnStream) -> str:
        if self._attempted:
            if self._baseline is not None:
                b_state, b_value = self._baseline
                b_state_t = cast(tuple[Any, ...], b_state)
                try:
                    if (
                        type(stream) is simple_stream.FrozenSimpleReturnStream
                        and _implementation_matches()
                        and stream.dataset_id == b_state_t[0][1]
                        and _freeze(stream.vintages) == b_state_t[1]
                    ):
                        packets = stream.packets
                        b_packets_state = b_state_t[2]
                        b_flat_fields = b_packets_state[4]
                        if type(packets) is tuple and len(packets) * len(
                            _PACKET_FIELDS
                        ) == len(b_flat_fields):
                            if set(map(type, packets)) == {
                                simple_stream.SimpleReturnPacket
                            }:
                                if all(
                                    map(
                                        operator.is_,
                                        chain.from_iterable(map(_GET, packets)),
                                        b_flat_fields,
                                    )
                                ):
                                    return b_value
                except (AttributeError, TypeError, ValueError, OverflowError):
                    pass
                before = _state(stream)
                if before is not None and before == b_state:
                    return b_value
            return stream.digest
        self._attempted = True
        before = _state(stream)
        value = stream.digest  # Preserve public failures; no cache is armed on error.
        if before is not None and _state(stream) == before:
            self._baseline = before, value
        return value
