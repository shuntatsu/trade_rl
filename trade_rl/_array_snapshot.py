"""Shared array admission before NumPy conversion can discard missingness."""

from __future__ import annotations

import inspect
from array import array
from collections import deque
from decimal import Decimal
from fractions import Fraction
from mmap import mmap
from typing import Protocol, TypeGuard

import numpy as np

_ARRAY_PROTOCOL_ABSENT = object()


class _IndexableContainer(Protocol):
    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> object: ...


def _is_indexable_container(value: object) -> TypeGuard[_IndexableContainer]:
    # NumPy accepts duck-typed __getitem__ sequences without ABC registration.
    return (
        not isinstance(value, (str, bytes, bytearray))
        and callable(getattr(type(value), "__len__", None))
        and callable(getattr(type(value), "__getitem__", None))
    )


def _is_exact_type(value: object, *allowed: type) -> bool:
    # Foreign metaclasses may override equality; admission requires identity.
    actual = type(value)
    return any(actual is candidate for candidate in allowed)


def _has_opaque_numpy_protocol(value: object) -> bool:
    # Exact builtin values/containers cannot attach arbitrary NumPy descriptors.
    # Preserve their low-overhead admission in frequent replay/accounting calls.
    if _is_exact_type(
        value,
        bool,
        int,
        float,
        complex,
        str,
        bytes,
        bytearray,
        list,
        tuple,
        deque,
        Decimal,
        range,
        memoryview,
        array,
    ):
        return False
    # Both ordinary and dynamically supplied descriptors can hide a masked
    # buffer on the first probe, then publish it during NumPy conversion.
    try:
        value_type = type(value)
        if (
            inspect.getattr_static(value_type, "__getattr__", _ARRAY_PROTOCOL_ABSENT)
            is not _ARRAY_PROTOCOL_ABSENT
            or value_type.__getattribute__ is not object.__getattribute__
        ):
            return True
        for name in ("__array__", "__array_interface__", "__array_struct__"):
            if inspect.getattr_static(
                value, name, _ARRAY_PROTOCOL_ABSENT
            ) is not _ARRAY_PROTOCOL_ABSENT or hasattr(value, name):
                return True
        return False
    except Exception as exc:
        # An inaccessible user-defined buffer descriptor is not trustworthy.
        raise ValueError("opaque array-like protocol cannot be inspected") from exc


def _require_unmasked_origin(value: object, *, field_name: str) -> None:
    # Scalars and sequences are checked separately. Every array owner must
    # be inspectable through native links before reading numeric values.
    if not (issubclass(type(value), np.ndarray) or type(value) is memoryview):
        return
    while True:
        if issubclass(type(value), np.ma.MaskedArray):
            raise ValueError(f"{field_name} must not contain a masked array")
        if type(value) is memoryview:
            value = value.obj
        elif issubclass(type(value), np.ndarray):
            value = np.ndarray.base.__get__(value)
        elif value is None or _is_exact_type(value, bytes, bytearray, array, mmap):
            return
        else:
            # Do not inspect arbitrary attributes of opaque owners, including
            # the mutable wrapper used by NumPy stride views.
            raise ValueError(f"{field_name} has opaque array ownership")


def _checked_vector_snapshot(
    value: object, *, field_name: str, allow_object_arrays: bool
) -> object:
    # Snapshot each index once; casting the original dynamic container could
    # fetch a masked value after the validator observed an unmasked value.
    root: list[object] = [None]
    pending: list[tuple[object, list[object] | None, int]] = [(value, root, 0)]
    active: dict[int, object] = {}
    while pending:
        item, parent, index = pending.pop()
        if parent is None:
            active.pop(id(item), None)
            continue
        _require_unmasked_origin(item, field_name=field_name)
        if isinstance(item, np.ndarray):
            # Strip the ndarray subclass without converting its native dtype.
            # Overridden dtype/tolist must not bypass object admission.
            native = np.asarray(item)
            if not native.dtype.hasobject:
                parent[index] = native.copy()
                continue
            if not allow_object_arrays:
                raise TypeError(f"{field_name} must not contain object arrays")
            if native.dtype.kind != "O":
                raise ValueError(
                    f"{field_name} must not contain opaque array-like values"
                )
            if id(item) in active:
                raise ValueError(f"{field_name} must not contain cyclic arrays")
            active[id(item)] = item
            pending.append((item, None, 0))
            pending.append((native.tolist(), parent, index))
            continue
        if isinstance(item, np.generic):
            dtype = np.generic.dtype.__get__(item)
            if type(item) is not dtype.type or dtype.hasobject:
                raise ValueError(
                    f"{field_name} must not contain opaque scalar converters"
                )
            parent[index] = item
            continue
        if _has_opaque_numpy_protocol(item):
            raise ValueError(f"{field_name} must not contain opaque array-like values")
        if _is_indexable_container(item):
            if id(item) in active:
                raise ValueError(f"{field_name} must not contain cyclic sequences")
            active[id(item)] = item
            children = [item[pos] for pos in range(len(item))]
            snapshot: list[object] = [None] * len(children)
            parent[index] = snapshot
            pending.append((item, None, 0))
            for pos in range(len(children) - 1, -1, -1):
                pending.append((children[pos], snapshot, pos))
            continue
        # NumPy must not invoke an untrusted scalar __float__ or __index__
        # after validation. Native number/text atoms retain casting.
        if not _is_exact_type(
            item,
            bool,
            int,
            float,
            complex,
            str,
            bytes,
            bytearray,
            Decimal,
            Fraction,
        ):
            raise ValueError(f"{field_name} must not contain opaque scalar converters")
        parent[index] = item
    return root[0]


def checked_array_input(
    value: object, *, field_name: str, allow_object_arrays: bool = True
) -> object:
    """Preserve existing native-array fast path and reject hidden missingness.

    The caller still owns dtype, shape and finite-value validation. Information
    discarded by the caller before admission cannot be recovered here.
    Dtype-sensitive callers must refuse object arrays rather than infer a new
    dtype from their validated contents.
    """
    _require_unmasked_origin(value, field_name=field_name)
    return (
        value
        if isinstance(value, np.ndarray)
        and not np.ndarray.dtype.__get__(value).hasobject
        else _checked_vector_snapshot(
            value, field_name=field_name, allow_object_arrays=allow_object_arrays
        )
    )
