"""Field-closed mapping validation for public configuration contracts."""

from __future__ import annotations

from collections.abc import Mapping, Set
from dataclasses import fields
from typing import Any, get_origin, get_type_hints


def require_exact_fields(
    value: Mapping[str, Any],
    *,
    required: Set[str],
    optional: Set[str],
    field: str,
) -> dict[str, Any]:
    """Return a copy after rejecting unknown and missing field names."""

    keys = set(value)
    unknown = sorted(keys - set(required) - set(optional))
    if unknown:
        raise ValueError(f"{field} has unknown fields: {', '.join(unknown)}")
    missing = sorted(set(required) - keys)
    if missing:
        raise ValueError(f"{field} has missing required fields: {', '.join(missing)}")
    return dict(value)


def _restore_tuple_fields(value: dict[str, Any], model: Any) -> dict[str, Any]:
    resolved = dict(value)
    type_hints = get_type_hints(model)
    for name, item in tuple(resolved.items()):
        if isinstance(item, list) and get_origin(type_hints.get(name)) is tuple:
            resolved[name] = tuple(item)
    return resolved


def require_dataclass_fields(
    value: Mapping[str, Any],
    model: Any,
    *,
    field: str,
    excluded: Set[str] = frozenset(),
) -> dict[str, Any]:
    """Reject non-constructor names and restore JSON tuple containers."""

    allowed = {item.name for item in fields(model) if item.init} - set(excluded)
    validated = require_exact_fields(
        value,
        required=set(),
        optional=allowed,
        field=field,
    )
    return _restore_tuple_fields(validated, model)


__all__ = ["require_dataclass_fields", "require_exact_fields"]
