"""Opaque, collision-resistant Studio resource identities."""

from __future__ import annotations

import hashlib
import re
from typing import Final, Literal

ResourceKind = Literal["dataset", "config", "run"]
_RESOURCE_KINDS: Final = frozenset({"dataset", "config", "run"})
_RESOURCE_ID_RE: Final = re.compile(r"^(dataset|config|run)-[0-9a-f]{24}$")


def resource_id(kind: ResourceKind | str, relative_path: str, identity: str) -> str:
    """Bind a UI/API identity to its resource kind, location, and canonical identity."""

    if kind not in _RESOURCE_KINDS:
        raise ValueError(f"unsupported Studio resource kind: {kind}")
    if not isinstance(relative_path, str) or not relative_path:
        raise ValueError("resource relative_path must be a non-empty string")
    if not isinstance(identity, str) or not identity:
        raise ValueError("resource canonical identity must be a non-empty string")
    payload = f"studio-resource-v1\0{kind}\0{relative_path}\0{identity}".encode()
    return f"{kind}-{hashlib.sha256(payload).hexdigest()[:24]}"


def require_resource_id(value: str, *, kind: ResourceKind | None = None) -> str:
    """Validate one opaque Studio resource ID and optionally its expected kind."""

    if not isinstance(value, str) or _RESOURCE_ID_RE.fullmatch(value) is None:
        raise ValueError("Studio resource id is invalid")
    actual = value.split("-", 1)[0]
    if kind is not None and actual != kind:
        raise ValueError(f"Studio resource id kind must be {kind}")
    return value


__all__ = ["ResourceKind", "require_resource_id", "resource_id"]
