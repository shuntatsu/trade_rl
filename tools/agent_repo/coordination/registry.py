"""Canonical off-tree Coordinator registry payload; no GitHub I/O or authority."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

from tools.agent_repo.coordination.enrollment import EnrollmentProjection
from tools.agent_repo.coordination.github_state import TaskStatusRecord

_SCHEMA = "coordinator_registry_v1"
_PROJECTION_SCHEMA = "coordinator_enrollment_projection_v1"
_REPOSITORY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*$")
_REGISTRY_FIELDS = frozenset({"schema", "repository", "entries"})
_PROJECTION_FIELDS = frozenset(
    {"schema", "repository", "issue_number", "issue_node_id", "status"}
)
_STATUS_FIELDS = frozenset(TaskStatusRecord.__dataclass_fields__)


def _reject_duplicate_json_fields(
    fields: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for name, value in fields:
        if name in result:
            raise ValueError(f"duplicate field in registry JSON: {name}")
        result[name] = value
    return result


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _exact_fields(value: dict[str, Any], expected: frozenset[str], name: str) -> None:
    unknown = set(value) - expected
    if unknown:
        raise ValueError(f"unknown field in {name}: {sorted(unknown)[0]}")
    missing = expected - set(value)
    if missing:
        raise ValueError(f"missing field in {name}: {sorted(missing)[0]}")


def _projection_from_payload(value: Any) -> EnrollmentProjection:
    payload = _mapping(value, "registry entry")
    _exact_fields(payload, _PROJECTION_FIELDS, "registry entry")
    if payload["schema"] != _PROJECTION_SCHEMA:
        raise ValueError("registry entry schema is unsupported")

    status_payload = _mapping(payload["status"], "registry entry status")
    _exact_fields(status_payload, _STATUS_FIELDS, "registry entry status")
    try:
        status = TaskStatusRecord(**status_payload)
        return EnrollmentProjection(
            repository=payload["repository"],
            issue_number=payload["issue_number"],
            issue_node_id=payload["issue_node_id"],
            status=status,
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"registry entry is invalid: {error}") from error


@dataclass(frozen=True)
class CoordinatorRegistry:
    """Deterministic Task registry payload intended for a future dedicated ref.

    This value is only a serialization contract. Its existence or validity does
    not authenticate a Coordinator, prove roster completeness, authorize a write,
    or permit cleanup.
    """

    repository: str
    entries: tuple[EnrollmentProjection, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.repository, str) or not _REPOSITORY_RE.fullmatch(
            self.repository
        ):
            raise ValueError("registry repository must use owner/name form")

        entries = tuple(self.entries)
        if any(not isinstance(entry, EnrollmentProjection) for entry in entries):
            raise ValueError("registry entries must be EnrollmentProjection values")
        if any(entry.repository != self.repository for entry in entries):
            raise ValueError("registry entry repository differs from registry")
        task_ids = [entry.status.task_id for entry in entries]
        if len(set(task_ids)) != len(task_ids):
            raise ValueError("registry contains duplicate Task IDs")
        object.__setattr__(self, "entries", entries)

    def canonical_payload(self) -> dict[str, object]:
        ordered = sorted(self.entries, key=lambda entry: entry.status.task_id)
        return {
            "schema": _SCHEMA,
            "repository": self.repository,
            "entries": [entry.canonical_payload() for entry in ordered],
        }

    def canonical_bytes(self) -> bytes:
        return json.dumps(
            self.canonical_payload(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")

    def digest(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()


def parse_registry_bytes(data: bytes) -> CoordinatorRegistry:
    """Parse only exact canonical registry bytes, rejecting ambiguous JSON."""

    if type(data) is not bytes:
        raise ValueError("registry payload must be bytes")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("registry payload must be UTF-8") from error
    try:
        decoded = json.loads(text, object_pairs_hook=_reject_duplicate_json_fields)
    except json.JSONDecodeError as error:
        raise ValueError("registry JSON is invalid") from error

    payload = _mapping(decoded, "registry")
    _exact_fields(payload, _REGISTRY_FIELDS, "registry")
    if payload["schema"] != _SCHEMA:
        raise ValueError("registry schema is unsupported")
    entries = payload["entries"]
    if not isinstance(entries, list):
        raise ValueError("registry entries must be a list")

    registry = CoordinatorRegistry(
        repository=payload["repository"],
        entries=tuple(_projection_from_payload(entry) for entry in entries),
    )
    if registry.canonical_bytes() != data:
        raise ValueError("registry payload bytes are not canonical")
    return registry


__all__ = ["CoordinatorRegistry", "parse_registry_bytes"]
