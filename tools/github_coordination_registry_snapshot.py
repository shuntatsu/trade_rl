"""Read-only consistent registry-ref snapshot, not a Coordinator authority.

The supplied GitHub transport must already be bound to the intended repository.
Two ref and repository reads detect changes while a pinned canonical Git object
is retrieved, but do not establish protected refs or a complete Task roster.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from tools.agent_repo.coordination.registry import (
    CoordinatorRegistry,
    parse_registry_bytes,
)
from tools.github_coordination_registry_reader import (
    GitHubRegistryObjectReader,
    GitObjectReadApi,
)

_REPO_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*$")
_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_REF_RE = re.compile(r"^refs/heads/coordination/[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*$")


@dataclass(frozen=True)
class RegistryRefSnapshot:
    """A momentary read-only observation; no Task registration authority."""

    repository: str
    repository_node_id: str
    ref_name: str
    ref_oid: str
    registry: CoordinatorRegistry
    registry_digest: str
    ref_readback_verified: bool = field(default=True, init=False)
    authoritative_registry_established: bool = field(default=False, init=False)
    task_registry_complete: bool = field(default=False, init=False)
    cleanup_authorized: bool = field(default=False, init=False)


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a GitHub response object")
    return value


def _requested_ref(ref_name: str) -> str:
    if (
        not isinstance(ref_name, str)
        or not _REF_RE.fullmatch(ref_name)
        or any(part in {".", ".."} for part in ref_name.split("/"))
        or ".." in ref_name
        or ref_name.endswith(".lock")
    ):
        raise ValueError("ref must be a safe coordination branch ref")
    return ref_name


def _repository_state(
    api: GitObjectReadApi, expected_repository: str
) -> tuple[int, str]:
    value = _mapping(api._request_json("GET", ""), "repository identity")
    if value.get("full_name") != expected_repository:
        raise ValueError("repository identity differs from expected repository")
    number = value.get("id")
    node_id = value.get("node_id")
    if type(number) is not int or number < 1:
        raise ValueError("repository identity requires positive native id")
    if not isinstance(node_id, str) or not node_id.strip():
        raise ValueError("repository identity requires nonempty node_id")
    return number, node_id


def _ref_state(api: GitObjectReadApi, ref_name: str) -> str:
    suffix = ref_name.removeprefix("refs/")
    value = _mapping(
        api._request_json("GET", f"git/ref/{suffix}"),
        "coordination ref",
    )
    if value.get("ref") != ref_name:
        raise ValueError("coordination ref identity differs from requested ref")
    target = _mapping(value.get("object"), "coordination ref target")
    oid = target.get("sha")
    if target.get("type") != "commit":
        raise ValueError("coordination ref target must be a commit")
    if not isinstance(oid, str) or not _SHA_RE.fullmatch(oid) or oid == "0" * 40:
        raise ValueError("coordination ref must have a nonzero commit OID")
    return oid


def observe_registry_snapshot(
    api: GitObjectReadApi,
    *,
    expected_repository: str,
    ref_name: str,
    registry_path: str,
) -> RegistryRefSnapshot:
    """GET only; fail closed on changed or unavailable repository/ref/payload."""

    if (
        not isinstance(expected_repository, str)
        or not _REPO_RE.fullmatch(expected_repository)
    ):
        raise ValueError("expected repository must be owner/name")
    ref_name = _requested_ref(ref_name)
    reader = GitHubRegistryObjectReader(
        api,
        expected_repository=expected_repository,
        registry_path=registry_path,
    )

    initial_repository = _repository_state(api, expected_repository)
    initial_oid = _ref_state(api, ref_name)
    raw = reader.read_registry_bytes(expected_repository, initial_oid)
    registry = parse_registry_bytes(raw)
    observed_oid = _ref_state(api, ref_name)
    if observed_oid != initial_oid:
        raise ValueError("coordination ref changed during snapshot")
    observed_repository = _repository_state(api, expected_repository)
    if observed_repository != initial_repository:
        raise ValueError("repository identity changed during snapshot")

    return RegistryRefSnapshot(
        repository=expected_repository,
        repository_node_id=initial_repository[1],
        ref_name=ref_name,
        ref_oid=initial_oid,
        registry=registry,
        registry_digest=registry.digest(),
    )


__all__ = ["RegistryRefSnapshot", "observe_registry_snapshot"]
