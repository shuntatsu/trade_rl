"""GET-only Task enrollment preflight over stable GitHub observations.

Never grants writer, roster-completeness, lease reassignment or cleanup rights.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from tools.agent_repo.coordination.enrollment import (
    EnrollmentProjection,
    prepare_enrollment_projection,
)
from tools.agent_repo.coordination.github_state import TaskStatusRecord
from tools.agent_repo.coordination.leases import LeaseRecord
from tools.agent_repo.coordination.model import TaskPacket
from tools.agent_repo.coordination.registry import (
    RegistryTransitionPlan,
    plan_registry_transition,
)
from tools.github_coordination_registry_reader import GitObjectReadApi
from tools.github_coordination_registry_snapshot import observe_registry_snapshot

_REPOSITORY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*$")
_SHA = re.compile(r"^[0-9a-f]{40}$")


def _object(value: Any, label: str) -> dict[str, Any]:
    if type(value) is not dict or any(type(key) is not str for key in value):
        raise ValueError(f"{label} must be a GitHub response object with native keys")
    return value


def _positive_id(value: Any, label: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{label} must be a positive native integer")
    return value


def _text(value: Any, label: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{label} must be non-empty text")
    return value


def _repository_state(api: GitObjectReadApi, repository: str) -> tuple[int, str]:
    row = _object(api._request_json("GET", ""), "repository")
    name = row.get("full_name")
    if type(name) is not str or name != repository:
        raise ValueError("repository identity differs from expected repository")
    default_branch = row.get("default_branch")
    if type(default_branch) is not str or default_branch != "main":
        raise ValueError("repository default branch must remain main")
    return (
        _positive_id(row.get("id"), "repository id"),
        _text(row.get("node_id"), "repository node_id"),
    )


def _issue_state(
    api: GitObjectReadApi, repository: str, issue_number: int
) -> tuple[int, int, str, str]:
    row = _object(api._request_json("GET", f"issues/{issue_number}"), "Issue")
    number = row.get("number")
    state = row.get("state")
    repository_url = row.get("repository_url")
    if (
        type(number) is not int
        or number != issue_number
        or type(state) is not str
        or state != "open"
        or type(repository_url) is not str
        or repository_url != f"https://api.github.com/repos/{repository}"
        or "pull_request" in row
    ):
        raise ValueError("Issue identity, repository, open state or kind differs")
    return (
        _positive_id(row.get("id"), "Issue id"),
        issue_number,
        _text(row.get("node_id"), "Issue node ID"),
        _text(row.get("updated_at"), "Issue updated_at"),
    )


def _commit_ref(api: GitObjectReadApi, ref_name: str) -> str:
    if type(ref_name) is not str or not ref_name.startswith("refs/heads/"):
        raise ValueError("GitHub ref must be a branch ref")
    row = _object(
        api._request_json("GET", f"git/ref/{ref_name.removeprefix('refs/')}"),
        "GitHub branch ref",
    )
    observed_ref = row.get("ref")
    if type(observed_ref) is not str or observed_ref != ref_name:
        raise ValueError("GitHub branch ref identity differs from requested ref")
    target = _object(row.get("object"), "GitHub branch ref target")
    sha = target.get("sha")
    kind = target.get("type")
    if (
        type(kind) is not str
        or kind != "commit"
        or type(sha) is not str
        or not _SHA.fullmatch(sha)
        or sha == "0" * 40
    ):
        raise ValueError("GitHub branch ref must target an exact nonzero commit")
    return sha


@dataclass(frozen=True)
class _ReadState:
    repository: tuple[int, str]
    issue: tuple[int, int, str, str]
    main_sha: str
    branch_sha: str


def _read_state(
    api: GitObjectReadApi, *, repository: str, issue_number: int, lease_branch: str
) -> _ReadState:
    return _ReadState(
        repository=_repository_state(api, repository),
        issue=_issue_state(api, repository, issue_number),
        main_sha=_commit_ref(api, "refs/heads/main"),
        branch_sha=_commit_ref(api, f"refs/heads/{lease_branch}"),
    )


@dataclass(frozen=True)
class EnrollmentAudit:
    """Consistent but non-atomic GET observations; no durable mutation evidence."""

    repository: str
    projection: EnrollmentProjection
    plan: RegistryTransitionPlan
    registry_oid: str
    registry_digest: str
    main_sha: str
    branch_sha: str
    stable_readback_verified: bool = field(default=True, init=False)
    enrollment_persisted: bool = field(default=False, init=False)
    authoritative_registry_established: bool = field(default=False, init=False)
    task_registry_complete: bool = field(default=False, init=False)
    lease_reassignment_authorized: bool = field(default=False, init=False)
    cleanup_authorized: bool = field(default=False, init=False)


def audit_task_enrollment(
    api: GitObjectReadApi,
    *,
    packet: TaskPacket,
    lease: LeaseRecord,
    status: TaskStatusRecord,
    repository: str,
    registry_ref: str,
    registry_path: str,
    expected_issue_node_id: str,
    now: datetime,
) -> EnrollmentAudit:
    """Propose a Task transition using stable GET evidence, never persist it.

    GitHub has no atomic multi-resource GET snapshot. A second observation
    catches ordinary race/drift, but can't prove writer identity, roster
    completeness or exclusion of an ABA update between reads.
    """
    if (
        type(packet) is not TaskPacket
        or type(lease) is not LeaseRecord
        or type(status) is not TaskStatusRecord
    ):
        raise ValueError("Task, lease and status must use their native contracts")
    if type(repository) is not str or not _REPOSITORY.fullmatch(repository):
        raise ValueError("repository must use owner/name form")
    if type(packet.parent_issue) is not int or packet.parent_issue < 1:
        raise ValueError("Task requires a concrete parent Issue")
    if not isinstance(now, datetime) or now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must be timezone-aware")

    expected_issue_node_id = _text(expected_issue_node_id, "expected Issue node ID")

    initial = _read_state(
        api,
        repository=repository,
        issue_number=packet.parent_issue,
        lease_branch=lease.lease_branch,
    )
    if initial.issue[2] != expected_issue_node_id:
        raise ValueError("Issue immutable node identity differs")
    snapshot = observe_registry_snapshot(
        api,
        expected_repository=repository,
        ref_name=registry_ref,
        registry_path=registry_path,
    )
    if initial.repository[1] != snapshot.repository_node_id:
        raise ValueError("repository changed during enrollment")

    candidate = prepare_enrollment_projection(
        packet,
        lease,
        status,
        repository=repository,
        issue_number=packet.parent_issue,
        issue_node_id=initial.issue[2],
        current_main_sha=initial.main_sha,
        observed_branch_sha=initial.branch_sha,
        issue_is_open=True,
        now=now,
    )
    plan = plan_registry_transition(
        snapshot.registry,
        candidate,
        expected_before_digest=snapshot.registry_digest,
    )
    final = _read_state(
        api,
        repository=repository,
        issue_number=packet.parent_issue,
        lease_branch=lease.lease_branch,
    )
    if final != initial:
        raise ValueError("GitHub source changed during enrollment")
    if _commit_ref(api, registry_ref) != snapshot.ref_oid:
        raise ValueError("registry ref changed during enrollment")

    return EnrollmentAudit(
        repository=repository,
        projection=candidate,
        plan=plan,
        registry_oid=snapshot.ref_oid,
        registry_digest=snapshot.registry_digest,
        main_sha=initial.main_sha,
        branch_sha=initial.branch_sha,
    )


__all__ = ["EnrollmentAudit", "audit_task_enrollment"]
