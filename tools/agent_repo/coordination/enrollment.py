"""Non-authoritative Task enrollment preflight; no GitHub write or cleanup grant.

A successful preflight checks consistency of *supplied* observations only. It
cannot authenticate their source, prove roster completeness, or implement CAS.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from tools.agent_repo.coordination.github_state import TaskStatusRecord
from tools.agent_repo.coordination.leases import LeaseRecord, lease_is_current
from tools.agent_repo.coordination.model import ExecutionMode, TaskPacket, TaskPhase
from tools.agent_repo.coordination.state import validate_transition

_REPOSITORY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*$")
_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def _require_sha(value: str, field: str) -> None:
    if not isinstance(value, str) or not _SHA_RE.fullmatch(value):
        raise ValueError(f"{field} must be an exact 40-character Git SHA")


@dataclass(frozen=True)
class EnrollmentProjection:
    """Minimal issue identity plus the existing TaskStatusRecord, not a new Task DB."""

    repository: str
    issue_number: int
    issue_node_id: str
    status: TaskStatusRecord

    def __post_init__(self) -> None:
        if not isinstance(self.repository, str) or not _REPOSITORY_RE.fullmatch(
            self.repository
        ):
            raise ValueError("repository must be owner/name")
        if (
            type(self.issue_number) is not int
            or self.issue_number < 1
            or not isinstance(self.issue_node_id, str)
            or not self.issue_node_id.strip()
        ):
            raise ValueError(
                "issue reference must have an immutable node ID and number"
            )
        if not isinstance(self.status, TaskStatusRecord):
            raise ValueError("status must be a TaskStatusRecord")
        if type(self.status.task_revision) is not int:
            raise ValueError("Task revision must be an exact integer")
        if (
            self.status.owner is None
            or self.status.lease_epoch is None
            or self.status.lease_branch is None
            or self.status.head_sha is None
        ):
            raise ValueError("write enrollment requires a complete fenced lease")

    def canonical_payload(self) -> dict[str, object]:
        return {
            "schema": "coordinator_enrollment_projection_v1",
            "repository": self.repository,
            "issue_number": self.issue_number,
            "issue_node_id": self.issue_node_id,
            "status": self.status.canonical_payload(),
        }


def prepare_enrollment_projection(
    packet: TaskPacket,
    lease: LeaseRecord,
    status: TaskStatusRecord,
    *,
    repository: str,
    issue_number: int,
    issue_node_id: str,
    current_main_sha: str,
    observed_branch_sha: str,
    issue_is_open: bool,
    now: datetime,
) -> EnrollmentProjection:
    """Reject inconsistent snapshots; never grant authority to persist them."""

    _require_sha(current_main_sha, "current_main_sha")
    _require_sha(observed_branch_sha, "observed_branch_sha")
    if type(packet.task_revision) is not int or type(lease.task_revision) is not int:
        raise ValueError("Task and lease revisions must be exact integers")
    if type(packet.parent_issue) is not int or type(issue_number) is not int:
        raise ValueError("Task issue number must be an exact integer")
    if issue_is_open is not True or packet.parent_issue != issue_number:
        raise ValueError("open immutable Task issue reference does not match contract")
    if packet.execution_mode is not ExecutionMode.WRITE:
        raise ValueError("read_only tasks must not acquire write enrollment")
    if current_main_sha != packet.base_sha:
        raise ValueError("current main has changed since Task base SHA")
    if not lease_is_current(lease, packet, now=now):
        raise ValueError("lease is expired or does not match Task contract")
    if lease.head_sha is None or observed_branch_sha != lease.head_sha:
        raise ValueError("lease branch tip is missing or drifted")
    if status.phase is TaskPhase.COMPLETE:
        raise ValueError("completed Task cannot be newly enrolled as writable")
    if (
        status.task_id != packet.task_id
        or status.task_revision != packet.task_revision
        or status.task_contract_digest != packet.contract_digest()
        or status.base_sha != packet.base_sha
        or status.owner != lease.owner
        or status.lease_epoch != lease.epoch
        or status.lease_branch != lease.lease_branch
        or status.head_sha != lease.head_sha
    ):
        raise ValueError("Task status, packet and lease do not bind the same snapshot")
    return EnrollmentProjection(repository, issue_number, issue_node_id, status)


def validate_enrollment_transition(
    previous: EnrollmentProjection | None,
    candidate: EnrollmentProjection,
    *,
    enumerated_task_ids: Sequence[str],
) -> None:
    """Validate one local Task-record transition without granting persistence."""

    if (
        isinstance(enumerated_task_ids, (str, bytes))
        or any(not isinstance(task, str) or not task for task in enumerated_task_ids)
        or len(set(enumerated_task_ids)) != len(enumerated_task_ids)
    ):
        raise ValueError("registry roster is malformed or duplicated")
    task_id = candidate.status.task_id
    if previous is None:
        if candidate.status.phase is TaskPhase.COMPLETE:
            raise ValueError("completed Task cannot be newly enrolled")
        if task_id in enumerated_task_ids:
            raise ValueError("duplicate Task ID in registry")
        return
    if previous.status.task_id != task_id or task_id not in enumerated_task_ids:
        raise ValueError("registry entry is missing or belongs to another Task")
    if (
        previous.repository != candidate.repository
        or previous.issue_number != candidate.issue_number
        or previous.issue_node_id != candidate.issue_node_id
    ):
        raise ValueError("immutable Task issue reference changed")
    old, new = previous.status, candidate.status
    if new.task_revision < old.task_revision:
        raise ValueError("Task revision regressed")
    if new.task_revision > old.task_revision + 1:
        raise ValueError("Task revision skipped a predecessor")
    if new.task_revision > old.task_revision:
        # A changed contract invalidates the old fenced lease. New epochs are
        # deliberately refused until remote reconciliation has an authority.
        raise ValueError("Task revision bump requires a new reconciled lease")
    if new.base_sha != old.base_sha:
        raise ValueError("Task base changed without a revision bump")
    if old.phase is TaskPhase.COMPLETE:
        raise ValueError("complete is a terminal task phase")
    if new.phase != old.phase:
        validate_transition(old.phase, new.phase)
    if (
        new.task_revision == old.task_revision
        and new.task_contract_digest != old.task_contract_digest
    ):
        raise ValueError("Task digest changed without a revision bump")
    if new.lease_epoch is None or old.lease_epoch is None:
        raise ValueError("missing write lease epoch")
    if new.lease_epoch < old.lease_epoch:
        raise ValueError("lease epoch regressed")
    if new.lease_epoch != old.lease_epoch:
        raise ValueError("new lease epoch requires remote side-effect reconciliation")
    if new.owner != old.owner or new.lease_branch != old.lease_branch:
        raise ValueError("lease owner or fenced branch changed within an epoch")
    if new.head_sha != old.head_sha:
        raise ValueError("branch advance requires verified remote ancestry")


def require_local_registry_preflight(
    previous: EnrollmentProjection | None,
    candidate: EnrollmentProjection,
    *,
    expected_registry_oid: str,
    observed_registry_oid: str,
    enumerated_task_ids: Sequence[str],
) -> None:
    """Check one proposed transition; remote atomic CAS is still mandatory.

    The supplied roster is not proof that GitHub enumeration was complete or
    authenticated; this function is deliberately not an enrollment writer.
    """

    _require_sha(expected_registry_oid, "expected_registry_oid")
    _require_sha(observed_registry_oid, "observed_registry_oid")
    if expected_registry_oid != observed_registry_oid:
        raise ValueError("registry reference changed since expected version")
    validate_enrollment_transition(
        previous,
        candidate,
        enumerated_task_ids=enumerated_task_ids,
    )


def require_enrollment_readback(
    expected: EnrollmentProjection,
    observed: EnrollmentProjection,
    *,
    expected_registry_oid: str,
    observed_registry_oid: str,
    observed_branch_sha: str,
) -> None:
    """Reject stale or different readback; caller must independently trust I/O."""

    for value, name in (
        (expected_registry_oid, "expected_registry_oid"),
        (observed_registry_oid, "observed_registry_oid"),
        (observed_branch_sha, "observed_branch_sha"),
    ):
        _require_sha(value, name)
    if expected_registry_oid != observed_registry_oid:
        raise ValueError("registry readback OID changed")
    if expected != observed:
        raise ValueError("registry readback record differs from candidate")
    if expected.status.head_sha != observed_branch_sha:
        raise ValueError("lease branch SHA changed during readback")


__all__ = [
    "EnrollmentProjection",
    "prepare_enrollment_projection",
    "require_enrollment_readback",
    "require_local_registry_preflight",
    "validate_enrollment_transition",
]
