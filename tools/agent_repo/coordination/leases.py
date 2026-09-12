"""Lease fencing and recovery rules for coordinated write tasks."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from enum import Enum

from tools.agent_repo.coordination.model import ExecutionMode, TaskPacket

_FULL_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_FULL_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_TASK_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_OWNER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._ -]*$")


class ReconciliationDecision(str, Enum):
    RESUME = "resume"
    SALVAGE = "salvage"
    SUPERSEDE = "supersede"
    REASSIGN = "reassign"


def _require_aware(value: datetime, *, field_name: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")


def _require_sha(value: str, *, field_name: str) -> None:
    if not _FULL_SHA_RE.fullmatch(value):
        raise ValueError(f"{field_name} must be a lowercase 40-hex Git SHA")


def _require_digest(value: str, *, field_name: str) -> None:
    if not _FULL_DIGEST_RE.fullmatch(value):
        raise ValueError(f"{field_name} must be a lowercase 64-hex SHA-256 digest")


def _owner_slug(owner: str) -> str:
    if not _OWNER_RE.fullmatch(owner) or ".." in owner:
        raise ValueError("owner must contain only safe branch-name characters")
    slug = re.sub(r"[ ._]+", "-", owner.strip().lower())
    slug = re.sub(r"-+", "-", slug).strip("-")
    if not slug:
        raise ValueError("owner must produce a non-empty branch slug")
    return slug


def lease_branch_name(task_id: str, epoch: int, owner: str) -> str:
    """Return the branch fence for one concrete lease epoch."""

    if not _TASK_ID_RE.fullmatch(task_id):
        raise ValueError("task_id is invalid for a lease branch")
    if isinstance(epoch, bool) or epoch < 1:
        raise ValueError("epoch must be a positive integer")
    return f"agent/{task_id}/e{epoch:04d}-{_owner_slug(owner)}"


@dataclass(frozen=True)
class LeaseRecord:
    task_id: str
    task_revision: int
    task_contract_digest: str
    base_sha: str
    owner: str
    epoch: int
    lease_branch: str
    head_sha: str | None
    last_heartbeat_at: datetime
    expires_at: datetime
    ttl_seconds: float

    def __post_init__(self) -> None:
        if not _TASK_ID_RE.fullmatch(self.task_id):
            raise ValueError("task_id is invalid")
        if isinstance(self.task_revision, bool) or self.task_revision < 1:
            raise ValueError("task_revision must be a positive integer")
        _require_digest(self.task_contract_digest, field_name="task_contract_digest")
        _require_sha(self.base_sha, field_name="base_sha")
        if self.head_sha is not None:
            _require_sha(self.head_sha, field_name="head_sha")
        if isinstance(self.epoch, bool) or self.epoch < 1:
            raise ValueError("epoch must be a positive integer")
        expected_branch = lease_branch_name(self.task_id, self.epoch, self.owner)
        if self.lease_branch != expected_branch:
            raise ValueError("lease_branch does not match task/epoch/owner fence")
        _require_aware(self.last_heartbeat_at, field_name="last_heartbeat_at")
        _require_aware(self.expires_at, field_name="expires_at")
        if self.ttl_seconds <= 0.0:
            raise ValueError("ttl_seconds must be positive")
        if self.expires_at <= self.last_heartbeat_at:
            raise ValueError("expires_at must be after last_heartbeat_at")


@dataclass(frozen=True)
class LeaseReconciliation:
    task_id: str
    expired_epoch: int
    observed_branch: str
    observed_head_sha: str | None
    decision: ReconciliationDecision
    pr_ids: tuple[str, ...] = ()
    ci_ids: tuple[str, ...] = ()
    artifact_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not _TASK_ID_RE.fullmatch(self.task_id):
            raise ValueError("reconciliation task_id is invalid")
        if isinstance(self.expired_epoch, bool) or self.expired_epoch < 1:
            raise ValueError("reconciliation expired_epoch must be positive")
        if not self.observed_branch.strip():
            raise ValueError("reconciliation observed_branch must be non-empty")
        if self.observed_head_sha is not None:
            _require_sha(self.observed_head_sha, field_name="observed_head_sha")
        object.__setattr__(self, "decision", ReconciliationDecision(self.decision))
        for field_name in ("pr_ids", "ci_ids", "artifact_ids"):
            values = tuple(str(value) for value in getattr(self, field_name))
            if any(not value.strip() for value in values):
                raise ValueError(f"reconciliation {field_name} must be non-empty")
            if len(set(values)) != len(values):
                raise ValueError(f"reconciliation {field_name} contains duplicates")
            object.__setattr__(self, field_name, values)


def _validate_grant_time(now: datetime, ttl: timedelta) -> float:
    _require_aware(now, field_name="now")
    ttl_seconds = ttl.total_seconds()
    if ttl_seconds <= 0.0:
        raise ValueError("lease ttl must be positive")
    return ttl_seconds


def _validate_reconciliation(
    previous: LeaseRecord,
    reconciliation: LeaseReconciliation | None,
) -> LeaseReconciliation:
    if reconciliation is None:
        raise ValueError("expired lease requires reconciliation before reassignment")
    if (
        reconciliation.task_id != previous.task_id
        or reconciliation.expired_epoch != previous.epoch
        or reconciliation.observed_branch != previous.lease_branch
    ):
        raise ValueError("reconciliation does not match the expired lease")
    return reconciliation


def grant_lease(
    packet: TaskPacket,
    owner: str,
    *,
    epoch: int,
    now: datetime,
    ttl: timedelta,
    previous: LeaseRecord | None = None,
    reconciliation: LeaseReconciliation | None = None,
) -> LeaseRecord:
    """Grant or renew a fenced lease for a write task."""

    if packet.execution_mode is ExecutionMode.READ_ONLY:
        raise ValueError("read_only tasks do not receive writable leases")
    ttl_seconds = _validate_grant_time(now, ttl)
    owner_slug = _owner_slug(owner)
    del owner_slug  # validation only; branch construction performs canonical slugging

    if previous is None:
        if reconciliation is not None:
            raise ValueError("reconciliation is only valid with a previous lease")
        branch = lease_branch_name(packet.task_id, epoch, owner)
        return LeaseRecord(
            task_id=packet.task_id,
            task_revision=packet.task_revision,
            task_contract_digest=packet.contract_digest(),
            base_sha=packet.base_sha,
            owner=owner,
            epoch=epoch,
            lease_branch=branch,
            head_sha=None,
            last_heartbeat_at=now,
            expires_at=now + ttl,
            ttl_seconds=ttl_seconds,
        )

    if previous.task_id != packet.task_id:
        raise ValueError("previous lease belongs to a different task")
    if now < previous.expires_at:
        raise ValueError("active lease cannot be replaced")

    resolved = _validate_reconciliation(previous, reconciliation)
    if resolved.decision is ReconciliationDecision.RESUME:
        if owner != previous.owner:
            raise ValueError("resume must keep the previous owner")
        if epoch != previous.epoch:
            raise ValueError("resume must keep the previous epoch")
        if (
            previous.task_revision != packet.task_revision
            or previous.task_contract_digest != packet.contract_digest()
            or previous.base_sha != packet.base_sha
        ):
            raise ValueError("resume cannot cross task contract or base changes")
        return LeaseRecord(
            task_id=packet.task_id,
            task_revision=packet.task_revision,
            task_contract_digest=packet.contract_digest(),
            base_sha=packet.base_sha,
            owner=owner,
            epoch=epoch,
            lease_branch=previous.lease_branch,
            head_sha=resolved.observed_head_sha,
            last_heartbeat_at=now,
            expires_at=now + ttl,
            ttl_seconds=ttl_seconds,
        )

    if epoch != previous.epoch + 1:
        raise ValueError("reassignment must use the next epoch")
    branch = lease_branch_name(packet.task_id, epoch, owner)
    if branch == previous.lease_branch:
        raise ValueError("reassignment must use a new fenced lease branch")
    return LeaseRecord(
        task_id=packet.task_id,
        task_revision=packet.task_revision,
        task_contract_digest=packet.contract_digest(),
        base_sha=packet.base_sha,
        owner=owner,
        epoch=epoch,
        lease_branch=branch,
        head_sha=None,
        last_heartbeat_at=now,
        expires_at=now + ttl,
        ttl_seconds=ttl_seconds,
    )


def lease_is_current(
    lease: LeaseRecord,
    packet: TaskPacket,
    *,
    now: datetime,
) -> bool:
    """Return whether a lease still owns the exact current task contract."""

    _require_aware(now, field_name="now")
    return (
        lease.task_id == packet.task_id
        and lease.task_revision == packet.task_revision
        and lease.task_contract_digest == packet.contract_digest()
        and lease.base_sha == packet.base_sha
        and now < lease.expires_at
    )


def heartbeat(
    lease: LeaseRecord,
    *,
    head_sha: str,
    observed_at: datetime,
) -> LeaseRecord:
    """Renew one lease after observing progress at an exact remote HEAD."""

    _require_sha(head_sha, field_name="head_sha")
    _require_aware(observed_at, field_name="observed_at")
    if observed_at < lease.last_heartbeat_at:
        raise ValueError("heartbeat time must be monotonic")
    if observed_at >= lease.expires_at:
        raise ValueError("expired lease requires reconciliation before heartbeat")
    ttl = timedelta(seconds=lease.ttl_seconds)
    return replace(
        lease,
        head_sha=head_sha,
        last_heartbeat_at=observed_at,
        expires_at=observed_at + ttl,
    )


__all__ = [
    "LeaseRecord",
    "LeaseReconciliation",
    "ReconciliationDecision",
    "grant_lease",
    "heartbeat",
    "lease_branch_name",
    "lease_is_current",
]
