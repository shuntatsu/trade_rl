"""Strict durable projections for Agent Coordination Plane state."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass

from tools.agent_repo.coordination.leases import LeaseRecord, lease_branch_name
from tools.agent_repo.coordination.model import (
    TaskCondition,
    TaskPacket,
    TaskPhase,
)

_TASK_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_FULL_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_FULL_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_MARKER_RE = re.compile(r"<!-- agent-coordination:([A-Za-z0-9][A-Za-z0-9._-]*) -->")
_JSON_BLOCK_RE = re.compile(r"```json\n(.*?)\n```", re.DOTALL)


def _require_task_id(value: str) -> None:
    if not _TASK_ID_RE.fullmatch(value):
        raise ValueError("task_id is invalid")


def _require_sha(value: str, *, field_name: str) -> None:
    if not _FULL_SHA_RE.fullmatch(value):
        raise ValueError(f"{field_name} must be a lowercase 40-hex Git SHA")


def _require_digest(value: str) -> None:
    if not _FULL_DIGEST_RE.fullmatch(value):
        raise ValueError("task_contract_digest must be a lowercase 64-hex SHA-256 digest")


def _optional_text(value: str | None, *, field_name: str) -> None:
    if value is not None and not value.strip():
        raise ValueError(f"{field_name} must be non-empty when provided")


@dataclass(frozen=True)
class TaskStatusRecord:
    """Coordinator-owned projection sufficient to reconstruct one task status."""

    task_id: str
    task_revision: int
    task_contract_digest: str
    phase: TaskPhase
    condition: TaskCondition
    base_sha: str
    owner: str | None = None
    lease_epoch: int | None = None
    lease_branch: str | None = None
    head_sha: str | None = None
    pr_id: str | None = None
    checkpoint: str | None = None
    blocking_reason: str | None = None
    stale_reason: str | None = None

    def __post_init__(self) -> None:
        _require_task_id(self.task_id)
        if isinstance(self.task_revision, bool) or self.task_revision < 1:
            raise ValueError("task_revision must be a positive integer")
        _require_digest(self.task_contract_digest)
        object.__setattr__(self, "phase", TaskPhase(self.phase))
        object.__setattr__(self, "condition", TaskCondition(self.condition))
        _require_sha(self.base_sha, field_name="base_sha")
        if self.head_sha is not None:
            _require_sha(self.head_sha, field_name="head_sha")

        lease_values = (self.owner, self.lease_epoch, self.lease_branch)
        present = tuple(value is not None for value in lease_values)
        if any(present) and not all(present):
            raise ValueError("lease fields owner/lease_epoch/lease_branch must be all present")
        if all(present):
            assert self.owner is not None
            assert self.lease_epoch is not None
            assert self.lease_branch is not None
            if not self.owner.strip():
                raise ValueError("lease fields require a non-empty owner")
            if isinstance(self.lease_epoch, bool) or self.lease_epoch < 1:
                raise ValueError("lease fields require a positive lease_epoch")
            expected = lease_branch_name(self.task_id, self.lease_epoch, self.owner)
            if self.lease_branch != expected:
                raise ValueError("lease fields contain an invalid fenced branch")

        for field_name in (
            "pr_id",
            "checkpoint",
            "blocking_reason",
            "stale_reason",
        ):
            _optional_text(getattr(self, field_name), field_name=field_name)

    def canonical_payload(self) -> dict[str, object]:
        payload = asdict(self)
        payload["phase"] = self.phase.value
        payload["condition"] = self.condition.value
        return payload


_STATUS_FIELDS = frozenset(TaskStatusRecord.__dataclass_fields__)


def render_status_comment(record: TaskStatusRecord) -> str:
    """Render one canonical Coordinator-owned GitHub status comment."""

    marker = f"<!-- agent-coordination:{record.task_id} -->"
    payload = json.dumps(
        record.canonical_payload(),
        sort_keys=True,
        indent=2,
        ensure_ascii=False,
    )
    return f"{marker}\n```json\n{payload}\n```\n"


def parse_status_comment(text: str) -> TaskStatusRecord:
    """Parse a status projection, rejecting ambiguous or partial records."""

    markers = _MARKER_RE.findall(text)
    if len(markers) != 1:
        raise ValueError("status comment must contain exactly one coordination marker")
    blocks = _JSON_BLOCK_RE.findall(text)
    if len(blocks) != 1:
        raise ValueError("status comment must contain exactly one JSON block")
    try:
        decoded = json.loads(blocks[0])
    except json.JSONDecodeError as error:
        raise ValueError("status comment JSON is invalid") from error
    if not isinstance(decoded, dict):
        raise ValueError("status comment JSON must be an object")
    if any(not isinstance(key, str) for key in decoded):
        raise ValueError("status comment JSON keys must be strings")
    unknown = set(decoded) - _STATUS_FIELDS
    if unknown:
        raise ValueError(f"unknown field in status comment: {sorted(unknown)[0]}")
    missing = _STATUS_FIELDS - set(decoded)
    if missing:
        raise ValueError(f"missing field in status comment: {sorted(missing)[0]}")

    try:
        record = TaskStatusRecord(**decoded)
    except (TypeError, ValueError) as error:
        raise ValueError(f"status comment JSON is invalid: {error}") from error
    if record.task_id != markers[0]:
        raise ValueError("coordination marker task_id does not match JSON task_id")
    return record


@dataclass(frozen=True)
class HandoffPacket:
    """Recoverable task handoff bound to one exact task/lease snapshot."""

    task_id: str
    task_revision: int
    task_contract_digest: str
    lease_epoch: int
    last_good_head: str
    verified: tuple[str, ...] = ()
    pending: tuple[str, ...] = ()
    known_failures: tuple[str, ...] = ()
    do_not_repeat: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_task_id(self.task_id)
        if isinstance(self.task_revision, bool) or self.task_revision < 1:
            raise ValueError("task_revision must be a positive integer")
        _require_digest(self.task_contract_digest)
        if isinstance(self.lease_epoch, bool) or self.lease_epoch < 1:
            raise ValueError("lease_epoch must be a positive integer")
        _require_sha(self.last_good_head, field_name="last_good_head")
        for field_name in (
            "verified",
            "pending",
            "known_failures",
            "do_not_repeat",
            "evidence_ids",
        ):
            values = tuple(str(value) for value in getattr(self, field_name))
            if any(not value.strip() for value in values):
                raise ValueError(f"handoff {field_name} entries must be non-empty")
            if len(set(values)) != len(values):
                raise ValueError(f"handoff {field_name} contains duplicates")
            object.__setattr__(self, field_name, values)


def validate_handoff(
    handoff: HandoffPacket,
    packet: TaskPacket,
    lease: LeaseRecord,
) -> None:
    """Reject a handoff that does not describe the exact current lease snapshot."""

    if (
        handoff.task_id != packet.task_id
        or handoff.task_revision != packet.task_revision
        or handoff.task_contract_digest != packet.contract_digest()
        or lease.task_id != packet.task_id
        or lease.task_revision != packet.task_revision
        or lease.task_contract_digest != packet.contract_digest()
        or lease.base_sha != packet.base_sha
    ):
        raise ValueError("handoff task contract does not match current task/lease")
    if handoff.lease_epoch != lease.epoch:
        raise ValueError("handoff lease epoch does not match current lease epoch")
    if lease.head_sha is None or handoff.last_good_head != lease.head_sha:
        raise ValueError("handoff head does not match current lease head")


__all__ = [
    "HandoffPacket",
    "TaskStatusRecord",
    "parse_status_comment",
    "render_status_comment",
    "validate_handoff",
]
