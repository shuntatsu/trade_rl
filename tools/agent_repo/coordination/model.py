"""Immutable contracts for repository-local multi-agent coordination."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

_FULL_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_FULL_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_TASK_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_RESOURCE_KEY_RE = re.compile(r"^[a-z][a-z0-9_-]*:.+$")


class ExecutionMode(str, Enum):
    READ_ONLY = "read_only"
    WRITE = "write"


class TaskPhase(str, Enum):
    PLANNED = "planned"
    READY = "ready"
    EXECUTING = "executing"
    REVIEW = "review"
    VERIFICATION = "verification"
    INTEGRATION = "integration"
    COMPLETE = "complete"


class TaskCondition(str, Enum):
    HEALTHY = "healthy"
    BLOCKED = "blocked"
    STALE = "stale"
    FAILED = "failed"
    CONFLICTED = "conflicted"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class DependencyKind(str, Enum):
    HARD = "hard"
    EVIDENCE = "evidence"
    INTEGRATION = "integration"


class EvidenceKind(str, Enum):
    REVIEW = "review"
    TEST = "test"
    CI = "ci"
    ARTIFACT = "artifact"
    INTEGRATION = "integration"
    POST_MERGE = "post_merge"


class EvidenceResult(str, Enum):
    PASS = "pass"
    FAIL = "fail"


class Capability(str, Enum):
    REPOSITORY_READ = "repository_read"
    LEASE_BRANCH_WRITE = "lease_branch_write"
    DRAFT_PR_WRITE = "draft_pr_write"
    ISSUE_COMMENT_WRITE = "issue_comment_write"
    REVIEW_WRITE = "review_write"
    CI_READ = "ci_read"
    ARTIFACT_READ = "artifact_read"
    INTEGRATION_WRITE = "integration_write"
    MERGE_WRITE = "merge_write"


def _tuple_strings(values: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    return tuple(str(value) for value in values)


def _require_full_sha(value: str, *, field_name: str) -> None:
    if not _FULL_SHA_RE.fullmatch(value):
        raise ValueError(f"{field_name} must be a lowercase 40-hex Git SHA")


def _require_digest(value: str, *, field_name: str) -> None:
    if not _FULL_DIGEST_RE.fullmatch(value):
        raise ValueError(f"{field_name} must be a lowercase 64-hex SHA-256 digest")


@dataclass(frozen=True)
class WriteScope:
    allow: tuple[str, ...] = ()
    deny: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "allow", _tuple_strings(self.allow))
        object.__setattr__(self, "deny", _tuple_strings(self.deny))
        if len(set(self.allow)) != len(self.allow):
            raise ValueError("duplicate write allow path")
        if len(set(self.deny)) != len(self.deny):
            raise ValueError("duplicate write deny path")
        if any(not value.strip() for value in (*self.allow, *self.deny)):
            raise ValueError("write scope paths must be non-empty")

    def canonical_payload(self) -> dict[str, list[str]]:
        return {
            "allow": sorted(self.allow),
            "deny": sorted(self.deny),
        }


@dataclass(frozen=True)
class TaskDependency:
    task_id: str
    kind: DependencyKind
    requires_phase: TaskPhase = TaskPhase.COMPLETE

    def __post_init__(self) -> None:
        if not _TASK_ID_RE.fullmatch(self.task_id):
            raise ValueError("dependency task_id is invalid")
        object.__setattr__(self, "kind", DependencyKind(self.kind))
        object.__setattr__(self, "requires_phase", TaskPhase(self.requires_phase))

    def canonical_payload(self) -> dict[str, str]:
        return {
            "task_id": self.task_id,
            "kind": self.kind.value,
            "requires_phase": self.requires_phase.value,
        }


@dataclass(frozen=True)
class TaskPacket:
    task_id: str
    task_revision: int
    parent_issue: int | None
    title: str
    objective: str
    execution_mode: ExecutionMode
    non_goals: tuple[str, ...] = ()
    dependencies: tuple[TaskDependency, ...] = ()
    write_scope: WriteScope = field(default_factory=WriteScope)
    resource_keys: tuple[str, ...] = ()
    capabilities: tuple[Capability, ...] = (Capability.REPOSITORY_READ,)
    acceptance_criteria: tuple[str, ...] = ()
    test_oracle: tuple[str, ...] = ()
    base_sha: str = ""
    risk: RiskLevel = RiskLevel.MEDIUM
    deliverable: str = "pull_request"

    def __post_init__(self) -> None:
        if not _TASK_ID_RE.fullmatch(self.task_id):
            raise ValueError("task_id must be a non-empty stable identifier")
        if isinstance(self.task_revision, bool) or self.task_revision < 1:
            raise ValueError("task_revision must be a positive integer")
        if self.parent_issue is not None and (
            isinstance(self.parent_issue, bool) or self.parent_issue < 1
        ):
            raise ValueError("parent_issue must be a positive integer")
        if not self.objective.strip():
            raise ValueError("objective must be non-empty")
        if not self.title.strip():
            raise ValueError("title must be non-empty")
        _require_full_sha(self.base_sha, field_name="base_sha")

        object.__setattr__(self, "execution_mode", ExecutionMode(self.execution_mode))
        object.__setattr__(self, "risk", RiskLevel(self.risk))
        object.__setattr__(self, "non_goals", _tuple_strings(self.non_goals))
        object.__setattr__(self, "resource_keys", _tuple_strings(self.resource_keys))
        object.__setattr__(
            self, "acceptance_criteria", _tuple_strings(self.acceptance_criteria)
        )
        object.__setattr__(self, "test_oracle", _tuple_strings(self.test_oracle))
        object.__setattr__(
            self,
            "capabilities",
            tuple(Capability(value) for value in self.capabilities),
        )
        object.__setattr__(
            self,
            "dependencies",
            tuple(
                value
                if isinstance(value, TaskDependency)
                else TaskDependency(**value)  # type: ignore[arg-type]
                for value in self.dependencies
            ),
        )

        dependency_ids = [value.task_id for value in self.dependencies]
        if len(set(dependency_ids)) != len(dependency_ids):
            raise ValueError("duplicate dependency task_id")
        if self.task_id in dependency_ids:
            raise ValueError("task cannot depend on itself")
        if len(set(self.resource_keys)) != len(self.resource_keys):
            raise ValueError("duplicate resource key")
        if len(set(self.capabilities)) != len(self.capabilities):
            raise ValueError("duplicate capability")
        for key in self.resource_keys:
            if not _RESOURCE_KEY_RE.fullmatch(key):
                raise ValueError(f"invalid resource key: {key}")

        if self.execution_mode is ExecutionMode.READ_ONLY:
            if self.write_scope.allow:
                raise ValueError("read_only task cannot declare writable allow paths")
            forbidden = {
                Capability.LEASE_BRANCH_WRITE,
                Capability.DRAFT_PR_WRITE,
                Capability.INTEGRATION_WRITE,
                Capability.MERGE_WRITE,
            }
            if forbidden.intersection(self.capabilities):
                raise ValueError("read_only task cannot request write capabilities")
        elif not self.write_scope.allow:
            raise ValueError("write task requires at least one allow path")

    def canonical_payload(self) -> dict[str, Any]:
        dependencies = sorted(
            (value.canonical_payload() for value in self.dependencies),
            key=lambda value: (
                value["task_id"],
                value["kind"],
                value["requires_phase"],
            ),
        )
        return {
            "schema": "agent_task_v1",
            "task_id": self.task_id,
            "task_revision": self.task_revision,
            "parent_issue": self.parent_issue,
            "objective": self.objective,
            "execution_mode": self.execution_mode.value,
            "non_goals": list(self.non_goals),
            "dependencies": dependencies,
            "write_scope": self.write_scope.canonical_payload(),
            "resource_keys": sorted(self.resource_keys),
            "capabilities": sorted(value.value for value in self.capabilities),
            "acceptance_criteria": list(self.acceptance_criteria),
            "test_oracle": list(self.test_oracle),
            "base_sha": self.base_sha,
            "risk": self.risk.value,
            "deliverable": self.deliverable,
        }

    def contract_digest(self) -> str:
        encoded = json.dumps(
            self.canonical_payload(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class TaskStatus:
    phase: TaskPhase
    condition: TaskCondition = TaskCondition.HEALTHY
    reason: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "phase", TaskPhase(self.phase))
        object.__setattr__(self, "condition", TaskCondition(self.condition))
        if self.reason is not None and not self.reason.strip():
            raise ValueError("status reason must be non-empty when provided")


@dataclass(frozen=True)
class EvidenceRecord:
    evidence_kind: EvidenceKind
    task_id: str
    task_revision: int
    task_contract_digest: str
    base_sha: str
    head_sha: str
    identifier: str
    result: EvidenceResult
    main_sha: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence_kind", EvidenceKind(self.evidence_kind))
        object.__setattr__(self, "result", EvidenceResult(self.result))
        if not _TASK_ID_RE.fullmatch(self.task_id):
            raise ValueError("task_id is invalid")
        if isinstance(self.task_revision, bool) or self.task_revision < 1:
            raise ValueError("task_revision must be a positive integer")
        _require_digest(
            self.task_contract_digest,
            field_name="task_contract_digest",
        )
        _require_full_sha(self.base_sha, field_name="base_sha")
        _require_full_sha(self.head_sha, field_name="head_sha")
        if not self.identifier.strip():
            raise ValueError("evidence identifier must be non-empty")
        if self.main_sha is not None:
            _require_full_sha(self.main_sha, field_name="main_sha")
        if self.evidence_kind in {EvidenceKind.INTEGRATION, EvidenceKind.POST_MERGE}:
            if self.main_sha is None:
                raise ValueError("main_sha is required for integration evidence")


__all__ = [
    "Capability",
    "DependencyKind",
    "EvidenceKind",
    "EvidenceRecord",
    "EvidenceResult",
    "ExecutionMode",
    "RiskLevel",
    "TaskCondition",
    "TaskDependency",
    "TaskPacket",
    "TaskPhase",
    "TaskStatus",
    "WriteScope",
]
