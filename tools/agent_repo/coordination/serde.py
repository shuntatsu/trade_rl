"""Strict JSON payload decoding for Agent Coordination Plane operator surfaces."""

from __future__ import annotations

from collections.abc import Mapping

from tools.agent_repo.coordination.dashboard import DashboardSnapshot
from tools.agent_repo.coordination.github_state import TaskStatusRecord
from tools.agent_repo.coordination.model import (
    Capability,
    DependencyKind,
    ExecutionMode,
    RiskLevel,
    TaskCondition,
    TaskDependency,
    TaskPacket,
    TaskPhase,
    TaskStatus,
    WriteScope,
)

_TASK_PACKET_FIELDS = {
    "schema",
    "task_id",
    "task_revision",
    "parent_issue",
    "title",
    "objective",
    "execution_mode",
    "non_goals",
    "dependencies",
    "write_scope",
    "resource_keys",
    "capabilities",
    "acceptance_criteria",
    "test_oracle",
    "base_sha",
    "risk",
    "deliverable",
}
_STATUS_FIELDS = {"phase", "condition", "reason"}
_STATUS_RECORD_FIELDS = frozenset(TaskStatusRecord.__dataclass_fields__)
_DASHBOARD_FIELDS = frozenset(DashboardSnapshot.__dataclass_fields__)


def _mapping(value: object, *, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{field_name} must be a JSON object")
    return value


def _strict_fields(
    payload: Mapping[str, object],
    expected: set[str] | frozenset[str],
    *,
    field_name: str,
) -> None:
    unknown = set(payload) - set(expected)
    if unknown:
        raise ValueError(f"unknown field in {field_name}: {sorted(unknown)[0]}")
    missing = set(expected) - set(payload)
    if missing:
        raise ValueError(f"missing field in {field_name}: {sorted(missing)[0]}")


def _string(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    return value


def _optional_string(value: object, *, field_name: str) -> str | None:
    if value is None:
        return None
    return _string(value, field_name=field_name)


def _integer(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer")
    return value


def _optional_integer(value: object, *, field_name: str) -> int | None:
    if value is None:
        return None
    return _integer(value, field_name=field_name)


def _strings(value: object, *, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a JSON array")
    return tuple(_string(item, field_name=field_name) for item in value)


def _dependencies(value: object) -> tuple[TaskDependency, ...]:
    if not isinstance(value, list):
        raise ValueError("dependencies must be a JSON array")
    result: list[TaskDependency] = []
    for item in value:
        payload = _mapping(item, field_name="dependency")
        _strict_fields(
            payload,
            {"task_id", "kind", "requires_phase"},
            field_name="dependency",
        )
        result.append(
            TaskDependency(
                task_id=_string(payload["task_id"], field_name="dependency.task_id"),
                kind=DependencyKind(
                    _string(payload["kind"], field_name="dependency.kind")
                ),
                requires_phase=TaskPhase(
                    _string(
                        payload["requires_phase"],
                        field_name="dependency.requires_phase",
                    )
                ),
            )
        )
    return tuple(result)


def task_packet_from_payload(value: object) -> TaskPacket:
    """Decode one canonical agent_task_v1 payload without accepting extras."""

    payload = _mapping(value, field_name="task packet")
    _strict_fields(payload, _TASK_PACKET_FIELDS, field_name="task packet")
    if payload["schema"] != "agent_task_v1":
        raise ValueError("task packet schema must be agent_task_v1")
    write_scope_payload = _mapping(payload["write_scope"], field_name="write_scope")
    _strict_fields(
        write_scope_payload,
        {"allow", "deny"},
        field_name="write_scope",
    )
    parent_issue_raw = payload["parent_issue"]
    parent_issue = (
        None
        if parent_issue_raw is None
        else _integer(parent_issue_raw, field_name="parent_issue")
    )
    return TaskPacket(
        task_id=_string(payload["task_id"], field_name="task_id"),
        task_revision=_integer(payload["task_revision"], field_name="task_revision"),
        parent_issue=parent_issue,
        title=_string(payload["title"], field_name="title"),
        objective=_string(payload["objective"], field_name="objective"),
        execution_mode=ExecutionMode(
            _string(payload["execution_mode"], field_name="execution_mode")
        ),
        non_goals=_strings(payload["non_goals"], field_name="non_goals"),
        dependencies=_dependencies(payload["dependencies"]),
        write_scope=WriteScope(
            allow=_strings(write_scope_payload["allow"], field_name="write_scope.allow"),
            deny=_strings(write_scope_payload["deny"], field_name="write_scope.deny"),
        ),
        resource_keys=_strings(payload["resource_keys"], field_name="resource_keys"),
        capabilities=tuple(
            Capability(value)
            for value in _strings(payload["capabilities"], field_name="capabilities")
        ),
        acceptance_criteria=_strings(
            payload["acceptance_criteria"], field_name="acceptance_criteria"
        ),
        test_oracle=_strings(payload["test_oracle"], field_name="test_oracle"),
        base_sha=_string(payload["base_sha"], field_name="base_sha"),
        risk=RiskLevel(_string(payload["risk"], field_name="risk")),
        deliverable=_string(payload["deliverable"], field_name="deliverable"),
    )


def task_status_from_payload(value: object) -> TaskStatus:
    payload = _mapping(value, field_name="task status")
    _strict_fields(payload, _STATUS_FIELDS, field_name="task status")
    return TaskStatus(
        phase=TaskPhase(_string(payload["phase"], field_name="phase")),
        condition=TaskCondition(
            _string(payload["condition"], field_name="condition")
        ),
        reason=_optional_string(payload["reason"], field_name="reason"),
    )


def task_status_record_from_payload(value: object) -> TaskStatusRecord:
    payload = _mapping(value, field_name="task status record")
    _strict_fields(payload, _STATUS_RECORD_FIELDS, field_name="task status record")
    return TaskStatusRecord(
        task_id=_string(payload["task_id"], field_name="task_id"),
        task_revision=_integer(payload["task_revision"], field_name="task_revision"),
        task_contract_digest=_string(
            payload["task_contract_digest"], field_name="task_contract_digest"
        ),
        phase=TaskPhase(_string(payload["phase"], field_name="phase")),
        condition=TaskCondition(
            _string(payload["condition"], field_name="condition")
        ),
        base_sha=_string(payload["base_sha"], field_name="base_sha"),
        owner=_optional_string(payload["owner"], field_name="owner"),
        lease_epoch=_optional_integer(payload["lease_epoch"], field_name="lease_epoch"),
        lease_branch=_optional_string(
            payload["lease_branch"], field_name="lease_branch"
        ),
        head_sha=_optional_string(payload["head_sha"], field_name="head_sha"),
        pr_id=_optional_string(payload["pr_id"], field_name="pr_id"),
        checkpoint=_optional_string(payload["checkpoint"], field_name="checkpoint"),
        blocking_reason=_optional_string(
            payload["blocking_reason"], field_name="blocking_reason"
        ),
        stale_reason=_optional_string(
            payload["stale_reason"], field_name="stale_reason"
        ),
    )


def dashboard_snapshot_from_payload(value: object) -> DashboardSnapshot:
    payload = _mapping(value, field_name="dashboard task")
    unknown = set(payload) - set(_DASHBOARD_FIELDS)
    if unknown:
        raise ValueError(f"unknown field in dashboard task: {sorted(unknown)[0]}")
    required = {"task_id", "title", "phase", "condition"}
    missing = required - set(payload)
    if missing:
        raise ValueError(f"missing field in dashboard task: {sorted(missing)[0]}")
    return DashboardSnapshot(
        task_id=_string(payload["task_id"], field_name="task_id"),
        title=_string(payload["title"], field_name="title"),
        phase=TaskPhase(_string(payload["phase"], field_name="phase")),
        condition=TaskCondition(
            _string(payload["condition"], field_name="condition")
        ),
        owner=_optional_string(payload.get("owner"), field_name="owner"),
        pr_id=_optional_string(payload.get("pr_id"), field_name="pr_id"),
        dependency_summary=_optional_string(
            payload.get("dependency_summary"), field_name="dependency_summary"
        ),
        blocking_reason=_optional_string(
            payload.get("blocking_reason"), field_name="blocking_reason"
        ),
        stale_reason=_optional_string(
            payload.get("stale_reason"), field_name="stale_reason"
        ),
    )


def packet_sequence(value: object) -> tuple[TaskPacket, ...]:
    if not isinstance(value, list):
        raise ValueError("packets must be a JSON array")
    return tuple(task_packet_from_payload(item) for item in value)


def string_set(value: object, *, field_name: str) -> frozenset[str]:
    return frozenset(_strings(value, field_name=field_name))


def status_mapping(value: object) -> dict[str, TaskStatus]:
    payload = _mapping(value, field_name="statuses")
    return {key: task_status_from_payload(item) for key, item in sorted(payload.items())}


__all__ = [
    "dashboard_snapshot_from_payload",
    "packet_sequence",
    "status_mapping",
    "string_set",
    "task_packet_from_payload",
    "task_status_from_payload",
    "task_status_record_from_payload",
]
