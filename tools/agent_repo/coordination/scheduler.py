"""Deterministic ready-set selection for Agent Coordination Plane tasks."""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence

from tools.agent_repo.coordination.conflicts import ConflictLevel, classify_conflict
from tools.agent_repo.coordination.dependencies import DependencyGraph
from tools.agent_repo.coordination.model import (
    RiskLevel,
    TaskCondition,
    TaskPacket,
    TaskPhase,
    TaskStatus,
)

_RISK_ORDER = {
    RiskLevel.LOW: 0,
    RiskLevel.MEDIUM: 1,
    RiskLevel.HIGH: 2,
}


def ready_tasks(
    packets: Sequence[TaskPacket],
    statuses: Mapping[str, TaskStatus],
    *,
    leased_task_ids: Collection[str],
) -> tuple[str, ...]:
    """Return a deterministic concurrently execution-ready set of task ids."""

    graph = DependencyGraph(packets)
    packet_by_id = {packet.task_id: packet for packet in packets}
    leased = set(leased_task_ids)

    unknown_status = set(statuses) - set(packet_by_id)
    if unknown_status:
        raise ValueError(f"status references unknown task: {sorted(unknown_status)[0]}")
    unknown_leases = leased - set(packet_by_id)
    if unknown_leases:
        raise ValueError(f"lease references unknown task: {sorted(unknown_leases)[0]}")

    active = [
        packet_by_id[task_id]
        for task_id, status in statuses.items()
        if status.phase is TaskPhase.EXECUTING
        and status.condition is TaskCondition.HEALTHY
    ]
    candidates: list[TaskPacket] = []
    for packet in packets:
        status = statuses.get(packet.task_id)
        if status is None:
            raise ValueError(f"missing status for task: {packet.task_id}")
        if status.phase is not TaskPhase.READY:
            continue
        if status.condition is not TaskCondition.HEALTHY:
            continue
        if packet.task_id in leased:
            continue
        if not graph.execution_dependencies_satisfied(packet.task_id, statuses):
            continue
        if any(
            classify_conflict(packet, running) is ConflictLevel.HARD
            for running in active
        ):
            continue
        candidates.append(packet)

    candidates.sort(
        key=lambda packet: (
            -graph.descendant_count(packet.task_id),
            _RISK_ORDER[packet.risk],
            packet.task_id,
        )
    )

    selected: list[TaskPacket] = []
    for candidate in candidates:
        if any(
            classify_conflict(candidate, existing) is ConflictLevel.HARD
            for existing in selected
        ):
            continue
        selected.append(candidate)
    return tuple(packet.task_id for packet in selected)


__all__ = ["ready_tasks"]
