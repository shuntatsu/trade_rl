"""Dependency validation and scheduling use a single packet roster snapshot."""

from collections.abc import Iterator

import pytest

from tools.agent_repo.coordination.dependencies import DependencyGraph
from tools.agent_repo.coordination.model import (
    DependencyKind,
    ExecutionMode,
    TaskCondition,
    TaskDependency,
    TaskPacket,
    TaskPhase,
    TaskStatus,
)
from tools.agent_repo.coordination.scheduler import ready_tasks


def _packet(task_id: str, *, missing: bool = False) -> TaskPacket:
    return TaskPacket(
        task_id=task_id,
        task_revision=1,
        parent_issue=1067,
        title=task_id,
        objective="Synthetic roster snapshot validation",
        execution_mode=ExecutionMode.READ_ONLY,
        base_sha="a" * 40,
        dependencies=(TaskDependency("MISSING", DependencyKind.HARD),)
        if missing
        else (),
    )


class _ChangingRoster(list[TaskPacket]):
    def __init__(self, first: TaskPacket, later: TaskPacket) -> None:
        super().__init__([first])
        self.later = later
        self.observations = 0

    def __iter__(self) -> Iterator[TaskPacket]:
        self.observations += 1
        if self.observations == 1:
            return super().__iter__()
        return iter((self.later,))


def test_graph_rejects_unknown_dependency_in_first_snapshot() -> None:
    packets = _ChangingRoster(_packet("A", missing=True), _packet("A"))
    with pytest.raises(ValueError, match="unknown dependency MISSING for A"):
        DependencyGraph(packets)
    assert packets.observations == 1


def test_ready_tasks_uses_same_roster_as_dependency_graph() -> None:
    packets = _ChangingRoster(_packet("A"), _packet("B"))
    statuses = {"A": TaskStatus(TaskPhase.READY, TaskCondition.HEALTHY)}
    assert ready_tasks(packets, statuses, leased_task_ids=()) == ("A",)
    assert packets.observations == 1


def test_stable_roster_remains_valid() -> None:
    packets = (_packet("A"), _packet("B"))
    statuses = {
        packet.task_id: TaskStatus(TaskPhase.READY, TaskCondition.HEALTHY)
        for packet in packets
    }
    assert DependencyGraph(packets).task_ids == ("A", "B")
    assert ready_tasks(packets, statuses, leased_task_ids=()) == ("A", "B")
