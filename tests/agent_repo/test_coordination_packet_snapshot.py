"""Dependency validation and scheduling use a single local observation."""

from collections.abc import Iterator, Mapping
from dataclasses import replace

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


class _ChangingStatuses(Mapping[str, TaskStatus]):
    def __init__(
        self, first: dict[str, TaskStatus], later: dict[str, TaskStatus]
    ) -> None:
        self.first = first
        self.later = later
        self.iterations = 0
        self.reads: dict[str, int] = {}

    def __iter__(self) -> Iterator[str]:
        self.iterations += 1
        return iter(self.first if self.iterations == 1 else self.later)

    def __len__(self) -> int:
        return len(self.first)

    def __getitem__(self, key: str) -> TaskStatus:
        count = self.reads.get(key, 0) + 1
        self.reads[key] = count
        return self.first[key] if count == 1 else self.later.get(key, self.first[key])


def test_ready_tasks_does_not_lose_executing_holder_on_second_status_iteration() -> (
    None
):
    packets = tuple(
        replace(_packet(name), resource_keys=("side-effect:exclusive",))
        for name in ("A", "B")
    )
    first = {"A": TaskStatus(TaskPhase.EXECUTING), "B": TaskStatus(TaskPhase.READY)}
    statuses = _ChangingStatuses(first, {"B": first["B"]})
    # The later key observation omits the holder, though a direct lookup can
    # still observe A executing. The first coherent roster must govern admission.
    assert ready_tasks(packets, statuses, leased_task_ids=()) == ()
    assert statuses.iterations == 1
    assert statuses.reads == {"A": 1, "B": 1}


def test_ready_tasks_does_not_reobserve_dependency_status_after_admission() -> None:
    packets = (
        _packet("A"),
        replace(_packet("B"), dependencies=(TaskDependency("A", DependencyKind.HARD),)),
    )
    first = {"A": TaskStatus(TaskPhase.PLANNED), "B": TaskStatus(TaskPhase.READY)}
    later = {"A": TaskStatus(TaskPhase.COMPLETE), "B": first["B"]}
    statuses = _ChangingStatuses(first, later)
    assert ready_tasks(packets, statuses, leased_task_ids=()) == ()
    assert statuses.iterations == 1
    assert statuses.reads == {"A": 1, "B": 1}
    assert ready_tasks(packets, later, leased_task_ids=()) == ("B",)
