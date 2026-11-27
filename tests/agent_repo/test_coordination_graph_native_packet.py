"""Native task graph boundary regressions (synthetic only)."""

from dataclasses import replace
from typing import Any

import pytest

from tools.agent_repo.coordination.dependencies import DependencyGraph
from tools.agent_repo.coordination.model import (
    DependencyKind,
    ExecutionMode,
    TaskDependency,
    TaskPacket,
    TaskPhase,
    TaskStatus,
)
from tools.agent_repo.coordination.scheduler import ready_tasks


def _packet(task_id: str = "A") -> TaskPacket:
    return TaskPacket(
        task_id=task_id,
        task_revision=1,
        parent_issue=1067,
        title="Synthetic task",
        objective="Validate native graph inputs",
        execution_mode=ExecutionMode.READ_ONLY,
        base_sha="a" * 40,
    )


class _ForeignPacket(TaskPacket):
    pass


class _TrapPacket(TaskPacket):
    def __getattribute__(self, name: str) -> Any:
        raise AssertionError("foreign packet attribute accessed")


class _ForeignID(str):
    armed = False

    def __hash__(self) -> int:
        if self.armed:
            raise AssertionError("foreign identity hash accessed")
        return super().__hash__()


@pytest.mark.parametrize(
    "foreign",
    [
        lambda: _ForeignPacket(**vars(_packet())),
        lambda: object.__new__(_TrapPacket),
        lambda: type("DuckPacket", (), vars(_packet()))(),
    ],
)
def test_graph_refuses_foreign_packet_before_attributes(foreign: Any) -> None:
    with pytest.raises(ValueError, match="native TaskPacket"):
        DependencyGraph((foreign(),))


def test_scheduler_refuses_foreign_packet() -> None:
    packet = _ForeignPacket(**vars(_packet()))
    with pytest.raises(ValueError, match="native TaskPacket"):
        ready_tasks((packet,), {"A": TaskStatus(TaskPhase.READY)}, leased_task_ids=())


def test_graph_refuses_non_native_task_id() -> None:
    task_id = _ForeignID("A")
    packet = _packet(task_id)
    task_id.armed = True
    with pytest.raises(ValueError, match="native strings"):
        DependencyGraph((packet,))


def test_graph_refuses_non_native_dependency_id() -> None:
    task_id = _ForeignID("A")
    child = replace(
        _packet("B"),
        dependencies=(TaskDependency(task_id, DependencyKind.HARD),),
    )
    task_id.armed = True
    with pytest.raises(ValueError, match="native strings"):
        DependencyGraph((_packet(), child))


def test_native_graph_and_scheduler_still_accept_dependencies() -> None:
    child = replace(
        _packet("B"), dependencies=(TaskDependency("A", DependencyKind.HARD),)
    )
    packets = (_packet(), child)
    statuses = {"A": TaskStatus(TaskPhase.COMPLETE), "B": TaskStatus(TaskPhase.READY)}
    assert DependencyGraph(packets).task_ids == ("A", "B")
    assert ready_tasks(packets, statuses, leased_task_ids=()) == ("B",)
