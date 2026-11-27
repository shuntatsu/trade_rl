"""Malformed status records must not permit scheduler admission."""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pytest

from tests.agent_repo.test_coordination_scheduler import _packet
from tools.agent_repo.coordination.model import (
    DependencyKind,
    TaskCondition,
    TaskDependency,
    TaskPhase,
    TaskStatus,
)
from tools.agent_repo.coordination.scheduler import ready_tasks


class _TrapStatus(TaskStatus):
    def __getattribute__(self, name: str) -> object:
        if name in {"phase", "condition"}:
            raise AssertionError("foreign status attribute accessed")
        return super().__getattribute__(name)


class _StatusKey(str):
    pass


def test_duck_typed_ready_status_is_rejected() -> None:
    forged = cast(
        TaskStatus,
        SimpleNamespace(phase=TaskPhase.READY, condition=TaskCondition.HEALTHY),
    )
    with pytest.raises(ValueError, match="native TaskStatus"):
        ready_tasks((_packet("A"),), {"A": forged}, leased_task_ids=())


def test_duck_typed_dependency_status_is_rejected() -> None:
    forged = cast(
        TaskStatus,
        SimpleNamespace(phase=TaskPhase.COMPLETE, condition=TaskCondition.HEALTHY),
    )
    packets = (
        _packet("A"),
        _packet("B", dependencies=(TaskDependency("A", DependencyKind.HARD),)),
    )
    with pytest.raises(ValueError, match="native TaskStatus"):
        ready_tasks(
            packets,
            {"A": forged, "B": TaskStatus(TaskPhase.READY)},
            leased_task_ids=(),
        )


def test_status_subclass_is_rejected_before_attribute_access() -> None:
    forged = object.__new__(_TrapStatus)
    with pytest.raises(ValueError, match="native TaskStatus"):
        ready_tasks((_packet("A"),), {"A": forged}, leased_task_ids=())


def test_status_key_subclass_is_rejected() -> None:
    with pytest.raises(ValueError, match="native strings"):
        ready_tasks(
            (_packet("A"),),
            {_StatusKey("A"): TaskStatus(TaskPhase.READY)},
            leased_task_ids=(),
        )


def test_native_status_dependency_and_conflict_controls() -> None:
    packets = (
        _packet("A"),
        _packet("B", dependencies=(TaskDependency("A", DependencyKind.HARD),)),
    )
    statuses = {"A": TaskStatus(TaskPhase.PLANNED), "B": TaskStatus(TaskPhase.READY)}
    assert ready_tasks(packets, statuses, leased_task_ids=()) == ()
    statuses["A"] = TaskStatus(TaskPhase.COMPLETE)
    assert ready_tasks(packets, statuses, leased_task_ids=()) == ("B",)
