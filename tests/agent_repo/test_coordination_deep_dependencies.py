"""Regression tests for dependency graph validation beyond recursion depth."""

from __future__ import annotations

import sys

import pytest

from tools.agent_repo.coordination.dependencies import DependencyGraph
from tools.agent_repo.coordination.model import (
    DependencyKind,
    ExecutionMode,
    TaskDependency,
    TaskPacket,
)


def _packet(task_id: str, dependencies: tuple[TaskDependency, ...]) -> TaskPacket:
    return TaskPacket(
        task_id=task_id,
        task_revision=1,
        parent_issue=1067,
        title=task_id,
        objective="Synthetic graph validation",
        execution_mode=ExecutionMode.READ_ONLY,
        base_sha="a" * 40,
        dependencies=dependencies,
    )


def _chain(*, cyclic: bool) -> tuple[TaskPacket, ...]:
    length = sys.getrecursionlimit() + 10
    kinds = tuple(DependencyKind)
    return tuple(
        _packet(
            f"T{index:08d}",
            (TaskDependency(f"T{(index + 1) % length:08d}", kinds[index % len(kinds)]),)
            if index + 1 < length or cyclic
            else (),
        )
        for index in range(length)
    )


def test_deep_acyclic_dependency_graph() -> None:
    packets = _chain(cyclic=False)
    graph = DependencyGraph(packets)
    assert graph.task_ids == tuple(packet.task_id for packet in packets)


def test_deep_cycle_has_contract_error_not_recursion_error() -> None:
    with pytest.raises(ValueError, match="dependency graph contains a cycle"):
        DependencyGraph(_chain(cyclic=True))


def test_diamond_and_disconnected_component_are_acyclic() -> None:
    packets = (
        _packet(
            "A",
            (
                TaskDependency("B", DependencyKind.HARD),
                TaskDependency("C", DependencyKind.EVIDENCE),
            ),
        ),
        _packet("B", (TaskDependency("D", DependencyKind.INTEGRATION),)),
        _packet("C", (TaskDependency("D", DependencyKind.HARD),)),
        _packet("D", ()),
        _packet("E", ()),
    )
    assert DependencyGraph(packets).task_ids == ("A", "B", "C", "D", "E")
