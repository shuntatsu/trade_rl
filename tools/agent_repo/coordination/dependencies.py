"""Dependency graph rules for Agent Coordination Plane tasks."""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence

from tools.agent_repo.coordination.model import (
    DependencyKind,
    TaskCondition,
    TaskPacket,
    TaskPhase,
    TaskStatus,
)

_PHASE_ORDER = {
    TaskPhase.PLANNED: 0,
    TaskPhase.READY: 1,
    TaskPhase.EXECUTING: 2,
    TaskPhase.REVIEW: 3,
    TaskPhase.VERIFICATION: 4,
    TaskPhase.INTEGRATION: 5,
    TaskPhase.COMPLETE: 6,
}


class DependencyGraph:
    """Validated acyclic task dependency graph."""

    def __init__(self, packets: Sequence[TaskPacket]) -> None:
        self._packets = {packet.task_id: packet for packet in packets}
        if len(self._packets) != len(packets):
            raise ValueError("duplicate task_id in dependency graph")
        for packet in packets:
            for dependency in packet.dependencies:
                if dependency.task_id not in self._packets:
                    raise ValueError(
                        f"unknown dependency {dependency.task_id} for {packet.task_id}"
                    )
        self._assert_acyclic()

    @property
    def task_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._packets))

    def _assert_acyclic(self) -> None:
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(task_id: str) -> None:
            if task_id in visiting:
                raise ValueError("dependency graph contains a cycle")
            if task_id in visited:
                return
            visiting.add(task_id)
            for dependency in self._packets[task_id].dependencies:
                visit(dependency.task_id)
            visiting.remove(task_id)
            visited.add(task_id)

        for task_id in sorted(self._packets):
            visit(task_id)

    def execution_dependencies_satisfied(
        self,
        task_id: str,
        statuses: Mapping[str, TaskStatus],
    ) -> bool:
        """Return whether hard dependencies permit task execution."""

        packet = self._packets[task_id]
        for dependency in packet.dependencies:
            if dependency.kind is not DependencyKind.HARD:
                continue
            status = statuses.get(dependency.task_id)
            if status is None or status.condition is not TaskCondition.HEALTHY:
                return False
            if _PHASE_ORDER[status.phase] < _PHASE_ORDER[dependency.requires_phase]:
                return False
        return True

    def evidence_dependencies_satisfied(
        self,
        task_id: str,
        *,
        evidence_task_ids: Collection[str],
    ) -> bool:
        """Return whether evidence dependencies permit verification/completion."""

        evidence_ids = set(evidence_task_ids)
        return all(
            dependency.task_id in evidence_ids
            for dependency in self._packets[task_id].dependencies
            if dependency.kind is DependencyKind.EVIDENCE
        )

    def integration_dependencies_satisfied(
        self,
        task_id: str,
        statuses: Mapping[str, TaskStatus],
    ) -> bool:
        """Return whether integration dependencies permit integration."""

        packet = self._packets[task_id]
        for dependency in packet.dependencies:
            if dependency.kind is not DependencyKind.INTEGRATION:
                continue
            status = statuses.get(dependency.task_id)
            if (
                status is None
                or status.condition is not TaskCondition.HEALTHY
                or status.phase is not TaskPhase.COMPLETE
            ):
                return False
        return True

    def descendant_count(self, task_id: str) -> int:
        descendants: set[str] = set()
        frontier = [task_id]
        while frontier:
            parent = frontier.pop()
            for candidate in self._packets.values():
                if candidate.task_id in descendants:
                    continue
                if any(dep.task_id == parent for dep in candidate.dependencies):
                    descendants.add(candidate.task_id)
                    frontier.append(candidate.task_id)
        return len(descendants)


__all__ = ["DependencyGraph"]
