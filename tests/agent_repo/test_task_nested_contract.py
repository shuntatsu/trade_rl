from dataclasses import replace
from typing import cast

import pytest

from tools.agent_repo.coordination.model import (
    DependencyKind,
    ExecutionMode,
    TaskDependency,
    TaskPacket,
    WriteScope,
)


class MutableScope:
    allow = ("tools/**",)
    deny: tuple[str, ...] = ()

    def canonical_payload(self) -> dict[str, list[str]]:
        return {"allow": list(self.allow), "deny": list(self.deny)}


class MutableDependency:
    task_id = "T1067-parent"
    kind = DependencyKind.HARD

    def canonical_payload(self) -> dict[str, str]:
        return {"task_id": self.task_id, "kind": "hard", "requires_phase": "complete"}


def _packet() -> TaskPacket:
    return TaskPacket(
        task_id="T1067-child",
        task_revision=1,
        parent_issue=1067,
        title="Synthetic task",
        objective="Check immutable nested contracts",
        execution_mode=ExecutionMode.WRITE,
        write_scope=WriteScope(allow=("tools/**",)),
        base_sha="a" * 40,
    )


def test_packet_rejects_mutable_scope_impostor() -> None:
    with pytest.raises(ValueError, match="write_scope"):
        replace(_packet(), write_scope=cast(WriteScope, MutableScope()))


def test_packet_rejects_mutable_dependency_impostor() -> None:
    with pytest.raises(ValueError, match="dependencies"):
        replace(_packet(), dependencies=(cast(TaskDependency, MutableDependency()),))


def test_packet_rejects_null_scope() -> None:
    with pytest.raises(ValueError, match="write_scope"):
        replace(_packet(), write_scope=cast(WriteScope, None))


def test_packet_snapshots_valid_dependency_list() -> None:
    dependency = TaskDependency("T1067-parent", DependencyKind.HARD)
    roster = [dependency]
    packet = replace(_packet(), dependencies=cast(tuple[TaskDependency, ...], roster))
    digest = packet.contract_digest()
    roster.clear()
    assert packet.dependencies == (dependency,)
    assert packet.contract_digest() == digest


@pytest.mark.parametrize("value", [None, "", {}, set()])
def test_packet_rejects_malformed_dependency_container(value: object) -> None:
    with pytest.raises(ValueError, match="dependencies"):
        replace(_packet(), dependencies=cast(tuple[TaskDependency, ...], value))


def test_packet_rejects_scope_subclass_with_mutable_semantics() -> None:
    class ScopeSubclass(WriteScope):
        pass

    with pytest.raises(ValueError, match="write_scope"):
        replace(_packet(), write_scope=ScopeSubclass(allow=("tools/**",)))
