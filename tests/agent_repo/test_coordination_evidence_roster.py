"""Evidence dependency IDs must be explicit collection entries."""

from collections import UserString, deque

import pytest

from tools.agent_repo.coordination.dependencies import DependencyGraph
from tools.agent_repo.coordination.model import (
    DependencyKind,
    ExecutionMode,
    TaskDependency,
    TaskPacket,
)


def _graph() -> DependencyGraph:
    root = TaskPacket(
        task_id="A",
        task_revision=1,
        parent_issue=1067,
        title="A",
        objective="Synthetic evidence gate",
        execution_mode=ExecutionMode.READ_ONLY,
        base_sha="a" * 40,
    )
    child = TaskPacket(
        task_id="B",
        task_revision=1,
        parent_issue=1067,
        title="B",
        objective="Synthetic evidence consumer",
        execution_mode=ExecutionMode.READ_ONLY,
        base_sha="a" * 40,
        dependencies=(TaskDependency("A", DependencyKind.EVIDENCE),),
    )
    return DependencyGraph((root, child))


@pytest.mark.parametrize(
    "invalid", ["A", "AB", UserString("A"), {"A": False}, b"", ["A", 1], None]
)
def test_evidence_rejects_malformed_roster(invalid: object) -> None:
    with pytest.raises(ValueError, match="evidence_task_ids"):
        _graph().evidence_dependencies_satisfied(
            "B",
            evidence_task_ids=invalid,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "valid", [("A",), ["A"], {"A"}, frozenset({"A"}), deque(["A"])]
)
def test_evidence_accepts_explicit_collection(valid: object) -> None:
    assert _graph().evidence_dependencies_satisfied(
        "B",
        evidence_task_ids=valid,  # type: ignore[arg-type]
    )


def test_evidence_does_not_split_valid_multi_character_entry() -> None:
    assert not _graph().evidence_dependencies_satisfied("B", evidence_task_ids=("AB",))
    assert not _graph().evidence_dependencies_satisfied("B", evidence_task_ids=())
