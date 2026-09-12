from __future__ import annotations

import pytest

from tools.agent_repo.coordination.conflicts import ConflictLevel, classify_conflict
from tools.agent_repo.coordination.dependencies import DependencyGraph
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
from tools.agent_repo.coordination.scheduler import ready_tasks

BASE_SHA = "a" * 40


def _packet(
    task_id: str,
    *,
    mode: ExecutionMode = ExecutionMode.WRITE,
    dependencies: tuple[TaskDependency, ...] = (),
    resource_keys: tuple[str, ...] = (),
    risk: RiskLevel = RiskLevel.MEDIUM,
) -> TaskPacket:
    return TaskPacket(
        task_id=task_id,
        task_revision=1,
        parent_issue=500,
        title=task_id,
        objective=f"Implement {task_id}",
        execution_mode=mode,
        dependencies=dependencies,
        write_scope=(
            WriteScope(allow=(f"tools/{task_id}/**",))
            if mode is ExecutionMode.WRITE
            else WriteScope()
        ),
        resource_keys=resource_keys,
        capabilities=(
            (Capability.REPOSITORY_READ, Capability.LEASE_BRANCH_WRITE)
            if mode is ExecutionMode.WRITE
            else (Capability.REPOSITORY_READ,)
        ),
        acceptance_criteria=("works",),
        test_oracle=("deterministic",),
        base_sha=BASE_SHA,
        risk=risk,
    )


def _healthy(phase: TaskPhase) -> TaskStatus:
    return TaskStatus(phase=phase, condition=TaskCondition.HEALTHY)


def test_dependency_graph_rejects_missing_targets_and_cycles() -> None:
    missing = _packet(
        "T2",
        dependencies=(TaskDependency("T1", DependencyKind.HARD),),
    )
    with pytest.raises(ValueError, match="unknown dependency"):
        DependencyGraph((missing,))

    first = _packet(
        "T1",
        dependencies=(TaskDependency("T2", DependencyKind.HARD),),
    )
    second = _packet(
        "T2",
        dependencies=(TaskDependency("T1", DependencyKind.HARD),),
    )
    with pytest.raises(ValueError, match="cycle"):
        DependencyGraph((first, second))


def test_dependency_kinds_apply_at_distinct_phase_gates() -> None:
    root = _packet("T0")
    hard = _packet(
        "TH",
        dependencies=(
            TaskDependency(
                "T0",
                DependencyKind.HARD,
                requires_phase=TaskPhase.VERIFICATION,
            ),
        ),
    )
    evidence = _packet(
        "TE",
        dependencies=(TaskDependency("T0", DependencyKind.EVIDENCE),),
    )
    integration = _packet(
        "TI",
        dependencies=(TaskDependency("T0", DependencyKind.INTEGRATION),),
    )
    graph = DependencyGraph((root, hard, evidence, integration))

    statuses = {
        "T0": _healthy(TaskPhase.VERIFICATION),
        "TH": _healthy(TaskPhase.READY),
        "TE": _healthy(TaskPhase.READY),
        "TI": _healthy(TaskPhase.READY),
    }
    assert graph.execution_dependencies_satisfied("TH", statuses)
    assert graph.execution_dependencies_satisfied("TE", statuses)
    assert graph.execution_dependencies_satisfied("TI", statuses)

    assert not graph.evidence_dependencies_satisfied("TE", evidence_task_ids=())
    assert graph.evidence_dependencies_satisfied("TE", evidence_task_ids=("T0",))

    assert not graph.integration_dependencies_satisfied("TI", statuses)
    statuses["T0"] = _healthy(TaskPhase.COMPLETE)
    assert graph.integration_dependencies_satisfied("TI", statuses)


def test_evidence_and_integration_dependencies_allow_parallel_ready_execution() -> None:
    root = _packet("T0")
    evidence = _packet(
        "TE",
        dependencies=(TaskDependency("T0", DependencyKind.EVIDENCE),),
    )
    integration = _packet(
        "TI",
        dependencies=(TaskDependency("T0", DependencyKind.INTEGRATION),),
    )
    statuses = {
        "T0": _healthy(TaskPhase.EXECUTING),
        "TE": _healthy(TaskPhase.READY),
        "TI": _healthy(TaskPhase.READY),
    }

    assert ready_tasks(
        (root, evidence, integration),
        statuses,
        leased_task_ids=(),
    ) == ("TE", "TI")


def test_read_only_snapshot_work_does_not_conflict_on_shared_read_resources() -> None:
    left = _packet(
        "R1",
        mode=ExecutionMode.READ_ONLY,
        resource_keys=("identity:market-dataset", "authority:feature-computation"),
    )
    right = _packet(
        "R2",
        mode=ExecutionMode.READ_ONLY,
        resource_keys=("identity:market-dataset", "authority:feature-computation"),
    )
    assert classify_conflict(left, right) is ConflictLevel.NONE


def test_conflict_model_distinguishes_hard_soft_and_none() -> None:
    assert (
        classify_conflict(
            _packet("A", resource_keys=("file:docs/**",)),
            _packet("B", resource_keys=("file:docs/AGENTS.md",)),
        )
        is ConflictLevel.HARD
    )
    for key in (
        "identity:market-dataset",
        "schema:candidate-run-v2",
        "side-effect:github-task-status",
    ):
        assert (
            classify_conflict(
                _packet("A", resource_keys=(key,)),
                _packet("B", resource_keys=(key,)),
            )
            is ConflictLevel.HARD
        )
    for key in (
        "authority:experiment-analysis",
        "workflow:canonical-m2-bootstrap",
        "artifact:evidence-set",
    ):
        assert (
            classify_conflict(
                _packet("A", resource_keys=(key,)),
                _packet("B", resource_keys=(key,)),
            )
            is ConflictLevel.SOFT
        )
    assert (
        classify_conflict(
            _packet("A", resource_keys=("authority:a",)),
            _packet("B", resource_keys=("authority:b",)),
        )
        is ConflictLevel.NONE
    )


def test_conflict_model_fails_closed_for_unknown_resource_namespace() -> None:
    with pytest.raises(ValueError, match="resource namespace"):
        classify_conflict(
            _packet("A", resource_keys=("mystery:x",)),
            _packet("B"),
        )


def test_scheduler_returns_dependency_satisfied_nonconflicting_ready_set() -> None:
    root = _packet("T0")
    first = _packet(
        "T1",
        dependencies=(TaskDependency("T0", DependencyKind.HARD),),
        resource_keys=("identity:shared",),
        risk=RiskLevel.LOW,
    )
    second = _packet(
        "T2",
        dependencies=(TaskDependency("T0", DependencyKind.HARD),),
        resource_keys=("identity:shared",),
        risk=RiskLevel.HIGH,
    )
    independent = _packet(
        "T3",
        dependencies=(TaskDependency("T0", DependencyKind.HARD),),
        resource_keys=("authority:other",),
    )
    packets = (root, first, second, independent)
    statuses = {
        "T0": _healthy(TaskPhase.COMPLETE),
        "T1": _healthy(TaskPhase.READY),
        "T2": _healthy(TaskPhase.READY),
        "T3": _healthy(TaskPhase.READY),
    }

    result = ready_tasks(
        packets,
        statuses,
        leased_task_ids=frozenset(),
    )

    assert "T1" in result
    assert "T2" not in result
    assert "T3" in result


def test_scheduler_excludes_leased_blocked_and_active_hard_conflicts() -> None:
    active = _packet("ACTIVE", resource_keys=("identity:shared",))
    conflict = _packet("CONFLICT", resource_keys=("identity:shared",))
    leased = _packet("LEASED", resource_keys=("authority:leased",))
    blocked = _packet("BLOCKED", resource_keys=("authority:blocked",))
    ready = _packet("READY", resource_keys=("authority:ready",))
    packets = (active, conflict, leased, blocked, ready)
    statuses = {
        "ACTIVE": _healthy(TaskPhase.EXECUTING),
        "CONFLICT": _healthy(TaskPhase.READY),
        "LEASED": _healthy(TaskPhase.READY),
        "BLOCKED": TaskStatus(TaskPhase.READY, TaskCondition.BLOCKED, "external"),
        "READY": _healthy(TaskPhase.READY),
    }

    assert ready_tasks(
        packets,
        statuses,
        leased_task_ids=frozenset({"ACTIVE", "LEASED"}),
    ) == ("READY",)
