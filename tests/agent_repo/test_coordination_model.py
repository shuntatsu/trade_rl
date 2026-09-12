from __future__ import annotations

import pytest

from tools.agent_repo.coordination.model import (
    Capability,
    DependencyKind,
    EvidenceKind,
    EvidenceRecord,
    EvidenceResult,
    ExecutionMode,
    RiskLevel,
    TaskCondition,
    TaskDependency,
    TaskPacket,
    TaskPhase,
    TaskStatus,
    WriteScope,
)
from tools.agent_repo.coordination.state import evidence_is_current, validate_transition

BASE_SHA = "a" * 40
HEAD_SHA = "b" * 40
OTHER_HEAD_SHA = "c" * 40
MAIN_SHA = "d" * 40
OTHER_MAIN_SHA = "e" * 40


def _packet(
    *,
    revision: int = 1,
    objective: str = "Implement coordination state",
    title: str = "Coordination state",
    execution_mode: ExecutionMode = ExecutionMode.WRITE,
) -> TaskPacket:
    return TaskPacket(
        task_id="T500-01",
        task_revision=revision,
        parent_issue=500,
        title=title,
        objective=objective,
        execution_mode=execution_mode,
        non_goals=("no distributed lock service",),
        dependencies=(
            TaskDependency(
                task_id="T500-00",
                kind=DependencyKind.HARD,
                requires_phase=TaskPhase.COMPLETE,
            ),
        ),
        write_scope=(
            WriteScope(
                allow=("tools/agent_repo/**", "tests/agent_repo/**"),
                deny=("trade_rl/**",),
            )
            if execution_mode is ExecutionMode.WRITE
            else WriteScope()
        ),
        resource_keys=(
            "authority:agent-coordination-task-state",
            "workflow:agent-task-claim",
        ),
        capabilities=(
            Capability.REPOSITORY_READ,
            *(
                (Capability.LEASE_BRANCH_WRITE, Capability.DRAFT_PR_WRITE)
                if execution_mode is ExecutionMode.WRITE
                else ()
            ),
        ),
        acceptance_criteria=("duplicate claim is rejected",),
        test_oracle=("deterministic transition",),
        base_sha=BASE_SHA,
        risk=RiskLevel.MEDIUM,
        deliverable="pull_request",
    )


def _evidence(
    packet: TaskPacket,
    *,
    kind: EvidenceKind = EvidenceKind.REVIEW,
    head_sha: str = HEAD_SHA,
    main_sha: str | None = None,
) -> EvidenceRecord:
    return EvidenceRecord(
        evidence_kind=kind,
        task_id=packet.task_id,
        task_revision=packet.task_revision,
        task_contract_digest=packet.contract_digest(),
        base_sha=packet.base_sha,
        head_sha=head_sha,
        identifier="evidence-1",
        result=EvidenceResult.PASS,
        main_sha=main_sha,
    )


def test_task_contract_digest_is_deterministic_and_semantic() -> None:
    left = _packet()
    right = _packet()

    assert left.canonical_payload() == right.canonical_payload()
    assert left.contract_digest() == right.contract_digest()
    assert len(left.contract_digest()) == 64

    changed = _packet(objective="Changed semantic objective")
    assert changed.contract_digest() != left.contract_digest()

    presentation_only = _packet(title="Different display title")
    assert presentation_only.contract_digest() == left.contract_digest()


def test_task_packet_rejects_invalid_identity_and_scope() -> None:
    with pytest.raises(ValueError, match="task_id"):
        TaskPacket(
            task_id="",
            task_revision=1,
            parent_issue=500,
            title="x",
            objective="x",
            execution_mode=ExecutionMode.READ_ONLY,
            base_sha=BASE_SHA,
        )

    with pytest.raises(ValueError, match="task_revision"):
        _packet(revision=0)

    with pytest.raises(ValueError, match="base_sha"):
        TaskPacket(
            task_id="T500-01",
            task_revision=1,
            parent_issue=500,
            title="x",
            objective="x",
            execution_mode=ExecutionMode.READ_ONLY,
            base_sha="not-a-sha",
        )

    with pytest.raises(ValueError, match="read_only"):
        TaskPacket(
            task_id="T500-01",
            task_revision=1,
            parent_issue=500,
            title="x",
            objective="x",
            execution_mode=ExecutionMode.READ_ONLY,
            write_scope=WriteScope(allow=("tools/**",)),
            base_sha=BASE_SHA,
        )

    with pytest.raises(ValueError, match="write.*allow"):
        TaskPacket(
            task_id="T500-01",
            task_revision=1,
            parent_issue=500,
            title="x",
            objective="x",
            execution_mode=ExecutionMode.WRITE,
            base_sha=BASE_SHA,
        )


def test_task_packet_rejects_duplicate_contract_entries() -> None:
    with pytest.raises(ValueError, match="duplicate dependency"):
        TaskPacket(
            task_id="T500-01",
            task_revision=1,
            parent_issue=500,
            title="x",
            objective="x",
            execution_mode=ExecutionMode.WRITE,
            dependencies=(
                TaskDependency("T500-00", DependencyKind.HARD),
                TaskDependency("T500-00", DependencyKind.EVIDENCE),
            ),
            write_scope=WriteScope(allow=("tools/**",)),
            base_sha=BASE_SHA,
        )

    with pytest.raises(ValueError, match="duplicate resource key"):
        TaskPacket(
            task_id="T500-01",
            task_revision=1,
            parent_issue=500,
            title="x",
            objective="x",
            execution_mode=ExecutionMode.WRITE,
            write_scope=WriteScope(allow=("tools/**",)),
            resource_keys=("identity:x", "identity:x"),
            base_sha=BASE_SHA,
        )


def test_phase_transition_contract_is_explicit() -> None:
    for current, target in (
        (TaskPhase.PLANNED, TaskPhase.READY),
        (TaskPhase.READY, TaskPhase.EXECUTING),
        (TaskPhase.EXECUTING, TaskPhase.REVIEW),
        (TaskPhase.REVIEW, TaskPhase.VERIFICATION),
        (TaskPhase.VERIFICATION, TaskPhase.INTEGRATION),
        (TaskPhase.INTEGRATION, TaskPhase.COMPLETE),
        (TaskPhase.REVIEW, TaskPhase.EXECUTING),
        (TaskPhase.VERIFICATION, TaskPhase.EXECUTING),
        (TaskPhase.INTEGRATION, TaskPhase.VERIFICATION),
        (TaskPhase.INTEGRATION, TaskPhase.EXECUTING),
    ):
        validate_transition(current, target)

    with pytest.raises(ValueError, match="transition"):
        validate_transition(TaskPhase.READY, TaskPhase.COMPLETE)
    with pytest.raises(ValueError, match="terminal"):
        validate_transition(TaskPhase.COMPLETE, TaskPhase.EXECUTING)


def test_task_status_keeps_phase_and_condition_independent() -> None:
    status = TaskStatus(
        phase=TaskPhase.REVIEW,
        condition=TaskCondition.STALE,
        reason="head_changed",
    )
    assert status.phase is TaskPhase.REVIEW
    assert status.condition is TaskCondition.STALE
    assert status.reason == "head_changed"


def test_review_evidence_is_bound_to_exact_task_contract_and_head() -> None:
    packet = _packet()
    evidence = _evidence(packet)

    assert evidence_is_current(evidence, packet, head_sha=HEAD_SHA)
    assert not evidence_is_current(evidence, packet, head_sha=OTHER_HEAD_SHA)
    assert not evidence_is_current(evidence, _packet(revision=2), head_sha=HEAD_SHA)
    assert not evidence_is_current(
        evidence,
        _packet(objective="new objective"),
        head_sha=HEAD_SHA,
    )


def test_integration_evidence_is_also_bound_to_current_main() -> None:
    packet = _packet()
    evidence = _evidence(
        packet,
        kind=EvidenceKind.INTEGRATION,
        main_sha=MAIN_SHA,
    )

    assert evidence_is_current(
        evidence,
        packet,
        head_sha=HEAD_SHA,
        current_main_sha=MAIN_SHA,
    )
    assert not evidence_is_current(
        evidence,
        packet,
        head_sha=HEAD_SHA,
        current_main_sha=OTHER_MAIN_SHA,
    )
    assert not evidence_is_current(evidence, packet, head_sha=HEAD_SHA)


def test_evidence_rejects_malformed_sha_and_integration_without_main() -> None:
    packet = _packet()
    with pytest.raises(ValueError, match="head_sha"):
        EvidenceRecord(
            evidence_kind=EvidenceKind.REVIEW,
            task_id=packet.task_id,
            task_revision=packet.task_revision,
            task_contract_digest=packet.contract_digest(),
            base_sha=packet.base_sha,
            head_sha="bad",
            identifier="evidence-1",
            result=EvidenceResult.PASS,
        )

    with pytest.raises(ValueError, match="main_sha"):
        _evidence(packet, kind=EvidenceKind.INTEGRATION)
