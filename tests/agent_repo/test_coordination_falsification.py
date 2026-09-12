from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from tools.agent_repo.coordination.conflicts import ConflictLevel, classify_conflict
from tools.agent_repo.coordination.github_state import parse_status_comment
from tools.agent_repo.coordination.leases import (
    LeaseReconciliation,
    ReconciliationDecision,
    grant_lease,
    heartbeat,
    lease_is_current,
)
from tools.agent_repo.coordination.model import (
    Capability,
    EvidenceKind,
    EvidenceRecord,
    EvidenceResult,
    ExecutionMode,
    TaskCondition,
    TaskPacket,
    TaskPhase,
    TaskStatus,
    WriteScope,
)
from tools.agent_repo.coordination.scheduler import ready_tasks
from tools.agent_repo.coordination.state import (
    evidence_is_current,
    evidence_is_current_after_peer_integration,
    parent_completion_allowed,
)

BASE_SHA = "a" * 40
HEAD_A = "b" * 40
HEAD_B = "c" * 40
MAIN_A = "d" * 40
MAIN_B = "e" * 40
NOW = datetime(2026, 9, 12, 12, 0, tzinfo=UTC)


def _packet(
    task_id: str,
    *,
    objective: str = "coordinate one isolated task",
    revision: int = 1,
    mode: ExecutionMode = ExecutionMode.WRITE,
    resource_keys: tuple[str, ...] = (),
) -> TaskPacket:
    write_scope = (
        WriteScope()
        if mode is ExecutionMode.READ_ONLY
        else WriteScope(allow=("tools/agent_repo/**",))
    )
    capabilities = (
        (Capability.REPOSITORY_READ,)
        if mode is ExecutionMode.READ_ONLY
        else (Capability.REPOSITORY_READ, Capability.LEASE_BRANCH_WRITE)
    )
    return TaskPacket(
        task_id=task_id,
        task_revision=revision,
        parent_issue=500,
        title=f"Task {task_id}",
        objective=objective,
        execution_mode=mode,
        write_scope=write_scope,
        resource_keys=resource_keys,
        capabilities=capabilities,
        acceptance_criteria=("observable oracle",),
        test_oracle=("deterministic state transition",),
        base_sha=BASE_SHA,
    )


def _evidence(
    packet: TaskPacket,
    *,
    kind: EvidenceKind = EvidenceKind.REVIEW,
    head_sha: str = HEAD_A,
    main_sha: str | None = None,
) -> EvidenceRecord:
    return EvidenceRecord(
        evidence_kind=kind,
        task_id=packet.task_id,
        task_revision=packet.task_revision,
        task_contract_digest=packet.contract_digest(),
        base_sha=packet.base_sha,
        head_sha=head_sha,
        identifier="evidence:1",
        result=EvidenceResult.PASS,
        main_sha=main_sha,
    )


def test_two_workers_cannot_race_to_replace_one_active_lease() -> None:
    packet = _packet("T500-race")
    first = grant_lease(
        packet,
        "agent-a",
        epoch=1,
        now=NOW,
        ttl=timedelta(minutes=15),
    )

    with pytest.raises(ValueError, match="active lease"):
        grant_lease(
            packet,
            "agent-b",
            epoch=2,
            now=NOW + timedelta(minutes=1),
            ttl=timedelta(minutes=15),
            previous=first,
            reconciliation=LeaseReconciliation(
                task_id=packet.task_id,
                expired_epoch=first.epoch,
                observed_branch=first.lease_branch,
                observed_head_sha=None,
                decision=ReconciliationDecision.REASSIGN,
            ),
        )


def test_expired_lease_cannot_be_revived_by_late_heartbeat() -> None:
    packet = _packet("T500-heartbeat")
    lease = grant_lease(
        packet,
        "agent-a",
        epoch=1,
        now=NOW,
        ttl=timedelta(minutes=5),
    )

    with pytest.raises(ValueError, match="expired"):
        heartbeat(
            lease,
            head_sha=HEAD_A,
            observed_at=lease.expires_at,
        )


def test_old_epoch_is_fenced_after_reassignment_and_late_push() -> None:
    packet = _packet("T500-epoch")
    old = heartbeat(
        grant_lease(
            packet,
            "agent-a",
            epoch=1,
            now=NOW,
            ttl=timedelta(minutes=5),
        ),
        head_sha=HEAD_A,
        observed_at=NOW + timedelta(minutes=1),
    )
    reassigned_at = old.expires_at + timedelta(seconds=1)
    current = grant_lease(
        packet,
        "agent-b",
        epoch=2,
        now=reassigned_at,
        ttl=timedelta(minutes=5),
        previous=old,
        reconciliation=LeaseReconciliation(
            task_id=packet.task_id,
            expired_epoch=old.epoch,
            observed_branch=old.lease_branch,
            observed_head_sha=HEAD_A,
            decision=ReconciliationDecision.REASSIGN,
            pr_ids=("pr:1",),
            ci_ids=("ci:1",),
        ),
    )

    assert current.epoch == old.epoch + 1
    assert current.lease_branch != old.lease_branch
    assert not lease_is_current(old, packet, now=reassigned_at)
    assert lease_is_current(current, packet, now=reassigned_at)


def test_contract_drift_without_revision_bump_invalidates_lease_and_evidence() -> None:
    original = _packet("T500-contract", objective="original objective")
    changed = _packet("T500-contract", objective="changed objective")
    lease = grant_lease(
        original,
        "agent-a",
        epoch=1,
        now=NOW,
        ttl=timedelta(minutes=10),
    )
    evidence = _evidence(original)

    assert original.task_revision == changed.task_revision
    assert original.contract_digest() != changed.contract_digest()
    assert not lease_is_current(lease, changed, now=NOW + timedelta(minutes=1))
    assert not evidence_is_current(evidence, changed, head_sha=HEAD_A)


def test_old_head_review_and_old_main_integration_evidence_are_stale() -> None:
    packet = _packet("T500-evidence")
    review = _evidence(packet, head_sha=HEAD_A)
    integration = _evidence(
        packet,
        kind=EvidenceKind.INTEGRATION,
        head_sha=HEAD_B,
        main_sha=MAIN_A,
    )

    assert not evidence_is_current(review, packet, head_sha=HEAD_B)
    assert not evidence_is_current(
        integration,
        packet,
        head_sha=HEAD_B,
        current_main_sha=MAIN_B,
    )
    assert evidence_is_current(
        integration,
        packet,
        head_sha=HEAD_B,
        current_main_sha=MAIN_A,
    )


def test_soft_conflicting_peer_integration_invalidates_current_review_evidence(
) -> None:
    peer = _packet(
        "T500-peer",
        resource_keys=("file:tools/peer.py", "authority:shared-analysis"),
    )
    integrated = _packet(
        "T500-integrated",
        resource_keys=("file:tools/integrated.py", "authority:shared-analysis"),
    )
    independent = _packet(
        "T500-independent",
        resource_keys=("authority:unrelated",),
    )
    evidence = _evidence(peer)

    assert classify_conflict(peer, integrated) is ConflictLevel.SOFT
    assert evidence_is_current(evidence, peer, head_sha=HEAD_A)
    assert not evidence_is_current_after_peer_integration(
        evidence,
        peer,
        integrated,
        head_sha=HEAD_A,
    )
    assert evidence_is_current_after_peer_integration(
        evidence,
        peer,
        independent,
        head_sha=HEAD_A,
    )


def test_file_disjoint_tasks_sharing_identity_are_serialized() -> None:
    left = _packet(
        "T500-identity-a",
        resource_keys=("file:tools/a.py", "identity:market-dataset-feature-tensor"),
    )
    right = _packet(
        "T500-identity-b",
        resource_keys=("file:tools/b.py", "identity:market-dataset-feature-tensor"),
    )
    statuses = {
        left.task_id: TaskStatus(TaskPhase.READY, TaskCondition.HEALTHY),
        right.task_id: TaskStatus(TaskPhase.READY, TaskCondition.HEALTHY),
    }

    assert classify_conflict(left, right) is ConflictLevel.HARD
    assert ready_tasks(
        (left, right),
        statuses,
        leased_task_ids=(),
    ) == (left.task_id,)


def test_parent_completion_requires_children_and_parent_level_oracles() -> None:
    complete_children = {
        "T500-a": TaskStatus(TaskPhase.COMPLETE, TaskCondition.HEALTHY),
        "T500-b": TaskStatus(TaskPhase.COMPLETE, TaskCondition.HEALTHY),
    }

    assert not parent_completion_allowed(
        complete_children,
        parent_acceptance_satisfied=False,
        parent_invariants_satisfied=True,
    )
    assert not parent_completion_allowed(
        complete_children,
        parent_acceptance_satisfied=True,
        parent_invariants_satisfied=False,
    )
    assert parent_completion_allowed(
        complete_children,
        parent_acceptance_satisfied=True,
        parent_invariants_satisfied=True,
    )

    incomplete_children = dict(complete_children)
    incomplete_children["T500-b"] = TaskStatus(
        TaskPhase.VERIFICATION,
        TaskCondition.HEALTHY,
    )
    assert not parent_completion_allowed(
        incomplete_children,
        parent_acceptance_satisfied=True,
        parent_invariants_satisfied=True,
    )


def test_malformed_partial_status_projection_fails_closed() -> None:
    partial = (
        "<!-- agent-coordination:T500-bad -->\n"
        "```json\n"
        '{"task_id":"T500-bad","task_revision":1}\n'
        "```\n"
    )
    with pytest.raises(ValueError, match="missing field"):
        parse_status_comment(partial)


def test_read_only_task_cannot_obtain_write_lease() -> None:
    packet = _packet("T500-read", mode=ExecutionMode.READ_ONLY)
    with pytest.raises(ValueError, match="read_only"):
        grant_lease(
            packet,
            "agent-a",
            epoch=1,
            now=NOW,
            ttl=timedelta(minutes=5),
        )
