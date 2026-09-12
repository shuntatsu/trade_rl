from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from tools.agent_repo.coordination.leases import (
    LeaseReconciliation,
    ReconciliationDecision,
    grant_lease,
    heartbeat,
    lease_branch_name,
    lease_is_current,
)
from tools.agent_repo.coordination.model import (
    Capability,
    ExecutionMode,
    TaskPacket,
    WriteScope,
)

BASE_SHA = "a" * 40
HEAD_SHA = "b" * 40
NEXT_HEAD_SHA = "c" * 40
NOW = datetime(2026, 9, 12, 12, 0, tzinfo=UTC)
TTL = timedelta(minutes=15)


def _packet(*, revision: int = 1, objective: str = "write") -> TaskPacket:
    return TaskPacket(
        task_id="T500-03",
        task_revision=revision,
        parent_issue=500,
        title="Lease protocol",
        objective=objective,
        execution_mode=ExecutionMode.WRITE,
        write_scope=WriteScope(allow=("tools/agent_repo/**",)),
        capabilities=(Capability.REPOSITORY_READ, Capability.LEASE_BRANCH_WRITE),
        base_sha=BASE_SHA,
    )


def _read_only_packet() -> TaskPacket:
    return TaskPacket(
        task_id="T500-R",
        task_revision=1,
        parent_issue=500,
        title="Read only",
        objective="Inspect evidence",
        execution_mode=ExecutionMode.READ_ONLY,
        base_sha=BASE_SHA,
    )


def test_branch_name_is_epoch_fenced_and_slug_safe() -> None:
    assert lease_branch_name("T500-03", 3, "Agent A") == "agent/T500-03/e0003-agent-a"
    assert lease_branch_name("T500-03", 4, "Agent A") != lease_branch_name(
        "T500-03", 3, "Agent A"
    )
    with pytest.raises(ValueError, match="owner"):
        lease_branch_name("T500-03", 3, "../unsafe")


def test_grant_rejects_read_only_tasks_and_duplicate_active_lease() -> None:
    with pytest.raises(ValueError, match="read_only"):
        grant_lease(_read_only_packet(), "agent-a", epoch=1, now=NOW, ttl=TTL)

    packet = _packet()
    lease = grant_lease(packet, "agent-a", epoch=1, now=NOW, ttl=TTL)
    with pytest.raises(ValueError, match="active lease"):
        grant_lease(
            packet,
            "agent-b",
            epoch=2,
            now=NOW + timedelta(minutes=1),
            ttl=TTL,
            previous=lease,
        )


def test_expiry_alone_cannot_reassign_without_matching_reconciliation() -> None:
    packet = _packet()
    old = grant_lease(packet, "agent-a", epoch=1, now=NOW, ttl=TTL)
    expired_at = NOW + TTL + timedelta(seconds=1)

    with pytest.raises(ValueError, match="reconciliation"):
        grant_lease(
            packet,
            "agent-b",
            epoch=2,
            now=expired_at,
            ttl=TTL,
            previous=old,
        )

    wrong = LeaseReconciliation(
        task_id=packet.task_id,
        expired_epoch=99,
        observed_branch=old.lease_branch,
        observed_head_sha=None,
        decision=ReconciliationDecision.REASSIGN,
    )
    with pytest.raises(ValueError, match="reconciliation"):
        grant_lease(
            packet,
            "agent-b",
            epoch=2,
            now=expired_at,
            ttl=TTL,
            previous=old,
            reconciliation=wrong,
        )


def test_reassignment_increments_epoch_and_changes_branch() -> None:
    packet = _packet()
    old = grant_lease(packet, "agent-a", epoch=3, now=NOW, ttl=TTL)
    expired_at = NOW + TTL + timedelta(seconds=1)
    reconciliation = LeaseReconciliation(
        task_id=packet.task_id,
        expired_epoch=old.epoch,
        observed_branch=old.lease_branch,
        observed_head_sha=HEAD_SHA,
        pr_ids=("502",),
        ci_ids=("34600000000",),
        artifact_ids=("1000",),
        decision=ReconciliationDecision.REASSIGN,
    )
    new = grant_lease(
        packet,
        "agent-b",
        epoch=4,
        now=expired_at,
        ttl=TTL,
        previous=old,
        reconciliation=reconciliation,
    )

    assert new.epoch == 4
    assert new.owner == "agent-b"
    assert new.lease_branch != old.lease_branch
    assert new.lease_branch == "agent/T500-03/e0004-agent-b"

    with pytest.raises(ValueError, match="next epoch"):
        grant_lease(
            packet,
            "agent-c",
            epoch=6,
            now=expired_at,
            ttl=TTL,
            previous=old,
            reconciliation=reconciliation,
        )


def test_same_owner_resume_keeps_epoch_and_branch_only_after_reconciliation() -> None:
    packet = _packet()
    old = grant_lease(packet, "agent-a", epoch=2, now=NOW, ttl=TTL)
    expired_at = NOW + TTL + timedelta(seconds=1)
    reconciliation = LeaseReconciliation(
        task_id=packet.task_id,
        expired_epoch=old.epoch,
        observed_branch=old.lease_branch,
        observed_head_sha=HEAD_SHA,
        decision=ReconciliationDecision.RESUME,
    )
    resumed = grant_lease(
        packet,
        "agent-a",
        epoch=2,
        now=expired_at,
        ttl=TTL,
        previous=old,
        reconciliation=reconciliation,
    )
    assert resumed.epoch == old.epoch
    assert resumed.lease_branch == old.lease_branch


def test_lease_freshness_binds_revision_digest_base_and_expiry() -> None:
    packet = _packet()
    lease = grant_lease(packet, "agent-a", epoch=1, now=NOW, ttl=TTL)

    assert lease_is_current(lease, packet, now=NOW + timedelta(minutes=1))
    assert not lease_is_current(
        lease, _packet(revision=2), now=NOW + timedelta(minutes=1)
    )
    assert not lease_is_current(
        lease,
        _packet(objective="changed"),
        now=NOW + timedelta(minutes=1),
    )
    assert not lease_is_current(lease, packet, now=NOW + TTL)


def test_heartbeat_is_monotonic_and_updates_exact_head() -> None:
    packet = _packet()
    lease = grant_lease(packet, "agent-a", epoch=1, now=NOW, ttl=TTL)
    updated = heartbeat(
        lease,
        head_sha=NEXT_HEAD_SHA,
        observed_at=NOW + timedelta(minutes=2),
    )
    assert updated.head_sha == NEXT_HEAD_SHA
    assert updated.last_heartbeat_at == NOW + timedelta(minutes=2)
    assert updated.owner == lease.owner
    assert updated.epoch == lease.epoch

    with pytest.raises(ValueError, match="monotonic"):
        heartbeat(
            updated,
            head_sha=HEAD_SHA,
            observed_at=NOW + timedelta(minutes=1),
        )


def test_lease_requires_timezone_aware_time_and_positive_ttl() -> None:
    packet = _packet()
    with pytest.raises(ValueError, match="timezone-aware"):
        grant_lease(
            packet,
            "agent-a",
            epoch=1,
            now=datetime(2026, 9, 12, 12, 0),
            ttl=TTL,
        )
    with pytest.raises(ValueError, match="positive"):
        grant_lease(packet, "agent-a", epoch=1, now=NOW, ttl=timedelta())
