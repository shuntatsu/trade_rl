"""Read-only P0 falsifiers for Issue #1067; not remote CAS evidence."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from tools.agent_repo.coordination.enrollment import (
    EnrollmentProjection,
    prepare_enrollment_projection,
    require_enrollment_readback,
    require_local_registry_preflight,
)
from tools.agent_repo.coordination.github_state import TaskStatusRecord
from tools.agent_repo.coordination.leases import (
    LeaseRecord,
    grant_lease,
    heartbeat,
    lease_branch_name,
)
from tools.agent_repo.coordination.model import (
    ExecutionMode,
    TaskCondition,
    TaskPacket,
    TaskPhase,
    WriteScope,
)

BASE = "a" * 40
HEAD = "b" * 40
OID = "c" * 40
NOW = datetime(2026, 11, 1, 0, 0, tzinfo=UTC)


def _packet() -> TaskPacket:
    return TaskPacket(
        task_id="T1067-01",
        task_revision=1,
        parent_issue=1067,
        title="Task enrollment preflight",
        objective="Bind a Coordinator Task and fenced write lease",
        execution_mode=ExecutionMode.WRITE,
        write_scope=WriteScope(allow=("tools/agent_repo/**",)),
        base_sha=BASE,
    )


def _snapshot() -> tuple[TaskPacket, LeaseRecord, TaskStatusRecord]:
    packet = _packet()
    lease = grant_lease(packet, "agent-a", epoch=1, now=NOW, ttl=timedelta(minutes=15))
    lease = heartbeat(lease, head_sha=HEAD, observed_at=NOW + timedelta(minutes=1))
    status = TaskStatusRecord(
        task_id=packet.task_id,
        task_revision=packet.task_revision,
        task_contract_digest=packet.contract_digest(),
        phase=TaskPhase.EXECUTING,
        condition=TaskCondition.HEALTHY,
        base_sha=packet.base_sha,
        owner=lease.owner,
        lease_epoch=lease.epoch,
        lease_branch=lease.lease_branch,
        head_sha=lease.head_sha,
    )
    return packet, lease, status


def _enrollment() -> EnrollmentProjection:
    packet, lease, status = _snapshot()
    return prepare_enrollment_projection(
        packet,
        lease,
        status,
        repository="shuntatsu/trade_rl",
        issue_number=1067,
        issue_node_id="I_kwDOAUTH1067",
        current_main_sha=BASE,
        observed_branch_sha=HEAD,
        issue_is_open=True,
        now=NOW + timedelta(minutes=2),
    )


def test_preflight_binds_existing_task_packet_lease_and_status() -> None:
    enrolled = _enrollment()
    assert enrolled.status.task_id == "T1067-01"
    payload = enrolled.canonical_payload()
    assert payload["schema"] == "coordinator_enrollment_projection_v1"
    assert payload["status"] == enrolled.status.canonical_payload()
    require_local_registry_preflight(
        None,
        enrolled,
        expected_registry_oid=OID,
        observed_registry_oid=OID,
        enumerated_task_ids=(),
    )
    require_enrollment_readback(
        enrolled,
        enrolled,
        expected_registry_oid=OID,
        observed_registry_oid=OID,
        observed_branch_sha=HEAD,
    )


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"issue_is_open": False}, "open immutable"),
        ({"issue_number": 1068}, "open immutable"),
        ({"current_main_sha": "d" * 40}, "current main"),
        ({"observed_branch_sha": "d" * 40}, "lease branch tip"),
        ({"now": NOW + timedelta(hours=1)}, "lease is expired"),
    ],
)
def test_preflight_rejects_stale_issue_main_branch_or_lease(
    change: dict[str, object], message: str
) -> None:
    packet, lease, status = _snapshot()
    kwargs = dict(
        repository="shuntatsu/trade_rl",
        issue_number=1067,
        issue_node_id="I_kwDOAUTH1067",
        current_main_sha=BASE,
        observed_branch_sha=HEAD,
        issue_is_open=True,
        now=NOW + timedelta(minutes=2),
    )
    kwargs.update(change)
    with pytest.raises(ValueError, match=message):
        prepare_enrollment_projection(packet, lease, status, **kwargs)


def test_preflight_rejects_mismatched_status_and_read_only_contract() -> None:
    packet, lease, status = _snapshot()
    bad_status = replace(status, task_revision=status.task_revision + 1)
    with pytest.raises(ValueError, match="status, packet and lease"):
        prepare_enrollment_projection(
            packet,
            lease,
            bad_status,
            repository="shuntatsu/trade_rl",
            issue_number=1067,
            issue_node_id="I_kwDOAUTH1067",
            current_main_sha=BASE,
            observed_branch_sha=HEAD,
            issue_is_open=True,
            now=NOW + timedelta(minutes=2),
        )
    read_only = replace(
        packet, execution_mode=ExecutionMode.READ_ONLY, write_scope=WriteScope()
    )
    with pytest.raises(ValueError, match="read_only"):
        prepare_enrollment_projection(
            read_only,
            lease,
            status,
            repository="shuntatsu/trade_rl",
            issue_number=1067,
            issue_node_id="I_kwDOAUTH1067",
            current_main_sha=BASE,
            observed_branch_sha=HEAD,
            issue_is_open=True,
            now=NOW + timedelta(minutes=2),
        )


def test_initial_registration_preflight_rejects_duplicate_task_and_stale_ref() -> None:
    candidate = _enrollment()
    with pytest.raises(ValueError, match="duplicate Task ID"):
        require_local_registry_preflight(
            None,
            candidate,
            expected_registry_oid=OID,
            observed_registry_oid=OID,
            enumerated_task_ids=(candidate.status.task_id,),
        )
    with pytest.raises(ValueError, match="reference changed"):
        require_local_registry_preflight(
            None,
            candidate,
            expected_registry_oid=OID,
            observed_registry_oid="d" * 40,
            enumerated_task_ids=(),
        )
    with pytest.raises(ValueError, match="malformed or duplicated"):
        require_local_registry_preflight(
            None,
            candidate,
            expected_registry_oid=OID,
            observed_registry_oid=OID,
            enumerated_task_ids=("x", "x"),
        )


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("issue", "immutable"),
        ("digest", "digest changed"),
        ("epoch", "reconciliation"),
        ("owner", "owner or fenced"),
        ("head", "remote ancestry"),
        ("base", "Task base changed"),
    ],
)
def test_update_requires_stable_binding_and_fenced_epoch(
    case: str, message: str
) -> None:
    prior = _enrollment()
    status = prior.status
    if case == "issue":
        proposed = replace(prior, issue_node_id="I_kwFORGED")
    elif case == "digest":
        proposed = replace(prior, status=replace(status, task_contract_digest="d" * 64))
    elif case == "epoch":
        next_branch = lease_branch_name("T1067-01", 2, "agent-a")
        proposed = replace(
            prior, status=replace(status, lease_epoch=2, lease_branch=next_branch)
        )
    elif case == "owner":
        new_branch = lease_branch_name("T1067-01", 1, "agent-b")
        proposed = replace(
            prior, status=replace(status, owner="agent-b", lease_branch=new_branch)
        )
    elif case == "head":
        proposed = replace(prior, status=replace(status, head_sha="e" * 40))
    else:
        proposed = replace(prior, status=replace(status, base_sha="e" * 40))
    with pytest.raises(ValueError, match=message):
        require_local_registry_preflight(
            prior,
            proposed,
            expected_registry_oid=OID,
            observed_registry_oid=OID,
            enumerated_task_ids=(status.task_id,),
        )


def test_readback_refuses_mutation_stale_oid_and_branch_drift() -> None:
    wanted = _enrollment()
    cases = [
        (replace(wanted, issue_node_id="I_kwSPOOF"), OID, HEAD, "record differs"),
        (wanted, "d" * 40, HEAD, "readback OID"),
        (wanted, OID, "d" * 40, "branch SHA changed"),
    ]
    for observed, oid, branch, match in cases:
        with pytest.raises(ValueError, match=match):
            require_enrollment_readback(
                wanted,
                observed,
                expected_registry_oid=OID,
                observed_registry_oid=oid,
                observed_branch_sha=branch,
            )


def test_update_rejects_revision_regression() -> None:
    current = _enrollment()
    former = replace(current, status=replace(current.status, task_revision=2))
    with pytest.raises(ValueError, match="revision regressed"):
        require_local_registry_preflight(
            former,
            current,
            expected_registry_oid=OID,
            observed_registry_oid=OID,
            enumerated_task_ids=(current.status.task_id,),
        )
