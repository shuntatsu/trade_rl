"""Commit-bearing coordination records must not accept Git's null OID."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from tools.agent_repo.coordination.github_state import (
    HandoffPacket,
    TaskStatusRecord,
    parse_status_comment,
    render_status_comment,
)
from tools.agent_repo.coordination.leases import (
    LeaseReconciliation,
    ReconciliationDecision,
    grant_lease,
    heartbeat,
)
from tools.agent_repo.coordination.model import (
    EvidenceKind,
    EvidenceRecord,
    EvidenceResult,
    ExecutionMode,
    TaskCondition,
    TaskPacket,
    TaskPhase,
    WriteScope,
)

ZERO = "0" * 40
BASE = "a" * 40
HEAD = "b" * 40
NOW = datetime(2026, 11, 25, tzinfo=UTC)


def _records():
    packet = TaskPacket(
        task_id="T1067-oid",
        task_revision=1,
        parent_issue=1067,
        title="Commit identity",
        objective="Reject absence sentinel",
        execution_mode=ExecutionMode.WRITE,
        base_sha=BASE,
        write_scope=WriteScope(allow=("tools/",)),
    )
    lease = heartbeat(
        grant_lease(packet, "agent-a", epoch=1, now=NOW, ttl=timedelta(minutes=5)),
        head_sha=HEAD,
        observed_at=NOW,
    )
    digest = packet.contract_digest()
    return {
        "task": packet,
        "evidence": EvidenceRecord(
            evidence_kind=EvidenceKind.INTEGRATION,
            task_id=packet.task_id,
            task_revision=1,
            task_contract_digest=digest,
            base_sha=BASE,
            head_sha=HEAD,
            identifier="ci:1",
            result=EvidenceResult.PASS,
            main_sha=BASE,
        ),
        "lease": lease,
        "reconciliation": LeaseReconciliation(
            task_id=packet.task_id,
            expired_epoch=1,
            observed_branch=lease.lease_branch,
            observed_head_sha=HEAD,
            decision=ReconciliationDecision.RESUME,
        ),
        "status": TaskStatusRecord(
            task_id=packet.task_id,
            task_revision=1,
            task_contract_digest=digest,
            phase=TaskPhase.EXECUTING,
            condition=TaskCondition.HEALTHY,
            base_sha=BASE,
            head_sha=HEAD,
        ),
        "handoff": HandoffPacket(
            task_id=packet.task_id,
            task_revision=1,
            task_contract_digest=digest,
            lease_epoch=1,
            last_good_head=HEAD,
        ),
    }


@pytest.mark.parametrize(
    ("record_name", "field"),
    [
        ("task", "base_sha"),
        ("evidence", "base_sha"),
        ("evidence", "head_sha"),
        ("evidence", "main_sha"),
        ("lease", "base_sha"),
        ("lease", "head_sha"),
        ("reconciliation", "observed_head_sha"),
        ("status", "base_sha"),
        ("status", "head_sha"),
        ("handoff", "last_good_head"),
    ],
)
def test_commit_identity_rejects_null_oid(record_name, field):
    record = _records()[record_name]
    with pytest.raises(ValueError, match=field):
        replace(record, **{field: ZERO})
    # Preserve syntactically valid nonzero OIDs, including leading zeroes.
    assert getattr(replace(record, **{field: "0" * 39 + "1"}), field) == "0" * 39 + "1"


@pytest.mark.parametrize("field", ["base_sha", "head_sha"])
def test_status_parser_rejects_null_oid(field):
    status = _records()["status"]
    assert isinstance(status, TaskStatusRecord)
    rendered = render_status_comment(status)
    assert parse_status_comment(rendered) == status
    malformed = rendered.replace(
        f'"{field}": "{getattr(status, field)}"', f'"{field}": "{ZERO}"'
    )
    assert malformed != rendered
    with pytest.raises(ValueError, match=field):
        parse_status_comment(malformed)


def test_heartbeat_rejects_null_oid_and_preserves_optional_absence():
    from tools.agent_repo.coordination.leases import LeaseRecord

    lease = _records()["lease"]
    assert isinstance(lease, LeaseRecord)
    assert replace(lease, head_sha=None).head_sha is None
    with pytest.raises(ValueError, match="head_sha"):
        heartbeat(lease, head_sha=ZERO, observed_at=NOW)
