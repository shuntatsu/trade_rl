from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from tools.agent_repo.coordination.dashboard import (
    DashboardSnapshot,
    render_dashboard,
)
from tools.agent_repo.coordination.github_state import (
    HandoffPacket,
    TaskStatusRecord,
    parse_status_comment,
    render_status_comment,
    validate_handoff,
)
from tools.agent_repo.coordination.leases import grant_lease, heartbeat
from tools.agent_repo.coordination.model import (
    Capability,
    ExecutionMode,
    TaskCondition,
    TaskPacket,
    TaskPhase,
    WriteScope,
)

BASE_SHA = "a" * 40
HEAD_SHA = "b" * 40
MAIN_SHA = "c" * 40
NOW = datetime(2026, 9, 12, 12, 0, tzinfo=UTC)


def _packet() -> TaskPacket:
    return TaskPacket(
        task_id="T500-04",
        task_revision=2,
        parent_issue=500,
        title="GitHub projection",
        objective="Persist reconstructible coordination state",
        execution_mode=ExecutionMode.WRITE,
        write_scope=WriteScope(allow=("tools/agent_repo/**",)),
        capabilities=(Capability.REPOSITORY_READ, Capability.LEASE_BRANCH_WRITE),
        base_sha=BASE_SHA,
    )


def _lease():
    packet = _packet()
    lease = grant_lease(
        packet,
        "agent-a",
        epoch=3,
        now=NOW,
        ttl=timedelta(minutes=15),
    )
    return heartbeat(
        lease,
        head_sha=HEAD_SHA,
        observed_at=NOW + timedelta(minutes=1),
    )


def _record() -> TaskStatusRecord:
    packet = _packet()
    lease = _lease()
    return TaskStatusRecord(
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
        pr_id="502",
        checkpoint="minimal GREEN",
    )


def test_status_comment_round_trip_is_canonical_and_single_marker() -> None:
    record = _record()
    rendered = render_status_comment(record)

    assert rendered.count("<!-- agent-coordination:T500-04 -->") == 1
    assert parse_status_comment(rendered) == record
    assert render_status_comment(parse_status_comment(rendered)) == rendered


def test_status_comment_parser_fails_closed_on_malformed_or_ambiguous_input() -> None:
    rendered = render_status_comment(_record())
    with pytest.raises(ValueError, match="marker"):
        parse_status_comment(rendered + "\n" + rendered)
    with pytest.raises(ValueError, match="JSON"):
        parse_status_comment("<!-- agent-coordination:T500-04 -->\n```json\n{bad}\n```")

    unknown = rendered.replace(
        '"task_revision": 2,',
        '"unknown": true,\n  "task_revision": 2,',
    )
    with pytest.raises(ValueError, match="unknown field"):
        parse_status_comment(unknown)


def test_status_record_rejects_partial_lease_projection() -> None:
    packet = _packet()
    with pytest.raises(ValueError, match="lease fields"):
        TaskStatusRecord(
            task_id=packet.task_id,
            task_revision=packet.task_revision,
            task_contract_digest=packet.contract_digest(),
            phase=TaskPhase.EXECUTING,
            condition=TaskCondition.HEALTHY,
            base_sha=packet.base_sha,
            owner="agent-a",
        )


def test_handoff_binds_exact_task_contract_lease_epoch_and_head() -> None:
    packet = _packet()
    lease = _lease()
    handoff = HandoffPacket(
        task_id=packet.task_id,
        task_revision=packet.task_revision,
        task_contract_digest=packet.contract_digest(),
        lease_epoch=lease.epoch,
        last_good_head=HEAD_SHA,
        verified=("targeted tests green",),
        pending=("full CI",),
        known_failures=(),
        do_not_repeat=("root-cause investigation",),
        evidence_ids=("ci:34600000000", "pr:502"),
    )
    validate_handoff(handoff, packet, lease)

    stale_revision = HandoffPacket(
        task_id=handoff.task_id,
        task_revision=handoff.task_revision + 1,
        task_contract_digest=handoff.task_contract_digest,
        lease_epoch=handoff.lease_epoch,
        last_good_head=handoff.last_good_head,
    )
    with pytest.raises(ValueError, match="task contract"):
        validate_handoff(stale_revision, packet, lease)

    stale_epoch = HandoffPacket(
        task_id=handoff.task_id,
        task_revision=handoff.task_revision,
        task_contract_digest=handoff.task_contract_digest,
        lease_epoch=handoff.lease_epoch + 1,
        last_good_head=handoff.last_good_head,
    )
    with pytest.raises(ValueError, match="lease epoch"):
        validate_handoff(stale_epoch, packet, lease)

    stale_head = HandoffPacket(
        task_id=handoff.task_id,
        task_revision=handoff.task_revision,
        task_contract_digest=handoff.task_contract_digest,
        lease_epoch=handoff.lease_epoch,
        last_good_head="d" * 40,
    )
    with pytest.raises(ValueError, match="head"):
        validate_handoff(stale_head, packet, lease)


def test_dashboard_is_deterministic_and_surfaces_operator_state() -> None:
    snapshots = (
        DashboardSnapshot(
            task_id="T500-02",
            title="Scheduler",
            phase=TaskPhase.COMPLETE,
            condition=TaskCondition.HEALTHY,
        ),
        DashboardSnapshot(
            task_id="T500-01",
            title="Model",
            phase=TaskPhase.REVIEW,
            condition=TaskCondition.STALE,
            owner="agent-a",
            pr_id="502",
            dependency_summary="waits T500-00",
            stale_reason="head_changed",
        ),
    )
    rendered = render_dashboard("#500 Agent Coordination Plane", MAIN_SHA, snapshots)

    assert rendered.index("T500-01") < rendered.index("T500-02")
    assert "review / stale" in rendered
    assert "agent-a" in rendered
    assert "PR #502" in rendered
    assert "waits T500-00" in rendered
    assert "head_changed" in rendered
    assert MAIN_SHA in rendered
    assert "1 / 2 complete" in rendered
