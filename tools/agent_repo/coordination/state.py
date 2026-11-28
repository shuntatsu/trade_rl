"""State transition and evidence freshness rules for coordinated tasks."""

from __future__ import annotations

from collections.abc import Mapping

from tools.agent_repo.coordination.conflicts import ConflictLevel, classify_conflict
from tools.agent_repo.coordination.model import (
    EvidenceKind,
    EvidenceRecord,
    TaskCondition,
    TaskPacket,
    TaskPhase,
    TaskStatus,
)

_ALLOWED_TRANSITIONS: dict[TaskPhase, frozenset[TaskPhase]] = {
    TaskPhase.PLANNED: frozenset({TaskPhase.READY}),
    TaskPhase.READY: frozenset({TaskPhase.EXECUTING}),
    TaskPhase.EXECUTING: frozenset({TaskPhase.REVIEW}),
    TaskPhase.REVIEW: frozenset({TaskPhase.EXECUTING, TaskPhase.VERIFICATION}),
    TaskPhase.VERIFICATION: frozenset(
        {TaskPhase.EXECUTING, TaskPhase.INTEGRATION, TaskPhase.COMPLETE}
    ),
    TaskPhase.INTEGRATION: frozenset(
        {TaskPhase.EXECUTING, TaskPhase.VERIFICATION, TaskPhase.COMPLETE}
    ),
    TaskPhase.COMPLETE: frozenset(),
}


def validate_transition(current: TaskPhase, target: TaskPhase) -> None:
    """Raise when a requested phase transition is outside the v1 protocol."""

    if type(current) not in (TaskPhase, str) or type(target) not in (TaskPhase, str):
        raise ValueError("task phases must be native str or TaskPhase")
    current_phase = TaskPhase(current)
    target_phase = TaskPhase(target)
    if current_phase is TaskPhase.COMPLETE:
        raise ValueError("complete is a terminal task phase")
    if target_phase not in _ALLOWED_TRANSITIONS[current_phase]:
        raise ValueError(
            f"task phase transition is not allowed: "
            f"{current_phase.value} -> {target_phase.value}"
        )


def evidence_is_current(
    evidence: EvidenceRecord,
    packet: TaskPacket,
    *,
    head_sha: str,
    current_main_sha: str | None = None,
) -> bool:
    """Return freshness for native EvidenceRecord and TaskPacket snapshots.

    Subclasses and duck-typed objects are not snapshots: overridden fields or
    digest methods can detach the comparison from the original contract.
    """
    if type(evidence) is not EvidenceRecord or type(packet) is not TaskPacket:
        return False

    # Direct Python construction can preserve str subclasses whose equality
    # overrides do not reflect their stored identity.
    if any(
        type(value) is not str
        for value in (
            evidence.task_id,
            packet.task_id,
            evidence.task_contract_digest,
            evidence.base_sha,
            packet.base_sha,
        )
    ):
        return False
    if evidence.task_id != packet.task_id:
        return False
    if evidence.task_revision != packet.task_revision:
        return False
    expected_digest = packet.contract_digest()
    if (
        type(expected_digest) is not str
        or evidence.task_contract_digest != expected_digest
    ):
        return False
    if evidence.base_sha != packet.base_sha:
        return False
    if type(head_sha) is not str or type(evidence.head_sha) is not str:
        return False
    if evidence.head_sha != head_sha:
        return False
    if evidence.evidence_kind in {EvidenceKind.INTEGRATION, EvidenceKind.POST_MERGE}:
        return (
            type(current_main_sha) is str
            and type(evidence.main_sha) is str
            and evidence.main_sha == current_main_sha
        )
    return True


def evidence_is_current_after_peer_integration(
    evidence: EvidenceRecord,
    packet: TaskPacket,
    integrated_packet: TaskPacket,
    *,
    head_sha: str,
    current_main_sha: str | None = None,
) -> bool:
    """Reject stale evidence and non-native current/peer task snapshots."""

    if (
        type(evidence) is not EvidenceRecord
        or type(packet) is not TaskPacket
        or type(integrated_packet) is not TaskPacket
    ):
        return False

    if type(packet.task_id) is not str or type(integrated_packet.task_id) is not str:
        return False
    if packet.task_id == integrated_packet.task_id:
        raise ValueError("peer integration requires a distinct task")
    if classify_conflict(packet, integrated_packet) is not ConflictLevel.NONE:
        return False
    return evidence_is_current(
        evidence,
        packet,
        head_sha=head_sha,
        current_main_sha=current_main_sha,
    )


def parent_completion_allowed(
    child_statuses: Mapping[str, TaskStatus],
    *,
    parent_acceptance_satisfied: bool,
    parent_invariants_satisfied: bool,
) -> bool:
    """Require a native child-status snapshot and affirmative parent oracles.

    Caller-supplied rosters do not establish authoritative task completeness.
    """

    if type(child_statuses) is not dict:
        return False
    children = tuple(child_statuses.items())
    if any(
        type(key) is not str or type(status) is not TaskStatus
        for key, status in children
    ):
        return False
    children_complete = all(
        status.phase is TaskPhase.COMPLETE and status.condition is TaskCondition.HEALTHY
        for _, status in children
    )
    return (
        children_complete
        and parent_acceptance_satisfied is True
        and parent_invariants_satisfied is True
    )


__all__ = [
    "evidence_is_current",
    "evidence_is_current_after_peer_integration",
    "parent_completion_allowed",
    "validate_transition",
]
