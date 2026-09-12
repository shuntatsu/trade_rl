"""State transition and evidence freshness rules for coordinated tasks."""

from __future__ import annotations

from tools.agent_repo.coordination.model import (
    EvidenceKind,
    EvidenceRecord,
    TaskPacket,
    TaskPhase,
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
    """Return whether evidence is bound to the current exact task snapshot."""

    if evidence.task_id != packet.task_id:
        return False
    if evidence.task_revision != packet.task_revision:
        return False
    if evidence.task_contract_digest != packet.contract_digest():
        return False
    if evidence.base_sha != packet.base_sha:
        return False
    if evidence.head_sha != head_sha:
        return False
    if evidence.evidence_kind in {EvidenceKind.INTEGRATION, EvidenceKind.POST_MERGE}:
        return current_main_sha is not None and evidence.main_sha == current_main_sha
    return True


__all__ = ["evidence_is_current", "validate_transition"]
