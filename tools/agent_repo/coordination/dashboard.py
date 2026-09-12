"""Human-readable derived dashboard for Agent Coordination Plane tasks."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from tools.agent_repo.coordination.model import TaskCondition, TaskPhase

_FULL_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True)
class DashboardSnapshot:
    """Minimal task projection used only for operator display."""

    task_id: str
    title: str
    phase: TaskPhase
    condition: TaskCondition
    owner: str | None = None
    pr_id: str | None = None
    dependency_summary: str | None = None
    blocking_reason: str | None = None
    stale_reason: str | None = None

    def __post_init__(self) -> None:
        if not self.task_id.strip():
            raise ValueError("task_id must be non-empty")
        if not self.title.strip():
            raise ValueError("title must be non-empty")
        object.__setattr__(self, "phase", TaskPhase(self.phase))
        object.__setattr__(self, "condition", TaskCondition(self.condition))
        for field_name in (
            "owner",
            "pr_id",
            "dependency_summary",
            "blocking_reason",
            "stale_reason",
        ):
            value = getattr(self, field_name)
            if value is not None and not value.strip():
                raise ValueError(f"{field_name} must be non-empty when provided")


def render_dashboard(
    parent_title: str,
    main_sha: str,
    snapshots: Sequence[DashboardSnapshot],
) -> str:
    """Render a deterministic non-authoritative operator summary."""

    if not parent_title.strip():
        raise ValueError("parent_title must be non-empty")
    if not _FULL_SHA_RE.fullmatch(main_sha):
        raise ValueError("main_sha must be a lowercase 40-hex Git SHA")
    ordered = sorted(snapshots, key=lambda snapshot: snapshot.task_id)
    if len({snapshot.task_id for snapshot in ordered}) != len(ordered):
        raise ValueError("dashboard contains duplicate task_id")

    completed = sum(snapshot.phase is TaskPhase.COMPLETE for snapshot in ordered)
    lines = [
        parent_title,
        f"main: {main_sha}",
        f"{completed} / {len(ordered)} complete",
        "",
    ]
    for snapshot in ordered:
        lines.append(
            f"{snapshot.task_id}  {snapshot.phase.value} / {snapshot.condition.value}"
        )
        lines.append(f"  title: {snapshot.title}")
        if snapshot.owner is not None:
            lines.append(f"  owner: {snapshot.owner}")
        if snapshot.pr_id is not None:
            lines.append(f"  PR #{snapshot.pr_id}")
        if snapshot.dependency_summary is not None:
            lines.append(f"  dependency: {snapshot.dependency_summary}")
        if snapshot.blocking_reason is not None:
            lines.append(f"  blocked: {snapshot.blocking_reason}")
        if snapshot.stale_reason is not None:
            lines.append(f"  stale: {snapshot.stale_reason}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


__all__ = ["DashboardSnapshot", "render_dashboard"]
