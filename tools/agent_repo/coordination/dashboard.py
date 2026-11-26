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
        if type(self.task_id) is not str or not self.task_id.strip():
            raise ValueError("task_id must be non-empty")
        if type(self.title) is not str or not self.title.strip():
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
            if value is not None and (type(value) is not str or not value.strip()):
                raise ValueError(f"{field_name} must be non-empty when provided")


def _display_text(value: str) -> str:
    """Keep untrusted display text on one line without terminal controls."""

    return "".join(
        char if char.isprintable() else char.encode("unicode_escape").decode("ascii")
        for char in value
    )


def _display_identifier(value: str) -> str:
    """Disambiguate literal backslashes from escaped identifier characters."""

    return _display_text(value.replace("\\", "\\\\"))


def render_dashboard(
    parent_title: str,
    main_sha: str,
    snapshots: Sequence[DashboardSnapshot],
) -> str:
    """Render a deterministic non-authoritative operator summary."""

    if type(parent_title) is not str or not parent_title.strip():
        raise ValueError("parent_title must be non-empty")
    if (
        type(main_sha) is not str
        or not _FULL_SHA_RE.fullmatch(main_sha)
        or main_sha == "0" * 40
    ):
        raise ValueError("main_sha must be a nonzero lowercase 40-hex Git SHA")
    ordered = sorted(snapshots, key=lambda snapshot: snapshot.task_id)
    if len({snapshot.task_id for snapshot in ordered}) != len(ordered):
        raise ValueError("dashboard contains duplicate task_id")

    completed = sum(snapshot.phase is TaskPhase.COMPLETE for snapshot in ordered)
    lines = [
        _display_text(parent_title),
        f"main: {main_sha}",
        f"{completed} / {len(ordered)} complete",
        "",
    ]
    for snapshot in ordered:
        lines.append(
            f"{_display_identifier(snapshot.task_id)}  {snapshot.phase.value} / {snapshot.condition.value}"
        )
        lines.append(f"  title: {_display_text(snapshot.title)}")
        if snapshot.owner is not None:
            lines.append(f"  owner: {_display_text(snapshot.owner)}")
        if snapshot.pr_id is not None:
            lines.append(f"  PR #{_display_identifier(snapshot.pr_id)}")
        if snapshot.dependency_summary is not None:
            lines.append(f"  dependency: {_display_text(snapshot.dependency_summary)}")
        if snapshot.blocking_reason is not None:
            lines.append(f"  blocked: {_display_text(snapshot.blocking_reason)}")
        if snapshot.stale_reason is not None:
            lines.append(f"  stale: {_display_text(snapshot.stale_reason)}")
        lines.append("")
    return "\n".join(lines).rstrip("\n") + "\n"


__all__ = ["DashboardSnapshot", "render_dashboard"]
