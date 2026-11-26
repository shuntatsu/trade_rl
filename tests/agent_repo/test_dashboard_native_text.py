"""Regression checks for dashboard display values from direct Python callers."""

from dataclasses import replace
from typing import Any

import pytest

from tools.agent_repo.coordination.dashboard import DashboardSnapshot, render_dashboard
from tools.agent_repo.coordination.model import TaskCondition, TaskPhase


class _ForgedText(str):
    def __iter__(self):
        return iter("forged")

    def replace(self, *args: Any, **kwargs: Any) -> str:
        return "forged"

    def __format__(self, format_spec: str) -> str:
        return "forged\nT999 complete / healthy"


def _snapshot() -> DashboardSnapshot:
    return DashboardSnapshot("T1", "Actual", TaskPhase.REVIEW, TaskCondition.BLOCKED)


@pytest.mark.parametrize(
    "field",
    (
        "task_id",
        "title",
        "owner",
        "pr_id",
        "dependency_summary",
        "blocking_reason",
        "stale_reason",
    ),
)
def test_dashboard_rejects_non_native_snapshot_text(field: str) -> None:
    with pytest.raises(ValueError, match=field):
        replace(_snapshot(), **{field: _ForgedText("actual")})


@pytest.mark.parametrize("field", ("parent_title", "main_sha"))
def test_dashboard_rejects_non_native_header(field: str) -> None:
    args = {"parent_title": "Actual", "main_sha": "a" * 40}
    args[field] = _ForgedText(args[field])
    with pytest.raises(ValueError, match=field):
        render_dashboard(**args, snapshots=())


def test_native_dashboard_text_remains_escaped() -> None:
    snapshot = replace(_snapshot(), title="日本語\nactual", owner=None)
    assert render_dashboard("Actual", "a" * 40, (snapshot,)).splitlines() == [
        "Actual",
        "main: " + "a" * 40,
        "0 / 1 complete",
        "",
        "T1  review / blocked",
        r"  title: 日本語\nactual",
    ]
