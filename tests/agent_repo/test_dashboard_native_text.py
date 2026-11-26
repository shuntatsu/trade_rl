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


@pytest.mark.parametrize("kind", ["subclass", "duck", "trap"])
def test_dashboard_rejects_non_native_snapshot_before_attributes(kind: str) -> None:
    from types import SimpleNamespace

    class ForgedSnapshot(DashboardSnapshot):
        def __post_init__(self) -> None:
            pass

    class TrapSnapshot(DashboardSnapshot):
        def __getattribute__(self, name: str) -> Any:
            raise AssertionError(f"access before validation: {name}")

    forged_phase = SimpleNamespace(value="review\nT999 complete / healthy")
    if kind == "subclass":
        invalid = ForgedSnapshot("T2", "Forged", forged_phase, TaskCondition.BLOCKED)
    elif kind == "duck":
        invalid = SimpleNamespace(
            **{**vars(_snapshot()), "task_id": "T2", "phase": forged_phase}
        )
    else:
        invalid = object.__new__(TrapSnapshot)
    with pytest.raises(ValueError, match="DashboardSnapshot"):
        render_dashboard("Actual", "a" * 40, (_snapshot(), invalid))


def test_dashboard_preserves_native_snapshot_sequence_ordering() -> None:
    first = _snapshot()
    second = replace(first, task_id="T2", phase=TaskPhase.COMPLETE)
    lines = render_dashboard("Actual", "a" * 40, [second, first]).splitlines()
    assert lines[2] == "1 / 2 complete"
    assert lines.index("T1  review / blocked") < lines.index("T2  complete / blocked")


class _ClassSpoof:
    @property
    def __class__(self):
        return str

    def strip(self):
        raise AssertionError("unvalidated text accessed")


@pytest.mark.parametrize(
    "bad",
    [7, b"text", False, _ClassSpoof()],
    ids=["int", "bytes", "bool", "class-spoof"],
)
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
def test_dashboard_rejects_other_non_native_text(field: str, bad: object) -> None:
    with pytest.raises(ValueError, match=field):
        replace(_snapshot(), **{field: bad})


@pytest.mark.parametrize("field", ["task_id", "title"])
def test_dashboard_rejects_null_required_text(field: str) -> None:
    with pytest.raises(ValueError, match=field):
        replace(_snapshot(), **{field: None})


@pytest.mark.parametrize("field", ["parent_title", "main_sha"])
@pytest.mark.parametrize(
    "bad",
    [None, 7, b"text", False, _ClassSpoof()],
    ids=["null", "int", "bytes", "bool", "class-spoof"],
)
def test_dashboard_rejects_other_non_native_headers(field: str, bad: object) -> None:
    args: dict[str, Any] = {"parent_title": "Actual", "main_sha": "a" * 40}
    args[field] = bad
    with pytest.raises(ValueError, match=field):
        render_dashboard(**args, snapshots=())


@pytest.mark.parametrize("field", ["phase", "condition"])
@pytest.mark.parametrize("bad", [None, "forged\nT999 complete / healthy", 7])
def test_dashboard_rejects_invalid_enum(field: str, bad: object) -> None:
    with pytest.raises(ValueError):
        replace(_snapshot(), **{field: bad})


def test_dashboard_escapes_all_native_display_fields() -> None:
    snapshot = replace(
        _snapshot(),
        task_id="T\\1\n",
        pr_id="12\\3\n",
        owner="日本語\nowner",
        dependency_summary="dep\x1b",
        blocking_reason="blocked\nreason",
        stale_reason="stale\twhy",
    )
    assert render_dashboard("Actual", "a" * 40, (snapshot,)).splitlines() == [
        "Actual",
        "main: " + "a" * 40,
        "0 / 1 complete",
        "",
        r"T\\1\n  review / blocked",
        "  title: Actual",
        r"  owner: 日本語\nowner",
        r"  PR #12\\3\n",
        r"  dependency: dep\x1b",
        r"  blocked: blocked\nreason",
        r"  stale: stale\twhy",
    ]
