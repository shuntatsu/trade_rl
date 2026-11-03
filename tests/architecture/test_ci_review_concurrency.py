"""Review-trigger concurrency contract; GitHub scheduler behavior is separate."""

from __future__ import annotations

from pathlib import Path

import pytest

WORKFLOW = Path(__file__).resolve().parents[2] / ".github/workflows/ci.yml"


def _policy_terms() -> tuple[str, ...]:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert text.count("\nconcurrency:\n") == 1
    section = text.split("\nconcurrency:\n", 1)[1].split("\njobs:\n", 1)[0]
    assert "group: ci-${{ github.workflow }}-${{ github.ref }}" in section
    lines = [
        line.strip().removeprefix("cancel-in-progress: ")
        for line in section.splitlines()
        if line.strip().startswith("cancel-in-progress: ")
    ]
    assert len(lines) == 1
    assert lines[0].startswith("${{ ") and lines[0].endswith(" }}")
    return tuple(lines[0][4:-3].split(" || "))


def _would_cancel(event: str, action: str | None, state: str | None) -> bool:
    fields = {
        "github.event_name": event,
        "github.event.action": action,
        "github.event.review.state": state,
    }
    terms = _policy_terms()
    assert set(terms) == {
        "github.event_name != 'pull_request_review'",
        "github.event.action != 'submitted'",
        "github.event.review.state != 'commented'",
    }
    evaluated: list[bool] = []
    for term in terms:
        field, value = term.split(" != ", 1)
        assert field in fields and value.startswith("'") and value.endswith("'")
        evaluated.append(fields[field] != value[1:-1])
    return any(evaluated)


@pytest.mark.parametrize(
    ("event", "action", "review_state", "must_cancel"),
    [
        ("pull_request_review", "submitted", "commented", False),
        ("pull_request_review", "edited", "commented", True),
        ("pull_request_review", "dismissed", "commented", True),
        ("pull_request_review", "submitted", "changes_requested", True),
        ("pull_request_review", "submitted", "approved", True),
        ("pull_request_review", "submitted", None, True),
        ("pull_request", "synchronize", None, True),
        ("pull_request", "opened", None, True),
        ("push", None, None, True),
    ],
)
def test_only_submitted_comment_preserves_running_full(
    event: str, action: str | None, review_state: str | None, must_cancel: bool
) -> None:
    assert _would_cancel(event, action, review_state) is must_cancel


def test_review_triggers_and_required_gate_roster_are_preserved() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "types: [submitted, edited, dismissed]" in text
    assert "  pull_request:\n    branches:\n      - main" in text
    for job in (
        "Fast Push",
        "Lean Core",
        "PPO Runtime",
        "Human Guide",
        "Market Store PostgreSQL",
        "Generic Independent Research Review",
    ):
        assert f"name: {job}" in text
