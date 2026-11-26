from __future__ import annotations

import json
from dataclasses import replace

import pytest

from tools.agent_repo.coordination.github_state import (
    TaskStatusRecord,
    parse_status_comment,
    render_status_comment,
)
from tools.agent_repo.coordination.model import TaskCondition, TaskPhase


def _record() -> TaskStatusRecord:
    return TaskStatusRecord(
        task_id="T500-utf8",
        task_revision=1,
        task_contract_digest="a" * 64,
        phase=TaskPhase.REVIEW,
        condition=TaskCondition.HEALTHY,
        base_sha="b" * 40,
    )


@pytest.mark.parametrize(
    "field", ["pr_id", "checkpoint", "blocking_reason", "stale_reason"]
)
@pytest.mark.parametrize("value", ["bad\ud800", "bad\udfff", "\ud83d\ude00"])
def test_status_record_rejects_non_scalar_text(field: str, value: str) -> None:
    with pytest.raises(ValueError, match=field):
        replace(_record(), **{field: value})


def test_status_parser_rejects_escaped_lone_surrogate() -> None:
    payload = _record().canonical_payload()
    payload["checkpoint"] = "bad\ud800"
    text = (
        "<!-- agent-coordination:T500-utf8 -->\n```json\n"
        + json.dumps(payload)
        + "\n```\n"
    )
    with pytest.raises(ValueError, match="checkpoint"):
        parse_status_comment(text)


def test_status_comment_preserves_valid_unicode_utf8() -> None:
    record = replace(
        _record(), checkpoint="日本語 😀 e\u0301", stale_reason="line\nnext"
    )
    rendered = render_status_comment(record)
    assert "日本語 😀" in rendered
    assert parse_status_comment(rendered.encode("utf-8").decode("utf-8")) == record


def test_status_parser_accepts_escaped_valid_surrogate_pair() -> None:
    payload = _record().canonical_payload()
    payload["checkpoint"] = "😀"
    text = (
        "<!-- agent-coordination:T500-utf8 -->\n```json\n"
        + json.dumps(payload)
        + "\n```\n"
    )
    assert parse_status_comment(text).checkpoint == "😀"
