from __future__ import annotations

import json
from dataclasses import replace

import pytest

from tools.agent_repo.coordination.github_state import (
    TaskStatusRecord,
    parse_status_comment,
    render_status_comment,
)
from tools.agent_repo.coordination.leases import lease_branch_name
from tools.agent_repo.coordination.model import TaskCondition, TaskPhase


def _record() -> TaskStatusRecord:
    return TaskStatusRecord(
        task_id="T500-utf8",
        task_revision=1,
        task_contract_digest="a" * 64,
        phase=TaskPhase.REVIEW,
        condition=TaskCondition.HEALTHY,
        base_sha="b" * 40,
        head_sha="c" * 40,
        owner="agent-a",
        lease_epoch=1,
        lease_branch=lease_branch_name("T500-utf8", 1, "agent-a"),
    )


_FREE_TEXT_FIELDS = ("pr_id", "checkpoint", "blocking_reason", "stale_reason")
_IDENTITY_FIELDS = (
    "task_id",
    "task_contract_digest",
    "phase",
    "condition",
    "base_sha",
    "head_sha",
    "owner",
    "lease_branch",
)


@pytest.mark.parametrize("field", _FREE_TEXT_FIELDS)
@pytest.mark.parametrize("value", ["bad\ud800", "bad\udfff", "\ud83d\ude00"])
def test_status_record_rejects_non_scalar_text(field: str, value: str) -> None:
    with pytest.raises(ValueError, match=field):
        replace(_record(), **{field: value})


@pytest.mark.parametrize("field", _IDENTITY_FIELDS)
@pytest.mark.parametrize("value", ["bad\ud800", "bad\udfff", "\ud83d\ude00"])
def test_status_identity_already_rejects_non_scalar_text(
    field: str, value: str
) -> None:
    # Identity and enum fields have stricter ASCII/enum/fenced-branch contracts.
    with pytest.raises(ValueError):
        replace(_record(), **{field: value})


@pytest.mark.parametrize("field", _FREE_TEXT_FIELDS + _IDENTITY_FIELDS)
def test_status_parser_rejects_escaped_lone_surrogate(field: str) -> None:
    payload = _record().canonical_payload()
    payload[field] = "bad\ud800"
    text = (
        "<!-- agent-coordination:T500-utf8 -->\n```json\n"
        + json.dumps(payload)
        + "\n```\n"
    )
    assert r"\ud800" in text
    with pytest.raises(ValueError, match="status comment JSON is invalid"):
        parse_status_comment(text)


@pytest.mark.parametrize("field", _FREE_TEXT_FIELDS)
def test_status_comment_preserves_valid_unicode_utf8(field: str) -> None:
    record = replace(_record(), **{field: "日本語 😀 e\u0301 line\nnext"})
    rendered = render_status_comment(record)
    assert "日本語 😀" in rendered
    assert parse_status_comment(rendered.encode("utf-8").decode("utf-8")) == record


@pytest.mark.parametrize("field", _FREE_TEXT_FIELDS)
def test_status_parser_accepts_escaped_valid_surrogate_pair(field: str) -> None:
    payload = _record().canonical_payload()
    payload[field] = "😀"
    text = (
        "<!-- agent-coordination:T500-utf8 -->\n```json\n"
        + json.dumps(payload)
        + "\n```\n"
    )
    assert r"\ud83d\ude00" in text
    assert getattr(parse_status_comment(text), field) == "😀"


class _SpoofedEncoding(str):
    def encode(self, *args: object, **kwargs: object) -> bytes:
        return b"valid"


class _ChangingCopy(str):
    def __deepcopy__(self, memo: dict[int, object]) -> str:
        return "changed"


@pytest.mark.parametrize("field", _FREE_TEXT_FIELDS)
@pytest.mark.parametrize(
    "value", [_SpoofedEncoding("bad\ud800"), _ChangingCopy("original")]
)
def test_status_record_rejects_non_native_free_text(field: str, value: str) -> None:
    with pytest.raises(ValueError, match=field):
        replace(_record(), **{field: value})
