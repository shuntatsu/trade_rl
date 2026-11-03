"""Read-only GitHub Issue projection readback, never Coordinator authority."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import pytest

from tools.agent_repo.coordination.enrollment import EnrollmentProjection
from tools.agent_repo.coordination.github_state import (
    TaskStatusRecord,
    render_status_comment,
)
from tools.agent_repo.coordination.leases import lease_branch_name
from tools.agent_repo.coordination.model import TaskCondition, TaskPhase
from tools.github_coordination_issue_projection import observe_issue_projection

REPO = "shuntatsu/trade_rl"
ISSUE = 1067
NODE = "I_kwDO_TASK_1067"
AUTHOR = 148511409
STATUS = TaskStatusRecord(
    task_id="T1067-01",
    task_revision=1,
    task_contract_digest="e" * 64,
    phase=TaskPhase.EXECUTING,
    condition=TaskCondition.HEALTHY,
    base_sha="a" * 40,
    owner="agent-a",
    lease_epoch=1,
    lease_branch=lease_branch_name("T1067-01", 1, "agent-a"),
    head_sha="b" * 40,
)
PROJECTION = EnrollmentProjection(REPO, ISSUE, NODE, STATUS)


class FakeApi:
    def __init__(self) -> None:
        self.issue = dict(
            id=123,
            number=ISSUE,
            node_id=NODE,
            repository_url=f"https://api.github.com/repos/{REPO}",
            state="open",
            comments=2,
            updated_at="2026-11-03T12:00:00Z",
        )
        self.comments = [
            dict(
                id=15,
                user={"id": AUTHOR},
                body=render_status_comment(STATUS),
                updated_at="2026-11-03T11:59:00Z",
            ),
            dict(
                id=16,
                user={"id": 999},
                body="ordinary discussion",
                updated_at="2026-11-03T11:59:30Z",
            ),
        ]
        self.issue_reads = 0
        self.comment_reads = 0
        self.change_comment = False
        self.change_final_issue = False
        self.requests: list[tuple[str, str, dict[str, str] | None]] = []

    def _request_json(
        self, method: str, suffix: str, *, query: dict[str, str] | None = None
    ) -> object:
        self.requests.append((method, suffix, query))
        assert method == "GET"
        if suffix == f"issues/{ISSUE}":
            assert query is None
            self.issue_reads += 1
            result = deepcopy(self.issue)
            if self.issue_reads == 3 and self.change_final_issue:
                result["state"] = "closed"
            return result
        if suffix == f"issues/{ISSUE}/comments":
            assert query is not None and query["per_page"] == "100"
            self.comment_reads += 1
            rows = deepcopy(self.comments)
            if self.change_comment and self.comment_reads == 2:
                rows[0]["body"] += "changed"
            page = int(query["page"])
            return rows[(page - 1) * 100 : page * 100]
        raise AssertionError("unexpected endpoint")


def _read(api: FakeApi):
    return observe_issue_projection(
        api, projection=PROJECTION, expected_publisher_user_id=AUTHOR
    )


def test_single_matching_canonical_status_is_observation_only() -> None:
    api = FakeApi()
    result = _read(api)
    assert result.task_id == STATUS.task_id
    assert result.projection == PROJECTION
    assert result.comment_id == 15
    assert result.publisher_user_id == AUTHOR
    assert result.issue_readback_verified is True
    assert result.authoritative_registry_established is False
    assert result.task_registry_complete is False
    assert result.cleanup_authorized is False
    assert api.issue_reads == 3 and api.comment_reads == 2
    assert all(method == "GET" for method, _, _ in api.requests)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("state", "closed"),
        ("node_id", "I_DIFFERENT"),
        ("number", 1068),
        ("number", True),
        ("repository_url", "https://api.github.com/repos/other/repo"),
        ("comments", 1001),
        ("comments", 0),
        ("updated_at", ""),
    ],
)
def test_issue_identity_or_comment_count_rejected(field: str, value: object) -> None:
    api = FakeApi()
    api.issue[field] = value
    with pytest.raises(ValueError):
        _read(api)


@pytest.mark.parametrize(
    "change",
    ["wrong_author", "duplicate", "missing", "stale", "extra_prose", "malformed"],
)
def test_status_comment_must_match_unique_publisher(change: str) -> None:
    api = FakeApi()
    if change == "wrong_author":
        api.comments[0]["user"]["id"] = 100
    elif change == "duplicate":
        api.comments[1]["body"] = render_status_comment(STATUS)
    elif change == "missing":
        api.comments[0]["body"] = "not a status"
    elif change == "stale":
        api.comments[0]["body"] = render_status_comment(
            replace(STATUS, task_revision=2)
        )
    elif change == "extra_prose":
        api.comments[0]["body"] += "extra"
    else:
        api.comments[0]["body"] = "<!-- agent-coordination:T1067-01 -->"
    with pytest.raises(ValueError):
        _read(api)


def test_partial_pages_and_duplicate_ids_rejected() -> None:
    api = FakeApi()
    api.issue["comments"] = 3
    with pytest.raises(ValueError):
        _read(api)
    api = FakeApi()
    api.comments[1]["id"] = 15
    with pytest.raises(ValueError):
        _read(api)


def test_comment_edits_and_final_issue_drift_rejected() -> None:
    api = FakeApi()
    api.change_comment = True
    with pytest.raises(ValueError):
        _read(api)
    api = FakeApi()
    api.change_final_issue = True
    with pytest.raises(ValueError):
        _read(api)


@pytest.mark.parametrize("publisher", [0, True, 1.0, "148511409"])
def test_publisher_must_be_native_positive_integer(publisher: object) -> None:
    api = FakeApi()
    with pytest.raises(ValueError, match="publisher"):
        observe_issue_projection(
            api, projection=PROJECTION, expected_publisher_user_id=publisher
        )
    assert api.requests == []


def test_complete_multi_page_comment_roster_is_observed_twice() -> None:
    api = FakeApi()
    api.comments = [api.comments[0]] + [
        {
            "id": 20 + index,
            "user": {"id": 999},
            "body": f"other discussion {index}",
            "updated_at": "2026-11-03T11:59:30Z",
        }
        for index in range(100)
    ]
    api.issue["comments"] = len(api.comments)
    result = _read(api)
    assert result.comment_id == 15
    assert api.issue_reads == 3
    assert api.comment_reads == 4
    assert {query["page"] for _, _, query in api.requests if query is not None} == {
        "1",
        "2",
    }


@pytest.mark.parametrize(
    "changed", ["missing_body", "missing_user", "boolean_id", "missing_timestamp"]
)
def test_malformed_comment_fields_fail_closed(changed: str) -> None:
    api = FakeApi()
    if changed == "missing_body":
        api.comments[0].pop("body")
    elif changed == "missing_user":
        api.comments[0].pop("user")
    elif changed == "boolean_id":
        api.comments[0]["id"] = True
    else:
        api.comments[0]["updated_at"] = ""
    with pytest.raises(ValueError):
        _read(api)
