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


@pytest.mark.parametrize("count", [100, 1000])
def test_full_last_page_requires_empty_terminal_page(count: int) -> None:
    api = FakeApi()
    api.comments = [api.comments[0]] + [
        dict(id=20 + i, user={"id": 999}, body="discussion", updated_at="now")
        for i in range(count - 1)
    ]
    api.issue["comments"] = count
    result = _read(api)
    assert result.comment_id == 15
    pages = [query["page"] for _, _, query in api.requests if query is not None]
    assert pages == [str(i) for i in range(1, count // 100 + 2)] * 2


@pytest.mark.parametrize("count", [100, 1000])
@pytest.mark.parametrize("scan", [1, 2])
def test_extra_comment_after_declared_full_page_is_rejected(
    count: int, scan: int
) -> None:
    class ExtraPageApi(FakeApi):
        def _request_json(
            self, method: str, suffix: str, *, query: dict[str, str] | None = None
        ) -> object:
            result = super()._request_json(method, suffix, query=query)
            if (
                query is not None
                and int(query["page"]) == count // 100 + 1
                and self.issue_reads == scan
            ):
                return [
                    dict(
                        id=9999,
                        user={"id": AUTHOR},
                        body=render_status_comment(STATUS),
                        updated_at="now",
                    )
                ]
            return result

    api = ExtraPageApi()
    api.comments = [api.comments[0]] + [
        dict(id=20 + i, user={"id": 999}, body="discussion", updated_at="now")
        for i in range(count - 1)
    ]
    api.issue["comments"] = count
    with pytest.raises(ValueError, match="pagination"):
        _read(api)


@pytest.mark.parametrize(
    ("section", "field", "bad", "error"),
    [
        ("issue", "node_id", "wrong-node", "Issue identity"),
        (
            "issue",
            "repository_url",
            "https://api.github.com/repos/other/repo",
            "Issue identity",
        ),
        ("issue", "state", "closed", "Issue identity"),
        ("issue", "updated_at", "now", "Issue updated_at"),
        ("comment", "body", render_status_comment(STATUS), "Issue comment body"),
        ("comment", "updated_at", "now", "Issue comment body"),
    ],
)
def test_foreign_projection_text_rejected_before_hooks(section, field, bad, error):
    calls = []

    class ForeignText(str):
        def __eq__(self, other):
            calls.append("eq")
            return True

        def __ne__(self, other):
            calls.append("ne")
            return False

        def strip(self, *args):
            calls.append("strip")
            return super().strip(*args)

        def __contains__(self, other):
            calls.append("contains")
            return super().__contains__(other)

        def __format__(self, spec):
            calls.append("format")
            return super().__format__(spec)

        def __str__(self):
            calls.append("str")
            return super().__str__()

        def __len__(self):
            calls.append("len")
            return super().__len__()

        def __bool__(self):
            calls.append("bool")
            return True

    class Api(FakeApi):
        def _request_json(self, method, suffix, *, query=None):
            result = super()._request_json(method, suffix, query=query)
            if section == "issue" and query is None:
                result[field] = ForeignText(bad)
            if section == "comment" and query is not None:
                result[0][field] = ForeignText(bad)
            return result

    with pytest.raises(ValueError, match=error):
        _read(Api())
    assert calls == []


@pytest.mark.parametrize(
    ("section", "error"),
    [
        ("issue", "Issue must be an object"),
        ("comment", "Issue comment must be an object"),
        ("publisher", "comment publisher must be an object"),
        ("page", "Issue comments pagination"),
    ],
)
def test_foreign_projection_containers_rejected_before_hooks(section, error):
    calls = []

    class ForeignDict(dict):
        def get(self, *args):
            calls.append("get")
            return super().get(*args)

        def __iter__(self):
            calls.append("dict iter")
            return super().__iter__()

        def __getitem__(self, key):
            calls.append("dict getitem")
            return super().__getitem__(key)

        def __contains__(self, key):
            calls.append("dict contains")
            return super().__contains__(key)

        def __bool__(self):
            calls.append("dict bool")
            return True

        def keys(self):
            calls.append("dict keys")
            return super().keys()

    class ForeignList(list):
        def __len__(self):
            calls.append("len")
            return super().__len__()

        def __iter__(self):
            calls.append("list iter")
            return super().__iter__()

        def __getitem__(self, key):
            calls.append("list getitem")
            return super().__getitem__(key)

        def __bool__(self):
            calls.append("list bool")
            return True

    class Api(FakeApi):
        def _request_json(self, method, suffix, *, query=None):
            result = super()._request_json(method, suffix, query=query)
            if query is None and section == "issue":
                return ForeignDict(result)
            if query is not None:
                if section == "page":
                    return ForeignList(result)
                if section == "comment":
                    result[0] = ForeignDict(result[0])
                if section == "publisher":
                    result[0]["user"] = ForeignDict(result[0]["user"])
            return result

    with pytest.raises(ValueError, match=error):
        _read(Api())
    assert calls == []


@pytest.mark.parametrize(
    ("section", "key", "error"),
    [
        ("issue", "node_id", "Issue must be an object"),
        ("comment", "body", "Issue comment must be an object"),
        ("publisher", "id", "comment publisher must be an object"),
    ],
)
def test_foreign_projection_key_rejected_before_lookup(section, key, error):
    calls = []

    class ForeignKey(str):
        def __hash__(self):
            calls.append("hash")
            return super().__hash__()

        def __eq__(self, other):
            calls.append("eq")
            return super().__eq__(other)

        def __str__(self):
            calls.append("str")
            return super().__str__()

    class Api(FakeApi):
        def _request_json(self, method, suffix, *, query=None):
            result = super()._request_json(method, suffix, query=query)
            row = None
            if query is None and section == "issue":
                row = result
            elif query is not None:
                if section == "comment":
                    row = result[0]
                elif section == "publisher":
                    row = result[0]["user"]
            if row is not None:
                row[ForeignKey(key)] = row.pop(key)
                # Dict insertion necessarily hashes the key. Check only the
                # consumer boundary, after the adversarial fixture is built.
                calls.clear()
            return result

    with pytest.raises(ValueError, match=error):
        _read(Api())
    assert calls == []


def test_foreign_projection_record_rejected_before_requests():
    class ForeignProjection(EnrollmentProjection):
        pass

    api = FakeApi()
    with pytest.raises(ValueError, match="EnrollmentProjection"):
        observe_issue_projection(
            api,
            projection=ForeignProjection(REPO, ISSUE, NODE, STATUS),
            expected_publisher_user_id=AUTHOR,
        )
    assert api.requests == []


@pytest.mark.parametrize(
    ("field", "error"),
    [
        ("repository", "repository must be owner/name"),
        ("issue_node_id", "issue reference must have an immutable node ID"),
    ],
)
def test_projection_constructor_rejects_foreign_identity_text(field, error):
    calls = []

    class ForeignText(str):
        def __str__(self):
            calls.append("str")
            return super().__str__()

        def __format__(self, spec):
            calls.append("format")
            return super().__format__(spec)

        def strip(self, *args):
            calls.append("strip")
            return super().strip(*args)

    with pytest.raises(ValueError, match=error):
        replace(PROJECTION, **{field: ForeignText(getattr(PROJECTION, field))})
    assert calls == []
