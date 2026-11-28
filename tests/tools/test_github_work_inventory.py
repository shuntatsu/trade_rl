"""Read-only GitHub work inventory safety and direction regressions."""

from __future__ import annotations

import pytest

from tools.agent_repo.coordination.github_state import (
    TaskStatusRecord,
    render_status_comment,
)
from tools.agent_repo.coordination.leases import lease_branch_name
from tools.agent_repo.coordination.model import TaskPhase
from tools.branch_hygiene import BranchInfo
from tools.github_work_inventory import (
    GitHubWorkApi,
    build_inventory,
    classify_ancestry,
)


class FakeWorkApi:
    repository = "shuntatsu/trade_rl"

    def __init__(self) -> None:
        self.issue_comments_by_id: dict[int, list[dict[str, object]]] = {}
        self.fail_comments = False
        self.pull_rows: list[dict[str, object]] = []
        self.issue_rows: list[dict[str, object]] = [{"number": 810}]
        self.branch_rows = (
            BranchInfo("main", "a" * 40, False),
            BranchInfo("feature/unregistered", "b" * 40, False),
            BranchInfo("provenance/branch-retention", "c" * 40, False),
        )

    def repository_default_branch(self) -> str:
        return "main"

    def branches(self) -> tuple[BranchInfo, ...]:
        return self.branch_rows

    def open_pulls(self) -> list[dict[str, object]]:
        return self.pull_rows

    def open_issues(self) -> list[dict[str, object]]:
        return [
            {
                "id": 100000 + int(row.get("number", 0)),
                "node_id": f"I_TEST_{row.get('number')}",
                "repository_url": f"https://api.github.com/repos/{self.repository}",
                "state": "open",
                "updated_at": "2026-11-04T00:00:00Z",
                "comments": len(
                    self.issue_comments_by_id.get(int(row.get("number", 0)), [])
                ),
                **row,
            }
            for row in self.issue_rows
        ]

    def issue_comments(self, issue_number: int) -> list[dict[str, object]]:
        if self.fail_comments:
            raise RuntimeError("GitHub API unreachable")
        return [
            {
                "id": 1000000 * issue_number + index + 1,
                "user": {"id": 148511409},
                "updated_at": "2026-11-04T00:00:00Z",
                **comment,
            }
            for index, comment in enumerate(
                self.issue_comments_by_id.get(issue_number, [])
            )
        ]

    def compare_head_to_main(self, head_sha: str, main_sha: str) -> str:
        assert main_sha == "a" * 40
        return "head_in_main" if head_sha == "d" * 40 else "main_in_head"


def test_work_inventory_public_adapter_exposes_no_mutating_operations() -> None:
    api = GitHubWorkApi(repository="owner/repo", token="fake")
    assert not hasattr(api, "delete_branches_with_leases")
    assert not hasattr(api, "create_retention_commit")
    assert not hasattr(api, "create_branch_ref")
    assert not hasattr(api, "update_branch_ref")
    assert not hasattr(api, "apply_cleanup")


def test_compare_direction_prevents_closing_ahead_pr() -> None:
    assert (
        classify_ancestry({"status": "ahead", "ahead_by": 10, "behind_by": 0})
        == "head_in_main"
    )
    assert (
        classify_ancestry({"status": "behind", "ahead_by": 0, "behind_by": 10})
        == "main_in_head"
    )
    assert (
        classify_ancestry({"status": "diverged", "ahead_by": 4, "behind_by": 2})
        == "diverged"
    )
    assert (
        classify_ancestry({"status": "identical", "ahead_by": 0, "behind_by": 0})
        == "identical"
    )
    with pytest.raises(ValueError, match="nonnegative integers"):
        classify_ancestry({"status": "ahead", "ahead_by": True, "behind_by": 0})


def test_compare_rejects_missing_or_conflicting_github_status() -> None:
    with pytest.raises(ValueError, match="compare status conflicts"):
        classify_ancestry({"ahead_by": 10, "behind_by": 0})
    with pytest.raises(ValueError, match="compare status conflicts"):
        classify_ancestry({"status": "behind", "ahead_by": 10, "behind_by": 0})
    with pytest.raises(ValueError, match="compare status conflicts"):
        classify_ancestry({"status": "ahead", "ahead_by": 0, "behind_by": 10})


def test_live_adapter_compare_uses_head_to_main_direction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = GitHubWorkApi(repository="owner/repo", token="fake")
    requested: list[tuple[str, str]] = []

    def fake_request(method: str, suffix: str) -> dict[str, object]:
        requested.append((method, suffix))
        return {"status": "behind", "ahead_by": 0, "behind_by": 13}

    monkeypatch.setattr(api._reader, "_request_json", fake_request)
    head_sha = "c" * 40
    main_sha = "a" * 40
    assert api.compare_head_to_main(head_sha, main_sha) == "main_in_head"
    assert requested == [("GET", f"compare/{head_sha}...{main_sha}")]


def test_local_pr_identity_handles_repository_name_casing() -> None:
    api = FakeWorkApi()
    api.repository = "Shuntatsu/Trade_RL"
    api.branch_rows += (BranchInfo("feature/current", "d" * 40, False),)
    api.pull_rows = [
        {
            "number": 123,
            "head": {
                "ref": "feature/current",
                "sha": "d" * 40,
                "repo": {"full_name": "shuntatsu/trade_rl"},
            },
            "base": {
                "ref": "main",
                "sha": "a" * 40,
                "repo": {"full_name": "shuntatsu/trade_rl"},
            },
        }
    ]

    report = build_inventory(api, compare_prs=True)  # type: ignore[arg-type]
    pull = report["pull_requests"][0]
    assert pull["head_ref"] == "feature/current"
    assert pull["relation_to_main"] == "head_in_main"
    assert report["cleanup_authorized"] is False


def test_live_adapter_open_query_uses_shared_client_pagination(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = GitHubWorkApi(repository="Shuntatsu/Trade_RL", token="fake")
    calls: list[tuple[str, str, dict[str, str] | None]] = []

    def fake_request(
        method: str, suffix: str, *, query: dict[str, str] | None = None
    ) -> list[object]:
        calls.append((method, suffix, query))
        return []

    monkeypatch.setattr(api._reader, "_request_json", fake_request)
    assert api.open_pulls() == []
    assert api.open_issues() == []
    assert calls == [
        (
            "GET",
            "pulls",
            {"per_page": "100", "page": "1", "state": "open"},
        ),
        (
            "GET",
            "issues",
            {"per_page": "100", "page": "1", "state": "open"},
        ),
    ]


def test_unregistered_branch_is_never_deletion_authority() -> None:
    report = build_inventory(FakeWorkApi())  # type: ignore[arg-type]
    assert report["cleanup_authorized"] is False
    assert report["task_registry_complete"] is False
    assert report["unclassified_branches"] == ["feature/unregistered"]


def test_open_stacked_pr_refs_are_protected_and_ancestry_is_explicit() -> None:
    api = FakeWorkApi()
    api.branch_rows += (
        BranchInfo("feature/stack", "d" * 40, False),
        BranchInfo("stack/base", "e" * 40, False),
    )
    api.pull_rows = [
        {
            "number": 999,
            "head": {
                "ref": "feature/stack",
                "sha": "d" * 40,
                "repo": {"full_name": api.repository},
            },
            "base": {
                "ref": "stack/base",
                "sha": "e" * 40,
                "repo": {"full_name": api.repository},
            },
        }
    ]
    report = build_inventory(api, compare_prs=True)  # type: ignore[arg-type]
    reasons = {row["name"]: row["reason"] for row in report["branches"]}
    assert reasons["feature/stack"] == "open_pr_ref"
    assert reasons["stack/base"] == "open_pr_ref"
    assert report["pull_requests"][0]["relation_to_main"] == "head_in_main"
    assert report["cleanup_authorized"] is False


@pytest.mark.parametrize(
    "head_repo",
    [
        pytest.param({"full_name": "someone/fork"}, id="foreign-fork"),
        pytest.param(None, id="unavailable-head-repo"),
    ],
)
def test_nonlocal_pr_ancestry_is_explicitly_uncompared(
    monkeypatch: pytest.MonkeyPatch,
    head_repo: dict[str, str] | None,
) -> None:
    api = FakeWorkApi()
    api.pull_rows = [
        {
            "number": 123,
            "head": {
                "ref": "feature/fork",
                "sha": "d" * 40,
                "repo": head_repo,
            },
            "base": {
                "ref": "main",
                "sha": "a" * 40,
                "repo": {"full_name": api.repository},
            },
        }
    ]

    def unexpected_compare(head_sha: str, main_sha: str) -> str:
        raise AssertionError("nonlocal head cannot be compared as a local SHA")

    monkeypatch.setattr(api, "compare_head_to_main", unexpected_compare)
    report = build_inventory(api, compare_prs=True)  # type: ignore[arg-type]
    pull = report["pull_requests"][0]
    assert pull["head_ref"] is None
    assert pull["relation_to_main"] == "nonlocal_head_uncompared"
    assert report["cleanup_authorized"] is False
    assert report["task_registry_complete"] is False


def test_local_pr_head_must_match_branch_but_base_sha_can_be_old() -> None:
    api = FakeWorkApi()
    api.branch_rows += (BranchInfo("feature/current", "d" * 40, False),)
    pr = {
        "number": 123,
        "head": {
            "ref": "feature/current",
            "sha": "d" * 40,
            "repo": {"full_name": api.repository},
        },
        "base": {
            "ref": "main",
            "sha": "b" * 40,
            "repo": {"full_name": api.repository},
        },
    }
    api.pull_rows = [pr]
    report = build_inventory(api)  # type: ignore[arg-type]
    assert report["pull_requests"][0]["base_sha"] == "b" * 40
    assert report["main_sha"] == "a" * 40
    assert report["pr_base_sha_semantics"] == "historical_ref_not_current_branch_tip"
    assert report["cleanup_authorized"] is False

    pr["head"] = {
        "ref": "feature/current",
        "sha": "c" * 40,
        "repo": {"full_name": api.repository},
    }
    with pytest.raises(RuntimeError, match="local PR head tip"):
        build_inventory(api)  # type: ignore[arg-type]

    pr["head"] = {
        "ref": "feature/missing",
        "sha": "d" * 40,
        "repo": {"full_name": api.repository},
    }
    with pytest.raises(RuntimeError, match="local PR head tip"):
        build_inventory(api)  # type: ignore[arg-type]


def test_coordinator_projection_only_protects_never_authorizes() -> None:
    api = FakeWorkApi()
    branch = lease_branch_name("T810-01", 2, "agent-a")
    api.branch_rows += (BranchInfo(branch, "f" * 40, False),)
    status = TaskStatusRecord(
        task_id="T810-01",
        task_revision=1,
        task_contract_digest="0" * 64,
        phase=TaskPhase.EXECUTING,
        condition="healthy",
        base_sha="a" * 40,
        owner="agent-a",
        lease_epoch=2,
        lease_branch=branch,
        head_sha="f" * 40,
    )
    api.issue_comments_by_id[810] = [{"body": render_status_comment(status)}]
    report = build_inventory(api)  # type: ignore[arg-type]
    reasons = {row["name"]: row["reason"] for row in report["branches"]}
    assert report["task_projection_count"] == 1
    assert reasons[branch] == "unverified_task_comment"
    assert report["task_projection_trusted"] is False
    assert report["cleanup_authorized"] is False


def test_duplicate_or_malformed_coordinator_state_fails_closed() -> None:
    api = FakeWorkApi()
    projection = TaskStatusRecord(
        task_id="T810-01",
        task_revision=1,
        task_contract_digest="0" * 64,
        phase=TaskPhase.EXECUTING,
        condition="healthy",
        base_sha="a" * 40,
    )
    api.issue_comments_by_id[810] = [{"body": render_status_comment(projection)}] * 2
    with pytest.raises(RuntimeError, match="duplicate Coordinator"):
        build_inventory(api)  # type: ignore[arg-type]

    api.issue_comments_by_id[810] = [
        {"body": "<!-- agent-coordination:T810-01 --> malformed"}
    ]
    with pytest.raises(RuntimeError, match="invalid Coordinator"):
        build_inventory(api)  # type: ignore[arg-type]


def test_work_inventory_rejects_changed_branch_tip_during_readback() -> None:
    api = FakeWorkApi()
    first = api.branch_rows
    calls = 0

    def moving_branches() -> tuple[BranchInfo, ...]:
        nonlocal calls
        calls += 1
        if calls == 1:
            return first
        return (
            first[0],
            BranchInfo("feature/unregistered", "e" * 40, False),
            first[2],
        )

    api.branches = moving_branches  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="branch tips or protection changed"):
        build_inventory(api)  # type: ignore[arg-type]


def test_work_inventory_rejects_pr_created_during_readback() -> None:
    api = FakeWorkApi()
    calls = 0

    def changing_prs() -> list[dict[str, object]]:
        nonlocal calls
        calls += 1
        if calls == 1:
            return []
        return [
            {
                "number": 1057,
                "head": {"sha": "a" * 40, "ref": "feature/new", "repo": None},
                "base": {
                    "ref": "main",
                    "sha": "a" * 40,
                    "repo": {"full_name": api.repository},
                },
            }
        ]

    api.open_pulls = changing_prs  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="open PR refs changed"):
        build_inventory(api)  # type: ignore[arg-type]


def test_work_inventory_rejects_duplicate_pr_and_issue_pages() -> None:
    api = FakeWorkApi()
    pr = {
        "number": 100,
        "head": {
            "sha": "d" * 40,
            "ref": "feature/pr",
            "repo": {"full_name": api.repository},
        },
        "base": {
            "ref": "main",
            "sha": "a" * 40,
            "repo": {"full_name": api.repository},
        },
    }
    api.branch_rows += (BranchInfo("feature/pr", "d" * 40, False),)
    api.pull_rows = [pr, pr]
    with pytest.raises(RuntimeError, match="duplicate open PR"):
        build_inventory(api)  # type: ignore[arg-type]

    api.pull_rows = []
    api.issue_rows = [{"number": 810}, {"number": 810}]
    with pytest.raises(RuntimeError, match="duplicate open Issue"):
        build_inventory(api)  # type: ignore[arg-type]


def test_partial_github_failure_does_not_produce_snapshot() -> None:
    api = FakeWorkApi()
    api.fail_comments = True
    with pytest.raises(RuntimeError, match="GitHub API unreachable"):
        build_inventory(api)  # type: ignore[arg-type]


def test_work_inventory_opt_in_marks_unfinished_workflow_branches() -> None:
    api = FakeWorkApi()
    calls = 0

    def active() -> frozenset[str]:
        nonlocal calls
        calls += 1
        return frozenset({"feature/unregistered", "main"})

    setattr(api, "unfinished_workflow_branch_names", active)
    report = build_inventory(api, include_workflow_refs=True)  # type: ignore[arg-type]
    reasons = {row["name"]: row["reason"] for row in report["branches"]}
    assert reasons["feature/unregistered"] == "unfinished_workflow_ref"
    assert report["unfinished_workflow_branches"] == ["feature/unregistered", "main"]
    assert report["unfinished_workflow_branch_names_checked"] is True
    assert report["workflow_activity_checked"] is False
    assert report["task_registry_complete"] is False
    assert report["cleanup_authorized"] is False
    assert "feature/unregistered" not in report["unclassified_branches"]
    assert calls == 2


def test_work_inventory_legacy_default_does_not_scan_workflows() -> None:
    api = FakeWorkApi()

    def forbidden_scan() -> frozenset[str]:
        raise AssertionError("default inventory must not start a new API scan")

    setattr(api, "unfinished_workflow_branch_names", forbidden_scan)
    report = build_inventory(api)  # type: ignore[arg-type]
    assert "unfinished_workflow_branches" not in report
    assert "unfinished_workflow_branch_names_checked" not in report
    assert report["workflow_activity_checked"] is False
    assert report["unclassified_branches"] == ["feature/unregistered"]


def test_work_inventory_workflow_branch_drift_is_not_silent() -> None:
    api = FakeWorkApi()
    calls = 0

    def drift() -> frozenset[str]:
        nonlocal calls
        calls += 1
        return frozenset({"main"}) if calls == 1 else frozenset()

    setattr(api, "unfinished_workflow_branch_names", drift)
    with pytest.raises(RuntimeError, match="unfinished workflow refs changed"):
        build_inventory(api, include_workflow_refs=True)  # type: ignore[arg-type]


def test_work_inventory_missing_workflow_branch_is_unknown_not_safe() -> None:
    api = FakeWorkApi()
    setattr(
        api,
        "unfinished_workflow_branch_names",
        lambda: frozenset({"deleted-while-pending"}),
    )
    with pytest.raises(RuntimeError, match="missing local branches"):
        build_inventory(api, include_workflow_refs=True)  # type: ignore[arg-type]


def test_work_inventory_workflow_api_failure_rejects_snapshot() -> None:
    api = FakeWorkApi()

    def transport_failure() -> frozenset[str]:
        raise RuntimeError("GitHub workflow pagination failed")

    setattr(api, "unfinished_workflow_branch_names", transport_failure)
    with pytest.raises(RuntimeError, match="pagination failed"):
        build_inventory(api, include_workflow_refs=True)  # type: ignore[arg-type]


def test_work_inventory_rechecks_branch_tips_after_workflow_scan() -> None:
    api = FakeWorkApi()
    setattr(api, "unfinished_workflow_branch_names", lambda: frozenset({"main"}))
    first = api.branch_rows
    calls = 0

    def moving_branches() -> tuple[BranchInfo, ...]:
        nonlocal calls
        calls += 1
        if calls <= 2:
            return first
        return (
            first[0],
            BranchInfo("feature/unregistered", "e" * 40, False),
            first[2],
        )

    api.branches = moving_branches  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="changed during workflow scan"):
        build_inventory(api, include_workflow_refs=True)  # type: ignore[arg-type]


def test_live_work_inventory_workflow_scan_reuses_branch_hygiene_reader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = GitHubWorkApi(repository="owner/repo", token="fake")
    calls = 0

    def active_workflows() -> frozenset[str]:
        nonlocal calls
        calls += 1
        return frozenset({"main"})

    monkeypatch.setattr(api._reader, "active_workflow_branch_names", active_workflows)
    assert api.unfinished_workflow_branch_names() == frozenset({"main"})
    assert calls == 1


def _complete_issue_row(
    *, number: int = 810, comments: int = 1, updated_at: str = "2026-11-04T00:00:00Z"
) -> dict[str, object]:
    return {
        "number": number,
        "id": number + 100000,
        "node_id": f"I_ISSUE_{number}",
        "state": "open",
        "updated_at": updated_at,
        "comments": comments,
    }


def _ordinary_comment(
    *, comment_id: int = 901, body: str = "Ordinary discussion", author_id: int = 77
) -> dict[str, object]:
    return {
        "id": comment_id,
        "user": {"id": author_id},
        "body": body,
        "updated_at": "2026-11-04T00:01:00Z",
    }


def test_issue_comment_snapshot_is_read_twice_without_promoting_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = FakeWorkApi()
    api.issue_rows = [_complete_issue_row()]
    api.issue_comments_by_id[810] = [_ordinary_comment()]
    reads: list[int] = []
    original = api.issue_comments

    def tracked_comments(issue_number: int) -> list[dict[str, object]]:
        reads.append(issue_number)
        return original(issue_number)

    monkeypatch.setattr(api, "issue_comments", tracked_comments)
    report = build_inventory(api)  # type: ignore[arg-type]
    assert reads == [810, 810]
    assert report["task_registry_complete"] is False
    assert report["cleanup_authorized"] is False


def test_edited_comment_body_or_author_between_passes_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for field, replacement in (("body", "edited"), ("user", {"id": 78})):
        api = FakeWorkApi()
        api.issue_rows = [_complete_issue_row()]
        comment = _ordinary_comment()
        calls = 0

        def changing_comments(issue_number: int) -> list[dict[str, object]]:
            nonlocal calls
            calls += 1
            row = dict(comment)
            if calls == 2:
                row[field] = replacement
            return [row]

        monkeypatch.setattr(api, "issue_comments", changing_comments)
        with pytest.raises((ValueError, RuntimeError), match="comment"):
            build_inventory(api)  # type: ignore[arg-type]


def test_issue_updated_at_drift_with_same_issue_number_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = FakeWorkApi()
    api.issue_rows = [_complete_issue_row()]
    api.issue_comments_by_id[810] = [_ordinary_comment()]
    calls = 0

    def moving_issues() -> list[dict[str, object]]:
        nonlocal calls
        calls += 1
        return [
            _complete_issue_row(
                updated_at="2026-11-04T00:01:00Z"
                if calls == 2
                else "2026-11-04T00:00:00Z"
            )
        ]

    monkeypatch.setattr(api, "open_issues", moving_issues)
    with pytest.raises((ValueError, RuntimeError), match="Issue"):
        build_inventory(api)  # type: ignore[arg-type]


def test_incomplete_comment_count_does_not_produce_work_inventory() -> None:
    api = FakeWorkApi()
    api.issue_rows = [_complete_issue_row(comments=2)]
    api.issue_comments_by_id[810] = [_ordinary_comment()]
    with pytest.raises((ValueError, RuntimeError), match="comment"):
        build_inventory(api)  # type: ignore[arg-type]


def test_duplicate_comment_ids_are_rejected_even_without_task_marker() -> None:
    api = FakeWorkApi()
    api.issue_rows = [_complete_issue_row(comments=2)]
    api.issue_comments_by_id[810] = [
        _ordinary_comment(),
        _ordinary_comment(body="other"),
    ]
    with pytest.raises((ValueError, RuntimeError), match="comment"):
        build_inventory(api)  # type: ignore[arg-type]


def test_new_task_marker_after_first_issue_comment_observation_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = FakeWorkApi()
    api.issue_rows = [_complete_issue_row(comments=1)]
    calls = 0
    old = _ordinary_comment()

    def changing_comments(issue_number: int) -> list[dict[str, object]]:
        nonlocal calls
        calls += 1
        return (
            [old]
            if calls == 1
            else [_ordinary_comment(body="<!-- agent-coordination:T810-NEW -->")]
        )

    monkeypatch.setattr(api, "issue_comments", changing_comments)
    with pytest.raises((ValueError, RuntimeError), match="comment"):
        build_inventory(api)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "invalid",
    [
        {"id": True},
        {"id": 0},
        {"user": {"id": True}},
        {"user": None},
        {"updated_at": ""},
    ],
)
def test_malformed_comment_identity_does_not_produce_work_inventory(
    invalid: dict[str, object],
) -> None:
    api = FakeWorkApi()
    api.issue_rows = [_complete_issue_row()]
    api.issue_comments_by_id[810] = [{**_ordinary_comment(), **invalid}]
    with pytest.raises((ValueError, RuntimeError), match="comment"):
        build_inventory(api)  # type: ignore[arg-type]


def test_issue_identity_collisions_across_distinct_numbers_are_rejected() -> None:
    for collision in ("id", "node_id"):
        api = FakeWorkApi()
        first = _complete_issue_row(number=810, comments=0)
        second = _complete_issue_row(number=811, comments=0)
        second[collision] = first[collision]
        api.issue_rows = [first, second]
        with pytest.raises((ValueError, RuntimeError), match="Issue identity"):
            build_inventory(api)  # type: ignore[arg-type]


def test_comment_id_reused_across_open_issues_is_rejected() -> None:
    api = FakeWorkApi()
    api.issue_rows = [
        _complete_issue_row(number=810, comments=1),
        _complete_issue_row(number=811, comments=1),
    ]
    api.issue_comments_by_id[810] = [_ordinary_comment(comment_id=100)]
    api.issue_comments_by_id[811] = [_ordinary_comment(comment_id=100)]
    with pytest.raises((ValueError, RuntimeError), match="comment ID"):
        build_inventory(api)  # type: ignore[arg-type]


def test_issue_comment_mutation_during_second_pass_rejected_without_task_marker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = FakeWorkApi()
    api.issue_rows = [_complete_issue_row(comments=1)]
    original = _ordinary_comment()
    count = 0

    def drop_comments(issue_number: int) -> list[dict[str, object]]:
        nonlocal count
        count += 1
        return [original] if count == 1 else []

    monkeypatch.setattr(api, "issue_comments", drop_comments)
    with pytest.raises((ValueError, RuntimeError), match="comment"):
        build_inventory(api)  # type: ignore[arg-type]


def test_live_comment_adapter_refuses_more_than_bounded_pages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = GitHubWorkApi(repository="shuntatsu/trade_rl", token="fake")
    called_pages: list[int] = []

    def fake_request(
        method: str, suffix: str, *, query: dict[str, str] | None = None
    ) -> list[dict[str, object]]:
        assert method == "GET"
        assert suffix == "issues/810/comments"
        assert query is not None and query["per_page"] == "100"
        page = int(query["page"])
        called_pages.append(page)
        assert page <= 11
        return [{"body": "comment"}] * (100 if page <= 10 else 1)

    monkeypatch.setattr(api._reader, "_request_json", fake_request)
    with pytest.raises(RuntimeError, match="comment.*limit|comment.*bound"):
        api.issue_comments(810)
    assert called_pages == list(range(1, 12))


def test_live_comment_adapter_accepts_exact_limit_with_empty_terminal_page(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = GitHubWorkApi(repository="shuntatsu/trade_rl", token="fake")
    called_pages: list[int] = []

    def fake_request(
        method: str, suffix: str, *, query: dict[str, str] | None = None
    ) -> list[dict[str, object]]:
        assert method == "GET" and suffix == "issues/810/comments"
        assert query is not None
        page = int(query["page"])
        called_pages.append(page)
        return [{"body": "comment"}] * (100 if page <= 10 else 0)

    monkeypatch.setattr(api._reader, "_request_json", fake_request)
    assert len(api.issue_comments(810)) == 1000
    assert called_pages == list(range(1, 12))


def test_issue_list_bound_to_selected_repository_and_refuses_foreign_issue() -> None:
    api = FakeWorkApi()
    api.issue_rows = [
        {
            **_complete_issue_row(comments=0),
            "repository_url": "https://api.github.com/repos/another/project",
        }
    ]
    with pytest.raises((ValueError, RuntimeError), match="open Issue repository"):
        build_inventory(api)  # type: ignore[arg-type]


@pytest.mark.parametrize("include_workflow_refs", [False, True])
def test_work_inventory_rejects_default_branch_drift(
    include_workflow_refs: bool,
) -> None:
    api = FakeWorkApi()
    selected = "main"
    events: list[str] = []
    original_branches = api.branches

    def default_branch() -> str:
        events.append("default")
        return selected

    def branches() -> tuple[BranchInfo, ...]:
        nonlocal selected
        events.append("branches")
        rows = original_branches()
        if events.count("branches") == (3 if include_workflow_refs else 2):
            selected = "feature/unregistered"
        return rows

    def workflow_refs() -> frozenset[str]:
        events.append("workflow")
        return frozenset({"main"})

    api.repository_default_branch = default_branch  # type: ignore[method-assign]
    api.branches = branches  # type: ignore[method-assign]
    setattr(api, "unfinished_workflow_branch_names", workflow_refs)
    with pytest.raises(RuntimeError, match="default branch changed"):
        build_inventory(api, include_workflow_refs=include_workflow_refs)  # type: ignore[arg-type]
    expected = ["default", "branches", "branches"]
    if include_workflow_refs:
        expected += ["workflow", "workflow", "branches"]
    assert events == [*expected, "default"]


@pytest.mark.parametrize("include_workflow_refs", [False, True])
@pytest.mark.parametrize("default", ["main", "feature/unregistered"])
def test_work_inventory_stable_default_branch(
    include_workflow_refs: bool, default: str
) -> None:
    api = FakeWorkApi()
    calls = 0

    def default_branch() -> str:
        nonlocal calls
        calls += 1
        return default

    api.repository_default_branch = default_branch  # type: ignore[method-assign]
    setattr(api, "unfinished_workflow_branch_names", lambda: frozenset({"main"}))
    report = build_inventory(api, include_workflow_refs=include_workflow_refs)  # type: ignore[arg-type]
    assert calls == 2
    assert report["main_sha"] == ("a" if default == "main" else "b") * 40
    assert [
        row["name"] for row in report["branches"] if row["reason"] == "default"
    ] == [default]
    assert report["cleanup_authorized"] is False
    assert report["task_registry_complete"] is False


def test_work_inventory_default_branch_readback_failure() -> None:
    api = FakeWorkApi()
    calls = 0

    def default_branch() -> str:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("default branch API unavailable")
        return "main"

    api.repository_default_branch = default_branch  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="default branch API unavailable"):
        build_inventory(api)  # type: ignore[arg-type]


@pytest.mark.parametrize("location", ["main", "branch", "PR head", "PR base"])
def test_inventory_rejects_zero_oid_as_commit(location: str) -> None:
    api = FakeWorkApi()
    zero = "0" * 40
    if location == "main":
        api.branch_rows = (BranchInfo("main", zero, False), *api.branch_rows[1:])
    elif location == "branch":
        api.branch_rows += (BranchInfo("feature/zero", zero, False),)
    else:
        api.pull_rows = [
            {
                "number": 123,
                "head": {
                    "ref": "feature/fork",
                    "sha": zero if location == "PR head" else "d" * 40,
                    "repo": {"full_name": "someone/fork"},
                },
                "base": {
                    "ref": "main",
                    "sha": zero if location == "PR base" else "a" * 40,
                    "repo": {"full_name": api.repository},
                },
            }
        ]
    with pytest.raises(ValueError, match="exact commit SHA"):
        build_inventory(api)  # type: ignore[arg-type]


@pytest.mark.parametrize("location", ["head", "main"])
def test_compare_rejects_zero_oid_before_http(
    monkeypatch: pytest.MonkeyPatch, location: str
) -> None:
    api = GitHubWorkApi(repository="owner/repo", token="fake")

    def unexpected_request(*args: object, **kwargs: object) -> object:
        raise AssertionError("zero OID must be rejected before HTTP")

    monkeypatch.setattr(api._reader, "_request_json", unexpected_request)
    with pytest.raises(ValueError, match="exact commit SHA"):
        api.compare_head_to_main(
            "0" * 40 if location == "head" else "a" * 40,
            "0" * 40 if location == "main" else "b" * 40,
        )


class _FalseZero(int):
    def __eq__(self, other: object) -> bool:
        return other == 0


class _CompareStatus(str):
    def __ne__(self, other: object) -> bool:
        return False


class _CompareDict(dict[str, object]):
    def get(self, key, default=None):
        raise AssertionError("foreign lookup must not run")


@pytest.mark.parametrize(
    "payload",
    [
        {"status": "ahead", "ahead_by": 4, "behind_by": _FalseZero(3)},
        {"status": _CompareStatus("behind"), "ahead_by": 4, "behind_by": 0},
        _CompareDict(status="ahead", ahead_by=4, behind_by=0),
        {"status": "ahead", "ahead_by": 4, "behind_by": 0, _CompareStatus("other"): 1},
    ],
)
def test_compare_rejects_foreign_identity_before_comparison(payload) -> None:
    with pytest.raises(ValueError):
        classify_ancestry(payload)


class _ForeignShaFormat(str):
    def __format__(self, spec: str) -> str:
        return "f" * 40


class _ForeignIssueFormat(int):
    def __format__(self, spec: str) -> str:
        return "999"


@pytest.mark.parametrize("location", ["head", "main", "issue"])
def test_inventory_rejects_foreign_url_identity_before_request(
    monkeypatch: pytest.MonkeyPatch, location: str
) -> None:
    api = GitHubWorkApi(repository="owner/repo", token="fake")
    requested: list[str] = []

    def fake_request(method: str, suffix: str, **kwargs: object) -> object:
        requested.append(suffix)
        if suffix.startswith("compare/"):
            return {"status": "behind", "ahead_by": 0, "behind_by": 1}
        return []

    monkeypatch.setattr(api._reader, "_request_json", fake_request)
    try:
        if location == "issue":
            with pytest.raises(
                ValueError, match="Issue number must be a positive integer"
            ):
                api.issue_comments(_ForeignIssueFormat(123))
            for invalid in (True, False):
                with pytest.raises(
                    ValueError, match="Issue number must be a positive integer"
                ):
                    api.issue_comments(invalid)
        else:
            with pytest.raises(
                ValueError, match=f"{location}_sha must be an exact commit SHA"
            ):
                api.compare_head_to_main(
                    _ForeignShaFormat("a" * 40) if location == "head" else "a" * 40,
                    _ForeignShaFormat("b" * 40) if location == "main" else "b" * 40,
                )
    finally:
        assert not requested, f"foreign identity reached HTTP: {requested}"
    assert api.issue_comments(123) == []
    assert api.compare_head_to_main("a" * 40, "b" * 40) == "main_in_head"
    assert requested == ["issues/123/comments", f"compare/{'a' * 40}...{'b' * 40}"]


class _ForgedRepositoryName(str):
    def casefold(self) -> str:
        return "shuntatsu/trade_rl"


class _ForgedRepositoryUrl(str):
    def casefold(self) -> str:
        return "https://api.github.com/repos/shuntatsu/trade_rl"


class _ForeignInventoryObject(dict[str, object]):
    def get(self, key, default=None):
        raise AssertionError("foreign inventory lookup must not run")


@pytest.mark.parametrize(
    "location", ["issue-url", "issue-object", "pr-base", "pr-head", "pr-object"]
)
def test_inventory_rejects_foreign_repository_identity(
    monkeypatch: pytest.MonkeyPatch, location: str
) -> None:
    api = FakeWorkApi()
    if location.startswith("issue"):
        row = {
            **_complete_issue_row(comments=0),
            "repository_url": (
                _ForgedRepositoryUrl("https://api.github.com/repos/foreign/project")
                if location == "issue-url"
                else f"https://api.github.com/repos/{api.repository}"
            ),
        }
        item = _ForeignInventoryObject(row) if location == "issue-object" else row
        monkeypatch.setattr(api, "open_issues", lambda: [item])
    else:
        api.branch_rows += (BranchInfo("feature/current", "d" * 40, False),)
        row = {
            "number": 123,
            "head": {
                "ref": "feature/current",
                "sha": "d" * 40,
                "repo": {
                    "full_name": _ForgedRepositoryName("foreign/project")
                    if location == "pr-head"
                    else api.repository
                },
            },
            "base": {
                "ref": "main",
                "sha": "a" * 40,
                "repo": {
                    "full_name": _ForgedRepositoryName("foreign/project")
                    if location == "pr-base"
                    else api.repository
                },
            },
        }
        item = _ForeignInventoryObject(row) if location == "pr-object" else row
        monkeypatch.setattr(api, "open_pulls", lambda: [item])
    with pytest.raises(
        ValueError, match="native|repository identity|different repository"
    ):
        build_inventory(api, compare_prs=True)  # type: ignore[arg-type]


class _ForeignText(str):
    pass


class _ForgedOpenState:
    def __ne__(self, other: object) -> bool:
        return False


class _ForeignInventoryList(list):
    pass


@pytest.mark.parametrize("field", ["node_id", "state", "updated_at"])
def test_inventory_rejects_foreign_issue_identity_text(field: str) -> None:
    api = FakeWorkApi()
    row = _complete_issue_row(comments=0)
    row[field] = (
        _ForgedOpenState() if field == "state" else _ForeignText(str(row[field]))
    )
    api.issue_rows = [row]
    with pytest.raises(ValueError, match="Issue identity or state|Issue updated_at"):
        build_inventory(api)  # type: ignore[arg-type]


@pytest.mark.parametrize("field", ["body", "updated_at"])
def test_inventory_rejects_foreign_comment_identity_text(field: str) -> None:
    api = FakeWorkApi()
    comment = _ordinary_comment()
    comment[field] = _ForeignText(str(comment[field]))
    api.issue_comments_by_id[810] = [comment]
    with pytest.raises(ValueError, match="Issue comment body or updated_at"):
        build_inventory(api)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "location",
    [
        "head-ref",
        "base-ref",
        "head-object",
        "base-object",
        "head-repo-object",
        "base-repo-object",
        "foreign-key",
    ],
)
def test_inventory_rejects_foreign_nested_pr_identity(location: str) -> None:
    api = FakeWorkApi()
    api.branch_rows += (BranchInfo("feature/current", "d" * 40, False),)
    head = {
        "ref": "feature/current",
        "sha": "d" * 40,
        "repo": {"full_name": api.repository},
    }
    base = {"ref": "main", "sha": "a" * 40, "repo": {"full_name": api.repository}}
    pr = {"number": 123, "head": head, "base": base}
    if location == "head-ref":
        head["ref"] = _ForeignText("feature/current")
    elif location == "base-ref":
        base["ref"] = _ForeignText("main")
    elif location == "head-object":
        pr["head"] = _ForeignInventoryObject(head)
    elif location == "base-object":
        pr["base"] = _ForeignInventoryObject(base)
    elif location == "head-repo-object":
        head["repo"] = _ForeignInventoryObject(head["repo"])
    elif location == "base-repo-object":
        base["repo"] = _ForeignInventoryObject(base["repo"])
    else:
        pr[_ForeignText("extra")] = 1
    api.pull_rows = [pr]
    with pytest.raises(ValueError, match="native|branch missing"):
        build_inventory(api, compare_prs=True)  # type: ignore[arg-type]


def test_inventory_rejects_foreign_repository_before_reader_creation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tools.github_work_inventory as inventory

    monkeypatch.setattr(
        inventory,
        "GitHubApi",
        lambda **kwargs: pytest.fail("foreign repository reached reader"),
    )
    with pytest.raises(ValueError, match="repository.*native"):
        GitHubWorkApi(repository=_ForeignText("owner/repo"), token="fake")


@pytest.mark.parametrize("source", ["open_pulls", "open_issues", "issue_comments"])
def test_inventory_rejects_foreign_roster_container(
    monkeypatch: pytest.MonkeyPatch, source: str
) -> None:
    api = FakeWorkApi()
    original = getattr(api, source)
    monkeypatch.setattr(
        api, source, lambda *args: _ForeignInventoryList(original(*args))
    )
    with pytest.raises((ValueError, RuntimeError), match="native|count differs"):
        build_inventory(api)  # type: ignore[arg-type]


def test_inventory_rejects_foreign_comment_page(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = GitHubWorkApi(repository="owner/repo", token="fake")
    monkeypatch.setattr(
        api._reader, "_request_json", lambda *a, **kw: _ForeignInventoryList()
    )
    with pytest.raises(RuntimeError, match="page has invalid size or shape"):
        api.issue_comments(123)


@pytest.mark.parametrize(
    "field", ["name", "protected", "default", "repository", "container"]
)
def test_inventory_rejects_foreign_branch_or_adapter_identity(
    monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    api = FakeWorkApi()
    if field == "name":
        api.branch_rows = (
            BranchInfo(_ForeignText("main"), "a" * 40, False),
            *api.branch_rows[1:],
        )
    elif field == "protected":
        api.branch_rows = (BranchInfo("main", "a" * 40, 0), *api.branch_rows[1:])
    elif field == "default":
        monkeypatch.setattr(
            api, "repository_default_branch", lambda: _ForeignText("main")
        )
    elif field == "repository":
        api.repository = _ForeignText(api.repository)
    else:
        monkeypatch.setattr(api, "branches", lambda: list(api.branch_rows))
    with pytest.raises(ValueError, match="native"):
        build_inventory(api)  # type: ignore[arg-type]
