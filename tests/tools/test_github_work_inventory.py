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
        return self.issue_rows

    def issue_comments(self, issue_number: int) -> list[dict[str, object]]:
        if self.fail_comments:
            raise RuntimeError("GitHub API unreachable")
        return self.issue_comments_by_id.get(issue_number, [])

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
                "head": {"sha": "a" * 40, "ref": "feature/new"},
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
