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
from tools.github_work_inventory import build_inventory, classify_ancestry


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


def test_compare_direction_prevents_closing_ahead_pr() -> None:
    assert classify_ancestry({"ahead_by": 10, "behind_by": 0}) == "head_in_main"
    assert classify_ancestry({"ahead_by": 0, "behind_by": 10}) == "main_in_head"
    assert classify_ancestry({"ahead_by": 4, "behind_by": 2}) == "diverged"
    assert classify_ancestry({"ahead_by": 0, "behind_by": 0}) == "identical"
    with pytest.raises(ValueError):
        classify_ancestry({"ahead_by": True, "behind_by": 0})


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
    assert reasons[branch] == "coordinator_projection"
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


def test_partial_github_failure_does_not_produce_snapshot() -> None:
    api = FakeWorkApi()
    api.fail_comments = True
    with pytest.raises(RuntimeError, match="GitHub API unreachable"):
        build_inventory(api)  # type: ignore[arg-type]
