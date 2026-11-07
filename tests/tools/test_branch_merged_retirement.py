from __future__ import annotations

from dataclasses import dataclass

import pytest

from tools.branch_hygiene import (
    BranchInfo,
    MergedPullRequestSource,
    OpenPullRequestRefs,
    apply_merged_retirement,
    plan_merged_retirement,
)

A = "a" * 40
B = "b" * 40
C = "c" * 40
D = "d" * 40
E = "e" * 40


def _refs(name: str = "feature/open", sha: str = C) -> OpenPullRequestRefs:
    return OpenPullRequestRefs(name, sha, "main", A)


def test_only_main_ancestors_without_current_owners_are_eligible() -> None:
    branches = (
        BranchInfo("main", A, False),
        BranchInfo("feature/integrated", B, False),
        BranchInfo("feature/open", C, False),
        BranchInfo("feature/unfinished-ci", B, False),
        BranchInfo("feature/divergent", D, False),
        BranchInfo("feature/protected", B, True),
        BranchInfo("research/integrated", B, False),
        BranchInfo("provenance/branch-retention", B, False),
        BranchInfo("infra/retained", B, False),
    )
    decisions = plan_merged_retirement(
        branches,
        default_branch="main",
        open_pull_request_refs=(_refs(),),
        active_workflow_branches=frozenset({"feature/unfinished-ci"}),
        reachable_from_main=frozenset({A, B, C}),
        merged_pr_sources=(MergedPullRequestSource("feature/integrated", B),),
    )
    by_name = {item.branch.name: item for item in decisions}
    assert {item.branch.name for item in decisions if item.action == "delete"} == {
        "feature/integrated"
    }
    assert by_name["feature/open"].reason == "open-pr-ref"
    assert by_name["feature/unfinished-ci"].reason == "active-workflow"
    assert by_name["feature/divergent"].reason == "not-in-main-history"
    assert by_name["research/integrated"].action == "keep"
    assert by_name["infra/retained"].action == "keep"



@pytest.mark.parametrize(
    "branch_name",
    (
        "fix/m1-target-sizing-representability-20261106",
        "fix/m1-order-intent-notional-underflow-20261106",
        "fix/m1-liquidity-zero-notional-20261106",
        "fix/m1-bookstate-strict-underflow-atomicity-20261106",
        "fix/m1-carry-cashflow-underflow-20261106",
        "fix/windows-job-child-pid-marker-20261105",
    ),
)
def test_m1_merged_source_ref_is_preserved_even_when_main_reachable(
    branch_name: str,
) -> None:
    branch = BranchInfo(branch_name, B, False)
    decisions = plan_merged_retirement(
        (BranchInfo("main", A, False), branch),
        default_branch="main",
        open_pull_request_refs=(),
        active_workflow_branches=frozenset(),
        reachable_from_main=frozenset({A, B}),
        merged_pr_sources=(MergedPullRequestSource(branch_name, B),),
    )
    result = next(item for item in decisions if item.branch.name == branch_name)
    assert result.action == "keep"
    assert result.reason == "m1-source-preservation"


@pytest.mark.parametrize(
    "invalid",
    [
        (BranchInfo("main", A, False), BranchInfo("main", B, False)),
        (BranchInfo("feature/a", A, False),),
        (BranchInfo("main", "no-sha", False),),
    ],
)
def test_missing_duplicate_or_malformed_main_inventory_is_rejected(
    invalid: tuple[BranchInfo, ...],
) -> None:
    with pytest.raises(ValueError):
        plan_merged_retirement(
            invalid,
            default_branch="main",
            open_pull_request_refs=(),
            active_workflow_branches=frozenset(),
            reachable_from_main=frozenset({A, B}),
        )


@dataclass
class FakeApi:
    _repository: str = "shuntatsu/trade_rl"

    def __post_init__(self) -> None:
        self.current = [
            BranchInfo("main", A, False),
            BranchInfo("feature/done", B, False),
        ]
        self.pr_refs: tuple[OpenPullRequestRefs, ...] = ()
        self.running: frozenset[str] = frozenset()
        self.merged = [MergedPullRequestSource("feature/done", B)]
        self.deleted: list[str] = []
        self.post_fail = False

    def branches(self) -> tuple[BranchInfo, ...]:
        return tuple(self.current)

    def open_pull_request_refs(self) -> tuple[OpenPullRequestRefs, ...]:
        return self.pr_refs

    def active_workflow_branch_names(self) -> frozenset[str]:
        return self.running

    def merged_pr_source_refs(self) -> tuple[MergedPullRequestSource, ...]:
        return tuple(self.merged)

    def delete_branches_with_leases(self, branches: tuple[BranchInfo, ...]) -> None:
        self.deleted.extend(item.name for item in branches)
        if not self.post_fail:
            self.current = [
                entry for entry in self.current if entry.name not in self.deleted
            ]


def _plan(api: FakeApi):
    return plan_merged_retirement(
        api.branches(),
        default_branch="main",
        open_pull_request_refs=(),
        active_workflow_branches=frozenset(),
        reachable_from_main=frozenset({A, B}),
        merged_pr_sources=api.merged_pr_source_refs(),
    )


def _apply(api: FakeApi):
    return apply_merged_retirement(
        api,
        _plan(api),
        default_branch="main",
        expected_main_sha=A,
        reachable_from_main=frozenset({A, B}),
        max_deletes=10,
    )


def test_exact_tip_and_post_delete_readback() -> None:
    api = FakeApi()
    assert _apply(api) == ("feature/done",)
    assert api.deleted == ["feature/done"]
    assert [entry.name for entry in api.current] == ["main"]


@pytest.mark.parametrize(
    "change",
    [
        "branch-moved",
        "main-moved",
        "new-pr",
        "unfinished-workflow",
        "protected",
        "wrong-repo",
        "merged-evidence-disappeared",
    ],
)
def test_pre_delete_changes_abort_without_mutating(change: str) -> None:
    api = FakeApi()
    decisions = _plan(api)
    if change == "branch-moved":
        api.current[1] = BranchInfo("feature/done", C, False)
    elif change == "main-moved":
        api.current[0] = BranchInfo("main", D, False)
    elif change == "new-pr":
        api.pr_refs = (_refs("feature/done", B),)
    elif change == "unfinished-workflow":
        api.running = frozenset({"feature/done"})
    elif change == "protected":
        api.current[1] = BranchInfo("feature/done", B, True)
    elif change == "merged-evidence-disappeared":
        api.merged = []
    else:
        api._repository = "different/repo"
    with pytest.raises((RuntimeError, ValueError)):
        apply_merged_retirement(
            api,
            decisions,
            default_branch="main",
            expected_main_sha=A,
            reachable_from_main=frozenset({A, B}),
        )
    assert api.deleted == []


def test_missing_post_delete_readback_is_not_success() -> None:
    api = FakeApi()
    api.post_fail = True
    with pytest.raises(RuntimeError, match="post-delete"):
        _apply(api)


def test_bulk_legacy_apply_remains_disabled_without_explicit_merged_mode() -> None:
    from tools.branch_hygiene import main

    with pytest.raises(RuntimeError, match="apply is disabled"):
        main(["--repository", "shuntatsu/trade_rl", "--apply"])


def test_retirement_origin_requires_exact_read_and_push_urls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import subprocess

    from tools.branch_hygiene import require_retirement_origin

    urls = iter(
        [
            "https://github.com/shuntatsu/trade_rl.git",
            "https://github.com/shuntatsu/trade_rl.git",
        ]
    )

    def ok_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args[0], 0, next(urls) + "\n", "")

    monkeypatch.setattr(subprocess, "run", ok_run)
    require_retirement_origin()

    urls = iter(
        [
            "https://github.com/shuntatsu/trade_rl.git",
            "https://github.com/shuntatsu/trade-strategy-workbench.git",
        ]
    )
    with pytest.raises(RuntimeError, match="exact trade_rl HTTPS origin"):
        require_retirement_origin()


def test_apply_respects_max_delete_and_keeps_other_integrated_refs() -> None:
    api = FakeApi()
    api.current.extend(
        [
            BranchInfo("feature/done2", C, False),
            BranchInfo("feature/done3", D, False),
        ]
    )
    decisions = plan_merged_retirement(
        api.branches(),
        default_branch="main",
        open_pull_request_refs=(),
        active_workflow_branches=frozenset(),
        reachable_from_main=frozenset({A, B, C, D}),
        merged_pr_sources=tuple(api.merged)
        + (
            MergedPullRequestSource("feature/done2", C),
            MergedPullRequestSource("feature/done3", D),
        ),
    )
    api.merged.extend(
        [
            MergedPullRequestSource("feature/done2", C),
            MergedPullRequestSource("feature/done3", D),
        ]
    )
    deleted = apply_merged_retirement(
        api,
        decisions,
        default_branch="main",
        expected_main_sha=A,
        reachable_from_main=frozenset({A, B, C, D}),
        max_deletes=2,
        batch_size=1,
    )
    assert deleted == ("feature/done", "feature/done2")
    assert "feature/done3" in [branch.name for branch in api.current]


def test_unsupported_delete_bound_is_rejected_before_any_write() -> None:
    api = FakeApi()
    with pytest.raises(ValueError, match="bounds"):
        apply_merged_retirement(
            api,
            _plan(api),
            default_branch="main",
            expected_main_sha=A,
            reachable_from_main=frozenset({A, B}),
            max_deletes=101,
        )
    assert api.deleted == []


def test_fresh_unclaimed_branch_at_main_commit_must_not_be_deleted() -> None:
    # A new writer may create a branch before committing or opening a PR.
    # Main ancestry alone does not prove that branch is retired work.
    fresh = BranchInfo("feature/new-active", A, False)
    decisions = plan_merged_retirement(
        (BranchInfo("main", A, False), fresh),
        default_branch="main",
        open_pull_request_refs=(),
        active_workflow_branches=frozenset(),
        reachable_from_main=frozenset({A}),
    )
    result = next(item for item in decisions if item.branch.name == fresh.name)
    assert result.action == "keep"
    assert result.reason == "no-exact-merged-pr-source"


def test_merged_pr_evidence_must_bind_branch_name_and_exact_sha() -> None:
    branch = BranchInfo("feature/merged", B, False)
    for evidence in (
        (),
        (MergedPullRequestSource("feature/unrelated", B),),
        (MergedPullRequestSource("feature/merged", C),),
    ):
        decisions = plan_merged_retirement(
            (BranchInfo("main", A, False), branch),
            default_branch="main",
            open_pull_request_refs=(),
            active_workflow_branches=frozenset(),
            reachable_from_main=frozenset({A, B, C}),
            merged_pr_sources=evidence,
        )
        actual = next(item for item in decisions if item.branch.name == branch.name)
        assert actual.action == "keep"
        assert actual.reason == "no-exact-merged-pr-source"


def test_only_closed_and_merged_local_pr_final_heads_count_as_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tools.branch_hygiene import GitHubApi

    api = GitHubApi(
        repository="shuntatsu/trade_rl", token="fake", api_url="https://github.invalid"
    )

    def entry(
        name: str,
        sha: str,
        merged: bool = True,
        head_repo: str | None = "shuntatsu/trade_rl",
    ) -> dict[str, object]:
        return {
            "state": "closed",
            "merged_at": "2026-11-06T00:00:00Z" if merged else None,
            "head": {
                "ref": name,
                "sha": sha,
                "repo": {"full_name": head_repo} if head_repo else None,
            },
            "base": {"ref": "main", "repo": {"full_name": "shuntatsu/trade_rl"}},
        }

    monkeypatch.setattr(
        api,
        "_list_pages",
        lambda *args, **kwargs: [
            entry("feature/merged", B),
            entry("feature/closed-unmerged", B, merged=False),
            entry("feature/fork", B, head_repo="different/repo"),
            entry("feature/deleted", B, head_repo=None),
        ],
    )
    assert api.merged_pr_source_refs() == (
        MergedPullRequestSource("feature/merged", B),
    )
