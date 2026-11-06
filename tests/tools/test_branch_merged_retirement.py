from __future__ import annotations

from dataclasses import dataclass

import pytest

from tools.branch_hygiene import (
    BranchInfo,
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
        self.deleted: list[str] = []
        self.post_fail = False

    def branches(self) -> tuple[BranchInfo, ...]:
        return tuple(self.current)

    def open_pull_request_refs(self) -> tuple[OpenPullRequestRefs, ...]:
        return self.pr_refs

    def active_workflow_branch_names(self) -> frozenset[str]:
        return self.running

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
