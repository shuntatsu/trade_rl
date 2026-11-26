from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tools.branch_hygiene import (
    BranchInfo,
    GitHubApi,
    MergedPullRequestSource,
    OpenPullRequestRefs,
    plan_merged_retirement,
)


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def _graph(repo: Path) -> tuple[str, str, str, str, str]:
    """Real normal merge with a constituent commit and a separate divergence."""
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "synthetic@example.invalid")
    _git(repo, "config", "user.name", "Synthetic")
    _git(repo, "commit", "--allow-empty", "-qm", "initial")
    base = _git(repo, "rev-parse", "HEAD")
    _git(repo, "switch", "-qc", "constituent")
    _git(repo, "commit", "--allow-empty", "-qm", "constituent")
    source = _git(repo, "rev-parse", "HEAD")
    _git(repo, "commit", "--allow-empty", "-qm", "carrier")
    carrier = _git(repo, "rev-parse", "HEAD")
    _git(repo, "switch", "-q", "main")
    _git(repo, "merge", "--no-ff", "-qm", "accepted merge", carrier)
    merged = _git(repo, "rev-parse", "HEAD")
    _git(repo, "switch", "-qc", "divergent", base)
    _git(repo, "commit", "--allow-empty", "-qm", "not integrated")
    divergent = _git(repo, "rev-parse", "HEAD")
    _git(repo, "switch", "-q", "main")
    return base, source, carrier, merged, divergent


def _entry(
    number: int,
    name: str,
    sha: str,
    *,
    merged_sha: str | None = None,
    base: str = "main",
    repo: str = "shuntatsu/trade_rl",
) -> dict:
    return {
        "number": number,
        "state": "closed",
        "merged_at": "2026-11-26T00:00:00Z" if merged_sha else None,
        "merge_commit_sha": merged_sha,
        "head": {"ref": name, "sha": sha, "repo": {"full_name": repo}},
        "base": {"ref": base, "repo": {"full_name": "shuntatsu/trade_rl"}},
    }


@pytest.mark.parametrize(
    "carrier_fault", [None, "wrong_head", "wrong_merge", "nested", "unmerged", "fork"]
)
def test_closed_constituents_need_real_main_merge_proof(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, carrier_fault: str | None
) -> None:
    base, source, carrier, merged, divergent = _graph(tmp_path)
    monkeypatch.chdir(tmp_path)
    branches = (
        BranchInfo("main", merged, False),
        BranchInfo("fix/constituent", source, False),
        BranchInfo("fix/nested-constituent", source, False),
        BranchInfo("fix/old-main", base, False),
        BranchInfo("fix/no-pr", source, False),
        BranchInfo("fix/divergent", divergent, False),
        BranchInfo("fix/moved", source, False),
        BranchInfo("fix/foreign", source, False),
    )
    accepted = _entry(
        10,
        "fix/carrier",
        source if carrier_fault == "wrong_head" else carrier,
        merged_sha=(
            base
            if carrier_fault == "wrong_merge"
            else None
            if carrier_fault == "unmerged"
            else merged
        ),
        base="fix/intermediate" if carrier_fault == "nested" else "main",
        repo="other/repo" if carrier_fault == "fork" else "shuntatsu/trade_rl",
    )
    records = [
        accepted,
        _entry(1, "fix/constituent", source),
        _entry(2, "fix/nested-constituent", source, base="fix/other"),
        _entry(3, "fix/old-main", base),
        _entry(4, "fix/divergent", divergent),
        _entry(5, "fix/moved", base),
        _entry(6, "fix/foreign", source, repo="other/repo"),
    ]
    api = GitHubApi(
        repository="shuntatsu/trade_rl", token="fake", api_url="https://github.invalid"
    )
    monkeypatch.setattr(api, "_list_pages", lambda *a, **kw: records)
    result = api.consolidated_pr_source_refs(branches, main_sha=merged)
    assert [
        (x.name, x.sha, x.source_pr_number, x.carrier_pr_number) for x in result
    ] == (
        [("fix/constituent", source, 1, 10), ("fix/nested-constituent", source, 2, 10)]
        if carrier_fault is None
        else []
    )


def test_consolidated_proof_cannot_override_live_or_reserved_refs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base, source, carrier, merged, divergent = _graph(tmp_path)
    monkeypatch.chdir(tmp_path)
    names = (
        "fix/retire",
        "fix/open",
        "fix/active",
        "fix/protected",
        "research/saved",
        "fix/m1-retain",
        "fix/issue1112-m1-audit-20261123",
        "fix/issue1112-junit-nested-counts-20261124",
        "audit/m1-retained",
        "coordination/task-registry",
        "task/lease",
        "lease/task",
    )
    branches = (
        BranchInfo("main", merged, False),
        *(BranchInfo(n, source, n == "fix/protected") for n in names),
    )
    api = GitHubApi(
        repository="shuntatsu/trade_rl", token="fake", api_url="https://github.invalid"
    )
    monkeypatch.setattr(
        api,
        "_list_pages",
        lambda *a, **kw: [
            _entry(100, "fix/carrier", carrier, merged_sha=merged),
            *(_entry(i, n, source) for i, n in enumerate(names, 1)),
        ],
    )
    proof = api.consolidated_pr_source_refs(branches, main_sha=merged)
    result = plan_merged_retirement(
        branches,
        default_branch="main",
        open_pull_request_refs=(
            OpenPullRequestRefs("fix/open", source, "main", merged),
        ),
        active_workflow_branches=frozenset({"fix/active"}),
        reachable_from_main=frozenset({base, source, carrier, merged}),
        merged_pr_sources=(),
        consolidated_pr_sources=proof,
    )
    assert [(x.branch.name, x.reason) for x in result if x.action == "delete"] == [
        ("fix/retire", "exact-closed-pr-consolidated-in-main")
    ]


def test_consolidated_roster_duplicate_pr_number_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, source, carrier, merged, _ = _graph(tmp_path)
    monkeypatch.chdir(tmp_path)
    api = GitHubApi(
        repository="shuntatsu/trade_rl", token="fake", api_url="https://github.invalid"
    )
    row = _entry(1, "fix/constituent", source)
    monkeypatch.setattr(
        api,
        "_list_pages",
        lambda *a, **kw: [
            _entry(10, "fix/carrier", carrier, merged_sha=merged),
            row,
            row,
        ],
    )
    with pytest.raises(RuntimeError, match="duplicate.*pull request"):
        api.consolidated_pr_source_refs(
            (
                BranchInfo("main", merged, False),
                BranchInfo("fix/constituent", source, False),
            ),
            main_sha=merged,
        )


@pytest.mark.parametrize("select_both", [False, True])
def test_apply_requires_fresh_consolidated_proof_before_every_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, select_both: bool
) -> None:
    from tools.branch_hygiene import apply_merged_retirement, reachable_commits

    _, source, carrier, merged, _ = _graph(tmp_path)
    monkeypatch.chdir(tmp_path)
    api = GitHubApi(
        repository="shuntatsu/trade_rl", token="fake", api_url="https://github.invalid"
    )
    branches = [
        BranchInfo("main", merged, False),
        BranchInfo("fix/first", source, False),
        BranchInfo("fix/second", source, False),
        BranchInfo("fix/ordinary", carrier, False),
    ]
    records = [
        _entry(10, "fix/carrier", carrier, merged_sha=merged),
        _entry(1, "fix/first", source),
        _entry(2, "fix/second", source),
    ]
    deleted: list[str] = []
    monkeypatch.setattr(api, "_list_pages", lambda *a, **kw: records)
    monkeypatch.setattr(api, "branches", lambda: tuple(branches))
    monkeypatch.setattr(api, "open_pull_request_refs", lambda: ())
    monkeypatch.setattr(api, "active_workflow_branch_names", lambda: frozenset())
    ordinary = (MergedPullRequestSource("fix/ordinary", carrier),)
    monkeypatch.setattr(api, "merged_pr_source_refs", lambda: ordinary)
    proof = api.consolidated_pr_source_refs(branches, main_sha=merged)
    reachable = reachable_commits((merged,))
    decisions = plan_merged_retirement(
        branches,
        default_branch="main",
        open_pull_request_refs=(),
        active_workflow_branches=frozenset(),
        reachable_from_main=reachable,
        consolidated_pr_sources=proof,
        merged_pr_sources=ordinary,
    )

    def delete(batch: tuple[BranchInfo, ...]) -> None:
        assert len(batch) == 1
        deleted.append(batch[0].name)
        branches.remove(batch[0])
        # The other exact source's closed evidence is no longer in the roster.
        records.pop()

    monkeypatch.setattr(api, "delete_branches_with_leases", delete)
    # Opt-in is mandatory even with a proof in the preview.
    with pytest.raises(RuntimeError, match="changed before retirement"):
        apply_merged_retirement(
            api,
            decisions,
            default_branch="main",
            expected_main_sha=merged,
            reachable_from_main=reachable,
            batch_size=1,
        )
    assert deleted == []
    with pytest.raises(ValueError, match="explicit exact ref"):
        apply_merged_retirement(
            api,
            decisions,
            default_branch="main",
            expected_main_sha=merged,
            reachable_from_main=reachable,
            batch_size=1,
            include_consolidated_prs=True,
        )
    assert deleted == []
    for invalid in (
        [f"fix/first={merged}"],
        [f"fix/unknown={source}"],
        [f"fix/ordinary={carrier}"],
        ["fix/first=bad-sha"],
        [f"fix/first={source}", f"fix/first={source}"],
    ):
        with pytest.raises(ValueError):
            apply_merged_retirement(
                api,
                decisions,
                default_branch="main",
                expected_main_sha=merged,
                reachable_from_main=reachable,
                include_consolidated_prs=True,
                approved_consolidated_refs=invalid,
            )
        assert deleted == []
    scope = [f"fix/first={source}"]
    if select_both:
        scope.append(f"fix/second={source}")
    kwargs = dict(
        default_branch="main",
        expected_main_sha=merged,
        reachable_from_main=reachable,
        batch_size=1,
        include_consolidated_prs=True,
        approved_consolidated_refs=scope,
    )
    if select_both:
        with pytest.raises(RuntimeError, match="changed before retirement"):
            apply_merged_retirement(api, decisions, **kwargs)
    else:
        assert apply_merged_retirement(api, decisions, **kwargs) == ("fix/first",)
    assert deleted == ["fix/first"]
    assert [x.name for x in branches] == ["main", "fix/second", "fix/ordinary"]


def test_carrier_uses_actual_second_parent_when_multiple_prs_share_merge_sha(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, source, carrier, merged, _ = _graph(tmp_path)
    monkeypatch.chdir(tmp_path)
    api = GitHubApi(
        repository="shuntatsu/trade_rl", token="fake", api_url="https://github.invalid"
    )
    monkeypatch.setattr(
        api,
        "_list_pages",
        lambda *a, **kw: [
            _entry(4, "fix/redundant", source, merged_sha=merged),
            _entry(20, "fix/carrier", carrier, merged_sha=merged),
            _entry(21, "fix/carrier-alias", carrier, merged_sha=merged),
            _entry(1, "fix/constituent", source),
        ],
    )
    proof = api.consolidated_pr_source_refs(
        (
            BranchInfo("main", merged, False),
            BranchInfo("fix/constituent", source, False),
        ),
        main_sha=merged,
    )
    assert len(proof) == 1 and proof[0].carrier_pr_number == 20


def test_apply_refuses_main_change_during_consolidated_proof(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tools.branch_hygiene import apply_merged_retirement, reachable_commits

    _, source, carrier, merged, divergent = _graph(tmp_path)
    monkeypatch.chdir(tmp_path)
    api = GitHubApi(
        repository="shuntatsu/trade_rl", token="fake", api_url="https://github.invalid"
    )
    branches = [
        BranchInfo("main", merged, False),
        BranchInfo("fix/constituent", source, False),
    ]
    monkeypatch.setattr(
        api,
        "_list_pages",
        lambda *a, **kw: [
            _entry(10, "fix/carrier", carrier, merged_sha=merged),
            _entry(1, "fix/constituent", source),
        ],
    )
    monkeypatch.setattr(api, "branches", lambda: tuple(branches))
    monkeypatch.setattr(api, "open_pull_request_refs", lambda: ())
    monkeypatch.setattr(api, "active_workflow_branch_names", lambda: frozenset())
    monkeypatch.setattr(api, "merged_pr_source_refs", lambda: ())
    original = api.consolidated_pr_source_refs
    proof = original(branches, main_sha=merged)
    reachable = reachable_commits((merged,))
    decisions = plan_merged_retirement(
        branches,
        default_branch="main",
        open_pull_request_refs=(),
        active_workflow_branches=frozenset(),
        reachable_from_main=reachable,
        consolidated_pr_sources=proof,
    )

    def drift(entries, *, main_sha):
        result = original(entries, main_sha=main_sha)
        branches[0] = BranchInfo("main", divergent, False)
        return result

    monkeypatch.setattr(api, "consolidated_pr_source_refs", drift)
    monkeypatch.setattr(
        api,
        "delete_branches_with_leases",
        lambda *_: pytest.fail("drift reached deletion"),
    )
    with pytest.raises(RuntimeError, match="main changed during consolidated proof"):
        apply_merged_retirement(
            api,
            decisions,
            default_branch="main",
            expected_main_sha=merged,
            reachable_from_main=reachable,
            include_consolidated_prs=True,
            approved_consolidated_refs=[f"fix/constituent={source}"],
        )


def test_unscoped_consolidated_cli_apply_refuses_before_api() -> None:
    from tools.branch_hygiene import main

    with pytest.raises(ValueError, match="explicit exact ref"):
        main(
            [
                "--repository",
                "shuntatsu/trade_rl",
                "--retire-merged",
                "--apply",
                "--include-consolidated",
            ]
        )


def test_octopus_first_introduction_is_not_normal_carrier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base, source, carrier, _, divergent = _graph(tmp_path)
    _git(tmp_path, "reset", "--hard", base)
    # Independent third parent with a real commit. commit-tree makes the
    # parent-count oracle independent of git merge strategy selection.
    tree = _git(tmp_path, "rev-parse", f"{base}^{{tree}}")
    octopus = _git(
        tmp_path,
        "commit-tree",
        tree,
        "-p",
        base,
        "-p",
        carrier,
        "-p",
        divergent,
        "-m",
        "octopus",
    )
    assert len(_git(tmp_path, "show", "-s", "--format=%P", octopus).split()) == 3
    monkeypatch.chdir(tmp_path)
    api = GitHubApi(
        repository="shuntatsu/trade_rl", token="fake", api_url="https://github.invalid"
    )
    monkeypatch.setattr(
        api,
        "_list_pages",
        lambda *a, **kw: [
            _entry(10, "fix/carrier", carrier, merged_sha=octopus),
            _entry(1, "fix/constituent", source),
        ],
    )
    assert (
        api.consolidated_pr_source_refs(
            (
                BranchInfo("main", octopus, False),
                BranchInfo("fix/constituent", source, False),
            ),
            main_sha=octopus,
        )
        == ()
    )
