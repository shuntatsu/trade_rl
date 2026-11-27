"""The owner-approved November 27 retirement is a fixed, isolated exception."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from tools import branch_hygiene as hygiene

MAIN = "a" * 40
AUDIT = "b" * 40
CLOUD = "c" * 40
SUCCESSOR = "d" * 40
M1_NAME = "fix/m1-example"
CLOUD_NAME = "research/cloud-example"
ANCHOR = "prepare/cloud-example"


@pytest.fixture
def roster(monkeypatch: pytest.MonkeyPatch) -> tuple[hygiene.BranchInfo, ...]:
    monkeypatch.setattr(hygiene, "_APPROVED_20261127_MAIN", ((M1_NAME, AUDIT),))
    monkeypatch.setattr(hygiene, "_APPROVED_20261127_CLOUD", ((CLOUD_NAME, CLOUD),))
    monkeypatch.setattr(hygiene, "_APPROVED_20261127_SUCCESSOR", (ANCHOR, SUCCESSOR))
    return (
        hygiene.BranchInfo("main", MAIN, False),
        hygiene.BranchInfo(M1_NAME, AUDIT, False),
        hygiene.BranchInfo(CLOUD_NAME, CLOUD, False),
        hygiene.BranchInfo(ANCHOR, SUCCESSOR, False),
        hygiene.BranchInfo("fix/unselected", AUDIT, False),
        hygiene.BranchInfo("research/other", CLOUD, False),
    )


def selected(
    roster: tuple[hygiene.BranchInfo, ...],
    *,
    opened: tuple[hygiene.OpenPullRequestRefs, ...] = (),
    running: frozenset[str] = frozenset(),
    main_history: frozenset[str] = frozenset((MAIN, AUDIT)),
    cloud_history: frozenset[str] = frozenset((SUCCESSOR, CLOUD)),
    sources: tuple[hygiene.MergedPullRequestSource, ...] = (
        hygiene.MergedPullRequestSource(M1_NAME, AUDIT),
    ),
) -> tuple[hygiene.BranchInfo, ...]:
    return hygiene.plan_approved_20261127_retirement(
        roster,
        expected_main_sha=MAIN,
        open_pull_request_refs=opened,
        active_workflow_branches=running,
        reachable_from_main=main_history,
        reachable_from_successor=cloud_history,
        merged_pr_sources=sources,
    )


def test_only_fixed_exact_sources_selected(
    roster: tuple[hygiene.BranchInfo, ...],
) -> None:
    assert {b.name for b in selected(roster)} == {M1_NAME, CLOUD_NAME}
    ordinary = hygiene.plan_merged_retirement(
        roster,
        default_branch="main",
        open_pull_request_refs=(),
        active_workflow_branches=frozenset(),
        reachable_from_main=frozenset((MAIN, AUDIT)),
        merged_pr_sources=(hygiene.MergedPullRequestSource(M1_NAME, AUDIT),),
    )
    assert not any(d.action == "delete" for d in ordinary)


@pytest.mark.parametrize("index", [1, 2, 3])
def test_changed_exact_tip_or_successor_aborts(
    roster: tuple[hygiene.BranchInfo, ...], index: int
) -> None:
    changed = list(roster)
    changed[index] = replace(changed[index], sha="e" * 40)
    with pytest.raises((ValueError, RuntimeError)):
        selected(tuple(changed))


@pytest.mark.parametrize("index", [1, 2])
def test_protected_selected_branch_aborts(
    roster: tuple[hygiene.BranchInfo, ...], index: int
) -> None:
    changed = list(roster)
    changed[index] = replace(changed[index], protected=True)
    with pytest.raises((ValueError, RuntimeError)):
        selected(tuple(changed))


@pytest.mark.parametrize("name", [M1_NAME, CLOUD_NAME])
def test_open_and_running_candidates_are_not_retired(
    roster: tuple[hygiene.BranchInfo, ...], name: str
) -> None:
    with pytest.raises((ValueError, RuntimeError)):
        selected(roster, running=frozenset((name,)))
    with pytest.raises((ValueError, RuntimeError)):
        selected(
            roster,
            opened=(hygiene.OpenPullRequestRefs(name, AUDIT, "main", MAIN),),
        )
    with pytest.raises((ValueError, RuntimeError)):
        selected(
            roster,
            opened=(hygiene.OpenPullRequestRefs("other", AUDIT, name, MAIN),),
        )


def test_missing_evidence_and_unmerged_m1_rejected(
    roster: tuple[hygiene.BranchInfo, ...],
) -> None:
    with pytest.raises((ValueError, RuntimeError)):
        selected(roster, sources=())
    with pytest.raises((ValueError, RuntimeError)):
        selected(roster, main_history=frozenset((MAIN,)))
    with pytest.raises((ValueError, RuntimeError)):
        selected(roster, cloud_history=frozenset((SUCCESSOR,)))


def test_missing_successor_and_duplicate_roster_rejected(
    roster: tuple[hygiene.BranchInfo, ...],
) -> None:
    with pytest.raises((ValueError, RuntimeError)):
        selected(tuple(b for b in roster if b.name != ANCHOR))
    with pytest.raises((ValueError, RuntimeError)):
        selected((*roster, roster[1]))


def test_completed_batch_is_idempotent(roster: tuple[hygiene.BranchInfo, ...]) -> None:
    assert (
        selected(tuple(b for b in roster if b.name not in {M1_NAME, CLOUD_NAME})) == ()
    )


def test_backup_must_be_real_git_directory(tmp_path: Path) -> None:
    with pytest.raises((ValueError, RuntimeError)):
        hygiene._verify_approved_20261127_backup(tmp_path, ())


def test_main_ancestor_also_requires_exact_pr_identity(
    roster: tuple[hygiene.BranchInfo, ...],
) -> None:
    with pytest.raises(RuntimeError, match="proof missing"):
        selected(
            roster,
            sources=(hygiene.MergedPullRequestSource("different-name", AUDIT),),
        )


def test_consolidated_m1_proof_is_accepted(
    roster: tuple[hygiene.BranchInfo, ...],
) -> None:
    actual = hygiene.plan_approved_20261127_retirement(
        roster,
        expected_main_sha=MAIN,
        open_pull_request_refs=(),
        active_workflow_branches=frozenset(),
        reachable_from_main=frozenset((MAIN, AUDIT)),
        reachable_from_successor=frozenset((SUCCESSOR, CLOUD)),
        consolidated_pr_sources=(
            hygiene.ConsolidatedPullRequestSource(M1_NAME, AUDIT, 1, 2),
        ),
    )
    assert {b.name for b in actual} == {M1_NAME, CLOUD_NAME}


def test_unapproved_names_at_same_tip_remain(
    roster: tuple[hygiene.BranchInfo, ...],
) -> None:
    assert "fix/unselected" not in {b.name for b in selected(roster)}
    assert "research/other" not in {b.name for b in selected(roster)}
    assert ANCHOR not in {b.name for b in selected(roster)}


def test_real_backup_ref_and_object_verification(tmp_path: Path) -> None:
    import subprocess

    repo = tmp_path / "backup.git"
    subprocess.run(
        ["git", "init", "--bare", str(repo)], check=True, capture_output=True
    )
    cmd = ["git", f"--git-dir={repo}"]
    tree = subprocess.check_output([*cmd, "mktree"], input="", text=True).strip()
    oid = subprocess.check_output(
        [
            *cmd,
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit-tree",
            tree,
            "-m",
            "synthetic",
        ],
        text=True,
    ).strip()
    name = "research/synthetic"
    subprocess.run([*cmd, "update-ref", "refs/remotes/origin/" + name, oid], check=True)
    hygiene._verify_approved_20261127_backup(repo, ((name, oid),))
    with pytest.raises(RuntimeError, match="backup ref differs"):
        hygiene._verify_approved_20261127_backup(repo, ((name, "f" * 40),))
    (repo / "objects/info/alternates").write_text(str(tmp_path / "objects"))
    with pytest.raises(ValueError, match="alternate"):
        hygiene._verify_approved_20261127_backup(repo, ((name, oid),))


@pytest.mark.parametrize("mode", ["preview", "apply", "drift", "post-drift", "limit"])
def test_execution_keeps_unselected_and_uses_fresh_roster(
    roster: tuple[hygiene.BranchInfo, ...],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mode: str,
) -> None:
    api = hygiene.GitHubApi(
        repository="shuntatsu/trade_rl",
        token="synthetic",
        api_url="https://api.github.com",
    )
    monkeypatch.setattr(hygiene, "require_retirement_origin", lambda: None)
    backup_calls = []
    monkeypatch.setattr(
        hygiene,
        "_verify_approved_20261127_backup",
        lambda path, required: backup_calls.append((path, required)),
    )
    graft_path = tmp_path / "nonexistent-grafts"

    def history(*args: str) -> str:
        if args == ("rev-parse", "refs/remotes/origin/main"):
            return MAIN
        if args == ("rev-parse", "--is-shallow-repository"):
            return "false"
        if args == ("rev-parse", "--git-path", "info/grafts"):
            return str(graft_path)
        if args == ("rev-parse", f"refs/remotes/origin/{ANCHOR}"):
            return SUCCESSOR
        if args == ("rev-list", MAIN):
            return f"{MAIN}\\n{AUDIT}".replace("\\n", "\n")
        if args == ("rev-list", SUCCESSOR):
            return f"{SUCCESSOR}\\n{CLOUD}".replace("\\n", "\n")
        raise AssertionError(args)

    monkeypatch.setattr(hygiene, "_git_history", history)
    monkeypatch.setattr(
        hygiene.subprocess,
        "check_output",
        lambda args: Path(hygiene.__file__).read_bytes(),
    )
    monkeypatch.setattr(api, "repository_default_branch", lambda: "main")
    monkeypatch.setattr(
        api,
        "merged_pr_source_refs",
        lambda: (hygiene.MergedPullRequestSource(M1_NAME, AUDIT),),
    )
    monkeypatch.setattr(api, "consolidated_pr_source_refs", lambda *a, **kw: ())
    monkeypatch.setattr(api, "open_pull_request_refs", lambda: ())
    monkeypatch.setattr(api, "active_workflow_branch_names", lambda: frozenset())

    deleted_batches = []
    expected_remaining = tuple(b for b in roster if b.name not in {M1_NAME, CLOUD_NAME})
    changed = tuple(replace(b, sha="e" * 40) if b.name == ANCHOR else b for b in roster)
    rosters = iter(
        (
            roster,
            changed if mode == "drift" else roster,
            roster if mode == "post-drift" else expected_remaining,
        )
    )
    monkeypatch.setattr(api, "branches", lambda: next(rosters))
    monkeypatch.setattr(
        api,
        "delete_branches_with_leases",
        lambda branches: deleted_batches.append(tuple(branches)),
    )

    def run():
        return hygiene.run_approved_20261127_retirement(
            api,
            apply=mode != "preview",
            backup_git_dir=tmp_path,
            max_deletes=1 if mode == "limit" else 16,
        )

    if mode in {"drift", "limit", "post-drift"}:
        with pytest.raises((RuntimeError, ValueError)):
            run()
        assert len(deleted_batches) == (1 if mode == "post-drift" else 0)
    else:
        planned, deleted = run()
        assert {b.name for b in planned} == {M1_NAME, CLOUD_NAME}
        if mode == "apply":
            assert set(deleted) == {M1_NAME, CLOUD_NAME}
            assert deleted_batches == [planned]
        else:
            assert deleted == ()
            assert deleted_batches == []
    assert backup_calls == [(tmp_path, ((CLOUD_NAME, CLOUD), (ANCHOR, SUCCESSOR)))]


@pytest.mark.parametrize(
    "args",
    [
        ["--retire-approved-20261127", "--backup-git-dir", "/tmp/backup"],
        ["--retire-merged", "--retire-approved-20261127"],
        ["--retire-merged", "--backup-git-dir", "/tmp/backup"],
        [
            "--retire-merged",
            "--retire-approved-20261127",
            "--backup-git-dir",
            "/tmp/backup",
            "--include-consolidated",
        ],
    ],
)
def test_mode_arguments_reject_unapproved_combinations(args: list[str]) -> None:
    with pytest.raises((RuntimeError, ValueError)):
        hygiene.main(
            ["--repository", "shuntatsu/trade_rl", "--token", "synthetic", *args]
        )


def test_atomic_git_delete_lease_refuses_moved_source(tmp_path: Path) -> None:
    """Use actual Git transport locally: a stale member must not delete its peer."""
    import subprocess

    bare = tmp_path / "remote.git"
    local = tmp_path / "local"
    subprocess.run(
        ["git", "init", "--bare", str(bare)], check=True, capture_output=True
    )
    subprocess.run(["git", "init", str(local)], check=True, capture_output=True)
    cmd = ["git", "-C", str(local)]
    tree = subprocess.check_output([*cmd, "mktree"], input="", text=True).strip()
    first = subprocess.check_output(
        [
            *cmd,
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit-tree",
            tree,
            "-m",
            "first",
        ],
        text=True,
    ).strip()
    second = subprocess.check_output(
        [
            *cmd,
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit-tree",
            tree,
            "-p",
            first,
            "-m",
            "second",
        ],
        text=True,
    ).strip()
    subprocess.run(
        [
            *cmd,
            "push",
            str(bare),
            first + ":refs/heads/one",
            second + ":refs/heads/two",
        ],
        check=True,
        capture_output=True,
    )
    lease = [
        *cmd,
        "push",
        "--atomic",
        "--porcelain",
        "--force-with-lease=refs/heads/one:" + first,
        "--force-with-lease=refs/heads/two:" + first,
        str(bare),
        ":refs/heads/one",
        ":refs/heads/two",
    ]
    failed = subprocess.run(lease, capture_output=True)
    assert failed.returncode != 0
    assert subprocess.check_output(
        ["git", f"--git-dir={bare}", "rev-parse", "refs/heads/one", "refs/heads/two"],
        text=True,
    ).splitlines() == [first, second]
    lease[lease.index("--force-with-lease=refs/heads/two:" + first)] = (
        "--force-with-lease=refs/heads/two:" + second
    )
    subprocess.run(lease, check=True, capture_output=True)
    assert (
        subprocess.check_output(
            [
                "git",
                f"--git-dir={bare}",
                "for-each-ref",
                "--format=%(refname)",
                "refs/heads",
            ],
            text=True,
        )
        == ""
    )
