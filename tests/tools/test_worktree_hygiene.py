"""Worktree hygiene is exercised against real temporary git repositories."""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Sequence, cast

import pytest

from tools import worktree_hygiene as wh
from tools.worktree_hygiene import OpenPullRequest

PrProvider = Callable[[], Sequence[OpenPullRequest]]


def git(cwd: Path, *args: str) -> str:
    process = subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=False
    )
    assert process.returncode == 0, process.stderr
    return process.stdout.strip()


@dataclass
class Env:
    repo: Path
    origin: Path
    root: Path

    @property
    def scope(self) -> Path:
        return self.repo / ".claude" / "worktrees"

    def add(self, name: str, *, parent: Path | None = None) -> Path:
        path = (parent or self.scope) / name
        git(self.repo, "worktree", "add", "-q", "-b", f"feat/{name}", str(path), "main")
        return path

    def age(self, worktree: Path, hours: float = 240.0) -> None:
        """Backdate every file and directory (ignored ones too, ``.git`` excluded)."""
        old = time.time() - hours * 3600.0
        for dirpath, dirnames, filenames in os.walk(worktree):
            if Path(dirpath) == worktree:
                dirnames[:] = [name for name in dirnames if name != ".git"]
                filenames = [name for name in filenames if name != ".git"]
            for name in (*filenames, *dirnames):
                os.utime(os.path.join(dirpath, name), (old, old))
        os.utime(worktree, (old, old))

    def commit(self, worktree: Path, name: str = "extra.txt") -> str:
        (worktree / name).write_text(f"{name}\n", encoding="utf-8", newline="\n")
        git(worktree, "add", name)
        git(worktree, "commit", "-q", "-m", f"add {name}")
        return git(worktree, "rev-parse", "HEAD")

    def merge_to_origin_main(self, worktree: Path, *, update_local: bool) -> None:
        git(worktree, "push", "-q", "origin", "HEAD:main")
        git(self.repo, "fetch", "-q", "origin")
        if update_local:
            git(self.repo, "merge", "-q", "--ff-only", "origin/main")

    def run(
        self,
        *args: str,
        provider: PrProvider | None = None,
        cwd: Path | None = None,
    ) -> int:
        return wh.main(
            list(args),
            pr_heads_provider=provider or (lambda: []),
            cwd=cwd or self.repo,
        )

    def exists(self, worktree: Path) -> bool:
        return worktree.is_dir() and str(worktree.name) in git(
            self.repo, "worktree", "list", "--porcelain"
        )


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Env:
    config = tmp_path / "gitconfig"
    config.write_text(
        "[user]\n\tname = Test\n\temail = test@example.invalid\n"
        "[init]\n\tdefaultBranch = main\n"
        "[core]\n\tautocrlf = false\n"
        "[commit]\n\tgpgsign = false\n",
        encoding="utf-8",
        newline="\n",
    )
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(config))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    origin = tmp_path / "origin.git"
    repo = tmp_path / "repo"
    git(tmp_path, "init", "-q", "--bare", str(origin))
    git(tmp_path, "init", "-q", str(repo))
    git(repo, "remote", "add", "origin", str(origin))
    (repo / "README.md").write_text("hello\n", encoding="utf-8", newline="\n")
    (repo / ".gitignore").write_text(
        "__pycache__/\n.pytest_cache/\nartifacts/\n*.log\n",
        encoding="utf-8",
        newline="\n",
    )
    git(repo, "add", ".")
    git(repo, "commit", "-q", "-m", "init")
    git(repo, "push", "-q", "-u", "origin", "main")
    exclude = repo / ".git" / "info" / "exclude"
    exclude.write_text(".claude/\n", encoding="utf-8", newline="\n")
    return Env(repo=repo, origin=origin, root=tmp_path)


def plan(env: Env, capsys: pytest.CaptureFixture[str], *args: str) -> dict[str, object]:
    capsys.readouterr()
    code = env.run("--json", *args)
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert isinstance(payload, dict)
    return payload


def reasons(payload: dict[str, object]) -> dict[str, str]:
    worktrees = payload["worktrees"]
    assert isinstance(worktrees, list)
    return {Path(item["path"]).name: item["reason"] for item in worktrees}


def test_clean_merged_idle_worktree_is_removed(
    env: Env, capsys: pytest.CaptureFixture[str]
) -> None:
    wt = env.add("done")
    env.age(wt)
    (env.repo / "wip.txt").write_text("x\n", encoding="utf-8", newline="\n")
    git(env.repo, "add", "wip.txt")
    git(env.repo, "stash", "push", "-q", "-m", "keep-me")
    stash_before = git(env.repo, "stash", "list")

    assert env.run("--apply") == 0

    assert not wt.exists()
    assert not env.exists(wt)
    assert git(env.repo, "stash", "list") == stash_before
    assert "feat/done" in git(env.repo, "branch", "--list", "feat/done")
    out = capsys.readouterr().out
    assert "- removed: 1" in out and "- failed: 0" in out


def test_default_mode_is_read_only(
    env: Env, capsys: pytest.CaptureFixture[str]
) -> None:
    wt = env.add("done")
    env.age(wt)
    payload = plan(env, capsys)
    assert reasons(payload)["done"] == "removable"
    assert payload["mode"] == "dry-run"
    assert wt.is_dir()


def test_main_worktree_is_listed_and_kept(
    env: Env, capsys: pytest.CaptureFixture[str]
) -> None:
    payload = plan(env, capsys)
    assert reasons(payload)["repo"] == "main-worktree"


def test_dirty_tracked_file_is_kept(
    env: Env, capsys: pytest.CaptureFixture[str]
) -> None:
    wt = env.add("dirty")
    (wt / "README.md").write_text("changed\n", encoding="utf-8", newline="\n")
    env.age(wt)
    assert reasons(plan(env, capsys))["dirty"] == "dirty-tracked"
    assert env.run("--apply") == 0
    assert wt.is_dir()


def test_untracked_file_is_kept_but_ignored_file_is_not_dirty(
    env: Env, capsys: pytest.CaptureFixture[str]
) -> None:
    untracked = env.add("untracked")
    (untracked / "scratch.txt").write_text("x\n", encoding="utf-8", newline="\n")
    ignored = env.add("ignored")
    (ignored / "__pycache__").mkdir()
    (ignored / "__pycache__" / "m.pyc").write_bytes(b"\0")
    env.age(untracked)
    env.age(ignored)
    found = reasons(plan(env, capsys))
    assert found["untracked"] == "untracked-files"
    assert found["ignored"] == "removable"


def test_unmerged_commit_is_kept(env: Env, capsys: pytest.CaptureFixture[str]) -> None:
    wt = env.add("ahead")
    env.commit(wt)
    env.age(wt)
    assert reasons(plan(env, capsys))["ahead"] == "unmerged-commits"
    assert env.run("--apply") == 0
    assert wt.is_dir()


def test_open_pr_head_is_kept(env: Env, capsys: pytest.CaptureFixture[str]) -> None:
    wt = env.add("pr")
    sha = env.commit(wt)
    env.merge_to_origin_main(wt, update_local=True)
    env.age(wt)
    pulls = [OpenPullRequest(7, "feat/pr", sha)]
    code = env.run("--json", provider=lambda: pulls)
    assert code == 0
    found = reasons(json.loads(capsys.readouterr().out))
    assert found["pr"] == "open-pr-head"
    assert env.run("--apply", provider=lambda: pulls) == 0
    assert wt.is_dir()


def test_locked_worktree_is_kept(env: Env, capsys: pytest.CaptureFixture[str]) -> None:
    wt = env.add("locked")
    env.age(wt)
    git(env.repo, "worktree", "lock", "--reason", "busy", str(wt))
    assert reasons(plan(env, capsys))["locked"] == "locked"
    assert env.run("--apply") == 0
    assert wt.is_dir()


def test_recently_modified_worktree_is_kept(
    env: Env, capsys: pytest.CaptureFixture[str]
) -> None:
    wt = env.add("fresh")
    assert reasons(plan(env, capsys))["fresh"] == "recently-modified"
    assert reasons(plan(env, capsys, "--min-idle-hours", "0"))["fresh"] == "removable"
    assert env.run("--apply") == 0
    assert wt.is_dir()


def test_worktree_outside_scope_is_kept(
    env: Env, capsys: pytest.CaptureFixture[str]
) -> None:
    elsewhere = env.root / "elsewhere"
    elsewhere.mkdir()
    wt = env.add("outside", parent=elsewhere)
    env.age(wt)
    assert reasons(plan(env, capsys))["outside"] == "outside-scope"
    assert env.run("--apply") == 0
    assert wt.is_dir()


def test_current_worktree_is_kept(env: Env, capsys: pytest.CaptureFixture[str]) -> None:
    wt = env.add("self")
    other = env.add("other")
    env.age(wt)
    env.age(other)
    capsys.readouterr()
    assert env.run("--json", cwd=wt) == 0
    found = reasons(json.loads(capsys.readouterr().out))
    assert found["self"] == "current-worktree"
    assert found["other"] == "removable"


def test_gh_failure_removes_nothing(
    env: Env, capsys: pytest.CaptureFixture[str]
) -> None:
    wt = env.add("done")
    env.age(wt)

    def broken() -> Sequence[OpenPullRequest]:
        raise wh.WorktreeHygieneError("gh pr list failed: no auth")

    assert env.run("--apply", provider=broken) == 2
    assert wt.is_dir()
    assert "nothing was removed" in capsys.readouterr().err


def test_gh_json_is_validated_fail_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fake(
        stdout: str, code: int = 0
    ) -> Callable[..., subprocess.CompletedProcess[str]]:
        def run(args: Sequence[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
            assert list(args[:4]) == ["gh", "pr", "list", "--state"]
            assert "--limit" in args and "200" in args
            return subprocess.CompletedProcess(list(args), code, stdout, "boom")

        return run

    good = [{"number": 3, "headRefName": "a", "headRefOid": "A" * 40}]
    monkeypatch.setattr(wh, "_run", fake(json.dumps(good)))
    assert wh.gh_open_pull_requests(tmp_path) == [OpenPullRequest(3, "a", "a" * 40)]
    for bad in ("not json", "{}", json.dumps([{"number": 1}])):
        monkeypatch.setattr(wh, "_run", fake(bad))
        with pytest.raises(wh.WorktreeHygieneError):
            wh.gh_open_pull_requests(tmp_path)
    monkeypatch.setattr(wh, "_run", fake("[]", 1))
    with pytest.raises(wh.WorktreeHygieneError):
        wh.gh_open_pull_requests(tmp_path)
    full = [
        {"number": i, "headRefName": "b", "headRefOid": "b" * 40} for i in range(200)
    ]
    monkeypatch.setattr(wh, "_run", fake(json.dumps(full)))
    with pytest.raises(wh.WorktreeHygieneError, match="truncated"):
        wh.gh_open_pull_requests(tmp_path)


def test_worktree_dirtied_between_plan_and_apply_is_kept(env: Env) -> None:
    wt = env.add("race")
    other = env.add("calm")
    env.age(wt)
    env.age(other)
    origin_main = wh.resolve_origin_main(env.repo)
    decisions = wh.plan_worktrees(
        env.repo, env.repo, open_pulls=[], origin_main_sha=origin_main
    )
    assert {d.worktree.path.name: d.action for d in decisions}["race"] == wh.REMOVABLE

    (wt / "README.md").write_text("late edit\n", encoding="utf-8", newline="\n")
    results = wh.apply_removals(env.repo, decisions)

    by_name = {item.path.name: item for item in results}
    assert by_name["race"].outcome == "kept"
    assert by_name["race"].reason == "recheck-dirty"
    assert wt.is_dir()
    assert by_name["calm"].outcome == "removed"
    assert not other.exists()


def test_worktree_locked_or_moved_between_plan_and_apply_is_kept(env: Env) -> None:
    locked = env.add("late-lock")
    moved = env.add("late-commit")
    env.age(locked)
    env.age(moved)
    decisions = wh.plan_worktrees(
        env.repo,
        env.repo,
        open_pulls=[],
        origin_main_sha=wh.resolve_origin_main(env.repo),
    )
    git(env.repo, "worktree", "lock", str(locked))
    env.commit(moved)
    results = {item.path.name: item for item in wh.apply_removals(env.repo, decisions)}
    assert results["late-lock"].reason == "recheck-locked"
    assert results["late-commit"].reason == "recheck-head-changed"
    assert locked.is_dir() and moved.is_dir()


def test_delete_local_branches_uses_safe_d(
    env: Env, capsys: pytest.CaptureFixture[str]
) -> None:
    merged = env.add("merged")
    env.commit(merged)
    env.merge_to_origin_main(merged, update_local=True)
    env.age(merged)
    refused = env.add("diverged")
    env.commit(refused, "other.txt")
    env.merge_to_origin_main(refused, update_local=False)
    env.age(refused)
    git(env.repo, "push", "-q", "origin", "feat/merged", "feat/diverged")

    code = env.run("--apply", "--delete-local-branches")

    out = capsys.readouterr().out
    assert code == 0
    assert not merged.exists() and not refused.exists()
    branches = git(env.repo, "branch", "--list", "feat/*")
    assert "feat/merged" not in branches  # fully merged into HEAD: deleted by -d
    assert "feat/diverged" in branches  # -d refused; never escalated to -D
    assert "refused by git branch -d" in out
    remote = git(env.repo, "ls-remote", "--heads", "origin")
    assert "refs/heads/feat/merged" in remote and "refs/heads/feat/diverged" in remote


def test_branches_are_kept_without_flag(env: Env) -> None:
    wt = env.add("plain")
    env.age(wt)
    assert env.run("--apply") == 0
    assert not wt.exists()
    assert "feat/plain" in git(env.repo, "branch", "--list", "feat/plain")


def test_remove_failure_is_reported_and_exit_nonzero(
    env: Env, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    wt = env.add("stuck")
    env.age(wt)
    real = wh._git

    def failing(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
        if args[:2] == ("worktree", "remove"):
            assert "--force" not in args and "-f" not in args
            return subprocess.CompletedProcess(list(args), 1, "", "in use")
        return real(cwd, *args)

    monkeypatch.setattr(wh, "_git", failing)
    assert env.run("--apply") == 1
    assert wt.is_dir()
    assert "- failed: 1" in capsys.readouterr().out


def test_fetch_updates_origin_main(
    env: Env, capsys: pytest.CaptureFixture[str]
) -> None:
    other = env.root / "other-clone"
    git(env.root, "clone", "-q", str(env.origin), str(other))
    (other / "new.txt").write_text("n\n", encoding="utf-8", newline="\n")
    git(other, "add", "new.txt")
    git(other, "commit", "-q", "-m", "advance")
    git(other, "push", "-q", "origin", "main")
    new_sha = git(other, "rev-parse", "HEAD")

    stale = plan(env, capsys)
    assert stale["origin_main"] != new_sha
    fresh = plan(env, capsys, "--fetch")
    assert fresh["origin_main"] == new_sha


def test_missing_origin_main_fails_closed(
    env: Env, capsys: pytest.CaptureFixture[str]
) -> None:
    wt = env.add("done")
    env.age(wt)
    git(env.repo, "update-ref", "-d", "refs/remotes/origin/main")
    assert env.run("--apply") == 2
    assert wt.is_dir()


def test_delete_branches_requires_apply(env: Env) -> None:
    with pytest.raises(SystemExit):
        wh.parse_args(["--delete-local-branches"])


def test_parse_worktree_list_flags() -> None:
    text = (
        "worktree /r\nHEAD " + "a" * 40 + "\nbranch refs/heads/main\n\n"
        "worktree /r/.claude/worktrees/x\nHEAD " + "b" * 40 + "\ndetached\n"
        "locked busy\n\n"
        "worktree /r/.claude/worktrees/y\nHEAD "
        + "c"
        * 40
        + "\nbranch refs/heads/f/y\n"
        "prunable gitdir file points to non-existent location\n"
    )
    main, locked, prunable = wh.parse_worktree_list(text)
    assert main.is_main and main.branch == "main"
    assert locked.locked and locked.branch is None and not locked.is_main
    assert prunable.prunable and prunable.branch == "f/y"


def _nested_child(env: Env, parent: Path) -> Path:
    """A worktree nested in ``parent``; the shared exclude hides it from status."""
    child = env.add("child", parent=parent / ".claude" / "worktrees")
    (child / "precious.txt").write_text("uncommitted\n", encoding="utf-8", newline="\n")
    return child


def test_plain_git_remove_would_destroy_nested_worktree(env: Env) -> None:
    """Documents the hazard: git's own clean check cannot see the nested child."""
    parent = env.add("parent")
    child = _nested_child(env, parent)
    env.age(parent)
    assert git(parent, "status", "--porcelain", "--untracked-files=all") == ""
    git(env.repo, "worktree", "remove", str(parent))
    assert not (child / "precious.txt").exists()


def test_parent_with_nested_worktree_is_kept(
    env: Env, capsys: pytest.CaptureFixture[str]
) -> None:
    parent = env.add("parent")
    child = _nested_child(env, parent)
    env.age(parent)
    found = reasons(plan(env, capsys, "--allow-ignored-data"))
    assert found["parent"] == "contains-nested-worktree"
    assert env.run("--apply", "--allow-ignored-data") == 0
    assert (child / "precious.txt").read_text(encoding="utf-8") == "uncommitted\n"


def test_nested_git_directory_in_ignored_path_is_kept(
    env: Env, capsys: pytest.CaptureFixture[str]
) -> None:
    wt = env.add("holder")
    clone = wt / "artifacts" / "clone"
    clone.mkdir(parents=True)
    git(clone, "init", "-q")
    env.age(wt)
    found = reasons(plan(env, capsys, "--allow-ignored-data"))
    assert found["holder"] == "contains-nested-worktree"
    assert env.run("--apply", "--allow-ignored-data") == 0
    assert (clone / ".git").is_dir()


def test_nested_worktree_added_between_plan_and_apply_is_kept(env: Env) -> None:
    parent = env.add("parent")
    env.age(parent)
    decisions = wh.plan_worktrees(
        env.repo,
        env.repo,
        open_pulls=[],
        origin_main_sha=wh.resolve_origin_main(env.repo),
    )
    assert {d.worktree.path.name: d.action for d in decisions}["parent"] == wh.REMOVABLE
    child = _nested_child(env, parent)
    (result,) = wh.apply_removals(env.repo, decisions)
    assert result.reason == "recheck-nested-worktree"
    assert (child / "precious.txt").exists()


def test_ignored_data_is_kept_with_example_paths(
    env: Env, capsys: pytest.CaptureFixture[str]
) -> None:
    wt = env.add("data")
    (wt / "artifacts").mkdir()
    (wt / "artifacts" / "model.bin").write_bytes(b"x")
    (wt / "run.log").write_text("log\n", encoding="utf-8", newline="\n")
    (wt / "__pycache__").mkdir()
    (wt / "__pycache__" / "m.pyc").write_bytes(b"\0")
    (wt / ".pytest_cache").mkdir()
    (wt / ".pytest_cache" / "README.md").write_text("c\n", encoding="utf-8")
    env.age(wt)
    payload = plan(env, capsys)
    assert reasons(payload)["data"] == "ignored-data"
    worktrees = payload["worktrees"]
    assert isinstance(worktrees, list)
    detail = next(i["detail"] for i in worktrees if i["path"].endswith("data"))
    assert "artifacts/" in detail and "run.log" in detail
    assert "__pycache__" not in detail and ".pytest_cache" not in detail
    assert env.run("--apply") == 0
    assert (wt / "artifacts" / "model.bin").exists()


def test_allow_ignored_data_flag_overrides(
    env: Env, capsys: pytest.CaptureFixture[str]
) -> None:
    wt = env.add("data")
    (wt / "artifacts").mkdir()
    (wt / "artifacts" / "model.bin").write_bytes(b"x")
    env.age(wt)
    assert reasons(plan(env, capsys))["data"] == "ignored-data"
    assert reasons(plan(env, capsys, "--allow-ignored-data"))["data"] == "removable"
    assert env.run("--apply", "--allow-ignored-data") == 0
    assert not wt.exists()


def test_ignored_data_created_between_plan_and_apply_is_kept(env: Env) -> None:
    wt = env.add("late")
    env.age(wt)
    decisions = wh.plan_worktrees(
        env.repo,
        env.repo,
        open_pulls=[],
        origin_main_sha=wh.resolve_origin_main(env.repo),
    )
    (wt / "run.log").write_text("late\n", encoding="utf-8", newline="\n")
    (result,) = wh.apply_removals(env.repo, decisions)
    assert result.reason == "recheck-ignored-data"
    assert wt.is_dir()


def test_idle_check_counts_ignored_files(
    env: Env, capsys: pytest.CaptureFixture[str]
) -> None:
    cache = env.add("cache")
    (cache / "__pycache__").mkdir()
    (cache / "__pycache__" / "m.pyc").write_bytes(b"\0")
    env.age(cache)
    assert reasons(plan(env, capsys))["cache"] == "removable"
    now = time.time()
    os.utime(cache / "__pycache__" / "m.pyc", (now, now))
    assert reasons(plan(env, capsys))["cache"] == "recently-modified"
    data = env.add("data")
    (data / "run.log").write_text("log\n", encoding="utf-8", newline="\n")
    env.age(data)
    os.utime(data / "run.log", (now, now))
    found = reasons(plan(env, capsys, "--allow-ignored-data"))
    assert found["data"] == "recently-modified"


def test_in_use_worktree_is_kept_when_rename_probe_fails(
    env: Env, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    wt = env.add("busy")
    env.age(wt)
    real_rename = os.rename

    def deny(src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> None:
        if Path(src) == wt:
            raise PermissionError("handle open")
        real_rename(src, dst)

    monkeypatch.setattr(wh.os, "rename", deny)
    assert env.run("--apply") == 0
    assert wt.is_dir()
    assert "[in-use]" in capsys.readouterr().out


@pytest.mark.skipif(os.name != "nt", reason="open handles block renames on Windows")
def test_open_handle_blocks_removal_on_windows(
    env: Env, capsys: pytest.CaptureFixture[str]
) -> None:
    wt = env.add("handle")
    env.age(wt)
    with open(wt / "README.md", encoding="utf-8"):
        assert env.run("--apply") == 0
    assert wt.is_dir()
    assert "[in-use]" in capsys.readouterr().out


def test_rename_probe_restores_directory(env: Env) -> None:
    wt = env.add("probe")
    assert wh._probe_in_use(wt) is None
    assert wt.is_dir() and not list(wt.parent.glob("*hygiene-probe*"))


@pytest.mark.parametrize(
    "marker, is_dir",
    [
        ("rebase-merge", True),
        ("rebase-apply", True),
        ("sequencer", True),
        ("MERGE_HEAD", False),
        ("CHERRY_PICK_HEAD", False),
        ("REVERT_HEAD", False),
        ("BISECT_LOG", False),
    ],
)
def test_operation_in_progress_is_kept(
    env: Env,
    capsys: pytest.CaptureFixture[str],
    marker: str,
    is_dir: bool,
) -> None:
    wt = env.add("busy")
    env.age(wt)
    gitdir = Path(git(wt, "rev-parse", "--absolute-git-dir"))
    if is_dir:
        (gitdir / marker).mkdir()
    else:
        (gitdir / marker).write_text("x\n", encoding="utf-8", newline="\n")
    payload = plan(env, capsys)
    assert reasons(payload)["busy"] == "operation-in-progress"
    assert env.run("--apply") == 0
    assert wt.is_dir()


def test_operation_started_between_plan_and_apply_is_kept(env: Env) -> None:
    wt = env.add("busy")
    env.age(wt)
    decisions = wh.plan_worktrees(
        env.repo,
        env.repo,
        open_pulls=[],
        origin_main_sha=wh.resolve_origin_main(env.repo),
    )
    gitdir = Path(git(wt, "rev-parse", "--absolute-git-dir"))
    (gitdir / "MERGE_HEAD").write_text("x\n", encoding="utf-8", newline="\n")
    (result,) = wh.apply_removals(env.repo, decisions)
    assert result.reason == "recheck-operation-in-progress"
    assert wt.is_dir()


def test_branch_name_of_open_pr_is_kept_even_with_different_head(
    env: Env, capsys: pytest.CaptureFixture[str]
) -> None:
    wt = env.add("renamed")
    env.age(wt)
    pulls = [OpenPullRequest(9, "feat/renamed", "f" * 40)]
    assert env.run("--json", provider=lambda: pulls) == 0
    found = reasons(json.loads(capsys.readouterr().out))
    assert found["renamed"] == "open-pr-branch"
    assert env.run("--apply", provider=lambda: pulls) == 0
    assert wt.is_dir()


def test_error_after_removal_reports_actual_count_and_exits_one(
    env: Env,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    wt = env.add("gone")
    env.age(wt)

    def boom(repo_root: Path) -> str | None:
        raise OSError("disk went away")

    monkeypatch.setattr(wh, "prune_worktrees", boom)
    assert env.run("--apply") == 1
    captured = capsys.readouterr()
    assert not wt.exists()
    assert "- removed: 1" in captured.out
    assert "nothing was removed" not in captured.err
    assert "1 worktree(s) removed" in captured.err


def test_oserror_for_one_worktree_does_not_stop_the_rest(
    env: Env,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    first = env.add("aaa")
    second = env.add("bbb")
    env.age(first)
    env.age(second)
    real = wh._probe_in_use

    def flaky(path: Path) -> str | None:
        if path.name == "aaa":
            raise OSError("antivirus")
        return real(path)

    monkeypatch.setattr(wh, "_probe_in_use", flaky)
    assert env.run("--apply") == 1
    assert first.is_dir() and not second.exists()
    out = capsys.readouterr().out
    assert "- removed: 1" in out and "- failed: 1" in out


def test_scope_prefix_lookalike_is_outside(tmp_path: Path) -> None:
    scope = tmp_path / ".claude" / "worktrees"
    (scope / "a").mkdir(parents=True)
    (tmp_path / ".claude" / "worktrees-evil" / "a").mkdir(parents=True)
    assert wh._is_within(scope / "a", scope)
    assert not wh._is_within(tmp_path / ".claude" / "worktrees-evil" / "a", scope)
    assert not wh._is_within(scope, scope)


@pytest.mark.skipif(os.name != "nt", reason="Windows paths are case-insensitive")
def test_windows_case_and_slash_variants_are_equivalent(
    env: Env, capsys: pytest.CaptureFixture[str]
) -> None:
    wt = env.add("MixedCase")
    other = env.add("other")
    env.age(wt)
    env.age(other)
    variant = str(wt).replace("\\", "/").swapcase()
    assert wh._is_within(variant, str(env.scope).swapcase())
    assert wh._norm(variant) == wh._norm(wt)
    capsys.readouterr()
    assert env.run("--json", cwd=Path(variant)) == 0
    found = reasons(json.loads(capsys.readouterr().out))
    assert found["MixedCase"] == "current-worktree"
    assert found["other"] == "removable"


def make_dir_link(link: Path, target: Path) -> None:
    """Directory junction on Windows (no privilege needed), symlink elsewhere."""
    link.parent.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        import _winapi

        _winapi.CreateJunction(str(target), str(link))  # type: ignore[attr-defined,unused-ignore]
    else:
        os.symlink(target, link, target_is_directory=True)


def _victim(env: Env) -> Path:
    victim = env.root / "victim"
    victim.mkdir()
    (victim / "precious.txt").write_text("keep\n", encoding="utf-8", newline="\n")
    return victim


@pytest.mark.parametrize("flags", [(), ("--allow-ignored-data",)])
def test_cache_named_link_to_outside_is_kept(
    env: Env, capsys: pytest.CaptureFixture[str], flags: tuple[str, ...]
) -> None:
    """Git for Windows follows junctions on remove and deletes their targets.

    A Windows junction named ``__pycache__`` is treated as an ignored directory,
    so the link check decides.  On POSIX git sees a symlink as a file, so the
    untracked-file check may keep the worktree first; either way it is KEEP.
    """
    victim = _victim(env)
    wt = env.add("linked")
    make_dir_link(wt / "pkg" / "__pycache__", victim)
    env.age(wt)
    allowed = (
        {"contains-link"} if os.name == "nt" else {"contains-link", "untracked-files"}
    )
    assert reasons(plan(env, capsys, *flags))["linked"] in allowed
    assert env.run("--apply", *flags) == 0
    assert wt.is_dir()
    assert (victim / "precious.txt").read_text(encoding="utf-8") == "keep\n"


def test_link_in_ignored_directory_is_decided_by_link_check(
    env: Env, capsys: pytest.CaptureFixture[str]
) -> None:
    """The link sits under an ignored directory, so only the link check can keep it."""
    victim = _victim(env)
    wt = env.add("ignored-link")
    make_dir_link(wt / "artifacts" / "pkg" / "__pycache__", victim)
    env.age(wt)
    assert reasons(plan(env, capsys))["ignored-link"] == "ignored-data"
    flag = "--allow-ignored-data"
    assert reasons(plan(env, capsys, flag))["ignored-link"] == "contains-link"
    assert env.run("--apply", flag) == 0
    assert wt.is_dir()
    assert (victim / "precious.txt").read_text(encoding="utf-8") == "keep\n"


def test_link_created_between_plan_and_apply_is_kept(env: Env) -> None:
    victim = _victim(env)
    wt = env.add("late-link")
    env.age(wt)
    decisions = wh.plan_worktrees(
        env.repo,
        env.repo,
        open_pulls=[],
        origin_main_sha=wh.resolve_origin_main(env.repo),
        allow_ignored_data=True,
    )
    make_dir_link(wt / "artifacts" / "__pycache__", victim)
    (result,) = wh.apply_removals(env.repo, decisions, allow_ignored_data=True)
    assert result.reason == "recheck-contains-link"
    assert (victim / "precious.txt").exists() and wt.is_dir()


def test_link_helper_detects_links_on_every_platform(env: Env) -> None:
    victim = _victim(env)
    link = env.root / "a-link"
    make_dir_link(link, victim)
    assert wh._is_link(str(link), os.lstat(link))
    assert not wh._is_link(str(victim), os.lstat(victim))
    scan_root = env.root / "scan"
    (scan_root / "sub").mkdir(parents=True)
    make_dir_link(scan_root / "sub" / "cache", victim)
    found = wh._scan_tree(scan_root)
    assert found.link == scan_root / "sub" / "cache"
    assert found.nested is None


def test_link_helper_honours_reparse_point_attribute(tmp_path: Path) -> None:
    plain = tmp_path / "plain"
    plain.mkdir()
    real = os.lstat(plain)
    fake = SimpleNamespace(
        st_mode=real.st_mode,
        st_file_attributes=wh._REPARSE_POINT,
    )
    assert wh._is_link(str(plain), cast(os.stat_result, fake))
    clear = SimpleNamespace(st_mode=real.st_mode, st_file_attributes=0)
    assert not wh._is_link(str(plain), cast(os.stat_result, clear))


def test_leftover_probe_directory_refuses_apply(
    env: Env, capsys: pytest.CaptureFixture[str]
) -> None:
    wt = env.add("done")
    env.age(wt)
    leftover = env.scope / "stuck.hygiene-probe-123"
    leftover.mkdir()
    capsys.readouterr()
    assert env.run("--apply") == 2
    captured = capsys.readouterr()
    assert str(leftover) in captured.err and "nothing was removed" in captured.err
    assert wt.is_dir()
    assert env.run() == 0  # the read-only plan is still allowed


def test_symlink_to_file_is_kept(env: Env, capsys: pytest.CaptureFixture[str]) -> None:
    wt = env.add("symfile")
    target = env.root / "outside.txt"
    target.write_text("x\n", encoding="utf-8", newline="\n")
    (wt / "__pycache__").mkdir()
    try:
        os.symlink(target, wt / "__pycache__" / "lnk")
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation is not permitted here")
    env.age(wt)
    assert reasons(plan(env, capsys, "--allow-ignored-data"))["symfile"] == (
        "contains-link"
    )
    assert env.run("--apply", "--allow-ignored-data") == 0
    assert target.exists() and wt.is_dir()


def test_probe_restore_failure_stops_run_and_skips_prune(
    env: Env, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    first = env.add("aaa")
    second = env.add("bbb")
    env.age(first)
    env.age(second)
    real_rename = os.rename
    calls: list[str] = []

    def rename(src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> None:
        if Path(src) == first:
            real_rename(src, dst)
        elif Path(dst) == first:
            raise PermissionError("cannot restore")
        else:
            real_rename(src, dst)

    def prune(repo_root: Path) -> str | None:
        calls.append("prune")
        return None

    monkeypatch.setattr(wh.os, "rename", rename)
    monkeypatch.setattr(wh, "prune_worktrees", prune)
    assert env.run("--apply") == 1
    captured = capsys.readouterr()
    probe = first.with_name(f"{first.name}.hygiene-probe-{os.getpid()}")
    assert probe.is_dir() and not first.exists()
    assert str(probe) in captured.out and "RESTORE MANUALLY" in captured.out
    assert calls == []
    assert second.is_dir()
    assert "- removed: 0" in captured.out
    real_rename(probe, first)


def test_ignored_data_short_circuits_before_tree_walk(
    env: Env, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    wt = env.add("data")
    (wt / "run.log").write_text("log\n", encoding="utf-8", newline="\n")
    env.age(wt)

    def forbidden(path: Path) -> object:
        raise AssertionError("full-tree walk must not run for ignored-data")

    monkeypatch.setattr(wh, "_scan_tree", forbidden)
    assert reasons(plan(env, capsys))["data"] == "ignored-data"
