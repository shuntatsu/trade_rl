"""Worktree hygiene is exercised against real temporary git repositories."""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

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
        old = time.time() - hours * 3600.0
        for name in git(worktree, "ls-files").splitlines():
            os.utime(worktree / name, (old, old))

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
    (repo / ".gitignore").write_text("__pycache__/\n", encoding="utf-8", newline="\n")
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
