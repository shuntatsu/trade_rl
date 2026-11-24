"""Exercise the real Git exact-SHA deletion lease against an isolated bare repo."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tools.branch_hygiene import (
    BranchInfo,
    _push_branches_with_leases,
    _retirement_push_env,
)


def _git(*args: str, cwd: Path | None = None) -> str:
    command = ["git", *(["-C", str(cwd)] if cwd is not None else []), *args]
    return subprocess.run(
        command, text=True, capture_output=True, check=True
    ).stdout.strip()


@pytest.fixture
def isolated_git(tmp_path: Path) -> tuple[Path, Path, str]:
    remote = tmp_path / "isolated.git"
    checkout = tmp_path / "checkout"
    _git("init", "--bare", str(remote))
    _git("init", "-b", "main", str(checkout))
    _git("config", "user.name", "Fixture", cwd=checkout)
    _git("config", "user.email", "fixture@example.invalid", cwd=checkout)
    (checkout / "sample.txt").write_text("seed\n", encoding="utf-8")
    _git("add", "sample.txt", cwd=checkout)
    _git("commit", "-m", "initial", cwd=checkout)
    _git("remote", "add", "origin", str(remote), cwd=checkout)
    _git("push", "origin", "main", cwd=checkout)
    _git("push", "origin", "HEAD:refs/heads/feat/merged", cwd=checkout)
    return checkout, remote, _git("rev-parse", "HEAD", cwd=checkout)


def _remote_tip(remote: Path) -> str | None:
    data = _git("ls-remote", str(remote), "refs/heads/feat/merged")
    return data.split()[0] if data else None


def test_real_atomic_exact_sha_lease_deletes_only_verified_tip(
    isolated_git: tuple[Path, Path, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout, remote, sha = isolated_git
    monkeypatch.chdir(checkout)
    assert _remote_tip(remote) == sha
    _push_branches_with_leases(
        (BranchInfo("feat/merged", sha, False),),
        env=_retirement_push_env("fixture-token"),
    )
    assert _remote_tip(remote) is None
    assert _git("rev-parse", "refs/heads/main", cwd=remote) == sha


def test_real_stale_sha_lease_cannot_delete_a_changed_remote_branch(
    isolated_git: tuple[Path, Path, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout, remote, old_sha = isolated_git
    _git("checkout", "-b", "feat/merged", cwd=checkout)
    (checkout / "sample.txt").write_text("new work\n", encoding="utf-8")
    _git("commit", "-am", "advance", cwd=checkout)
    new_sha = _git("rev-parse", "HEAD", cwd=checkout)
    _git("push", "origin", "feat/merged", cwd=checkout)
    monkeypatch.chdir(checkout)
    assert new_sha != old_sha
    with pytest.raises(RuntimeError, match="conditional branch deletion failed"):
        _push_branches_with_leases(
            (BranchInfo("feat/merged", old_sha, False),),
            env=_retirement_push_env("fixture-token"),
        )
    assert _remote_tip(remote) == new_sha
