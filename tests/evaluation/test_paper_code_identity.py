"""Retained paper checkout identity and foreign-source refusal tests."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from trade_rl.evaluation.paper import code_identity as seal
from trade_rl.evaluation.paper.code_identity import CodeIdentity

GIT_ENV = {
    "GIT_AUTHOR_NAME": "t",
    "GIT_AUTHOR_EMAIL": "t@example.com",
    "GIT_COMMITTER_NAME": "t",
    "GIT_COMMITTER_EMAIL": "t@example.com",
}


def git(repo: Path, *args: str) -> str:
    import os

    completed = subprocess.run(
        ["git", *args],
        cwd=repo,
        env={**os.environ, **GIT_ENV},
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout.strip()


def make_repo(path: Path) -> tuple[Path, str, str]:
    path.mkdir()
    git(path, "init", "-q")
    (path / "tracked.txt").write_text("one\n", encoding="utf-8")
    git(path, "add", "tracked.txt")
    git(path, "commit", "-q", "-m", "root")
    root = git(path, "rev-parse", "HEAD")
    (path / "tracked.txt").write_text("two\n", encoding="utf-8")
    git(path, "commit", "-q", "-am", "second")
    return path, root, git(path, "rev-parse", "HEAD")


def test_code_identity_is_clean_only_without_modified_or_untracked_files(
    tmp_path: Path,
) -> None:
    repo, root, head = make_repo(tmp_path / "repo")
    assert seal.current_code_identity(repo, root_commit=root) == CodeIdentity(
        head, True
    )
    (repo / "scratch.txt").write_text("untracked\n", encoding="utf-8")
    assert seal.current_code_identity(repo, root_commit=root) == CodeIdentity(
        head, False
    )
    (repo / "scratch.txt").unlink()
    (repo / "tracked.txt").write_text("edited\n", encoding="utf-8")
    assert seal.current_code_identity(repo, root_commit=root) == CodeIdentity(
        head, False
    )


def test_code_identity_ignores_ignored_files_and_requires_the_pinned_root(
    tmp_path: Path,
) -> None:
    repo, root, head = make_repo(tmp_path / "repo")
    (repo / ".gitignore").write_text("*.log\n", encoding="utf-8")
    git(repo, "add", ".gitignore")
    git(repo, "commit", "-q", "-m", "ignore")
    head = git(repo, "rev-parse", "HEAD")
    (repo / "build.log").write_text("ignored\n", encoding="utf-8")
    assert seal.current_code_identity(repo, root_commit=root) == CodeIdentity(
        head, True
    )
    # another repository (different root commit) is not the trade_rl repository
    assert seal.current_code_identity(repo, root_commit="f" * 40) == CodeIdentity(
        None, False
    )


def test_code_identity_is_unavailable_outside_a_repository(tmp_path: Path) -> None:
    assert seal.current_code_identity(tmp_path) == CodeIdentity(None, False)


def test_a_foreign_copy_of_the_module_is_refused_even_with_the_right_root_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Isolates the module-location check: the root commit matches, the path does not."""

    repo, root, head = make_repo(tmp_path / "repo")
    foreign = repo / "vendor" / "copy" / "pkg" / "seal.py"
    foreign.parent.mkdir(parents=True)
    foreign.write_text("# a copy of the module inside a trade_rl-like repo\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "vendor copy")
    head = git(repo, "rev-parse", "HEAD")
    assert seal.current_code_identity(repo, root_commit=root) == CodeIdentity(
        head, True
    )

    monkeypatch.setattr(seal, "__file__", str(foreign))
    assert foreign.resolve().parents[3] == repo.resolve()
    # root commit is right and the tree is clean, but this file is not the repository's
    # tracked trade_rl/evaluation/paper/code_identity.py
    assert seal.current_code_identity(None, root_commit=root) == CodeIdentity(
        None, False
    )

    # control: the same module file at the tracked location is accepted
    tracked = repo / "trade_rl" / "evaluation" / "paper" / "code_identity.py"
    tracked.parent.mkdir(parents=True)
    tracked.write_text("# tracked copy\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "tracked location")
    head = git(repo, "rev-parse", "HEAD")
    monkeypatch.setattr(seal, "__file__", str(tracked))
    assert seal.current_code_identity(None, root_commit=root) == CodeIdentity(
        head, True
    )


def test_the_real_checkout_identity_matches_git_when_history_is_available() -> None:
    root = Path(__file__).resolve().parents[2]

    def real_git(*args: str) -> str | None:
        try:
            done = subprocess.run(
                ["git", *args], cwd=root, capture_output=True, text=True, check=True
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        return done.stdout

    roots = real_git("rev-list", "--max-parents=0", "HEAD")
    if roots is None or seal.TRADE_RL_ROOT_COMMIT not in roots.split():
        pytest.skip(
            "not a full git checkout of the trade_rl repository (shallow or no git)"
        )
    identity = seal.current_code_identity()
    head = real_git("rev-parse", "HEAD")
    status = real_git("status", "--porcelain", "--untracked-files=all")
    assert head is not None and status is not None
    assert identity.git_head == head.strip()
    # reports a dirty tree as dirty (and a clean one as clean); never silently clean
    assert identity.tracked_tree_clean == (status.strip() == "")
