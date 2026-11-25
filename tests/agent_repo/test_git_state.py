from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from tools.agent_repo.git_state import read_git_state


def _git(repository: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _init_repository(root: Path) -> None:
    _git(root, "init", "-b", "main")
    _git(root, "config", "user.name", "Test User")
    _git(root, "config", "user.email", "test@example.com")
    (root / "trade_rl").mkdir()
    (root / "trade_rl" / "a.py").write_text("VALUE = 1\n", encoding="utf-8")
    (root / "docs" / "specs").mkdir(parents=True)
    (root / "docs" / "specs" / "active.md").write_text(
        "# Active\n\nStatus: Active\n", encoding="utf-8"
    )
    (root / ".github" / "workflows").mkdir(parents=True)
    (root / ".github" / "workflows" / "ci.yml").write_text(
        "name: CI\n", encoding="utf-8"
    )
    _git(root, "add", ".")
    _git(root, "commit", "-m", "baseline")


def test_read_git_state_reports_exact_worktree_and_base_facts(tmp_path: Path) -> None:
    _init_repository(tmp_path)
    _git(tmp_path, "switch", "-c", "feature")
    (tmp_path / "trade_rl" / "a.py").write_text("VALUE = 2\n", encoding="utf-8")
    (tmp_path / "scratch.txt").write_text("scratch\n", encoding="utf-8")

    state = read_git_state(tmp_path, base_ref="main")

    assert state.branch == "feature"
    assert len(state.head) == 40
    assert state.base_ref == "main"
    assert state.merge_base is not None
    assert state.dirty_paths == ("trade_rl/a.py",)
    assert state.untracked_paths == ("scratch.txt",)
    assert state.changed_paths == ("trade_rl/a.py",)
    assert state.active_docs == ("docs/specs/active.md",)
    assert state.workflow_paths == (".github/workflows/ci.yml",)


@pytest.mark.skipif(
    os.name == "nt",
    reason="Windows filenames cannot contain ASCII tab characters",
)
def test_read_git_state_preserves_exact_staged_path_with_tab(tmp_path: Path) -> None:
    _init_repository(tmp_path)
    odd_path = tmp_path / "trade_rl" / "odd\tname.py"
    odd_path.write_text("VALUE = 1\n", encoding="utf-8")
    _git(tmp_path, "add", "trade_rl/odd\tname.py")
    _git(tmp_path, "commit", "-m", "add odd path")

    odd_path.write_text("VALUE = 2\n", encoding="utf-8")
    _git(tmp_path, "add", "trade_rl/odd\tname.py")

    state = read_git_state(tmp_path)

    assert state.dirty_paths == ("trade_rl/odd\tname.py",)
    assert state.changed_paths == ("trade_rl/odd\tname.py",)


@pytest.mark.skipif(
    os.name == "nt",
    reason="Windows filenames cannot contain ASCII tab characters",
)
def test_read_git_state_preserves_exact_untracked_path_with_tab(tmp_path: Path) -> None:
    _init_repository(tmp_path)
    odd_path = tmp_path / "scratch\tname.txt"
    odd_path.write_text("scratch\n", encoding="utf-8")

    state = read_git_state(tmp_path)

    assert state.untracked_paths == ("scratch\tname.txt",)


@pytest.mark.skipif(
    os.name == "nt",
    reason="Windows filenames cannot contain ASCII tab characters",
)
def test_read_git_state_preserves_exact_base_changed_path_with_tab(
    tmp_path: Path,
) -> None:
    _init_repository(tmp_path)
    _git(tmp_path, "switch", "-c", "feature")
    odd_path = tmp_path / "trade_rl" / "branch\tname.py"
    odd_path.write_text("VALUE = 1\n", encoding="utf-8")
    _git(tmp_path, "add", "trade_rl/branch\tname.py")
    _git(tmp_path, "commit", "-m", "add branch path")

    state = read_git_state(tmp_path, base_ref="main")

    assert state.changed_paths == ("trade_rl/branch\tname.py",)


def test_read_git_state_reports_detached_head(tmp_path: Path) -> None:
    _init_repository(tmp_path)
    head = _git(tmp_path, "rev-parse", "HEAD")
    _git(tmp_path, "checkout", "--detach", head)

    state = read_git_state(tmp_path)

    assert state.branch is None
    assert state.head == head


def test_read_git_state_fails_closed_for_missing_explicit_base(tmp_path: Path) -> None:
    _init_repository(tmp_path)

    with pytest.raises(subprocess.CalledProcessError):
        read_git_state(tmp_path, base_ref="missing-base")


@pytest.mark.skipif(os.name == "nt", reason="Windows forbids CR/LF in filenames")
@pytest.mark.parametrize("separator", ["\r", "\r\n"])
def test_read_git_state_preserves_cr_in_all_path_sources(
    tmp_path: Path, separator: str
) -> None:
    _init_repository(tmp_path)
    _git(tmp_path, "switch", "-c", "feature")
    committed = f"trade_rl/committed{separator}name.py"
    staged = f"trade_rl/staged{separator}name.py"
    unstaged = f"trade_rl/unstaged{separator}name.py"
    untracked = f"scratch{separator}name.txt"
    for name in (committed, staged, unstaged):
        (tmp_path / name).write_text("VALUE = 1\n", encoding="utf-8")
    _git(tmp_path, "add", committed, staged, unstaged)
    _git(tmp_path, "commit", "-m", "add CR paths")
    for name in (staged, unstaged, untracked):
        (tmp_path / name).write_text("VALUE = 2\n", encoding="utf-8")
    _git(tmp_path, "add", staged)

    state = read_git_state(tmp_path, base_ref="main")

    assert state.dirty_paths == tuple(sorted((staged, unstaged)))
    assert state.untracked_paths == (untracked,)
    assert state.changed_paths == tuple(sorted((committed, staged, unstaged)))


@pytest.mark.skipif(os.name == "nt", reason="Windows forbids CR/LF in filenames")
def test_read_git_state_does_not_alias_cr_and_lf_paths(tmp_path: Path) -> None:
    _init_repository(tmp_path)
    paths = ("trade_rl/alias\rname.py", "trade_rl/alias\nname.py")
    for name in paths:
        (tmp_path / name).write_text("VALUE = 1\n", encoding="utf-8")
    _git(tmp_path, "add", *paths)

    state = read_git_state(tmp_path)

    assert state.dirty_paths == tuple(sorted(paths))
    assert state.changed_paths == tuple(sorted(paths))


@pytest.mark.parametrize("committed", [False, True])
@pytest.mark.parametrize("renames", ["true", "false", "copies"])
def test_read_git_state_reports_both_rename_paths(
    tmp_path: Path, committed: bool, renames: str
) -> None:
    _init_repository(tmp_path)
    _git(tmp_path, "config", "diff.renames", renames)
    _git(tmp_path, "switch", "-c", "feature")
    old = "trade_rl/a.py"
    new = "trade_rl/renamed.py"
    _git(tmp_path, "mv", old, new)
    if committed:
        _git(tmp_path, "commit", "-m", "rename")

    state = read_git_state(tmp_path, base_ref="main")

    assert state.dirty_paths == (() if committed else (old, new))
    assert state.changed_paths == (old, new)


@pytest.mark.parametrize("branch", ["feature", "topic/feature"])
def test_read_git_state_reports_literal_branch_when_tag_shadows_name(
    tmp_path: Path, branch: str
) -> None:
    _init_repository(tmp_path)
    _git(tmp_path, "switch", "-c", branch)
    _git(tmp_path, "tag", branch)
    _git(tmp_path, "config", "core.warnAmbiguousRefs", "true")

    state = read_git_state(tmp_path)

    assert state.branch == branch
    assert state.head == _git(tmp_path, "rev-parse", "HEAD")
