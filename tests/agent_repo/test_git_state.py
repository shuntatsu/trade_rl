from __future__ import annotations

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
