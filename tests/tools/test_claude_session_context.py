"""Read-only SessionStart hook behavior and Claude Code wiring contracts."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from tools.claude_session_context import session_context

ROOT = Path(__file__).resolve().parents[2]


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo with spaces"
    root.mkdir()
    _git(root, "init", "--quiet", "--initial-branch=main")
    (root / "tracked.txt").write_text("v1\n", encoding="utf-8")
    _git(root, "add", "tracked.txt")
    _git(
        root,
        "-c",
        "user.name=Test Agent",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "--quiet",
        "-m",
        "initial",
    )
    return root


def test_session_context_refuses_non_repository(tmp_path: Path) -> None:
    result = session_context(tmp_path)
    assert "Repository root unavailable" in result
    assert "NOT been checked" in result


def test_session_context_clean_root_warns_before_writing(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    head = _git(repo, "rev-parse", "HEAD")
    result = session_context(repo)
    assert "branch: main" in result
    assert "On main: inspect only" in result
    assert f"HEAD: {head}" in result
    assert "working tree: clean" in result
    assert "cached origin/main: unavailable" in result
    assert "Cached refs may be stale" in result
    assert "exact-head CI/reviews separately" in result
    assert "this hook cannot authorize economic runs" in result


def test_session_context_dirty_names_are_private(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    (repo / ".env.local").write_text("SECRET\n", encoding="utf-8")
    _git(
        repo,
        "update-ref",
        "refs/remotes/origin/main",
        _git(repo, "rev-parse", "HEAD"),
    )
    result = session_context(repo)
    assert "changes present (preserve them)" in result
    assert "cached origin/main: " + _git(repo, "rev-parse", "HEAD") in result
    assert ".env.local" not in result
    assert "SECRET" not in result


def test_session_context_handles_detached_head(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    _git(repo, "checkout", "--detach", "--quiet", "HEAD")
    assert "branch: DETACHED" in session_context(repo)


def test_claude_entry_hook_and_skill_are_single_authority() -> None:
    entry = (ROOT / "CLAUDE.md").read_text(encoding="utf-8")
    settings = json.loads((ROOT / ".claude/settings.json").read_text(encoding="utf-8"))
    skill = (ROOT / ".claude/skills/research-challenge/SKILL.md").read_text(
        encoding="utf-8"
    )

    assert entry.startswith("@AGENTS.md\n")
    assert "docs/AGENTS.md" in entry
    assert skill.startswith("---\nname: research-challenge\n")
    assert "not an independent review" in skill
    assert "disable-model-invocation: true" in skill
    hooks = settings["hooks"]["SessionStart"]
    assert len(hooks) == 1
    assert hooks[0]["matcher"] == "startup|resume|compact|clear|fork"
    assert hooks[0]["hooks"] == [
        {
            "type": "command",
            "command": "python",
            "args": ["${CLAUDE_PROJECT_DIR}/tools/claude_session_context.py"],
            "timeout": 15,
        }
    ]


def test_session_context_enforces_non_main_orientation(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    _git(repo, "checkout", "-b", "feature/safe-review", "--quiet")
    result = session_context(repo)
    assert "branch: feature/safe-review" in result
    assert "Never modify main directly" in result


def test_direct_exec_hook_on_worktree_root_with_spaces(tmp_path: Path) -> None:
    parent = _repo(tmp_path)
    (parent / "tools").mkdir()
    (parent / "tools/claude_session_context.py").write_bytes(
        (ROOT / "tools/claude_session_context.py").read_bytes()
    )
    _git(parent, "add", "tools/claude_session_context.py")
    _git(
        parent,
        "-c",
        "user.name=Test Agent",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "--quiet",
        "-m",
        "add hook",
    )
    worktree = tmp_path / "separate worktree with spaces"
    _git(parent, "worktree", "add", "--detach", str(worktree), "HEAD")
    settings = json.loads((ROOT / ".claude/settings.json").read_text(encoding="utf-8"))
    hook = settings["hooks"]["SessionStart"][0]["hooks"][0]
    script = hook["args"][0].replace("${CLAUDE_PROJECT_DIR}", str(worktree))
    env = {**os.environ, "CLAUDE_PROJECT_DIR": str(worktree)}
    # Exercise exactly the direct-exec argument shape Claude Code supplies,
    # substituting the expected Python interpreter for a platform-specific PATH.
    assert hook["command"] == "python"
    result = subprocess.run(
        [sys.executable, script],
        cwd=worktree,
        env=env,
        input=json.dumps({"hook_event_name": "SessionStart", "source": "fork"}),
        capture_output=True,
        text=True,
        check=True,
        timeout=hook["timeout"],
    )
    assert "branch: DETACHED" in result.stdout
    assert "working tree: clean" in result.stdout
    assert "local snapshot only" in result.stdout
    assert "cannot authorize economic runs" in result.stdout
    assert result.stderr == ""


def test_hook_keeps_git_index_unchanged(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    index = repo / ".git" / "index"
    before = (index.read_bytes(), index.stat().st_mtime_ns)
    assert "working tree: clean" in session_context(repo)
    after = (index.read_bytes(), index.stat().st_mtime_ns)
    assert after == before


def test_session_context_rejects_bare_repository(tmp_path: Path) -> None:
    bare = tmp_path / "bare"
    subprocess.run(
        ["git", "init", "--quiet", "--bare", str(bare)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Repository root unavailable" in session_context(bare)
