"""Read-only SessionStart hook behavior and Claude Code wiring contracts."""

from __future__ import annotations

import json
import subprocess
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


def test_session_context_prints_local_snapshot_without_authorizing(tmp_path: Path) -> None:
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
    assert "never execute economic runs" in result


def test_session_context_exposes_dirty_flag_but_not_filenames(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    (repo / ".env.local").write_text("SECRET\n", encoding="utf-8")
    _git(repo, "update-ref", "refs/remotes/origin/main", _git(repo, "rev-parse", "HEAD"))
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
    hooks = settings["hooks"]["SessionStart"]
    assert len(hooks) == 1
    assert hooks[0]["matcher"] == "startup|resume|compact|clear"
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
