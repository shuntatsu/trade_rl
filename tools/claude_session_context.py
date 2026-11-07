"""Print bounded, read-only local Git context for a Claude Code SessionStart hook.

This is only session orientation. It cannot verify GitHub PRs, current remote main,
review receipts, or CI and must never be treated as a write/merge authorization.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def _git(root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=3,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def session_context(root: Path) -> str:
    """Return local branch, HEAD and dirty state without exposing filenames."""
    prefix = _git(root, "rev-parse", "--show-prefix")
    if prefix is None or prefix != "":
        return (
            "[trade_rl / Claude Code] Repository root unavailable. "
            "Verify the working directory and read AGENTS.md before changing files. "
            "GitHub state and authorization have NOT been checked."
        )

    branch = _git(root, "symbolic-ref", "--quiet", "--short", "HEAD")
    head = _git(root, "rev-parse", "--verify", "HEAD")
    status = _git(root, "status", "--porcelain=v1", "--untracked-files=normal")
    cached_main = _git(root, "rev-parse", "--verify", "refs/remotes/origin/main")

    if status is None:
        working_tree = "unknown (git status failed)"
    else:
        working_tree = "clean" if status == "" else "changes present (preserve them)"

    safe_branch = branch if branch is not None else "DETACHED"
    main_warning = (
        "On main: inspect only; use a dedicated branch/worktree for any change."
        if safe_branch == "main"
        else "Never modify main directly or override another worktree. "
        "Check ownership before changes."
    )
    return "\n".join(
        (
            "[trade_rl / Claude Code | local snapshot only]",
            f"branch: {safe_branch}",
            f"HEAD: {head if head is not None else 'unavailable'}",
            f"working tree: {working_tree}",
            f"cached origin/main: {cached_main if cached_main is not None else 'unavailable'}",
            main_warning,
            "Cached refs may be stale. Check current GitHub main, open PRs, "
            "research authority and exact-head CI/reviews separately before writes or merge.",
            "Follow AGENTS.md and docs/AGENTS.md; never execute economic runs, "
            "read sealed data, or authorize live trading from this hook.",
        )
    )


def main() -> None:
    root = Path(os.environ.get("CLAUDE_PROJECT_DIR") or Path(__file__).resolve().parents[1])
    print(session_context(root))


if __name__ == "__main__":
    main()
