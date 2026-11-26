"""Source-bound checkout identity shared by retained paper operations."""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Final

AUTHORIZATION_PATTERN: Final = re.compile(
    r"^https://github\.com/shuntatsu/trade_rl/issues/810#issuecomment-[1-9][0-9]*$"
)
TRADE_RL_ROOT_COMMIT: Final = "8e7a42170d7504d7b41c5555e99754f9c56d8adb"


@dataclass(frozen=True)
class CodeIdentity:
    """The checked-out code: HEAD commit and whether the working tree is unmodified.

    ``tracked_tree_clean`` is false when any tracked file is modified OR any
    non-ignored untracked file exists.
    """

    git_head: str | None
    tracked_tree_clean: bool


_UNAVAILABLE: Final = CodeIdentity(None, False)


def current_code_identity(
    repo_root: Path | None = None, *, root_commit: str = TRADE_RL_ROOT_COMMIT
) -> CodeIdentity:
    """Identify the trade_rl checkout this code runs from (no network).

    Unavailable (``git_head`` None) unless: ``git rev-parse --show-toplevel`` resolves a
    repository; with the default ``repo_root`` this very module file is the tracked
    ``trade_rl/evaluation/paper/code_identity.py`` of that repository (so a non-editable
    install, or a copy inside some other enclosing repository, is refused); and the
    repository history contains ``root_commit``, the trade_rl root commit. Untracked
    non-ignored files make the tree dirty.
    """

    module_file = Path(__file__).resolve()
    start = repo_root or module_file.parents[3]

    def run(*args: str) -> str | None:
        try:
            completed = subprocess.run(
                ["git", *args], cwd=start, capture_output=True, text=True, check=True
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        return completed.stdout

    toplevel_text = run("rev-parse", "--show-toplevel")
    if toplevel_text is None or not toplevel_text.strip():
        return _UNAVAILABLE
    toplevel = Path(toplevel_text.strip()).resolve()
    if repo_root is None:
        expected = toplevel / "trade_rl" / "evaluation" / "paper" / "code_identity.py"
        if module_file != expected.resolve():
            return _UNAVAILABLE
    roots = run("rev-list", "--max-parents=0", "HEAD")
    if roots is None or root_commit not in roots.split():
        return _UNAVAILABLE
    head = run("rev-parse", "HEAD")
    status = run("status", "--porcelain", "--untracked-files=all")
    if head is None or status is None or not head.strip():
        return _UNAVAILABLE
    return CodeIdentity(head.strip(), status.strip() == "")
