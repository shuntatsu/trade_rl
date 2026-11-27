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
    repository; with the default ``repo_root`` this very module file is a regular
    blob tracked in HEAD at ``trade_rl/evaluation/paper/code_identity.py`` of that
    repository (so a non-editable install or a foreign copy is refused); and the
    observed HEAD history contains ``root_commit``, the trade_rl root commit.
    Untracked non-ignored files make the tree dirty. This is a local Git observation,
    not an atomic checkout snapshot or an authentication mechanism.
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
    head = run("rev-parse", "HEAD")
    if head is None or not head.strip():
        return _UNAVAILABLE
    head = head.strip()
    if repo_root is None:
        relative = "trade_rl/evaluation/paper/code_identity.py"
        expected = toplevel / relative
        if module_file != expected:
            return _UNAVAILABLE
        # A clean status does not prove an ignored module exists in HEAD.
        # Require the exact path to be a regular tracked blob, not a symlink.
        entry = run("ls-tree", "-z", head, "--", relative)
        if entry is None:
            return _UNAVAILABLE
        metadata, separator, name = entry.partition("\t")
        fields = metadata.split()
        if (
            separator != "\t"
            or name != relative + "\0"
            or len(fields) != 3
            or fields[0] not in {"100644", "100755"}
            or fields[1] != "blob"
        ):
            return _UNAVAILABLE
    roots = run("rev-list", "--max-parents=0", head)
    if roots is None or root_commit not in roots.split():
        return _UNAVAILABLE
    status = run("status", "--porcelain", "--untracked-files=all")
    if status is None or run("rev-parse", "HEAD") != head + "\n":
        return _UNAVAILABLE
    return CodeIdentity(head, status.strip() == "")
