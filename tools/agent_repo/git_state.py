"""Read exact local Git facts for repository-agent preflight."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from tools.agent_repo.path_safety import (
    checked_repo_directory,
    checked_repo_file,
    checked_repo_path,
)


@dataclass(frozen=True, slots=True)
class GitState:
    """Source-derived Git/worktree facts for one repository snapshot."""

    branch: str | None
    head: str
    base_ref: str | None
    merge_base: str | None
    dirty_paths: tuple[str, ...]
    untracked_paths: tuple[str, ...]
    changed_paths: tuple[str, ...]
    active_docs: tuple[str, ...]
    workflow_paths: tuple[str, ...]


def _git(repository: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _lines(value: str) -> tuple[str, ...]:
    return tuple(sorted(line for line in value.splitlines() if line))


def _dirty_paths(repository: Path) -> tuple[str, ...]:
    status = _git(repository, "status", "--porcelain=v1", "--untracked-files=no")
    paths: set[str] = set()
    for line in status.splitlines():
        if len(line) < 4:
            continue
        path = line[3:]
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        paths.add(path)
    return tuple(sorted(paths))


def _active_docs(repository: Path) -> tuple[str, ...]:
    result: list[str] = []
    for directory in (repository / "docs" / "specs", repository / "docs" / "plans"):
        if not directory.exists() and not directory.is_symlink():
            continue
        directory = checked_repo_directory(repository, directory)
        for path in sorted(directory.rglob("*.md")):
            path = checked_repo_file(repository, path)
            if "Status: Active" in path.read_text(encoding="utf-8"):
                result.append(path.relative_to(repository).as_posix())
    return tuple(sorted(result))


def _workflow_paths(repository: Path) -> tuple[str, ...]:
    directory = repository / ".github" / "workflows"
    if not directory.exists() and not directory.is_symlink():
        return ()
    directory = checked_repo_directory(repository, directory)
    result: list[str] = []
    for path in sorted(directory.iterdir()):
        path = checked_repo_path(repository, path)
        if path.is_file():
            result.append(path.relative_to(repository).as_posix())
    return tuple(result)


def read_git_state(repository: Path, *, base_ref: str | None = None) -> GitState:
    """Return current Git/worktree facts without mutating the repository."""

    root = repository.resolve()
    head = _git(root, "rev-parse", "HEAD").strip()
    try:
        branch_value = _git(root, "symbolic-ref", "--short", "-q", "HEAD").strip()
    except subprocess.CalledProcessError:
        branch: str | None = None
    else:
        branch = branch_value or None

    dirty_paths = _dirty_paths(root)
    untracked_paths = _lines(_git(root, "ls-files", "--others", "--exclude-standard"))

    merge_base: str | None = None
    base_changed: tuple[str, ...] = ()
    if base_ref is not None:
        merge_base = _git(root, "merge-base", "HEAD", base_ref).strip()
        base_changed = _lines(_git(root, "diff", "--name-only", f"{merge_base}...HEAD"))

    changed_paths = tuple(sorted(set(base_changed) | set(dirty_paths)))
    return GitState(
        branch=branch,
        head=head,
        base_ref=base_ref,
        merge_base=merge_base,
        dirty_paths=dirty_paths,
        untracked_paths=untracked_paths,
        changed_paths=changed_paths,
        active_docs=_active_docs(root),
        workflow_paths=_workflow_paths(root),
    )


__all__ = ["GitState", "read_git_state"]
