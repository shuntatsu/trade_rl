"""Repository-local path containment for source-derived Agent tooling."""

from __future__ import annotations

from pathlib import Path


def checked_repo_path(repository: Path, path: Path) -> Path:
    """Return a repository-contained path while rejecting every symlink component."""

    root = repository.resolve()
    candidate = path if path.is_absolute() else root / path
    try:
        relative = candidate.relative_to(root)
    except ValueError as error:
        raise ValueError(f"path escapes repository: {path}") from error

    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"repository inspection rejects symlink: {path}")

    resolved = current.resolve(strict=True)
    if not resolved.is_relative_to(root):
        raise ValueError(f"path escapes repository: {path}")
    return current


def checked_repo_file(repository: Path, path: Path) -> Path:
    """Require one regular, non-symlink file within the repository."""

    candidate = checked_repo_path(repository, path)
    if not candidate.is_file():
        raise ValueError(f"repository inspection requires a regular file: {path}")
    return candidate


def checked_repo_directory(repository: Path, path: Path) -> Path:
    """Require one non-symlink directory within the repository."""

    candidate = checked_repo_path(repository, path)
    if not candidate.is_dir():
        raise ValueError(f"repository inspection requires a directory: {path}")
    return candidate


__all__ = ["checked_repo_directory", "checked_repo_file", "checked_repo_path"]
