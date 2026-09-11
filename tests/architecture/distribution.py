"""Exact tracked Python-source closure across checkout, sdist and wheel.

Archives are inspected, never extracted or imported. This is a package-source
contract, not a substitute for installed-package or optional-capability tests.
"""

from __future__ import annotations

import argparse
import hashlib
import stat
import subprocess
import tarfile
import zipfile
from pathlib import Path, PurePosixPath


def _git_sources(repository: Path) -> dict[str, str]:
    tracked = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", "-z", "HEAD", "--", "trade_rl"],
        cwd=repository,
        check=True,
        capture_output=True,
    ).stdout.decode("utf-8")
    names = {name for name in tracked.split("\0") if name.endswith(".py")}
    present = {
        path.relative_to(repository).as_posix()
        for path in (repository / "trade_rl").rglob("*.py")
    }
    if not names or names != present:
        raise ValueError("worktree source roster differs from Git (missing/untracked)")
    diff = subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", "trade_rl"], cwd=repository
    )
    if diff.returncode == 1:
        raise ValueError("worktree source bytes differ from Git HEAD")
    diff.check_returncode()
    result: dict[str, str] = {}
    for name in sorted(names):
        path = repository / name
        if not path.is_file() or any(
            part.is_symlink() for part in (path, *path.parents)
        ):
            raise ValueError("package source must be a regular file without symlinks")
        result[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def _safe_name(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if (
        not name
        or not path.parts
        or "\\" in name
        or path.is_absolute()
        or ".." in path.parts
        or path.as_posix() != name.rstrip("/")
    ):
        raise ValueError(f"unsafe archive path: {name}")
    return path


def _archive_sources(archive: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    seen: set[str] = set()

    def register(name: str) -> PurePosixPath:
        path = _safe_name(name)
        key = path.as_posix()
        if key in seen:
            raise ValueError(f"duplicate archive member: {name}")
        seen.add(key)
        return path

    if archive.suffix == ".whl":
        with zipfile.ZipFile(archive) as wheel:
            for info in wheel.infolist():
                path = register(info.filename)
                kind = stat.S_IFMT(info.external_attr >> 16)
                if kind == stat.S_IFLNK:
                    raise ValueError("wheel must not contain symlink members")
                if info.is_dir() or path.suffix != ".py":
                    continue
                if kind not in (0, stat.S_IFREG):
                    raise ValueError("wheel Python source must be a regular file")
                if path.parts[0] != "trade_rl":
                    raise ValueError("unexpected Python source outside package path")
                result[path.as_posix()] = hashlib.sha256(wheel.read(info)).hexdigest()
    elif archive.name.endswith(".tar.gz"):
        roots: set[str] = set()
        with tarfile.open(archive, "r:gz") as sdist:
            for member in sdist:
                path = register(member.name)
                roots.add(path.parts[0])
                if member.issym() or member.islnk():
                    raise ValueError(
                        "sdist must not contain symlink or hardlink members"
                    )
                if len(path.parts) < 2 or path.parts[1] != "trade_rl":
                    continue
                if path.suffix != ".py":
                    continue
                if not member.isfile():
                    raise ValueError("sdist Python source must be a regular file")
                handle = sdist.extractfile(member)
                if handle is None:
                    raise ValueError("sdist Python source is unreadable")
                with handle:
                    name = PurePosixPath(*path.parts[1:]).as_posix()
                    result[name] = hashlib.sha256(handle.read()).hexdigest()
        if len(roots) > 1:
            raise ValueError("sdist must have one archive root path")
    else:
        raise ValueError("expected .whl or .tar.gz distribution")
    return result


def verify_distribution(repository: Path, archive: Path) -> None:
    expected = _git_sources(repository)
    actual = _archive_sources(archive)
    if expected != actual:
        missing = sorted(expected.keys() - actual.keys())
        extra = sorted(actual.keys() - expected.keys())
        changed = sorted(
            name
            for name in expected.keys() & actual.keys()
            if expected[name] != actual[name]
        )
        raise ValueError(
            f"distribution source mismatch: missing={missing}, extra={extra}, changed={changed}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("archives", nargs="+", type=Path)
    args = parser.parse_args()
    for archive in args.archives:
        verify_distribution(args.repository, archive)
        print(f"tracked Python source closure: {archive.name}: OK")


if __name__ == "__main__":
    main()
