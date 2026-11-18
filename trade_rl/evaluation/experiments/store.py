"""Append-only filesystem mechanics for controlled experiment Studies."""

from __future__ import annotations

import errno
import json
import os
import shutil
import stat
import sys
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO, cast

from trade_rl.artifacts.atomic_write import (
    _temporary_name_suffix,
    atomic_rename_directory,
)
from trade_rl.artifacts.canonical import canonical_json_bytes
from trade_rl.artifacts.verified_file import (
    _open_directory_descriptor_nofollow,
    _trusted_process_directory_fd,
    open_regular_binary,
)
from trade_rl.evaluation.experiments.errors import (
    ArtifactIntegrityError,
    InvalidExperimentStateError,
    PublishedArtifactCleanupError,
)

_LOCK_NAME = ".mutation.lock"


def _require_no_study_parent_traversal(path: Path) -> None:
    """Reject lexical dot-dot before it can erase a symlink ancestor."""

    if ".." in path.parts:
        raise ArtifactIntegrityError("Study path must not contain parent traversal")


def _lock_file(handle: BinaryIO) -> None:
    if os.name == "nt":
        import msvcrt

        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        locking = getattr(msvcrt, "locking")
        lock_mode = getattr(msvcrt, "LK_LOCK")
        locking(handle.fileno(), lock_mode, 1)
        return

    import fcntl

    flock = getattr(fcntl, "flock")
    lock_ex = getattr(fcntl, "LOCK_EX")
    flock(handle.fileno(), lock_ex)


def _unlock_file(handle: BinaryIO) -> None:
    if os.name == "nt":
        import msvcrt

        handle.seek(0)
        locking = getattr(msvcrt, "locking")
        unlock_mode = getattr(msvcrt, "LK_UNLCK")
        locking(handle.fileno(), unlock_mode, 1)
        return

    import fcntl

    flock = getattr(fcntl, "flock")
    lock_un = getattr(fcntl, "LOCK_UN")
    flock(handle.fileno(), lock_un)


def _open_mutation_lock(root: Path) -> BinaryIO:
    """Open the Study lock without following symlinks on supported POSIX hosts."""

    _require_no_study_parent_traversal(Path(root))
    source = Path(os.path.abspath(root))
    flags = os.O_RDWR | os.O_CREAT | os.O_APPEND | getattr(os, "O_BINARY", 0)
    if os.name == "posix":
        # A hostile FIFO must not block the open before fstat rejects its type.
        flags |= getattr(os, "O_NONBLOCK", 0)
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    directory_flag = getattr(os, "O_DIRECTORY", 0)
    descriptor = -1
    try:
        if (
            os.name == "posix"
            and nofollow
            and directory_flag
            and os.open in os.supports_dir_fd
        ):
            parent_fd = _open_directory_descriptor_nofollow(source)
            try:
                descriptor = os.open(
                    _LOCK_NAME, flags | nofollow, 0o666, dir_fd=parent_fd
                )
            finally:
                os.close(parent_fd)
        else:
            # Non-POSIX fallback checks static links; it cannot close the
            # concurrent replacement race without descriptor-relative opens.
            for directory in (source, *source.parents):
                if directory.is_symlink() or directory.is_junction():
                    raise ArtifactIntegrityError(
                        "Study mutation lock parent must not be a symlink or junction"
                    )
            lock_path = source / _LOCK_NAME
            if lock_path.is_symlink() or lock_path.is_junction() or lock_path.is_dir():
                raise ArtifactIntegrityError(
                    "Study mutation lock must be a regular file"
                )
            descriptor = os.open(lock_path, flags | nofollow, 0o666)
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ArtifactIntegrityError("Study mutation lock must be a regular file")
        handle = os.fdopen(descriptor, "a+b")
        descriptor = -1
        return handle
    except OSError as error:
        if error.errno in (errno.ELOOP, errno.ENOTDIR, errno.EISDIR):
            raise ArtifactIntegrityError(
                "Study mutation lock must be a regular file "
                "in non-symlinked directories"
            ) from error
        raise
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _assert_pinned_publication_parent(path: Path, descriptor: int) -> None:
    """Reject a changed parent pathname even while its old inode is open."""

    # Reopen through the shared no-follow chain: a trusted in-scope
    # /proc/self/fd/N capability is valid even though its name is a symlink.
    # Ordinary, untrusted symlink ancestors still fail closed.
    try:
        current_fd = _open_directory_descriptor_nofollow(path)
        try:
            current = os.fstat(current_fd)
        finally:
            os.close(current_fd)
    except OSError as error:
        raise ArtifactIntegrityError("Study publication parent changed") from error
    pinned = os.fstat(descriptor)
    current_identity = (current.st_dev, current.st_ino)
    pinned_identity = (pinned.st_dev, pinned.st_ino)
    if not stat.S_ISDIR(current.st_mode) or current_identity != pinned_identity:
        raise ArtifactIntegrityError("Study publication parent changed")


@contextmanager
def _pinned_publication_parent(
    parent: Path,
) -> Iterator[tuple[Path, Callable[[], None]]]:
    """Anchor publication paths to descriptor-verified POSIX directories."""

    _require_no_study_parent_traversal(parent)
    if os.name != "posix":
        # Portable static checks cannot prevent a concurrent replacement.
        def check_portable() -> None:
            if parent.is_symlink() or parent.is_junction() or not parent.is_dir():
                raise ArtifactIntegrityError(
                    "Study publication parent must be a regular directory"
                )

        check_portable()
        yield parent, check_portable
        return

    nofollow = getattr(os, "O_NOFOLLOW", 0)
    directory = getattr(os, "O_DIRECTORY", 0)
    if not (nofollow and directory and os.open in os.supports_dir_fd):
        raise ArtifactIntegrityError(
            "Study publication requires descriptor-relative no-follow support"
        )

    source = Path(os.path.abspath(parent))
    descriptor = -1
    try:
        # Only setup failures may be classified as a bad publication parent.
        # An OSError raised by the caller's publication body must propagate.
        try:
            descriptor = _open_directory_descriptor_nofollow(source)
            pinned: Path | None = None
            for prefix in ("/proc/self/fd", "/dev/fd"):
                candidate = Path(prefix) / str(descriptor)
                if candidate.is_dir():
                    pinned = candidate
                    break
            if pinned is None:
                raise ArtifactIntegrityError(
                    "Study publication requires a usable directory-fd path"
                )

            def check_parent() -> None:
                _assert_pinned_publication_parent(source, descriptor)

            check_parent()
        except OSError as error:
            if error.errno in (errno.ELOOP, errno.ENOTDIR, errno.EISDIR):
                raise ArtifactIntegrityError(
                    "Study publication parent must be a non-symlinked directory"
                ) from error
            raise

        with _trusted_process_directory_fd(descriptor):
            yield pinned, check_parent
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _mkdir_study_root_nofollow(root: Path) -> None:
    """Create a Study root without following any symlinked ancestor.

    Open the nearest existing directory with the shared no-follow capability
    helper, then create each missing child relative to its held descriptor.
    """

    source = Path(os.path.abspath(root))
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    directory = getattr(os, "O_DIRECTORY", 0)
    if (
        os.name != "posix"
        or not nofollow
        or not directory
        or os.open not in os.supports_dir_fd
        or os.mkdir not in os.supports_dir_fd
    ):
        # Non-POSIX fallback checks static symlinks and junctions. It cannot
        # claim atomic protection against concurrent replacements.
        for ancestor in (source, *source.parents):
            if ancestor.is_symlink() or ancestor.is_junction():
                raise ArtifactIntegrityError("Study root has a linked ancestor")
        source.mkdir(parents=True, exist_ok=True)
        return

    missing: list[str] = []
    ancestor = source
    while not ancestor.exists() and not ancestor.is_symlink():
        missing.append(ancestor.name)
        ancestor = ancestor.parent

    descriptor = -1
    try:
        descriptor = _open_directory_descriptor_nofollow(ancestor)
        flags = getattr(os, "O_PATH", os.O_RDONLY) | directory | nofollow
        for name in reversed(missing):
            try:
                os.mkdir(name, dir_fd=descriptor)
            except FileExistsError:
                # A competing creator is acceptable only if it made a real
                # directory; O_NOFOLLOW on the next open enforces that.
                pass
            child = os.open(name, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        # The root may itself be an internally registered process-FD alias
        # for a builder stage. A lexical lstat would see the alias symlink;
        # the shared opener authenticates the exact FD and its directory inode.
        current_fd = _open_directory_descriptor_nofollow(source)
        try:
            current = os.fstat(current_fd)
        finally:
            os.close(current_fd)
        opened = os.fstat(descriptor)
        current_identity = (current.st_dev, current.st_ino)
        opened_identity = (opened.st_dev, opened.st_ino)
        if not stat.S_ISDIR(current.st_mode) or current_identity != opened_identity:
            raise ArtifactIntegrityError("Study root changed during creation")
    except OSError as error:
        if error.errno in (errno.ELOOP, errno.ENOTDIR, errno.EISDIR, errno.ENOENT):
            raise ArtifactIntegrityError(
                "Study root must have non-symlinked directory ancestors"
            ) from error
        raise
    finally:
        if descriptor >= 0:
            os.close(descriptor)


class StudyStore:
    """Own append-only publication, safe paths, and the Study mutation lock."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        _require_no_study_parent_traversal(self.root)
        _mkdir_study_root_nofollow(self.root)
        self._thread_lock = threading.RLock()
        self._lock_depth = 0

    def _relative_path(self, relative: str | Path) -> Path:
        value = Path(relative)
        if value.is_absolute():
            raise ValueError("Study path must be relative")
        if not value.parts or value == Path("."):
            raise ValueError("Study path must be a non-empty relative path")
        if ".." in value.parts:
            raise ValueError("Study path must not contain parent traversal")
        if value.parts[0] == _LOCK_NAME:
            raise ValueError("Study path is reserved for the mutation lock")
        return value

    def _checked_target(self, relative: str | Path) -> Path:
        _require_no_study_parent_traversal(self.root)
        value = self._relative_path(relative)
        current = self.root
        for part in value.parts[:-1]:
            current = current / part
            if current.is_symlink():
                raise ArtifactIntegrityError("Study path parent must not be a symlink")
            if current.exists() and not current.is_dir():
                raise ArtifactIntegrityError(
                    "Study path parent must be a regular directory"
                )
        return self.root / value

    def _mkdir_checked_parents(self, target: Path) -> None:
        """Create missing parents without redirecting mkdir after a path swap."""

        relative_parent = target.parent.relative_to(self.root)
        if not relative_parent.parts:
            return

        nofollow = getattr(os, "O_NOFOLLOW", 0)
        directory_flag = getattr(os, "O_DIRECTORY", 0)
        if (
            os.name == "posix"
            and nofollow
            and directory_flag
            and os.open in os.supports_dir_fd
            and os.mkdir in os.supports_dir_fd
        ):
            descriptor = -1
            flags = getattr(os, "O_PATH", os.O_RDONLY) | directory_flag | nofollow
            try:
                # Hold the original Study directory and every intermediate
                # parent across mkdir. A lexical path swap cannot move these
                # writes outside the opened directory hierarchy.
                descriptor = _open_directory_descriptor_nofollow(self.root)
                for name in relative_parent.parts:
                    try:
                        os.mkdir(name, dir_fd=descriptor)
                    except FileExistsError:
                        # Accept only an existing real directory: the next
                        # O_NOFOLLOW open rejects symlinks and ordinary files.
                        pass
                    child = os.open(name, flags, dir_fd=descriptor)
                    os.close(descriptor)
                    descriptor = child
                _assert_pinned_publication_parent(target.parent, descriptor)
            except OSError as error:
                if error.errno in (
                    errno.ELOOP,
                    errno.ENOTDIR,
                    errno.EISDIR,
                    errno.ENOENT,
                ):
                    raise ArtifactIntegrityError(
                        "Study path parent must be a non-symlinked directory"
                    ) from error
                raise
            finally:
                if descriptor >= 0:
                    os.close(descriptor)
            return

        # Portable fallback is a static check only. It does not claim
        # atomic resistance to a concurrent parent directory replacement.
        current = self.root
        for part in relative_parent.parts:
            current = current / part
            if current.is_symlink() or current.is_junction():
                raise ArtifactIntegrityError("Study path parent must not be a symlink")
            if current.exists():
                if not current.is_dir():
                    raise ArtifactIntegrityError(
                        "Study path parent must be a regular directory"
                    )
                continue
            current.mkdir()

    @contextmanager
    def mutation_lock(self) -> Iterator[None]:
        """Serialize Study mutation across processes and support same-store nesting."""

        with self._thread_lock:
            _require_no_study_parent_traversal(self.root)
            if self._lock_depth > 0:
                self._lock_depth += 1
                try:
                    yield
                finally:
                    self._lock_depth -= 1
                return

            if self.root.is_symlink() or not self.root.is_dir():
                # A nested Study can use a thread-scoped /proc/self/fd/N
                # directory alias. Authenticate its registered inode rather
                # than rejecting the symlink spelling. Ordinary symlinks and
                # unregistered aliases still fail closed; the lock opener
                # independently checks the directory chain again.
                if os.name != "posix":
                    raise ArtifactIntegrityError(
                        "Study root must be a regular directory"
                    )
                try:
                    root_fd = _open_directory_descriptor_nofollow(self.root)
                except OSError as error:
                    raise ArtifactIntegrityError(
                        "Study root must be a regular directory"
                    ) from error
                else:
                    os.close(root_fd)
            lock_path = self.root / _LOCK_NAME
            if lock_path.is_symlink():
                raise ArtifactIntegrityError(
                    "Study mutation lock must not be a symlink"
                )
            with _open_mutation_lock(self.root) as handle:
                _lock_file(handle)
                self._lock_depth = 1
                try:
                    yield
                finally:
                    self._lock_depth = 0
                    _unlock_file(handle)

    def publish_json_once(self, relative: str | Path, value: object) -> Path:
        """Atomically publish canonical JSON without replacing existing evidence."""

        target = self._checked_target(relative)
        self._mkdir_checked_parents(target)
        with _pinned_publication_parent(target.parent) as (parent, check_parent):
            pinned_target = parent / target.name
            check_parent()
            if pinned_target.exists() or pinned_target.is_symlink():
                raise InvalidExperimentStateError(
                    f"Study artifact already exists: {relative}"
                )

            staging = parent / f".{target.name}.staging-{_temporary_name_suffix()}"
            try:
                check_parent()
                with staging.open("xb") as handle:
                    handle.write(canonical_json_bytes(value))
                    handle.flush()
                    os.fsync(handle.fileno())
                    check_parent()
                    if pinned_target.exists() or pinned_target.is_symlink():
                        raise InvalidExperimentStateError(
                            f"Study artifact already exists: {relative}"
                        )
                    # Keep the verified source inode open through publication.
                    # On Linux, a directory-fd argument forces os.link to use
                    # linkat(AT_SYMLINK_FOLLOW), resolving this process's file
                    # descriptor rather than the replaceable staging basename.
                    try:
                        if os.name == "posix" and sys.platform == "linux":
                            stage_alias = Path(f"/proc/self/fd/{handle.fileno()}")
                            if not stage_alias.is_file():
                                raise ArtifactIntegrityError(
                                    "Study staging file descriptor unavailable"
                                )
                            parent_descriptor = _open_directory_descriptor_nofollow(
                                parent
                            )
                            try:
                                os.link(
                                    stage_alias,
                                    pinned_target,
                                    src_dir_fd=parent_descriptor,
                                    follow_symlinks=True,
                                )
                            finally:
                                os.close(parent_descriptor)
                        else:
                            # Non-Linux hard links preserve exclusivity only;
                            # they do not guarantee source-inode immutability
                            # under a non-cooperating staging-name replacement.
                            os.link(staging, pinned_target)
                    except FileExistsError as error:
                        raise InvalidExperimentStateError(
                            f"Study artifact already exists: {relative}"
                        ) from error
            except BaseException:
                try:
                    staging.unlink(missing_ok=True)
                except OSError:
                    # Preserve the primary publication failure.
                    pass
                raise

            # Publication is committed; never report a cleanup error as a
            # safe-to-retry ordinary failed write.
            try:
                staging.unlink(missing_ok=True)
            except OSError as error:
                raise PublishedArtifactCleanupError(
                    f"Study artifact was published, but staging cleanup failed: "
                    f"{relative}; do not retry publication"
                ) from error
            try:
                check_parent()
            except ArtifactIntegrityError as error:
                raise PublishedArtifactCleanupError(
                    f"Study artifact may have been published into a detached "
                    f"parent: {relative}; do not retry publication"
                ) from error
        return target

    def read_json(self, relative: str | Path) -> dict[str, object]:
        """Read one immutable JSON object through the safe Study path boundary."""

        target = self._checked_target(relative)
        if target.is_symlink() or not target.is_file():
            raise ArtifactIntegrityError("Study artifact must be a regular file")
        # Open the verified file descriptor before reading: a later symlink
        # substitution must not redirect this read to a different artifact.
        try:
            with open_regular_binary(target, field="Study artifact") as handle:
                encoded = handle.read()
        except ValueError as error:
            raise ArtifactIntegrityError(
                "Study artifact must be a regular file"
            ) from error
        except OSError as error:
            raise ArtifactIntegrityError("Study artifact JSON is malformed") from error
        try:
            raw = cast(object, json.loads(encoded.decode("utf-8")))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ArtifactIntegrityError("Study artifact JSON is malformed") from error
        if not isinstance(raw, dict) or any(not isinstance(key, str) for key in raw):
            raise ArtifactIntegrityError("Study artifact must contain a JSON object")
        return cast(dict[str, object], raw)

    def publish_directory_once(
        self,
        relative: str | Path,
        builder: Callable[[Path], None],
    ) -> Path:
        """Build private staging, then publish once without parent redirection."""

        target = self._checked_target(relative)
        self._mkdir_checked_parents(target)
        with _pinned_publication_parent(target.parent) as (parent, check_parent):
            with _pinned_publication_parent(self.root) as (pinned_root, check_root):
                pinned_target = parent / target.name
                check_parent()
                check_root()
                if pinned_target.exists() or pinned_target.is_symlink():
                    raise InvalidExperimentStateError(
                        f"Study artifact already exists: {relative}"
                    )

                # Stage in the Study root to isolate builder writes from
                # substitution of a nested final-target parent directory.
                name = f".{target.name}.staging-{_temporary_name_suffix()}"
                pinned_staging = pinned_root / name
                check_parent()
                check_root()
                pinned_staging.mkdir()
                staging_descriptor = -1
                try:
                    check_root()
                    if os.name == "posix":
                        # The pinned root alone cannot protect the staging
                        # leaf name. Hold its inode across the builder callback.
                        staging_descriptor = _open_directory_descriptor_nofollow(
                            pinned_staging
                        )
                        stage_alias: Path | None = None
                        for prefix in ("/proc/self/fd", "/dev/fd"):
                            candidate = Path(prefix) / str(staging_descriptor)
                            if candidate.is_dir():
                                stage_alias = candidate
                                break
                        if stage_alias is None:
                            raise ArtifactIntegrityError(
                                "Study staging requires a usable directory-fd path"
                            )
                        with _trusted_process_directory_fd(staging_descriptor):
                            builder(stage_alias)
                        # Refuse to publish a replaced staging directory name.
                        _assert_pinned_publication_parent(
                            pinned_staging, staging_descriptor
                        )
                    else:
                        # Portable static checks cannot prevent leaf swaps.
                        builder(pinned_staging)
                    check_root()
                    check_parent()
                    if pinned_target.exists() or pinned_target.is_symlink():
                        raise InvalidExperimentStateError(
                            f"Study artifact already exists: {relative}"
                        )
                    # Exclusive rename uses the fd-anchored POSIX paths.
                    atomic_rename_directory(pinned_staging, pinned_target)
                finally:
                    if staging_descriptor >= 0:
                        os.close(staging_descriptor)
                    if pinned_staging.exists() or pinned_staging.is_symlink():
                        if pinned_staging.is_dir() and not pinned_staging.is_symlink():
                            shutil.rmtree(pinned_staging)
                        else:
                            pinned_staging.unlink(missing_ok=True)
                try:
                    check_parent()
                    check_root()
                except ArtifactIntegrityError as error:
                    raise PublishedArtifactCleanupError(
                        f"Study directory may have been published into a "
                        f"detached parent: {relative}; do not retry publication"
                    ) from error
        return target


__all__ = ["StudyStore"]
