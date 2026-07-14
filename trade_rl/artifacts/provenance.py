"""Content-addressed runtime provenance captured for reproducible runs."""

from __future__ import annotations

import hashlib
import importlib.metadata
import platform
import re
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.common import require_sha256

_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(root: Path, *args: str) -> str | None:
    try:
        completed = subprocess.run(
            ("git", "-C", str(root), *args),
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


@dataclass(frozen=True, slots=True)
class RuntimeProvenance:
    digest: str
    git_commit: str
    git_dirty: bool
    lockfile_digest: str | None
    python_version: str
    platform_name: str
    hardware_name: str
    package_versions: tuple[tuple[str, str], ...]
    deterministic_seed_config_digest: str
    schema_version: str = "runtime_provenance_v1"

    def __post_init__(self) -> None:
        require_sha256(self.digest, field="provenance.digest")
        if not _GIT_SHA_RE.fullmatch(self.git_commit):
            raise ValueError("git_commit must be a 40-character lowercase SHA")
        if not isinstance(self.git_dirty, bool):
            raise ValueError("git_dirty must be a boolean")
        if self.lockfile_digest is not None:
            require_sha256(self.lockfile_digest, field="lockfile_digest")
        require_sha256(
            self.deterministic_seed_config_digest,
            field="deterministic_seed_config_digest",
        )
        if any(
            not value
            for value in (self.python_version, self.platform_name, self.hardware_name)
        ):
            raise ValueError("runtime provenance strings must be non-empty")
        if tuple(sorted(self.package_versions)) != self.package_versions:
            raise ValueError("package versions must use deterministic ordering")
        if self.schema_version != "runtime_provenance_v1":
            raise ValueError("unsupported runtime provenance schema")
        if self.digest != content_digest(self.digest_payload()):
            raise ValueError("runtime provenance digest mismatch")

    def digest_payload(self) -> dict[str, object]:
        return {
            "deterministic_seed_config_digest": self.deterministic_seed_config_digest,
            "git_commit": self.git_commit,
            "git_dirty": self.git_dirty,
            "hardware_name": self.hardware_name,
            "lockfile_digest": self.lockfile_digest,
            "package_versions": self.package_versions,
            "platform_name": self.platform_name,
            "python_version": self.python_version,
            "schema_version": self.schema_version,
        }


def capture_runtime_provenance(
    root: str | Path,
    *,
    git_commit: str | None = None,
    git_dirty: bool | None = None,
    deterministic_seed_config: object,
    package_versions: Mapping[str, str] | None = None,
    python_version: str | None = None,
    platform_name: str | None = None,
    hardware_name: str | None = None,
) -> RuntimeProvenance:
    """Capture deterministic provenance without embedding local filesystem paths."""

    repository = Path(root)
    resolved_commit = (
        git_commit or _git(repository, "rev-parse", "HEAD") or ""
    ).lower()
    if not _GIT_SHA_RE.fullmatch(resolved_commit):
        raise ValueError("a valid git commit is required for runtime provenance")
    resolved_dirty = git_dirty
    if resolved_dirty is None:
        status = _git(repository, "status", "--porcelain")
        if status is None:
            raise ValueError("git dirty state could not be determined")
        resolved_dirty = bool(status)

    lock_path = repository / "uv.lock"
    lock_digest = _sha256_file(lock_path) if lock_path.is_file() else None
    if package_versions is None:
        names = ("gymnasium", "numpy", "stable-baselines3", "torch")
        versions: dict[str, str] = {}
        for name in names:
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                versions[name] = "not-installed"
    else:
        versions = {str(key): str(value) for key, value in package_versions.items()}

    payload = {
        "deterministic_seed_config_digest": content_digest(deterministic_seed_config),
        "git_commit": resolved_commit,
        "git_dirty": resolved_dirty,
        "hardware_name": hardware_name
        or platform.processor()
        or platform.machine()
        or "unknown",
        "lockfile_digest": lock_digest,
        "package_versions": tuple(sorted(versions.items())),
        "platform_name": platform_name or platform.platform(),
        "python_version": python_version or sys.version.split()[0],
        "schema_version": "runtime_provenance_v1",
    }
    return RuntimeProvenance(digest=content_digest(payload), **payload)


__all__ = ["RuntimeProvenance", "capture_runtime_provenance"]
