"""Authoritative immutable serving-bundle registry."""

from __future__ import annotations

import contextlib
import json
import os
import re
import shutil
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from mars_lite.serving.bundle import ServingBundle, load_bundle

_VERSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


@dataclass(frozen=True)
class ActiveBundleRecord:
    version: str
    bundle_digest: str
    activated_at: float
    evidence_identity: str
    previous_version: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "bundle_digest": self.bundle_digest,
            "activated_at": self.activated_at,
            "evidence_identity": self.evidence_identity,
            "previous_version": self.previous_version,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ActiveBundleRecord":
        return cls(
            version=str(value["version"]),
            bundle_digest=str(value["bundle_digest"]),
            activated_at=float(value["activated_at"]),
            evidence_identity=str(value["evidence_identity"]),
            previous_version=(
                str(value["previous_version"])
                if value.get("previous_version") is not None
                else None
            ),
        )


class ModelRegistry:
    """Store immutable bundle versions and atomically select one active version."""

    def __init__(self, registry_dir: str | Path) -> None:
        self.root = Path(registry_dir)
        self.versions_dir = self.root / "versions"
        self.active_path = self.root / "active.json"
        self.history_path = self.root / "activation-history.jsonl"
        self.lock_path = self.root / ".registry.lock"
        self.root.mkdir(parents=True, exist_ok=True)
        self.versions_dir.mkdir(parents=True, exist_ok=True)
        self._thread_lock = threading.RLock()

    @contextlib.contextmanager
    def _lock(self, timeout: float = 5.0) -> Iterator[None]:
        with self._thread_lock:
            deadline = time.monotonic() + timeout
            fd: int | None = None
            while fd is None:
                try:
                    fd = os.open(
                        self.lock_path,
                        os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                        0o600,
                    )
                except FileExistsError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("timed out acquiring registry lock")
                    time.sleep(0.05)
            try:
                os.write(fd, str(os.getpid()).encode("ascii"))
                os.close(fd)
                fd = None
                yield
            finally:
                if fd is not None:
                    os.close(fd)
                self.lock_path.unlink(missing_ok=True)

    def version_dir(self, version: str) -> Path:
        self._validate_version(version)
        return self.versions_dir / version

    @staticmethod
    def _validate_version(version: str) -> None:
        if not _VERSION_RE.fullmatch(version):
            raise ValueError(f"invalid model version: {version!r}")

    @staticmethod
    def _fsync_dir(path: Path) -> None:
        if os.name == "nt":
            return
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _atomic_write_json(self, path: Path, value: dict[str, Any]) -> None:
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        data = (
            json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n"
        )
        try:
            with temporary.open("w", encoding="utf-8") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            self._fsync_dir(path.parent)
        finally:
            temporary.unlink(missing_ok=True)

    def register(self, source_dir: str | Path) -> ServingBundle:
        source = Path(source_dir)
        bundle = load_bundle(source)
        self._validate_version(bundle.version)
        target = self.version_dir(bundle.version)

        with self._lock():
            if target.exists():
                raise ValueError(f"version {bundle.version!r} is already registered")
            temporary = self.versions_dir / f".{bundle.version}.{uuid.uuid4().hex}.tmp"
            try:
                shutil.copytree(source, temporary)
                copied = load_bundle(temporary)
                if copied.bundle_digest != bundle.bundle_digest:
                    raise ValueError("registered copy bundle digest mismatch")
                os.replace(temporary, target)
                self._fsync_dir(self.versions_dir)
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary, ignore_errors=True)
        return load_bundle(target)

    def list_versions(self) -> list[str]:
        return sorted(
            path.name
            for path in self.versions_dir.iterdir()
            if path.is_dir() and not path.name.startswith(".")
        )

    def get_active_record(self) -> ActiveBundleRecord:
        if not self.active_path.is_file():
            raise LookupError("no active model")
        try:
            value = json.loads(self.active_path.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise TypeError
            return ActiveBundleRecord.from_dict(value)
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid active registry pointer") from exc

    def get_active_bundle(self) -> ServingBundle:
        record = self.get_active_record()
        bundle = load_bundle(self.version_dir(record.version))
        if bundle.bundle_digest != record.bundle_digest:
            raise ValueError("active pointer bundle digest mismatch")
        return bundle

    def _append_history(
        self, previous: ActiveBundleRecord | None, current: ActiveBundleRecord
    ) -> None:
        event = {
            "previous_version": previous.version if previous else None,
            "previous_digest": previous.bundle_digest if previous else None,
            "version": current.version,
            "bundle_digest": current.bundle_digest,
            "activated_at": current.activated_at,
            "evidence_identity": current.evidence_identity,
        }
        with self.history_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    event, sort_keys=True, separators=(",", ":"), allow_nan=False
                )
                + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())

    def _read_active_optional(self) -> ActiveBundleRecord | None:
        if not self.active_path.exists():
            return None
        return self.get_active_record()

    def activate(self, version: str, evidence_identity: str) -> ServingBundle:
        if not evidence_identity:
            raise ValueError("evidence_identity is required")
        with self._lock():
            target_dir = self.version_dir(version)
            if not target_dir.is_dir():
                raise KeyError(f"unknown model version: {version}")
            target = load_bundle(target_dir)
            previous = self._read_active_optional()
            record = ActiveBundleRecord(
                version=target.version,
                bundle_digest=target.bundle_digest,
                activated_at=time.time(),
                evidence_identity=evidence_identity,
                previous_version=previous.version if previous else None,
            )
            self._atomic_write_json(self.active_path, record.to_dict())
            self._append_history(previous, record)
            return target

    def rollback(self, target_version: str | None = None) -> ServingBundle:
        with self._lock():
            current = self.get_active_record()
            target_name = target_version or current.previous_version
            if target_name is None:
                raise LookupError("no previous active model to roll back to")
            target_dir = self.version_dir(target_name)
            if not target_dir.is_dir():
                raise KeyError(f"unknown model version: {target_name}")
            target = load_bundle(target_dir)
            record = ActiveBundleRecord(
                version=target.version,
                bundle_digest=target.bundle_digest,
                activated_at=time.time(),
                evidence_identity=f"rollback:{current.version}",
                previous_version=None,
            )
            self._atomic_write_json(self.active_path, record.to_dict())
            self._append_history(current, record)
            return target
