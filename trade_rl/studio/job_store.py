"""Cross-process atomic persistence and run reservations for Studio jobs."""

from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping

from trade_rl.studio.contracts import JobSummary
from trade_rl.studio.errors import IdentityConflict, ResourceNotFound

_TERMINAL = {"succeeded", "failed", "cancelled"}
_ALLOWED: dict[str, set[str]] = {
    "queued": {"running", "failed", "cancelled"},
    "running": {"succeeded", "failed", "cancelling"},
    "cancelling": {"cancelled", "failed"},
    "succeeded": set(),
    "failed": set(),
    "cancelled": set(),
}


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    data = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


@contextmanager
def _exclusive_lock(path: Path, *, timeout: float = 2.0) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout
    while True:
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            try:
                age = time.time() - path.stat().st_mtime
                if age > 30.0:
                    path.unlink(missing_ok=True)
                    continue
            except OSError:
                pass
            if time.monotonic() >= deadline:
                raise IdentityConflict(f"Studio job lock is busy: {path.name}")
            time.sleep(0.01)
            continue
        try:
            os.write(descriptor, f"{os.getpid()}\n".encode())
        finally:
            os.close(descriptor)
        break
    try:
        yield
    finally:
        path.unlink(missing_ok=True)


class JobStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.records = root
        self.locks = root / ".locks"
        self.reservations = root / ".reservations"
        self.root.mkdir(parents=True, exist_ok=True)
        self.locks.mkdir(parents=True, exist_ok=True)
        self.reservations.mkdir(parents=True, exist_ok=True)

    def _record_path(self, job_id: str) -> Path:
        if not job_id or Path(job_id).name != job_id:
            raise ResourceNotFound(f"unknown job: {job_id}")
        return self.records / f"{job_id}.json"

    def _record_lock(self, job_id: str) -> Path:
        return self.locks / f"{job_id}.lock"

    def read(self, job_id: str) -> JobSummary:
        path = self._record_path(job_id)
        if not path.is_file():
            raise ResourceNotFound(f"unknown job: {job_id}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return JobSummary.model_validate(payload)
        except (OSError, json.JSONDecodeError, ValueError) as error:
            raise ValueError(f"job record is invalid: {job_id}") from error

    def create(self, summary: JobSummary) -> JobSummary:
        with _exclusive_lock(self._record_lock(summary.id)):
            path = self._record_path(summary.id)
            if path.exists():
                raise IdentityConflict(f"job already exists: {summary.id}")
            _atomic_json(path, summary.model_dump(mode="json"))
        return summary

    def transition(
        self,
        job_id: str,
        *,
        expected: set[str],
        updates: Mapping[str, Any],
    ) -> JobSummary:
        with _exclusive_lock(self._record_lock(job_id)):
            current = self.read(job_id)
            if current.status not in expected:
                raise IdentityConflict(
                    f"job {job_id} cannot transition from {current.status}"
                )
            requested = updates.get("status", current.status)
            if (
                requested != current.status
                and requested not in _ALLOWED[current.status]
            ):
                raise IdentityConflict(
                    f"illegal Studio job transition: {current.status} -> {requested}"
                )
            resolved = current.model_copy(update=dict(updates))
            _atomic_json(
                self._record_path(job_id),
                resolved.model_dump(mode="json"),
            )
            return resolved

    def list(self) -> tuple[JobSummary, ...]:
        records: list[JobSummary] = []
        for path in self.records.glob("*.json"):
            try:
                records.append(self.read(path.stem))
            except (ResourceNotFound, ValueError):
                continue
        return tuple(sorted(records, key=lambda item: item.submitted_at, reverse=True))

    def _reservation_path(self, artifact_root: str, run_id: str) -> Path:
        digest = hashlib.sha256(
            f"studio-run-reservation-v1\0{artifact_root}\0{run_id}".encode()
        ).hexdigest()
        return self.reservations / f"{digest}.json"

    def reserve(
        self,
        *,
        artifact_root: str,
        run_id: str,
        job_id: str,
        owner_instance_id: str,
        created_at: str,
    ) -> None:
        path = self._reservation_path(artifact_root, run_id)
        payload = {
            "artifact_root": artifact_root,
            "created_at": created_at,
            "job_id": job_id,
            "owner_instance_id": owner_instance_id,
            "run_id": run_id,
            "schema_version": "studio_run_reservation_v1",
        }
        while True:
            try:
                descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            except FileExistsError as error:
                try:
                    existing = json.loads(path.read_text(encoding="utf-8"))
                    existing_job = self.read(str(existing.get("job_id", "")))
                except (OSError, json.JSONDecodeError, ResourceNotFound, ValueError):
                    raise IdentityConflict(
                        f"run is already reserved: {run_id}"
                    ) from error
                if existing_job.status in _TERMINAL:
                    path.unlink(missing_ok=True)
                    continue
                raise IdentityConflict(f"run is already reserved: {run_id}") from error
            try:
                os.write(
                    descriptor,
                    json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(),
                )
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            return

    def release(self, *, artifact_root: str, run_id: str, job_id: str) -> None:
        path = self._reservation_path(artifact_root, run_id)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if payload.get("job_id") == job_id:
            path.unlink(missing_ok=True)


__all__ = ["JobStore"]
