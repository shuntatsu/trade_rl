"""Persistent, restart-safe subprocess jobs for exploratory Studio training."""

from __future__ import annotations

import os
import re
import subprocess
import sys
import uuid
from collections import deque
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, cast

from trade_rl.artifacts.run_manifest import validate_training_run_directory
from trade_rl.studio.contracts import JobSummary, TrainingJobRequest
from trade_rl.studio.errors import (
    IdentityConflict,
    InvalidStudioRequest,
    JobOwnershipLost,
)
from trade_rl.studio.job_store import JobStore
from trade_rl.studio.settings import StudioSettings

_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_TERMINAL_STATES = {"succeeded", "failed", "cancelled"}


class ProcessHandle(Protocol):
    pid: int

    def poll(self) -> int | None: ...

    def terminate(self) -> None: ...

    def kill(self) -> None: ...

    def wait(self, timeout: float | None = None) -> int: ...


class CatalogProtocol(Protocol):
    def resolve_config(self, resource_id: str) -> Any: ...

    def resolve_dataset(self, resource_id: str) -> Any: ...


ProcessFactory = Callable[..., ProcessHandle]


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _default_process_factory(
    command: tuple[str, ...], *, cwd: Path, log_path: Path
) -> ProcessHandle:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab", buffering=0) as log_handle:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=os.name != "nt",
        )
    return cast(ProcessHandle, process)


def _pid_start_token(pid: int) -> str | None:
    path = Path(f"/proc/{pid}/stat")
    try:
        fields = path.read_text(encoding="utf-8").split()
    except OSError:
        return None
    return fields[21] if len(fields) > 21 else None


def _pid_alive(pid: int | None) -> bool:
    if pid is None or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _pid_matches(pid: int | None, token: str | None) -> bool:
    if not _pid_alive(pid):
        return False
    if pid is None or token is None:
        return True
    current = _pid_start_token(pid)
    return current is not None and current == token


class JobSupervisor:
    """Submit fixed training commands through an atomic, restart-safe job store."""

    def __init__(
        self,
        settings: StudioSettings,
        *,
        catalog: CatalogProtocol | None = None,
        process_factory: ProcessFactory = _default_process_factory,
        instance_id: str | None = None,
    ) -> None:
        self.settings = settings
        if catalog is None:
            from trade_rl.studio.catalog import StudioCatalog

            catalog = StudioCatalog(settings)
        self.catalog = catalog
        self.process_factory = process_factory
        self.instance_id = instance_id or f"studio-{uuid.uuid4().hex}"
        self.store = JobStore(settings.job_root)
        self._processes: dict[str, ProcessHandle] = {}

    def _log_path(self, job_id: str) -> Path:
        return self.settings.job_root / f"{job_id}.log"

    def _command(
        self,
        *,
        config_path: Path,
        dataset_path: Path,
        artifact_root: Path,
        run_id: str,
    ) -> tuple[str, ...]:
        return (
            sys.executable,
            "-c",
            "from trade_rl.cli import main; raise SystemExit(main())",
            "train",
            "run",
            "--config",
            str(config_path),
            "--dataset",
            str(dataset_path),
            "--output",
            str(artifact_root),
            "--run-id",
            run_id,
        )

    @staticmethod
    def _existing_run(artifact_root: Path, run_id: str) -> bool:
        return any(
            (artifact_root / namespace / run_id).exists()
            for namespace in ("runs", "failed", ".staging")
        )

    def submit_training(self, request: TrainingJobRequest) -> JobSummary:
        if request.run_id in {".", ".."} or not _RUN_ID_RE.fullmatch(request.run_id):
            raise InvalidStudioRequest("run_id contains unsupported characters")
        config = self.catalog.resolve_config(request.config_resource_id)
        dataset = self.catalog.resolve_dataset(request.dataset_resource_id)
        artifact_root = self.settings.run_roots[0]
        artifact_root.mkdir(parents=True, exist_ok=True)
        if self._existing_run(artifact_root, request.run_id):
            raise IdentityConflict(f"run already exists: {request.run_id}")

        job_id = (
            f"job-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}"
        )
        submitted_at = _utc_now()
        relative_root = self.settings.relative_path(artifact_root)
        self.store.reserve(
            artifact_root=relative_root,
            run_id=request.run_id,
            job_id=job_id,
            owner_instance_id=self.instance_id,
            created_at=submitted_at,
        )
        try:
            if self._existing_run(artifact_root, request.run_id):
                raise IdentityConflict(f"run already exists: {request.run_id}")
            config_digest = config.summary.config_digest
            if not isinstance(config_digest, str) or not config_digest:
                raise InvalidStudioRequest(
                    "resolved training config has no canonical digest"
                )
            dataset_id = dataset.summary.dataset_id
            if not isinstance(dataset_id, str) or not dataset_id:
                raise InvalidStudioRequest("resolved dataset has no canonical identity")
            queued = JobSummary(
                id=job_id,
                status="queued",
                run_id=request.run_id,
                config_resource_id=request.config_resource_id,
                dataset_resource_id=request.dataset_resource_id,
                config_digest=config_digest,
                dataset_id=dataset_id,
                config_path=self.settings.relative_path(config.path),
                dataset_path=self.settings.relative_path(dataset.path),
                artifact_root=relative_root,
                submitted_at=submitted_at,
                owner_instance_id=self.instance_id,
            )
            self.store.create(queued)
            command = self._command(
                config_path=config.path,
                dataset_path=dataset.path,
                artifact_root=artifact_root,
                run_id=request.run_id,
            )
            try:
                process = self.process_factory(
                    command,
                    cwd=self.settings.project_root,
                    log_path=self._log_path(job_id),
                )
            except Exception as error:
                failed = self.store.transition(
                    job_id,
                    expected={"queued"},
                    updates={
                        "status": "failed",
                        "completed_at": _utc_now(),
                        "error": f"worker start failed: {error}",
                    },
                )
                self.store.release(
                    artifact_root=relative_root,
                    run_id=request.run_id,
                    job_id=job_id,
                )
                return failed
            self._processes[job_id] = process
            return self.store.transition(
                job_id,
                expected={"queued"},
                updates={
                    "status": "running",
                    "started_at": _utc_now(),
                    "pid": process.pid,
                    "pid_start_token": _pid_start_token(process.pid),
                    "cancellable": True,
                },
            )
        except Exception:
            self.store.release(
                artifact_root=relative_root,
                run_id=request.run_id,
                job_id=job_id,
            )
            raise

    def _finish(
        self,
        summary: JobSummary,
        *,
        status: str,
        exit_code: int | None,
        error: str | None,
    ) -> JobSummary:
        finished = self.store.transition(
            summary.id,
            expected={summary.status},
            updates={
                "status": status,
                "completed_at": summary.completed_at or _utc_now(),
                "exit_code": exit_code,
                "cancellable": False,
                "error": error,
            },
        )
        self.store.release(
            artifact_root=summary.artifact_root,
            run_id=summary.run_id,
            job_id=summary.id,
        )
        return finished

    def _published_valid(self, path: Path) -> bool:
        if not path.is_dir():
            return False
        try:
            validate_training_run_directory(path)
        except (OSError, ValueError, TypeError):
            return False
        return True

    def _reconcile(self, summary: JobSummary) -> JobSummary:
        if summary.status in _TERMINAL_STATES:
            self.store.release(
                artifact_root=summary.artifact_root,
                run_id=summary.run_id,
                job_id=summary.id,
            )
            return summary.model_copy(update={"cancellable": False})
        artifact_root = self.settings.project_root / summary.artifact_root
        published = artifact_root / "runs" / summary.run_id
        failed = artifact_root / "failed" / summary.run_id
        process = self._processes.get(summary.id)
        if process is not None:
            exit_code = process.poll()
            if exit_code is None:
                return summary.model_copy(update={"cancellable": True})
            self._processes.pop(summary.id, None)
            if summary.status == "cancelling":
                return self._finish(
                    summary, status="cancelled", exit_code=exit_code, error=None
                )
            if exit_code == 0 and self._published_valid(published):
                return self._finish(
                    summary, status="succeeded", exit_code=0, error=None
                )
            error = (
                f"worker exited with code {exit_code}"
                if exit_code != 0
                else "worker exited without a valid published run"
            )
            return self._finish(
                summary, status="failed", exit_code=exit_code, error=error
            )
        if self._published_valid(published):
            return self._finish(summary, status="succeeded", exit_code=0, error=None)
        if failed.is_dir():
            return self._finish(
                summary,
                status="failed",
                exit_code=summary.exit_code,
                error=summary.error or "training run was isolated as failed",
            )
        if _pid_matches(summary.pid, summary.pid_start_token):
            return summary.model_copy(update={"cancellable": False})
        return self._finish(
            summary,
            status="failed",
            exit_code=summary.exit_code,
            error=summary.error or "worker is no longer running",
        )

    def get_job(self, job_id: str) -> JobSummary:
        return self._reconcile(self.store.read(job_id))

    def list_jobs(self) -> tuple[JobSummary, ...]:
        jobs: list[JobSummary] = []
        for summary in self.store.list():
            try:
                jobs.append(self._reconcile(summary))
            except (IdentityConflict, ValueError):
                jobs.append(summary.model_copy(update={"cancellable": False}))
        return tuple(sorted(jobs, key=lambda item: item.submitted_at, reverse=True))

    def cancel(self, job_id: str) -> JobSummary:
        summary = self.get_job(job_id)
        if summary.status in _TERMINAL_STATES:
            return summary
        process = self._processes.get(job_id)
        if process is None or summary.owner_instance_id != self.instance_id:
            raise JobOwnershipLost(
                "job process is not owned by this Studio instance and cannot be terminated"
            )
        cancelling = self.store.transition(
            job_id,
            expected={"queued", "running"},
            updates={"status": "cancelling", "cancellable": False},
        )
        process.terminate()
        try:
            exit_code = process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            process.kill()
            exit_code = process.wait(timeout=5.0)
        self._processes.pop(job_id, None)
        return self._finish(
            cancelling,
            status="cancelled",
            exit_code=exit_code,
            error=None,
        )

    def tail_log(
        self, job_id: str, *, limit: int = 200
    ) -> tuple[tuple[str, ...], bool]:
        self.get_job(job_id)
        if limit <= 0 or limit > 2_000:
            raise InvalidStudioRequest("log limit must be between 1 and 2000")
        path = self._log_path(job_id)
        if not path.is_file():
            return (), False
        lines: deque[str] = deque(maxlen=limit + 1)
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                lines.append(line.rstrip("\r\n"))
        truncated = len(lines) > limit
        if truncated:
            lines.popleft()
        return tuple(lines), truncated


__all__ = ["JobSupervisor"]
