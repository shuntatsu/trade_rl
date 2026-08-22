"""Persistent, restart-safe subprocess jobs for exploratory Studio training."""

from __future__ import annotations

import ctypes
import os
import re
import signal
import subprocess
import sys
import uuid
from collections import deque
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, Protocol, cast

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
_WINDOWS_NEW_PROCESS_GROUP: Final = getattr(
    subprocess,
    "CREATE_NEW_PROCESS_GROUP",
    0x00000200,
)
_PROCESS_STOP_TIMEOUT_SECONDS: Final = 5.0


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


def _process_group_options(platform_name: str) -> dict[str, object]:
    """Return the subprocess options that isolate one complete worker tree."""

    if platform_name == "nt":
        return {"creationflags": _WINDOWS_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def _default_process_factory(
    command: tuple[str, ...], *, cwd: Path, log_path: Path
) -> ProcessHandle:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab", buffering=0) as log_handle:
        if os.name == "nt":
            process = subprocess.Popen(
                command,
                cwd=cwd,
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                creationflags=_WINDOWS_NEW_PROCESS_GROUP,
            )
        else:
            process = subprocess.Popen(
                command,
                cwd=cwd,
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
    return cast(ProcessHandle, process)


def _windows_process_start_token(pid: int) -> str | None:
    loader = getattr(ctypes, "WinDLL", None)
    if loader is None:
        return None

    class FileTime(ctypes.Structure):
        _fields_ = (
            ("low", ctypes.c_uint32),
            ("high", ctypes.c_uint32),
        )

    kernel32 = loader("kernel32", use_last_error=True)
    open_process = kernel32.OpenProcess
    open_process.argtypes = (ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32)
    open_process.restype = ctypes.c_void_p
    get_process_times = kernel32.GetProcessTimes
    get_process_times.argtypes = (
        ctypes.c_void_p,
        ctypes.POINTER(FileTime),
        ctypes.POINTER(FileTime),
        ctypes.POINTER(FileTime),
        ctypes.POINTER(FileTime),
    )
    get_process_times.restype = ctypes.c_int
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = (ctypes.c_void_p,)
    close_handle.restype = ctypes.c_int

    handle = open_process(0x1000, 0, pid)
    if not handle:
        return None
    try:
        creation = FileTime()
        exit_time = FileTime()
        kernel_time = FileTime()
        user_time = FileTime()
        if not get_process_times(
            handle,
            ctypes.byref(creation),
            ctypes.byref(exit_time),
            ctypes.byref(kernel_time),
            ctypes.byref(user_time),
        ):
            return None
        return f"{creation.high:08x}{creation.low:08x}"
    finally:
        close_handle(handle)


def _pid_start_token(
    pid: int,
    *,
    platform_name: str | None = None,
) -> str | None:
    platform = os.name if platform_name is None else platform_name
    if platform == "nt":
        return _windows_process_start_token(pid)
    if platform != "posix":
        return None
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
    if not _pid_alive(pid) or pid is None or token is None:
        return False
    current = _pid_start_token(pid)
    return current is not None and current == token


def _wait_or_none(process: ProcessHandle) -> int | None:
    try:
        return process.wait(timeout=_PROCESS_STOP_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        return None


def _terminate_posix_process_tree(process: ProcessHandle) -> int:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.terminate()
    exit_code = _wait_or_none(process)
    if exit_code is not None:
        return exit_code
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        process.kill()
    return process.wait(timeout=_PROCESS_STOP_TIMEOUT_SECONDS)


def _run_taskkill(pid: int, *, force: bool) -> None:
    command = ["taskkill", "/PID", str(pid), "/T"]
    if force:
        command.append("/F")
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "failed to terminate Studio worker process tree: "
            f"{completed.stderr.strip() or completed.stdout.strip()}"
        )


def _terminate_windows_process_tree(process: ProcessHandle) -> int:
    _run_taskkill(process.pid, force=False)
    exit_code = _wait_or_none(process)
    if exit_code is not None:
        return exit_code
    _run_taskkill(process.pid, force=True)
    return process.wait(timeout=_PROCESS_STOP_TIMEOUT_SECONDS)


def _terminate_process_tree(process: ProcessHandle) -> int:
    if os.name == "nt":
        return _terminate_windows_process_tree(process)
    return _terminate_posix_process_tree(process)


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
        exit_code = _terminate_process_tree(process)
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
