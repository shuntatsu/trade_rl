"""Persistent local subprocess jobs for exploratory Studio training runs."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import uuid
from collections import deque
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, cast

from trade_rl.studio.contracts import JobSummary, TrainingJobRequest
from trade_rl.studio.settings import StudioSettings

_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_TERMINAL_STATES = {"succeeded", "failed", "cancelled"}


class ProcessHandle(Protocol):
    pid: int

    def poll(self) -> int | None: ...

    def terminate(self) -> None: ...

    def kill(self) -> None: ...

    def wait(self, timeout: float | None = None) -> int: ...


ProcessFactory = Callable[..., ProcessHandle]


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    temporary.replace(path)


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


class JobSupervisor:
    """Submit and reconcile fixed exploratory training commands."""

    def __init__(
        self,
        settings: StudioSettings,
        *,
        process_factory: ProcessFactory = _default_process_factory,
    ) -> None:
        self.settings = settings
        self.process_factory = process_factory
        self.settings.job_root.mkdir(parents=True, exist_ok=True)
        self._processes: dict[str, ProcessHandle] = {}

    def _record_path(self, job_id: str) -> Path:
        if not job_id or Path(job_id).name != job_id:
            raise KeyError(job_id)
        return self.settings.job_root / f"{job_id}.json"

    def _log_path(self, job_id: str) -> Path:
        return self.settings.job_root / f"{job_id}.log"

    def _read(self, job_id: str) -> dict[str, Any]:
        path = self._record_path(job_id)
        if not path.is_file():
            raise KeyError(f"unknown job: {job_id}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("job record must be a JSON object")
        return dict(payload)

    def _write(self, payload: dict[str, Any]) -> JobSummary:
        summary = JobSummary.model_validate(payload)
        _atomic_json(
            self._record_path(summary.id),
            summary.model_dump(mode="json", by_alias=False),
        )
        return summary

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

    def _existing_run(self, artifact_root: Path, run_id: str) -> bool:
        return any(
            (artifact_root / namespace / run_id).exists()
            for namespace in ("runs", "failed", ".staging")
        )

    def _active_run(self, run_id: str) -> bool:
        for path in self.settings.job_root.glob("*.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                summary = JobSummary.model_validate(payload)
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            if summary.run_id == run_id and summary.status not in _TERMINAL_STATES:
                return True
        return False

    def submit_training(self, request: TrainingJobRequest) -> JobSummary:
        if request.run_id in {".", ".."} or not _RUN_ID_RE.fullmatch(request.run_id):
            raise ValueError("run_id contains unsupported characters")
        config_path = self.settings.resolve_config_path(request.config_path)
        dataset_path = self.settings.resolve_dataset_path(request.dataset_path)
        artifact_root = self.settings.resolve_run_root(request.artifact_root)
        if not config_path.is_file():
            raise FileNotFoundError(f"training config does not exist: {request.config_path}")
        if not dataset_path.is_dir():
            raise FileNotFoundError(f"dataset does not exist: {request.dataset_path}")
        if self._existing_run(artifact_root, request.run_id) or self._active_run(
            request.run_id
        ):
            raise FileExistsError(f"run already exists or is active: {request.run_id}")

        job_id = f"job-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}"
        submitted_at = _utc_now()
        queued = JobSummary(
            id=job_id,
            status="queued",
            run_id=request.run_id,
            config_path=self.settings.relative_path(config_path),
            dataset_path=self.settings.relative_path(dataset_path),
            artifact_root=self.settings.relative_path(artifact_root),
            submitted_at=submitted_at,
        )
        self._write(queued.model_dump(mode="json"))
        command = self._command(
            config_path=config_path,
            dataset_path=dataset_path,
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
            return self._write(
                queued.model_copy(
                    update={
                        "status": "failed",
                        "completed_at": _utc_now(),
                        "error": f"worker start failed: {error}",
                    }
                ).model_dump(mode="json")
            )
        self._processes[job_id] = process
        return self._write(
            queued.model_copy(
                update={
                    "status": "running",
                    "started_at": _utc_now(),
                    "pid": process.pid,
                }
            ).model_dump(mode="json")
        )

    def _reconcile(self, summary: JobSummary) -> JobSummary:
        if summary.status in _TERMINAL_STATES:
            return summary
        artifact_root = self.settings.resolve_run_root(summary.artifact_root)
        published = artifact_root / "runs" / summary.run_id
        failed = artifact_root / "failed" / summary.run_id
        process = self._processes.get(summary.id)
        if process is not None:
            exit_code = process.poll()
            if exit_code is None:
                return summary
            self._processes.pop(summary.id, None)
            if summary.status == "cancelling":
                status = "cancelled"
                error = None
            elif exit_code == 0 and published.is_dir():
                status = "succeeded"
                error = None
            else:
                status = "failed"
                error = (
                    f"worker exited with code {exit_code}"
                    if exit_code != 0
                    else "worker exited without a published run"
                )
            return self._write(
                summary.model_copy(
                    update={
                        "status": status,
                        "completed_at": _utc_now(),
                        "exit_code": exit_code,
                        "error": error,
                    }
                ).model_dump(mode="json")
            )
        if published.is_dir():
            return self._write(
                summary.model_copy(
                    update={
                        "status": "succeeded",
                        "completed_at": summary.completed_at or _utc_now(),
                        "exit_code": 0,
                    }
                ).model_dump(mode="json")
            )
        if failed.is_dir():
            return self._write(
                summary.model_copy(
                    update={
                        "status": "failed",
                        "completed_at": summary.completed_at or _utc_now(),
                        "error": summary.error or "training run was isolated as failed",
                    }
                ).model_dump(mode="json")
            )
        if _pid_alive(summary.pid):
            return summary
        return self._write(
            summary.model_copy(
                update={
                    "status": "failed",
                    "completed_at": summary.completed_at or _utc_now(),
                    "error": summary.error or "worker is no longer running",
                }
            ).model_dump(mode="json")
        )

    def get_job(self, job_id: str) -> JobSummary:
        return self._reconcile(JobSummary.model_validate(self._read(job_id)))

    def list_jobs(self) -> tuple[JobSummary, ...]:
        jobs: list[JobSummary] = []
        for path in self.settings.job_root.glob("*.json"):
            try:
                jobs.append(self.get_job(path.stem))
            except (KeyError, OSError, ValueError, json.JSONDecodeError):
                continue
        return tuple(sorted(jobs, key=lambda item: item.submitted_at, reverse=True))

    def cancel(self, job_id: str) -> JobSummary:
        summary = self.get_job(job_id)
        if summary.status in _TERMINAL_STATES:
            return summary
        cancelling = self._write(
            summary.model_copy(update={"status": "cancelling"}).model_dump(mode="json")
        )
        process = self._processes.get(job_id)
        if process is None:
            if not _pid_alive(cancelling.pid):
                return self._write(
                    cancelling.model_copy(
                        update={"status": "cancelled", "completed_at": _utc_now()}
                    ).model_dump(mode="json")
                )
            raise RuntimeError("cannot safely terminate a worker not owned by this process")
        process.terminate()
        try:
            exit_code = process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            process.kill()
            exit_code = process.wait(timeout=5.0)
        self._processes.pop(job_id, None)
        return self._write(
            cancelling.model_copy(
                update={
                    "status": "cancelled",
                    "completed_at": _utc_now(),
                    "exit_code": exit_code,
                }
            ).model_dump(mode="json")
        )

    def tail_log(self, job_id: str, *, limit: int = 200) -> tuple[tuple[str, ...], bool]:
        self.get_job(job_id)
        if limit <= 0 or limit > 2_000:
            raise ValueError("log limit must be between 1 and 2000")
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
