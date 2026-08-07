"""Read-only projection of externally supervised Docker runs into Studio jobs."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from trade_rl.studio.catalog_common import mtime, read_json
from trade_rl.studio.contracts import JobSummary
from trade_rl.studio.errors import ResourceNotFound
from trade_rl.studio.settings import StudioSettings


class SupervisedJobCatalog:
    def __init__(self, settings: StudioSettings) -> None:
        self.settings = settings

    def _generation_roots(self) -> tuple[Path, ...]:
        roots: set[Path] = set()
        for configured in self.settings.run_roots:
            generations = (
                configured / "runs" if (configured / "runs").is_dir() else configured
            )
            if generations.is_dir():
                roots.update(path for path in generations.iterdir() if path.is_dir())
        return tuple(
            sorted(roots, key=lambda path: path.stat().st_mtime_ns, reverse=True)
        )

    def _job(self, generation: Path) -> JobSummary | None:
        heartbeat_path = generation / "heartbeat.json"
        payload = read_json(heartbeat_path)
        if payload is None or payload.get("state") != "running":
            return None
        observed = payload.get("observed_at")
        try:
            observed_at = datetime.fromisoformat(str(observed))
            age = (datetime.now(UTC) - observed_at).total_seconds()
        except (TypeError, ValueError):
            return None
        if age < 0.0 or age > 120.0:
            return None
        artifact_root = generation / "artifacts"
        staging = artifact_root / ".staging"
        candidates = (
            sorted(path for path in staging.iterdir() if path.is_dir())
            if staging.is_dir()
            else []
        )
        run_id = candidates[0].name if candidates else generation.name
        pid = payload.get("pid")
        provenance_path = generation / "entrypoint-provenance.json"
        started_at = (
            mtime(provenance_path)
            if provenance_path.is_file()
            else observed_at.isoformat()
        )
        return JobSummary(
            id=generation.name,
            status="running",
            run_id=run_id,
            config_resource_id="external-supervisor",
            dataset_resource_id="external-supervisor",
            config_digest="0" * 64,
            dataset_id="0" * 64,
            config_path="",
            dataset_path="",
            artifact_root=self.settings.relative_path(artifact_root),
            submitted_at=started_at,
            owner_instance_id="external-supervisor",
            started_at=started_at,
            pid=pid if isinstance(pid, int) and not isinstance(pid, bool) else None,
            cancellable=False,
        )

    def list(self) -> tuple[JobSummary, ...]:
        return tuple(
            job
            for generation in self._generation_roots()
            if (job := self._job(generation)) is not None
        )

    def get(self, job_id: str) -> JobSummary:
        for job in self.list():
            if job.id == job_id:
                return job
        raise ResourceNotFound(f"Studio job not found: {job_id}")


__all__ = ["SupervisedJobCatalog"]
