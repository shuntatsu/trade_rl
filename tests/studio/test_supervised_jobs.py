from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

from trade_rl.studio.settings import StudioSettings
from trade_rl.studio.supervised_jobs import SupervisedJobCatalog


def test_catalog_projects_fresh_supervisor_heartbeat_as_read_only_job(
    tmp_path: Path,
) -> None:
    generation = tmp_path / "var" / "runs" / "generation-1"
    staging = generation / "artifacts" / ".staging" / "walk-forward"
    staging.mkdir(parents=True)
    (generation / "heartbeat.json").write_text(
        json.dumps(
            {
                "state": "running",
                "observed_at": datetime.now(UTC).isoformat(),
                "pid": 27,
            }
        ),
        encoding="utf-8",
    )
    provenance = generation / "entrypoint-provenance.json"
    provenance.write_text("{}", encoding="utf-8")
    started_timestamp = datetime(2026, 8, 1, tzinfo=UTC).timestamp()
    os.utime(provenance, (started_timestamp, started_timestamp))
    settings = StudioSettings(
        project_root=tmp_path,
        dataset_roots=(tmp_path / "var" / "runs",),
        run_roots=(tmp_path / "var",),
        config_roots=(tmp_path / "examples",),
        job_root=tmp_path / "var" / "studio" / "jobs",
    )

    job = SupervisedJobCatalog(settings).list()[0]

    assert job.id == "generation-1"
    assert job.run_id == "walk-forward"
    assert job.artifact_root == "var/runs/generation-1/artifacts"
    assert job.status == "running"
    assert job.cancellable is False
    assert job.started_at == datetime.fromtimestamp(started_timestamp, UTC).isoformat()
    assert job.submitted_at == job.started_at
