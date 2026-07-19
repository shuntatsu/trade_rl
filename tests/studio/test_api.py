from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from tests.studio.helpers import write_dataset, write_run
from tests.studio.test_catalog import settings
from tests.studio.test_jobs import FakeFactory, prepare_inputs, request
from trade_rl.studio.api import create_app
from trade_rl.studio.jobs import JobSupervisor


def client(tmp_path: Path) -> tuple[TestClient, FakeFactory]:
    write_dataset(tmp_path / "datasets" / "btc")
    write_run(tmp_path / "research")
    prepare_inputs(tmp_path)
    factory = FakeFactory()
    supervisor = JobSupervisor(settings(tmp_path), process_factory=factory)
    return TestClient(create_app(settings(tmp_path), supervisor=supervisor)), factory


def test_read_endpoints_return_validated_real_artifacts_and_no_go(
    tmp_path: Path,
) -> None:
    api, _ = client(tmp_path)

    overview = api.get("/api/studio/overview")
    datasets = api.get("/api/studio/datasets")
    runs = api.get("/api/studio/runs")
    configs = api.get("/api/studio/configs")

    assert overview.status_code == 200
    assert overview.json()["assessment"]["status"] == "NO-GO"
    assert overview.json()["latestDataset"]["relativePath"] == "datasets/btc"
    assert datasets.json()["items"][0]["status"] == "VALID"
    assert runs.json()["items"][0]["id"] == "run-001"
    assert configs.json()["items"][0]["relativePath"] == "configs/training.json"
    assert "access-control-allow-origin" not in overview.headers


def test_training_job_lifecycle_is_exposed_over_api(tmp_path: Path) -> None:
    api, factory = client(tmp_path)

    created = api.post(
        "/api/studio/jobs/training",
        json=request()
        .model_copy(update={"run_id": "run-002"})
        .model_dump(by_alias=True),
    )

    assert created.status_code == 201
    job_id = created.json()["id"]
    assert created.json()["status"] == "running"
    assert api.get("/api/studio/jobs").json()["total"] == 1
    assert api.get(f"/api/studio/jobs/{job_id}").status_code == 200
    assert api.get(f"/api/studio/jobs/{job_id}/log").json()["lines"] == ["started"]

    cancelled = api.post(f"/api/studio/jobs/{job_id}/cancel")

    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "cancelled"
    assert factory.process.terminated is True


def test_api_maps_missing_duplicate_and_invalid_requests(tmp_path: Path) -> None:
    api, _ = client(tmp_path)

    assert api.get("/api/studio/jobs/missing").status_code == 404
    first = api.post(
        "/api/studio/jobs/training",
        json=request()
        .model_copy(update={"run_id": "run-002"})
        .model_dump(by_alias=True),
    )
    assert first.status_code == 201
    duplicate = api.post(
        "/api/studio/jobs/training",
        json=request()
        .model_copy(update={"run_id": "run-002"})
        .model_dump(by_alias=True),
    )
    assert duplicate.status_code == 409
    escaped = api.post(
        "/api/studio/jobs/training",
        json={
            "configPath": "../secret.json",
            "datasetPath": "datasets/btc",
            "artifactRoot": "research",
            "runId": "run-escape",
        },
    )
    assert escaped.status_code == 400


def test_audit_endpoints_compare_evidence_and_idle_serving(tmp_path: Path) -> None:
    api, _ = client(tmp_path)
    write_run(tmp_path / "research", run_id="run-002", algorithm="sac")

    comparison = api.get(
        "/api/studio/compare",
        params={"left_run_id": "run-001", "right_run_id": "run-002"},
    )
    evidence = api.get("/api/studio/runs/run-001/evidence")
    serving = api.get("/api/studio/serving")

    assert comparison.status_code == 200
    assert comparison.json()["leftRunId"] == "run-001"
    assert comparison.json()["rightRunId"] == "run-002"
    assert comparison.json()["productionStatus"] == "NO-GO"
    assert evidence.status_code == 200
    assert evidence.json()["runId"] == "run-001"
    assert evidence.json()["productionStatus"] == "NO-GO"
    assert serving.status_code == 200
    assert serving.json()["state"] == "IDLE"
    assert serving.json()["productionStatus"] == "NO-GO"


def test_audit_endpoints_reject_unknown_run_ids(tmp_path: Path) -> None:
    api, _ = client(tmp_path)

    compare = api.get(
        "/api/studio/compare",
        params={"left_run_id": "run-001", "right_run_id": "missing"},
    )
    evidence = api.get("/api/studio/runs/missing/evidence")

    assert compare.status_code == 404
    assert evidence.status_code == 404


def test_evidence_endpoint_reports_tampered_run_instead_of_hiding_it(tmp_path: Path) -> None:
    api, _ = client(tmp_path)
    run = tmp_path / "research" / "runs" / "run-001"
    (run / "walk-forward.json").write_text('{"tampered":true}', encoding="utf-8")

    response = api.get("/api/studio/runs/run-001/evidence")

    assert response.status_code == 200
    assert response.json()["status"] == "INVALID"
    assert response.json()["validationError"] is not None
