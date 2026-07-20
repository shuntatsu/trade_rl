from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from trade_rl.studio.api import create_app
from trade_rl.studio.catalog import StudioCatalog
from trade_rl.studio.jobs import JobSupervisor

from .helpers import write_dataset, write_run
from .test_catalog import settings
from .test_jobs import FakeCatalog, FakeFactory, request


def client(
    tmp_path: Path,
) -> tuple[TestClient, FakeFactory, FakeCatalog, StudioCatalog]:
    write_dataset(tmp_path / "datasets" / "btc")
    write_run(tmp_path / "research")
    real_catalog = StudioCatalog(settings(tmp_path))
    job_catalog = FakeCatalog(tmp_path)
    factory = FakeFactory()
    supervisor = JobSupervisor(
        settings(tmp_path),
        catalog=job_catalog,
        process_factory=factory,
    )
    return (
        TestClient(
            create_app(
                settings(tmp_path),
                catalog=real_catalog,
                supervisor=supervisor,
            )
        ),
        factory,
        job_catalog,
        real_catalog,
    )


def test_read_endpoints_return_collision_free_validated_resources_and_no_go(
    tmp_path: Path,
) -> None:
    api, _, _, _ = client(tmp_path)

    overview = api.get("/api/studio/overview")
    datasets = api.get("/api/studio/datasets")
    runs = api.get("/api/studio/runs")
    configs = api.get("/api/studio/configs")

    assert overview.status_code == 200
    assert overview.json()["assessment"]["status"] == "NO-GO"
    assert overview.json()["latestDataset"]["relativePath"] == "datasets/btc"
    assert datasets.json()["items"][0]["id"].startswith("dataset-")
    assert datasets.json()["items"][0]["datasetId"]
    assert runs.json()["items"][0]["id"].startswith("run-")
    assert runs.json()["items"][0]["runId"] == "run-001"
    assert configs.status_code == 200
    assert "access-control-allow-origin" not in overview.headers


def test_training_job_lifecycle_accepts_only_catalog_resource_identities(
    tmp_path: Path,
) -> None:
    api, factory, job_catalog, _ = client(tmp_path)

    created = api.post(
        "/api/studio/jobs/training",
        json=request(job_catalog, run_id="run-002").model_dump(by_alias=True),
    )

    assert created.status_code == 201
    job_id = created.json()["id"]
    assert created.json()["status"] == "running"
    assert created.json()["configResourceId"] == job_catalog.config.summary.id
    assert created.json()["datasetResourceId"] == job_catalog.dataset.summary.id
    assert api.get("/api/studio/jobs").json()["total"] == 1
    assert api.get(f"/api/studio/jobs/{job_id}").status_code == 200
    assert api.get(f"/api/studio/jobs/{job_id}/log").json()["lines"] == ["started"]

    cancelled = api.post(f"/api/studio/jobs/{job_id}/cancel")

    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "cancelled"
    assert factory.process.terminated is True


def test_api_returns_stable_typed_errors_for_missing_and_conflicting_resources(
    tmp_path: Path,
) -> None:
    api, _, job_catalog, _ = client(tmp_path)

    missing = api.get("/api/studio/jobs/missing")
    assert missing.status_code == 404
    assert missing.json()["detail"]["code"] == "resource_not_found"

    payload = request(job_catalog, run_id="run-002").model_dump(by_alias=True)
    assert api.post("/api/studio/jobs/training", json=payload).status_code == 201
    duplicate = api.post("/api/studio/jobs/training", json=payload)
    assert duplicate.status_code == 409
    assert duplicate.json()["detail"]["code"] == "identity_conflict"

    unknown = api.post(
        "/api/studio/jobs/training",
        json={
            "configResourceId": "config-000000000000000000000000",
            "datasetResourceId": job_catalog.dataset.summary.id,
            "runId": "run-missing",
        },
    )
    assert unknown.status_code == 404
    assert unknown.json()["detail"]["code"] == "resource_not_found"

    old_path_contract = api.post(
        "/api/studio/jobs/training",
        json={
            "configPath": "../secret.json",
            "datasetPath": "datasets/btc",
            "artifactRoot": "research",
            "runId": "run-escape",
        },
    )
    assert old_path_contract.status_code == 422


def test_audit_endpoints_use_resource_ids_and_report_comparison_eligibility(
    tmp_path: Path,
) -> None:
    api, _, _, catalog = client(tmp_path)
    write_run(tmp_path / "research", run_id="run-002", algorithm="sac")
    runs = {item.run_id: item for item in catalog.list_runs()}

    comparison = api.get(
        "/api/studio/compare",
        params={
            "left_resource_id": runs["run-001"].id,
            "right_resource_id": runs["run-002"].id,
        },
    )
    evidence = api.get(f"/api/studio/runs/{runs['run-001'].id}/evidence")
    serving = api.get("/api/studio/serving")

    assert comparison.status_code == 200
    assert comparison.json()["leftResourceId"] == runs["run-001"].id
    assert comparison.json()["rightResourceId"] == runs["run-002"].id
    assert comparison.json()["leftRunId"] == "run-001"
    assert comparison.json()["eligibility"]["status"] in {
        "COMPARABLE",
        "PARTIALLY_COMPARABLE",
    }
    assert comparison.json()["productionStatus"] == "NO-GO"
    assert evidence.status_code == 200
    assert evidence.json()["runResourceId"] == runs["run-001"].id
    assert evidence.json()["runId"] == "run-001"
    assert serving.status_code == 200
    assert serving.json()["state"] == "IDLE"


def test_audit_endpoints_reject_human_run_ids_and_unknown_resource_ids(
    tmp_path: Path,
) -> None:
    api, _, _, catalog = client(tmp_path)
    left = catalog.list_runs()[0].id

    compare = api.get(
        "/api/studio/compare",
        params={"left_resource_id": left, "right_resource_id": "run-001"},
    )
    evidence = api.get("/api/studio/runs/run-001/evidence")

    assert compare.status_code == 404
    assert compare.json()["detail"]["code"] == "resource_not_found"
    assert evidence.status_code == 404
    assert evidence.json()["detail"]["code"] == "resource_not_found"


def test_evidence_endpoint_reports_tampered_run_instead_of_hiding_it(
    tmp_path: Path,
) -> None:
    api, _, _, catalog = client(tmp_path)
    run = tmp_path / "research" / "runs" / "run-001"
    (run / "walk-forward.json").write_text('{"tampered":true}', encoding="utf-8")
    invalid = next(item for item in catalog.list_runs() if item.run_id == "run-001")

    response = api.get(f"/api/studio/runs/{invalid.id}/evidence")

    assert response.status_code == 200
    assert response.json()["status"] == "INVALID"
    assert response.json()["validationError"] is not None
