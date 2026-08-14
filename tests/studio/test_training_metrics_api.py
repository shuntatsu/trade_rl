from __future__ import annotations

from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

from .support import request, studio_client as client


def test_training_metrics_endpoints_return_status_and_allowlisted_scalars(
    tmp_path: Path,
) -> None:
    api, _, catalog, _ = client(tmp_path)
    created = api.post(
        "/api/studio/jobs/training",
        json=request(catalog, run_id="run-metrics").model_dump(by_alias=True),
    )
    job_id = created.json()["id"]
    run = (
        tmp_path
        / "research"
        / ".staging"
        / "run-metrics"
        / "members"
        / "member-000"
        / "tensorboard"
        / "seed-7-ppo"
    )
    writer = SummaryWriter(log_dir=run)
    writer.add_scalar("train/learning_rate", 1.2e-4, 100)
    writer.add_scalar("secret/internal", 999.0, 100)
    writer.close()

    status = api.get(
        f"/api/studio/jobs/{job_id}/training-metrics/status",
        params={"seed": 7},
    )
    assert status.status_code == 200
    assert status.json()["available"] is True
    assert status.json()["availableTags"] == ["train/learning_rate"]

    scalars = api.get(
        f"/api/studio/jobs/{job_id}/training-metrics/scalars",
        params=[("seed", "7"), ("tag", "train/learning_rate")],
    )
    assert scalars.status_code == 200
    assert scalars.json()["series"][0]["points"][0]["step"] == 100


def test_training_metrics_api_returns_empty_and_rejects_unknown_tags(
    tmp_path: Path,
) -> None:
    api, _, catalog, _ = client(tmp_path)
    created = api.post(
        "/api/studio/jobs/training",
        json=request(catalog, run_id="run-empty").model_dump(by_alias=True),
    )
    job_id = created.json()["id"]
    empty = api.get(f"/api/studio/jobs/{job_id}/training-metrics/status")
    assert empty.status_code == 200
    assert empty.json()["available"] is False

    unknown = api.get(
        f"/api/studio/jobs/{job_id}/training-metrics/scalars",
        params={"tag": "secret/internal"},
    )
    assert unknown.status_code == 400
    assert unknown.json()["detail"]["code"] == "invalid_request"
    assert (
        api.get("/api/studio/jobs/missing/training-metrics/status").status_code == 404
    )
