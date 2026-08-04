from __future__ import annotations

import json
from pathlib import Path

from .test_api import client
from .test_jobs import request


def test_behavior_cloning_progress_exposes_latest_epoch(tmp_path: Path) -> None:
    api, _, catalog, _ = client(tmp_path)
    created = api.post(
        "/api/studio/jobs/training",
        json=request(catalog, run_id="bc-live-001").model_dump(by_alias=True),
    ).json()
    progress = (
        tmp_path
        / "research"
        / ".staging"
        / "bc-live-001"
        / "fold-000"
        / "configuration-000"
        / "seed-7"
        / "behavior-cloning-progress.json"
    )
    progress.parent.mkdir(parents=True)
    progress.write_text(
        json.dumps(
            {
                "schema_version": "behavior_cloning_progress_v1",
                "phase": "training",
                "seed": 7,
                "epoch": 9,
                "total_epochs": 45,
                "best_epoch": 8,
                "elapsed_seconds": 120.0,
                "estimated_remaining_seconds": 480.0,
                "validation_loss": 0.125,
                "gate_loss": 0.1,
                "target_loss": 0.02,
                "composed_loss": 0.005,
                "gate_precision": 0.87,
                "gate_recall": 0.61,
                "activity_ratio": 0.93,
                "all_hold_collapse": False,
                "all_trade_collapse": False,
                "early_stopping": False,
                "updated_at": "2026-08-02T14:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )

    response = api.get(f"/api/studio/jobs/{created['id']}/behavior-cloning/progress")

    assert response.status_code == 200
    payload = response.json()
    assert payload["phase"] == "training"
    assert payload["epoch"] == 9
    assert payload["totalEpochs"] == 45
    assert payload["percent"] == 20.0
    assert payload["gatePrecision"] == 0.87
    assert payload["gateRecall"] == 0.61
    assert payload["fold"] == "fold-000"
    assert payload["configuration"] == "configuration-000"
    assert payload["seed"] == 7


def test_behavior_cloning_progress_infers_preparing_from_teacher(
    tmp_path: Path,
) -> None:
    api, _, catalog, _ = client(tmp_path)
    created = api.post(
        "/api/studio/jobs/training",
        json=request(catalog, run_id="bc-preparing").model_dump(by_alias=True),
    ).json()
    manifest = (
        tmp_path
        / "research"
        / ".staging"
        / "bc-preparing"
        / "fold-000"
        / "configuration-000"
        / "seed-3"
        / "teacher"
        / "manifest.json"
    )
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}", encoding="utf-8")

    response = api.get(f"/api/studio/jobs/{created['id']}/behavior-cloning/progress")

    assert response.status_code == 200
    assert response.json()["phase"] == "preparing"
    assert response.json()["seed"] == 3
