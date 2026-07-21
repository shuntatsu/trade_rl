from __future__ import annotations

import json
import math
from pathlib import Path

from .test_api import client
from .test_jobs import request


def checkpoint_payload() -> dict[str, object]:
    return {
        "schema_version": "checkpoint_selection_v2_seed_aware",
        "checkpoint_range": [100, 120],
        "candidates": [
            {
                "evaluation_digest": "a" * 64,
                "policy_digest": "b" * 64,
                "score": math.log1p(0.05),
                "seed": 3,
            },
            {
                "evaluation_digest": "c" * 64,
                "policy_digest": "d" * 64,
                "score": math.log1p(0.02),
                "seed": 11,
            },
        ],
        "seed_finalists": [
            {
                "checkpoint_evaluation_digest": "a" * 64,
                "checkpoint_score": math.log1p(0.05),
                "policy_digest": "b" * 64,
                "seed": 3,
            }
        ],
    }


def selection_path(tmp_path: Path, run_id: str) -> Path:
    return (
        tmp_path
        / "research"
        / ".staging"
        / run_id
        / "fold-000"
        / "candidates"
        / "residual"
        / "checkpoint-selection.json"
    )


def test_checkpoint_evaluations_are_identity_checked_and_seed_aware(
    tmp_path: Path,
) -> None:
    api, _, catalog, _ = client(tmp_path)
    created = api.post(
        "/api/studio/jobs/training",
        json=request(catalog, run_id="live-eval").model_dump(by_alias=True),
    ).json()
    path = selection_path(tmp_path, "live-eval")
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(checkpoint_payload()), encoding="utf-8")

    response = api.get(f"/api/studio/jobs/{created['id']}/checkpoint-evaluations")

    assert response.status_code == 200
    payload = response.json()
    assert payload["available"] is True
    assert payload["productionStatus"] == "NO-GO"
    assert [item["seed"] for item in payload["items"]] == [3, 11]
    assert payload["items"][0]["finalist"] is True
    assert payload["items"][1]["finalist"] is False
    assert payload["items"][0]["totalReturn"] == 0.05
    assert payload["items"][0]["checkpointRange"] == [100, 120]
    assert payload["items"][0]["source"].endswith("checkpoint-selection.json")


def test_checkpoint_evaluations_report_absent_and_reject_invalid_evidence(
    tmp_path: Path,
) -> None:
    api, _, catalog, _ = client(tmp_path)
    empty = api.post(
        "/api/studio/jobs/training",
        json=request(catalog, run_id="eval-empty").model_dump(by_alias=True),
    ).json()
    invalid = api.post(
        "/api/studio/jobs/training",
        json=request(catalog, run_id="eval-invalid").model_dump(by_alias=True),
    ).json()
    invalid_path = selection_path(tmp_path, "eval-invalid")
    invalid_path.parent.mkdir(parents=True)
    payload = checkpoint_payload()
    candidates = payload["candidates"]
    assert isinstance(candidates, list)
    assert isinstance(candidates[0], dict)
    candidates[0]["evaluation_digest"] = "not-a-digest"
    invalid_path.write_text(json.dumps(payload), encoding="utf-8")

    absent = api.get(f"/api/studio/jobs/{empty['id']}/checkpoint-evaluations")
    rejected = api.get(f"/api/studio/jobs/{invalid['id']}/checkpoint-evaluations")

    assert absent.status_code == 200
    assert absent.json() == {
        "available": False,
        "items": [],
        "productionStatus": "NO-GO",
    }
    assert rejected.status_code == 422
    assert rejected.json()["detail"]["code"] == "artifact_invalid"
