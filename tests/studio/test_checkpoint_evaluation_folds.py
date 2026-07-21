from __future__ import annotations

import json
from pathlib import Path

from .test_api import client
from .test_checkpoint_evaluations_api import checkpoint_payload
from .test_jobs import request


def path(tmp_path: Path, run_id: str, fold: str) -> Path:
    return (
        tmp_path
        / "research"
        / ".staging"
        / run_id
        / fold
        / "candidates"
        / "residual"
        / "checkpoint-selection.json"
    )


def test_checkpoint_evaluations_keep_fold_identity_and_lexical_order(
    tmp_path: Path,
) -> None:
    api, _, catalog, _ = client(tmp_path)
    created = api.post(
        "/api/studio/jobs/training",
        json=request(catalog, run_id="live-folds").model_dump(by_alias=True),
    ).json()
    for fold in ("fold-001", "fold-000"):
        selection = path(tmp_path, "live-folds", fold)
        selection.parent.mkdir(parents=True)
        selection.write_text(json.dumps(checkpoint_payload()), encoding="utf-8")

    response = api.get(f"/api/studio/jobs/{created['id']}/checkpoint-evaluations")

    assert response.status_code == 200
    seed_three = [item for item in response.json()["items"] if item["seed"] == 3]
    assert [item["fold"] for item in seed_three] == ["fold-000", "fold-001"]
    assert all(item["configuration"] == "residual" for item in seed_three)
