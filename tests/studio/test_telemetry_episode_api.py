from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from .test_api import client
from .test_jobs import request
from .test_telemetry_api import record, stream_path


def test_telemetry_api_exposes_explicit_and_legacy_episode_identity(
    tmp_path: Path,
) -> None:
    api, _, catalog, _ = client(tmp_path)
    created = api.post(
        "/api/studio/jobs/training",
        json=request(catalog, run_id="live-episode-api").model_dump(by_alias=True),
    ).json()
    stream = stream_path(tmp_path, "live-episode-api", 7)
    stream.parent.mkdir(parents=True, exist_ok=True)

    explicit = replace(record(1), episode_id=5).to_json_dict()
    legacy = record(2).to_json_dict()
    legacy.pop("episode_id")
    stream.write_text(
        "\n".join(
            (
                json.dumps(explicit, sort_keys=True, separators=(",", ":")),
                json.dumps(legacy, sort_keys=True, separators=(",", ":")),
            )
        )
        + "\n",
        encoding="utf-8",
    )

    response = api.get(
        f"/api/studio/jobs/{created['id']}/telemetry/events",
        params={"seed": 7, "limit": 10},
    )

    assert response.status_code == 200
    items = response.json()["items"]
    assert [item["sequence"] for item in items] == [1, 2]
    assert [item["episodeId"] for item in items] == [5, None]
