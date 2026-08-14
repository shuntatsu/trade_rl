from __future__ import annotations

import json
import os
from pathlib import Path
from uuid import UUID

from trade_rl.telemetry.training import TrainingTelemetryWriter

from .helpers import (
    telemetry_record as record,
    telemetry_stream_path as stream_path,
)
from .support import request, studio_client as client


def test_telemetry_status_and_cursor_page_are_scoped_to_known_job(
    tmp_path: Path,
) -> None:
    api, _, catalog, _ = client(tmp_path)
    created = api.post(
        "/api/studio/jobs/training",
        json=request(catalog, run_id="live-001").model_dump(by_alias=True),
    ).json()
    stream = stream_path(tmp_path, "live-001", 7)
    with TrainingTelemetryWriter(stream, flush_every=1) as writer:
        writer.append(record(1))
        writer.append(record(2))

    status = api.get(f"/api/studio/jobs/{created['id']}/telemetry/status")
    page = api.get(
        f"/api/studio/jobs/{created['id']}/telemetry/events",
        params={"after_sequence": 1, "limit": 10},
    )

    assert status.status_code == 200
    status_payload = status.json()
    generation = status_payload.pop("streamGeneration")
    assert str(UUID(generation)) == generation
    assert status_payload == {
        "available": True,
        "selectedSeed": 7,
        "availableSeeds": [7],
        "recordCount": 2,
        "lastSequence": 2,
        "malformedLines": 0,
        "sizeBytes": stream.stat().st_size,
        "source": "research/.staging/live-001/seed-7/telemetry/training-telemetry.jsonl",
    }
    assert page.status_code == 200
    assert page.json()["seed"] == 7
    assert [item["sequence"] for item in page.json()["items"]] == [2]
    assert page.json()["nextSequence"] == 2
    assert page.json()["truncated"] is False
    assert page.json()["streamGeneration"] == generation
    assert page.json()["resetRequired"] is False


def test_telemetry_can_select_independent_seed_streams(tmp_path: Path) -> None:
    api, _, catalog, _ = client(tmp_path)
    created = api.post(
        "/api/studio/jobs/training",
        json=request(catalog, run_id="live-multi").model_dump(by_alias=True),
    ).json()
    for seed, count in ((3, 2), (11, 3)):
        with TrainingTelemetryWriter(
            stream_path(tmp_path, "live-multi", seed), flush_every=1
        ) as writer:
            for sequence in range(1, count + 1):
                writer.append(record(sequence, seed=seed))

    default_status = api.get(
        f"/api/studio/jobs/{created['id']}/telemetry/status"
    ).json()
    selected_status = api.get(
        f"/api/studio/jobs/{created['id']}/telemetry/status",
        params={"seed": 11},
    ).json()
    selected_page = api.get(
        f"/api/studio/jobs/{created['id']}/telemetry/events",
        params={"seed": 11, "after_sequence": 1, "limit": 10},
    ).json()

    assert default_status["selectedSeed"] == 3
    assert default_status["availableSeeds"] == [3, 11]
    assert default_status["recordCount"] == 2
    assert selected_status["selectedSeed"] == 11
    assert selected_status["recordCount"] == 3
    assert selected_page["seed"] == 11
    assert [item["seed"] for item in selected_page["items"]] == [11, 11]
    assert [item["sequence"] for item in selected_page["items"]] == [2, 3]


def test_telemetry_reports_unavailable_and_rejects_unknown_job(tmp_path: Path) -> None:
    api, _, catalog, _ = client(tmp_path)
    created = api.post(
        "/api/studio/jobs/training",
        json=request(catalog, run_id="live-empty").model_dump(by_alias=True),
    ).json()

    unavailable = api.get(f"/api/studio/jobs/{created['id']}/telemetry/status")
    missing = api.get("/api/studio/jobs/missing/telemetry/status")

    assert unavailable.status_code == 200
    assert unavailable.json()["available"] is False
    assert unavailable.json()["selectedSeed"] is None
    assert unavailable.json()["availableSeeds"] == []
    assert unavailable.json()["source"] is None
    assert unavailable.json()["streamGeneration"] is None
    assert missing.status_code == 404
    assert missing.json()["detail"]["code"] == "resource_not_found"


def test_telemetry_rejects_multiple_streams_for_one_seed(tmp_path: Path) -> None:
    api, _, catalog, _ = client(tmp_path)
    created = api.post(
        "/api/studio/jobs/training",
        json=request(catalog, run_id="live-duplicate").model_dump(by_alias=True),
    ).json()
    for namespace in (".staging", "runs"):
        with TrainingTelemetryWriter(
            stream_path(tmp_path, "live-duplicate", 7, namespace=namespace),
            flush_every=1,
        ) as writer:
            writer.append(record(1))

    response = api.get(f"/api/studio/jobs/{created['id']}/telemetry/status")

    assert response.status_code != 200
    assert response.json()["detail"]["code"] == "artifact_invalid"


def test_old_stream_generation_requests_reset_without_returning_records(
    tmp_path: Path,
) -> None:
    api, _, catalog, _ = client(tmp_path)
    created = api.post(
        "/api/studio/jobs/training",
        json=request(catalog, run_id="live-reset").model_dump(by_alias=True),
    ).json()
    stream = stream_path(tmp_path, "live-reset", 7)
    with TrainingTelemetryWriter(stream, flush_every=1) as writer:
        writer.append(record(1))
        writer.append(record(2))
    old_generation = api.get(
        f"/api/studio/jobs/{created['id']}/telemetry/status"
    ).json()["streamGeneration"]

    replacement = tmp_path / "replacement.jsonl"
    replacement.write_text(
        json.dumps(record(1).to_json_dict(), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(replacement, stream)

    response = api.get(
        f"/api/studio/jobs/{created['id']}/telemetry/events",
        params={
            "seed": 7,
            "after_sequence": 2,
            "limit": 10,
            "stream_generation": old_generation,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["items"] == []
    assert payload["nextSequence"] == 0
    assert payload["resetRequired"] is True
    assert payload["streamGeneration"] != old_generation
    assert str(UUID(payload["streamGeneration"])) == payload["streamGeneration"]


def test_telemetry_events_reject_invalid_stream_generation(tmp_path: Path) -> None:
    api, _, catalog, _ = client(tmp_path)
    created = api.post(
        "/api/studio/jobs/training",
        json=request(catalog, run_id="live-invalid-generation").model_dump(
            by_alias=True
        ),
    ).json()

    response = api.get(
        f"/api/studio/jobs/{created['id']}/telemetry/events",
        params={"stream_generation": "not-a-uuid"},
    )

    assert response.status_code == 422
