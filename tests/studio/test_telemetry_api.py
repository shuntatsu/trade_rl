from __future__ import annotations

from pathlib import Path

from trade_rl.telemetry.training import TrainingTelemetryRecord, TrainingTelemetryWriter

from .test_api import client
from .test_jobs import request


def record(sequence: int) -> TrainingTelemetryRecord:
    return TrainingTelemetryRecord(
        sequence=sequence,
        recorded_at="2026-07-21T08:00:00+00:00",
        global_step=sequence * 32,
        environment_step=sequence,
        seed=7,
        environment_id=0,
        event_type="rollout",
        market_index=100 + sequence,
        market_time="2026-07-21T08:00:00.000000000",
        symbol="BTCUSDT",
        open=67_500.0,
        high=67_900.0,
        low=67_400.0,
        close=67_842.3,
        action=(0.4,),
        executed_target=(0.4,),
        weights_before=(0.2,),
        weights_after=(0.4,),
        portfolio_value=101_342.85,
        baseline_portfolio_value=100_400.0,
        reward=0.214,
        drawdown=0.0086,
        interval_cost=4.25,
        interval_return=0.0012,
        risk_reasons=(),
        emergency_deleverage=False,
        terminated=False,
        truncated=False,
    )


def test_telemetry_status_and_cursor_page_are_scoped_to_known_job(
    tmp_path: Path,
) -> None:
    api, _, catalog, _ = client(tmp_path)
    created = api.post(
        "/api/studio/jobs/training",
        json=request(catalog, run_id="live-001").model_dump(by_alias=True),
    ).json()
    stream = (
        tmp_path
        / "research"
        / ".staging"
        / "live-001"
        / "seed-7"
        / "telemetry"
        / "training-telemetry.jsonl"
    )
    with TrainingTelemetryWriter(stream, flush_every=1) as writer:
        writer.append(record(1))
        writer.append(record(2))

    status = api.get(f"/api/studio/jobs/{created['id']}/telemetry/status")
    page = api.get(
        f"/api/studio/jobs/{created['id']}/telemetry/events",
        params={"after_sequence": 1, "limit": 10},
    )

    assert status.status_code == 200
    assert status.json() == {
        "available": True,
        "recordCount": 2,
        "lastSequence": 2,
        "malformedLines": 0,
        "sizeBytes": stream.stat().st_size,
        "source": "research/.staging/live-001/seed-7/telemetry/training-telemetry.jsonl",
    }
    assert page.status_code == 200
    assert [item["sequence"] for item in page.json()["items"]] == [2]
    assert page.json()["nextSequence"] == 2
    assert page.json()["truncated"] is False


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
    assert unavailable.json()["source"] is None
    assert missing.status_code == 404
    assert missing.json()["detail"]["code"] == "resource_not_found"
