from __future__ import annotations

import json
from pathlib import Path

import pytest

from trade_rl.telemetry.training import (
    TrainingTelemetryRecord,
    TrainingTelemetryWriter,
    read_training_telemetry,
    training_telemetry_status,
)


def record(sequence: int, *, event_type: str = "rollout") -> TrainingTelemetryRecord:
    return TrainingTelemetryRecord(
        sequence=sequence,
        recorded_at="2026-07-21T08:00:00+00:00",
        global_step=sequence * 32,
        environment_step=sequence,
        seed=7,
        environment_id=0,
        event_type=event_type,
        market_index=100 + sequence,
        market_time="2026-07-21T07:55:00.000000000",
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


def test_writer_and_cursor_reader_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "training-telemetry.jsonl"
    with TrainingTelemetryWriter(path, flush_every=1) as writer:
        writer.append(record(1))
        writer.append(record(2, event_type="position"))

    page = read_training_telemetry(path, after_sequence=1, limit=10)

    assert [item.sequence for item in page.items] == [2]
    assert page.items[0].event_type == "position"
    assert page.next_sequence == 2
    assert page.truncated is False
    assert page.malformed_lines == 0
    assert page.sequence_gaps == ()


def test_writer_rejects_non_monotonic_sequence(tmp_path: Path) -> None:
    path = tmp_path / "training-telemetry.jsonl"
    with TrainingTelemetryWriter(path, flush_every=1) as writer:
        writer.append(record(2))
        with pytest.raises(ValueError, match="strictly increase"):
            writer.append(record(2))


def test_reader_reports_malformed_lines_and_sequence_gaps(tmp_path: Path) -> None:
    path = tmp_path / "training-telemetry.jsonl"
    path.write_text(
        "\n".join(
            (
                json.dumps(record(1).to_json_dict()),
                "{broken",
                json.dumps(record(4).to_json_dict()),
            )
        )
        + "\n",
        encoding="utf-8",
    )

    page = read_training_telemetry(path, after_sequence=0, limit=10)

    assert [item.sequence for item in page.items] == [1, 4]
    assert page.malformed_lines == 1
    assert page.sequence_gaps == ((2, 3),)


def test_status_is_unavailable_before_stream_exists(tmp_path: Path) -> None:
    status = training_telemetry_status(tmp_path / "missing.jsonl")

    assert status.available is False
    assert status.record_count == 0
    assert status.last_sequence == 0


def test_record_rejects_non_finite_market_values() -> None:
    with pytest.raises(ValueError, match="finite"):
        TrainingTelemetryRecord(**{**record(1).__dict__, "close": float("nan")})
