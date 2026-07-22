from __future__ import annotations

import json
from pathlib import Path

import pytest

from trade_rl.telemetry import (
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


@pytest.mark.parametrize(
    "field,value",
    (
        ("emergency_deleverage", "false"),
        ("terminated", 0),
        ("truncated", None),
    ),
)
def test_record_json_rejects_non_boolean_flags(field: str, value: object) -> None:
    payload = record(1).to_json_dict()
    payload[field] = value

    with pytest.raises(ValueError, match=field):
        TrainingTelemetryRecord.from_json_dict(payload)


def test_record_json_requires_all_boolean_flags() -> None:
    payload = record(1).to_json_dict()
    payload.pop("terminated")

    with pytest.raises(ValueError, match="terminated"):
        TrainingTelemetryRecord.from_json_dict(payload)


def test_second_poll_indexes_only_appended_bytes(tmp_path: Path) -> None:
    path = tmp_path / "training-telemetry.jsonl"
    with TrainingTelemetryWriter(path, flush_every=1) as writer:
        for sequence in range(1, 130):
            writer.append(record(sequence))

    first = read_training_telemetry(path, after_sequence=120, limit=20)
    indexed_size = path.stat().st_size
    assert [item.sequence for item in first.items] == list(range(121, 130))

    with TrainingTelemetryWriter(path, flush_every=1) as writer:
        writer.append(record(130))
    second = read_training_telemetry(path, after_sequence=129, limit=20)

    index_path = path.with_name(f"{path.name}.index.json")
    index_payload = json.loads(index_path.read_text(encoding="utf-8"))
    assert [item.sequence for item in second.items] == [130]
    assert index_payload["last_scan_start"] == indexed_size
    assert index_payload["indexed_size"] == path.stat().st_size


def test_index_rebuilds_after_stream_replacement(tmp_path: Path) -> None:
    path = tmp_path / "training-telemetry.jsonl"
    index_path = path.with_name(f"{path.name}.index.json")
    with TrainingTelemetryWriter(path, flush_every=1) as writer:
        writer.append(record(1))
        writer.append(record(2))
    assert training_telemetry_status(path).last_sequence == 2

    replacement = tmp_path / "replacement.jsonl"
    replacement.write_text(
        json.dumps(record(1).to_json_dict()) + "\n",
        encoding="utf-8",
    )
    replacement.replace(path)

    status = training_telemetry_status(path)
    rebuilt_payload = json.loads(index_path.read_text(encoding="utf-8"))
    page = read_training_telemetry(path, after_sequence=0, limit=10)
    next_payload = json.loads(index_path.read_text(encoding="utf-8"))

    assert status.record_count == 1
    assert status.last_sequence == 1
    assert rebuilt_payload["last_scan_start"] == 0
    assert [item.sequence for item in page.items] == [1]
    assert next_payload["last_scan_start"] == path.stat().st_size
