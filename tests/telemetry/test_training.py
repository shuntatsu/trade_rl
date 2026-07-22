from __future__ import annotations

import json
import os
import threading
from pathlib import Path

import pytest

from trade_rl.telemetry import (
    TrainingTelemetryRecord,
    TrainingTelemetryWriter,
    read_training_telemetry,
    training_telemetry_status,
)
from trade_rl.telemetry import indexed_training as indexed


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
    assert next_payload == rebuilt_payload


def _required_generation(value: str | None) -> str:
    assert value is not None
    return value


def test_generation_remains_stable_across_normal_append(tmp_path: Path) -> None:
    path = tmp_path / "training-telemetry.jsonl"
    with TrainingTelemetryWriter(path, flush_every=1) as writer:
        writer.append(record(1))
    first = training_telemetry_status(path)

    with TrainingTelemetryWriter(path, flush_every=1) as writer:
        writer.append(record(2))
    second = training_telemetry_status(path)

    assert _required_generation(first.stream_generation) == second.stream_generation


def test_old_generation_requests_reset_after_stream_replacement(tmp_path: Path) -> None:
    path = tmp_path / "training-telemetry.jsonl"
    with TrainingTelemetryWriter(path, flush_every=1) as writer:
        writer.append(record(1))
        writer.append(record(2))
    old_generation = _required_generation(
        training_telemetry_status(path).stream_generation
    )

    replacement = tmp_path / "replacement.jsonl"
    replacement.write_text(
        json.dumps(record(1).to_json_dict(), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(replacement, path)

    page = read_training_telemetry(
        path,
        after_sequence=2,
        limit=10,
        expected_generation=old_generation,
    )

    assert page.items == ()
    assert page.next_sequence == 0
    assert page.reset_required is True
    assert page.stream_generation not in (None, old_generation)


def test_index_loss_rotates_generation_and_invalidates_cursor(tmp_path: Path) -> None:
    path = tmp_path / "training-telemetry.jsonl"
    with TrainingTelemetryWriter(path, flush_every=1) as writer:
        writer.append(record(1))
    first = training_telemetry_status(path)
    old_generation = _required_generation(first.stream_generation)
    path.with_name(f"{path.name}.index.json").unlink()

    page = read_training_telemetry(
        path,
        after_sequence=1,
        limit=10,
        expected_generation=old_generation,
    )

    assert page.items == ()
    assert page.reset_required is True
    assert page.stream_generation not in (None, old_generation)


def test_expected_generation_requires_canonical_uuid(tmp_path: Path) -> None:
    path = tmp_path / "training-telemetry.jsonl"
    with TrainingTelemetryWriter(path, flush_every=1) as writer:
        writer.append(record(1))

    with pytest.raises(ValueError, match="expected_generation"):
        read_training_telemetry(
            path,
            after_sequence=0,
            limit=10,
            expected_generation="not-a-generation",
        )


def test_no_growth_polls_do_not_rewrite_index(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "training-telemetry.jsonl"
    with TrainingTelemetryWriter(path, flush_every=1) as writer:
        for sequence in range(1, 70):
            writer.append(record(sequence))
    assert training_telemetry_status(path).last_sequence == 69

    writes: list[int] = []

    def fail_on_write(_path: Path, _index: object) -> None:
        writes.append(1)

    monkeypatch.setattr(indexed, "_write_index", fail_on_write)

    status = training_telemetry_status(path)
    page = read_training_telemetry(path, after_sequence=64, limit=10)

    assert status.last_sequence == 69
    assert [item.sequence for item in page.items] == [65, 66, 67, 68, 69]
    assert writes == []


def test_page_parsing_does_not_hold_append_process_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "training-telemetry.jsonl"
    with TrainingTelemetryWriter(path, flush_every=1) as writer:
        for sequence in range(1, 131):
            writer.append(record(sequence))
    assert training_telemetry_status(path).last_sequence == 130

    original_parse = indexed._parse_record
    parse_started = threading.Event()
    release_parse = threading.Event()
    reader_done = threading.Event()
    writer_done = threading.Event()
    errors: list[BaseException] = []

    def blocking_parse(raw_line: bytes) -> TrainingTelemetryRecord:
        if not parse_started.is_set():
            parse_started.set()
            if not release_parse.wait(timeout=10.0):
                raise TimeoutError("page parse was not released")
        return original_parse(raw_line)

    monkeypatch.setattr(indexed, "_parse_record", blocking_parse)

    def read_page() -> None:
        try:
            read_training_telemetry(path, after_sequence=64, limit=20)
        except BaseException as error:  # pragma: no cover - asserted below
            errors.append(error)
        finally:
            reader_done.set()

    def append_record() -> None:
        try:
            with TrainingTelemetryWriter(path, flush_every=1) as writer:
                writer.append(record(131))
        except BaseException as error:  # pragma: no cover - asserted below
            errors.append(error)
        finally:
            writer_done.set()

    reader = threading.Thread(target=read_page)
    reader.start()
    assert parse_started.wait(timeout=10.0)

    writer = threading.Thread(target=append_record)
    writer.start()
    assert writer_done.wait(timeout=2.0), "writer remained blocked by page parsing"

    release_parse.set()
    reader.join(timeout=10.0)
    writer.join(timeout=10.0)

    assert reader_done.is_set()
    assert errors == []
    assert training_telemetry_status(path).last_sequence == 131


def test_near_tail_page_parses_at_most_one_checkpoint_stride(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "training-telemetry.jsonl"
    with TrainingTelemetryWriter(path, flush_every=256) as writer:
        for sequence in range(1, 4_097):
            writer.append(record(sequence))
    assert training_telemetry_status(path).last_sequence == 4_096

    original_parse = indexed._parse_record
    parsed = 0

    def counted_parse(raw_line: bytes) -> TrainingTelemetryRecord:
        nonlocal parsed
        parsed += 1
        return original_parse(raw_line)

    monkeypatch.setattr(indexed, "_parse_record", counted_parse)
    page = read_training_telemetry(path, after_sequence=4_080, limit=20)

    assert [item.sequence for item in page.items] == list(range(4_081, 4_097))
    assert parsed <= 80
