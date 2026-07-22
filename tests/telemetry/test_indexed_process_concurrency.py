from __future__ import annotations

import json
import multiprocessing
import os
import time
from pathlib import Path
from queue import Empty
from typing import Any

import pytest

from trade_rl.telemetry import (
    TrainingTelemetryRecord,
    TrainingTelemetryWriter,
    read_training_telemetry,
    training_telemetry_status,
)


def _record(sequence: int) -> TrainingTelemetryRecord:
    return TrainingTelemetryRecord(
        sequence=sequence,
        recorded_at="2026-07-22T09:00:00+00:00",
        global_step=sequence * 32,
        environment_step=sequence,
        seed=7,
        environment_id=0,
        event_type="rollout",
        market_index=100 + sequence,
        market_time="2026-07-22T08:55:00.000000000",
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


def _duplicate_writer_worker(
    path_value: str,
    ready: Any,
    start: Any,
    results: Any,
) -> None:
    writer: TrainingTelemetryWriter | None = None
    try:
        writer = TrainingTelemetryWriter(Path(path_value), flush_every=1)
        ready.put(("ready", os.getpid()))
        if not start.wait(timeout=20.0):
            results.put(("error", "TimeoutError", "start event was not released"))
            return
        writer.append(_record(1))
        writer.flush()
        results.put(("ok", os.getpid(), ""))
    except BaseException as error:  # pragma: no cover - asserted in parent process
        results.put(("error", type(error).__name__, str(error)))
    finally:
        if writer is not None:
            writer.close()


def _ordered_writer_worker(
    path_value: str,
    start: Any,
    results: Any,
    count: int,
) -> None:
    try:
        if not start.wait(timeout=20.0):
            raise TimeoutError("start event was not released")
        with TrainingTelemetryWriter(Path(path_value), flush_every=1) as writer:
            for sequence in range(1, count + 1):
                writer.append(_record(sequence))
                if sequence % 4 == 0:
                    time.sleep(0.001)
        results.put(("writer", "ok", count))
    except BaseException as error:  # pragma: no cover - asserted in parent process
        results.put(("writer", type(error).__name__, str(error)))


def _reader_worker(
    path_value: str,
    start: Any,
    results: Any,
    iterations: int,
) -> None:
    path = Path(path_value)
    try:
        if not start.wait(timeout=20.0):
            raise TimeoutError("start event was not released")
        last_seen = 0
        for _ in range(iterations):
            status = training_telemetry_status(path)
            page = read_training_telemetry(path, after_sequence=0, limit=2_000)
            sequences = [record.sequence for record in page.items]
            if sequences != sorted(set(sequences)):
                raise AssertionError("reader observed duplicate or unordered sequences")
            if status.available and status.last_sequence < last_seen:
                raise AssertionError("status last_sequence regressed")
            last_seen = max(last_seen, status.last_sequence)
            time.sleep(0.001)
        results.put(("reader", "ok", last_seen))
    except BaseException as error:  # pragma: no cover - asserted in parent process
        results.put(("reader", type(error).__name__, str(error)))


def _context() -> multiprocessing.context.SpawnContext:
    return multiprocessing.get_context("spawn")


def _join(processes: list[multiprocessing.Process], *, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    for process in processes:
        process.join(timeout=max(0.0, deadline - time.monotonic()))
    for process in processes:
        if process.is_alive():
            process.terminate()
            process.join(timeout=5.0)
            pytest.fail(f"process {process.pid} did not terminate")
        assert process.exitcode == 0


def _queue_items(queue: Any, expected: int) -> list[tuple[object, ...]]:
    items: list[tuple[object, ...]] = []
    for _ in range(expected):
        try:
            items.append(tuple(queue.get(timeout=10.0)))
        except Empty:
            pytest.fail(f"expected {expected} process results, received {len(items)}")
    return items


def test_duplicate_sequence_race_allows_exactly_one_process_append(
    tmp_path: Path,
) -> None:
    path = tmp_path / "training-telemetry.jsonl"
    context = _context()
    ready = context.Queue()
    start = context.Event()
    results = context.Queue()
    processes = [
        context.Process(
            target=_duplicate_writer_worker,
            args=(str(path), ready, start, results),
        )
        for _ in range(2)
    ]

    for process in processes:
        process.start()
    ready_items = _queue_items(ready, 2)
    assert all(item[0] == "ready" for item in ready_items)
    start.set()
    _join(processes)
    outcomes = _queue_items(results, 2)

    successful = [item for item in outcomes if item[0] == "ok"]
    rejected = [item for item in outcomes if item[0] == "error"]
    assert len(successful) == 1
    assert len(rejected) == 1
    assert rejected[0][1] == "ValueError"
    assert "strictly increase" in str(rejected[0][2])

    page = read_training_telemetry(path, after_sequence=0, limit=10)
    status = training_telemetry_status(path)
    assert [item.sequence for item in page.items] == [1]
    assert status.record_count == 1
    assert status.last_sequence == 1


def test_append_fails_closed_when_stream_has_incomplete_tail(tmp_path: Path) -> None:
    path = tmp_path / "training-telemetry.jsonl"
    path.write_bytes(
        (json.dumps(_record(1).to_json_dict(), sort_keys=True) + "\n").encode("utf-8")
        + b'{"schema_version":"training_telemetry_v1"'
    )

    with TrainingTelemetryWriter(path, flush_every=1) as writer:
        with pytest.raises(RuntimeError, match="incomplete trailing record"):
            writer.append(_record(2))

    page = read_training_telemetry(path, after_sequence=0, limit=10)
    assert [item.sequence for item in page.items] == [1]


def test_concurrent_readers_observe_consistent_index_snapshots(tmp_path: Path) -> None:
    path = tmp_path / "training-telemetry.jsonl"
    context = _context()
    start = context.Event()
    results = context.Queue()
    count = 96
    readers = 4
    processes = [
        context.Process(
            target=_ordered_writer_worker,
            args=(str(path), start, results, count),
        ),
        *[
            context.Process(
                target=_reader_worker,
                args=(str(path), start, results, 80),
            )
            for _ in range(readers)
        ],
    ]

    for process in processes:
        process.start()
    start.set()
    _join(processes, timeout=60.0)
    outcomes = _queue_items(results, readers + 1)

    assert all(item[1] == "ok" for item in outcomes), outcomes
    status = training_telemetry_status(path)
    page = read_training_telemetry(path, after_sequence=0, limit=2_000)
    index_path = path.with_name(f"{path.name}.index.json")
    index_payload = json.loads(index_path.read_text(encoding="utf-8"))

    assert status.record_count == count
    assert status.last_sequence == count
    assert [item.sequence for item in page.items] == list(range(1, count + 1))
    assert index_payload["record_count"] == count
    assert index_payload["last_sequence"] == count
    assert index_payload["indexed_size"] == path.stat().st_size


@pytest.mark.skipif(os.name == "nt", reason="Windows prevents replacing an open file")
def test_append_fails_closed_when_stream_identity_is_replaced(tmp_path: Path) -> None:
    path = tmp_path / "training-telemetry.jsonl"
    replacement = tmp_path / "replacement.jsonl"
    writer = TrainingTelemetryWriter(path, flush_every=1)
    try:
        writer.append(_record(1))
        replacement.write_text(
            json.dumps(_record(1).to_json_dict(), sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(replacement, path)

        with pytest.raises(RuntimeError, match="stream identity changed"):
            writer.append(_record(2))
    finally:
        writer.close()

    page = read_training_telemetry(path, after_sequence=0, limit=10)
    assert [item.sequence for item in page.items] == [1]
