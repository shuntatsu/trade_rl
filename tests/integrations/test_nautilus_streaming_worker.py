from __future__ import annotations

import json
from decimal import Decimal

import pytest

pytest.importorskip("nautilus_trader")

from trade_rl.integrations.nautilus.event_projection import SourceBar
from trade_rl.integrations.nautilus.historical_execution import (
    NautilusHistoricalTargetInterval,
)
from trade_rl.integrations.nautilus.historical_streaming import (
    NautilusHistoricalStreamingWorker,
)
from trade_rl.integrations.nautilus.historical_subprocess import (
    run_historical_target_intervals_subprocess,
)

_HOUR_NS = 60 * 60 * 1_000_000_000


def _flat_bar(open_ns: int) -> SourceBar:
    return SourceBar(
        open_ns=open_ns,
        close_ns=open_ns + _HOUR_NS,
        open_price=100.0,
        high_price=100.0,
        low_price=100.0,
        close_price=100.0,
        mark_price=100.0,
        index_price=100.0,
    )


def _intervals(
    targets: tuple[float, ...],
) -> tuple[NautilusHistoricalTargetInterval, ...]:
    return tuple(
        NautilusHistoricalTargetInterval(
            sequence=sequence,
            target_exposure=target,
            allocated_equity=1_000.0,
            source_bars=(_flat_bar((sequence - 1) * _HOUR_NS),),
        )
        for sequence, target in enumerate(targets, start=1)
    )


class _FakeProcess:
    def __init__(self) -> None:
        self.alive = True
        self.exitcode: int | None = None
        self.terminate_calls = 0
        self.join_calls = 0

    def is_alive(self) -> bool:
        return self.alive

    def terminate(self) -> None:
        self.terminate_calls += 1
        self.alive = False
        self.exitcode = -15

    def join(self, timeout: float | None = None) -> None:
        del timeout
        self.join_calls += 1


class _ResetConnection:
    def __init__(self) -> None:
        self.close_calls = 0

    def send_bytes(self, payload: bytes) -> None:
        del payload
        raise ConnectionResetError("child channel reset")

    def close(self) -> None:
        self.close_calls += 1


class _UnexpectedAckConnection:
    def __init__(self) -> None:
        self.close_calls = 0

    def send_bytes(self, payload: bytes) -> None:
        del payload

    def poll(self, timeout: float) -> bool:
        del timeout
        return True

    def recv_bytes(self) -> bytes:
        return json.dumps({"ok": True, "event": "unexpected"}).encode("utf-8")

    def close(self) -> None:
        self.close_calls += 1


def _uninitialized_worker(
    *,
    process: _FakeProcess,
    connection: _ResetConnection | _UnexpectedAckConnection,
) -> NautilusHistoricalStreamingWorker:
    worker = object.__new__(NautilusHistoricalStreamingWorker)
    worker._process = process
    worker._connection = connection
    worker._timeout_seconds = 0.01
    worker._closed = False
    worker._worker_pid = 1
    return worker


@pytest.mark.nautilus
@pytest.mark.parametrize(
    "targets",
    [
        (0.1, 0.0),
        (0.1, -0.1, 0.0),
        (0.1, 0.2, 0.05, 0.0),
    ],
    ids=("round-trip", "safe-sign-flip", "same-side-changes"),
)
def test_streaming_worker_matches_full_prefix_execution_in_one_child(
    targets: tuple[float, ...],
) -> None:
    intervals = _intervals(targets)
    expected = run_historical_target_intervals_subprocess(
        intervals,
        starting_balance=Decimal("1000"),
        no_trade_band=0.0,
    )

    with NautilusHistoricalStreamingWorker(
        starting_balance=Decimal("1000"),
        no_trade_band=0.0,
    ) as worker:
        results = tuple(worker.execute(interval) for interval in intervals)

        assert {result.worker_pid for result in results} == {worker.worker_pid}
        assert results[0].execution.terminal_position_lots == 1000
        assert results[-1].execution == expected.execution
        assert results[-1].execution.runtime_version == "1.230.0"
        assert results[-1].execution.terminal_position_lots == 0
        assert results[-1].execution.terminal_open_orders == 0


@pytest.mark.nautilus
def test_close_tolerates_transport_reset_after_child_failure_and_is_idempotent() -> (
    None
):
    process = _FakeProcess()
    connection = _ResetConnection()
    worker = _uninitialized_worker(process=process, connection=connection)

    worker.close()
    worker.close()

    assert process.terminate_calls == 1
    assert process.join_calls == 1
    assert connection.close_calls == 1


@pytest.mark.nautilus
def test_close_still_rejects_invalid_acknowledgement_from_live_child() -> None:
    process = _FakeProcess()
    connection = _UnexpectedAckConnection()
    worker = _uninitialized_worker(process=process, connection=connection)

    with pytest.raises(RuntimeError, match="unexpected event"):
        worker.close()

    assert process.terminate_calls == 1
    assert connection.close_calls == 1
