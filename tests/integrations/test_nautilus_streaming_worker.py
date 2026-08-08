from __future__ import annotations

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


def _round_trip() -> tuple[NautilusHistoricalTargetInterval, ...]:
    return (
        NautilusHistoricalTargetInterval(
            sequence=1,
            target_exposure=0.1,
            allocated_equity=1_000.0,
            source_bars=(_flat_bar(0),),
        ),
        NautilusHistoricalTargetInterval(
            sequence=2,
            target_exposure=0.0,
            allocated_equity=1_000.0,
            source_bars=(_flat_bar(_HOUR_NS),),
        ),
    )


@pytest.mark.nautilus
def test_streaming_worker_matches_full_prefix_execution_in_one_child() -> None:
    intervals = _round_trip()
    expected = run_historical_target_intervals_subprocess(
        intervals,
        starting_balance=Decimal("1000"),
        no_trade_band=0.0,
    )

    with NautilusHistoricalStreamingWorker(
        starting_balance=Decimal("1000"),
        no_trade_band=0.0,
    ) as worker:
        first = worker.execute(intervals[0])
        second = worker.execute(intervals[1])

        assert first.worker_pid == second.worker_pid == worker.worker_pid
        assert first.execution.terminal_position_lots == 1000
        assert second.execution == expected.execution
        assert second.execution.runtime_version == "1.230.0"
        assert second.execution.terminal_position_lots == 0
        assert second.execution.terminal_open_orders == 0
