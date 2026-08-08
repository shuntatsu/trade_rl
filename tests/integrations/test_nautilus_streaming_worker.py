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
