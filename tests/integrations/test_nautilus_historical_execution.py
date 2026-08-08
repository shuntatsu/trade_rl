from __future__ import annotations

from decimal import Decimal

import pytest

pytest.importorskip("nautilus_trader")

from trade_rl.integrations.nautilus.event_projection import SourceBar
from trade_rl.integrations.nautilus.historical_execution import (
    NautilusHistoricalTargetInterval,
    run_historical_target_intervals,
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


def _flat_round_trip_intervals() -> tuple[NautilusHistoricalTargetInterval, ...]:
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
def test_historical_targets_reconcile_sign_flip_and_capture_boundaries() -> None:
    intervals = (
        NautilusHistoricalTargetInterval(
            sequence=1,
            target_exposure=0.1,
            allocated_equity=1_000.0,
            source_bars=(_flat_bar(0),),
        ),
        NautilusHistoricalTargetInterval(
            sequence=2,
            target_exposure=-0.1,
            allocated_equity=1_000.0,
            source_bars=(_flat_bar(_HOUR_NS),),
        ),
        NautilusHistoricalTargetInterval(
            sequence=3,
            target_exposure=0.0,
            allocated_equity=1_000.0,
            source_bars=(_flat_bar(2 * _HOUR_NS),),
        ),
    )

    result = run_historical_target_intervals(
        intervals,
        snapshot_timestamps_ns=(_HOUR_NS, 2 * _HOUR_NS),
        starting_balance=Decimal("1000"),
        no_trade_band=0.0,
    )

    assert [fill.quantity_lots for fill in result.fills] == [
        1000,
        -1000,
        -1000,
        1000,
    ]
    assert [fill.position_lots for fill in result.fills] == [1000, 0, -1000, 0]
    assert [
        (snapshot.timestamp_ns, snapshot.signed_quantity)
        for snapshot in result.position_snapshots
    ] == [
        (_HOUR_NS, Decimal("1")),
        (2 * _HOUR_NS, Decimal("-1")),
    ]
    assert result.terminal_position_lots == 0
    assert result.terminal_open_orders == 0


@pytest.mark.nautilus
def test_historical_target_is_quantized_to_maintained_lot_increment() -> None:
    intervals = (
        NautilusHistoricalTargetInterval(
            sequence=1,
            target_exposure=-0.39736594653926315,
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

    result = run_historical_target_intervals_subprocess(
        intervals,
        starting_balance=Decimal("1000"),
        no_trade_band=0.0,
    ).execution

    assert [fill.quantity_lots for fill in result.fills] == [-3973, 3973]
    assert [fill.position_lots for fill in result.fills] == [-3973, 0]
    assert result.terminal_position_lots == 0
    assert result.terminal_open_orders == 0


@pytest.mark.nautilus
def test_historical_subprocess_runtime_uses_a_fresh_child_per_replay() -> None:
    intervals = _flat_round_trip_intervals()

    first = run_historical_target_intervals_subprocess(
        intervals,
        starting_balance=Decimal("1000"),
        no_trade_band=0.0,
    )
    second = run_historical_target_intervals_subprocess(
        intervals,
        starting_balance=Decimal("1000"),
        no_trade_band=0.0,
    )

    assert first.worker_pid != second.worker_pid
    assert first.execution == second.execution
    assert first.execution.runtime_version == "1.230.0"
    assert first.execution.terminal_position_lots == 0
    assert first.execution.terminal_open_orders == 0
