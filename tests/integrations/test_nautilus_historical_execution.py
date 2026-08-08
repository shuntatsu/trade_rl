from __future__ import annotations

from decimal import Decimal

import pytest

pytest.importorskip("nautilus_trader")

from trade_rl.integrations.nautilus.event_projection import SourceBar
from trade_rl.integrations.nautilus.historical_execution import (
    NautilusHistoricalTargetInterval,
    run_historical_target_intervals,
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


@pytest.mark.nautilus
def test_historical_target_intervals_execute_open_then_reduce_to_flat() -> None:
    intervals = (
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

    result = run_historical_target_intervals(
        intervals,
        starting_balance=Decimal("1000"),
        no_trade_band=0.0,
    )

    assert [fill.quantity_lots for fill in result.fills] == [1000, -1000]
    assert [fill.position_lots for fill in result.fills] == [1000, 0]
    assert result.terminal_position_lots == 0
    assert result.terminal_open_orders == 0
