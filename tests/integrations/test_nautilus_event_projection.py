from __future__ import annotations

import pytest

from trade_rl.integrations.nautilus.event_projection import (
    TARGET_ACTIVATION_DELAY_NS,
    MarketPhase,
    SourceBar,
    project_bar_events,
)


def _bar() -> SourceBar:
    return SourceBar(
        open_ns=1_000_000,
        close_ns=901_000_000_000,
        open_price=100.0,
        high_price=110.0,
        low_price=95.0,
        close_price=105.0,
        mark_price=104.5,
        index_price=104.25,
    )


def test_projection_activates_queued_target_only_after_open_book_update() -> None:
    events = project_bar_events(_bar(), activate_queued_target=True)
    phases = [event.phase for event in events]

    assert phases[0] is MarketPhase.OPEN_QUOTE
    assert phases[1] is MarketPhase.TARGET_ACTIVATION
    assert events[1].timestamp_ns == _bar().open_ns + TARGET_ACTIVATION_DELAY_NS
    assert events[0].timestamp_ns < events[1].timestamp_ns
    assert events[-2].phase is MarketPhase.CLOSE_QUOTE
    assert events[-1].phase is MarketPhase.POLICY_DECISION
    assert events[-2].timestamp_ns < events[-1].timestamp_ns


def test_primary_extreme_order_depends_only_on_ohlc_shape() -> None:
    low_first = project_bar_events(_bar(), activate_queued_target=False)
    assert [event.phase for event in low_first][1:3] == [
        MarketPhase.LOW,
        MarketPhase.HIGH,
    ]

    high_first_bar = SourceBar(
        open_ns=1_000_000,
        close_ns=901_000_000_000,
        open_price=108.0,
        high_price=110.0,
        low_price=95.0,
        close_price=105.0,
        mark_price=104.5,
        index_price=104.25,
    )
    high_first = project_bar_events(high_first_bar, activate_queued_target=False)
    assert [event.phase for event in high_first][1:3] == [
        MarketPhase.HIGH,
        MarketPhase.LOW,
    ]


def test_projection_has_strictly_increasing_noncolliding_timestamps() -> None:
    events = project_bar_events(_bar(), activate_queued_target=True)
    timestamps = [event.timestamp_ns for event in events]

    assert timestamps == sorted(timestamps)
    assert len(timestamps) == len(set(timestamps))


def test_projection_contains_mark_and_index_before_close() -> None:
    events = project_bar_events(_bar(), activate_queued_target=False)
    phases = [event.phase for event in events]

    assert phases[-4:] == [
        MarketPhase.INDEX,
        MarketPhase.MARK,
        MarketPhase.CLOSE_QUOTE,
        MarketPhase.POLICY_DECISION,
    ]


def test_invalid_source_bar_fails_closed() -> None:
    with pytest.raises(ValueError, match="timestamps"):
        project_bar_events(
            SourceBar(
                open_ns=10,
                close_ns=10,
                open_price=100.0,
                high_price=110.0,
                low_price=95.0,
                close_price=105.0,
                mark_price=104.5,
                index_price=104.25,
            ),
            activate_queued_target=True,
        )

    with pytest.raises(ValueError, match="OHLC"):
        project_bar_events(
            SourceBar(
                open_ns=1_000_000,
                close_ns=901_000_000_000,
                open_price=120.0,
                high_price=110.0,
                low_price=95.0,
                close_price=105.0,
                mark_price=104.5,
                index_price=104.25,
            ),
            activate_queued_target=True,
        )
