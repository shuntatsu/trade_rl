from __future__ import annotations

import pytest

pytest.importorskip("nautilus_trader")

from trade_rl.integrations.nautilus.event_projection import (
    MarketPhase,
    SourceBar,
    project_bar_events,
)
from trade_rl.integrations.nautilus.instrument import build_maintained_btcusdt_perpetual
from trade_rl.integrations.nautilus.quote_projection import build_quote_tick


@pytest.mark.nautilus
def test_projected_mid_price_becomes_tick_aligned_two_sided_quote() -> None:
    instrument = build_maintained_btcusdt_perpetual()
    bar = SourceBar(
        open_ns=1_000_000,
        close_ns=901_000_000_000,
        open_price=100.0,
        high_price=110.0,
        low_price=95.0,
        close_price=105.0,
        mark_price=104.5,
        index_price=104.25,
    )
    event = project_bar_events(bar, activate_queued_target=False)[0]

    quote = build_quote_tick(
        event,
        instrument=instrument,
        half_spread_ticks=1,
        displayed_size=1.0,
    )

    assert event.phase is MarketPhase.OPEN_QUOTE
    assert str(quote.instrument_id) == "BTCUSDT-PERP.BINANCE"
    assert str(quote.bid_price) == "99.9"
    assert str(quote.ask_price) == "100.1"
    assert str(quote.bid_size) == "1.000"
    assert str(quote.ask_size) == "1.000"
    assert quote.ts_event == bar.open_ns
    assert quote.ts_init == bar.open_ns


@pytest.mark.nautilus
def test_non_price_projection_phase_is_rejected() -> None:
    instrument = build_maintained_btcusdt_perpetual()
    event = project_bar_events(
        SourceBar(
            open_ns=1_000_000,
            close_ns=901_000_000_000,
            open_price=100.0,
            high_price=110.0,
            low_price=95.0,
            close_price=105.0,
            mark_price=104.5,
            index_price=104.25,
        ),
        activate_queued_target=True,
    )[1]

    assert event.phase is MarketPhase.TARGET_ACTIVATION
    with pytest.raises(ValueError, match="price-bearing"):
        build_quote_tick(
            event,
            instrument=instrument,
            half_spread_ticks=1,
            displayed_size=1.0,
        )
