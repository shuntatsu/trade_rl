from __future__ import annotations

import pytest

pytest.importorskip("nautilus_trader")

from nautilus_trader.adapters.binance import BINANCE_VENUE
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.model import Money
from nautilus_trader.model.currencies import USDT
from nautilus_trader.model.enums import AccountType, OmsType

from trade_rl.integrations.nautilus.event_projection import (
    SourceBar,
    project_bar_events,
)
from trade_rl.integrations.nautilus.instrument import (
    build_maintained_btcusdt_perpetual,
)
from trade_rl.integrations.nautilus.quote_projection import build_quote_tick


@pytest.mark.nautilus
def test_projected_quote_stream_replays_through_backtest_engine() -> None:
    engine = BacktestEngine()
    try:
        engine.add_venue(
            venue=BINANCE_VENUE,
            oms_type=OmsType.NETTING,
            account_type=AccountType.MARGIN,
            base_currency=USDT,
            starting_balances=[Money(100_000, USDT)],
        )
        instrument = build_maintained_btcusdt_perpetual()
        engine.add_instrument(instrument)

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
        projected = project_bar_events(bar, activate_queued_target=False)
        quotes = [
            build_quote_tick(
                event,
                instrument=instrument,
                half_spread_ticks=1,
                displayed_size=1.0,
            )
            for event in projected
            if event.price is not None and event.phase.value not in {"index", "mark"}
        ]

        engine.add_data(quotes, sort=True)
        engine.run()

        latest = engine.cache.quote_tick(instrument.id)
        assert latest is not None
        assert int(latest.ts_event) == bar.close_ns
        assert str(latest.bid_price) == "104.9"
        assert str(latest.ask_price) == "105.1"
    finally:
        engine.dispose()
