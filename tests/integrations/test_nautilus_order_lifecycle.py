from __future__ import annotations

import pytest

pytest.importorskip("nautilus_trader")

from nautilus_trader.adapters.binance import BINANCE_VENUE
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.model import Money
from nautilus_trader.model.currencies import USDT
from nautilus_trader.model.enums import AccountType, OmsType, OrderSide, TimeInForce
from nautilus_trader.trading.strategy import Strategy

from trade_rl.integrations.nautilus.event_projection import (
    MarketPhase,
    SourceBar,
    project_bar_events,
)
from trade_rl.integrations.nautilus.instrument import build_maintained_btcusdt_perpetual
from trade_rl.integrations.nautilus.quote_projection import build_quote_tick


class _FlatLongFlatProbe(Strategy):
    def __init__(self, instrument_id: object) -> None:
        super().__init__()
        self.instrument_id = instrument_id
        self._quote_count = 0

    def on_start(self) -> None:
        self.subscribe_quote_ticks(self.instrument_id)

    def on_quote_tick(self, tick: object) -> None:
        instrument = self.cache.instrument(self.instrument_id)
        assert instrument is not None
        self._quote_count += 1
        if self._quote_count == 1:
            order = self.order_factory.market(
                instrument_id=self.instrument_id,
                order_side=OrderSide.BUY,
                quantity=instrument.make_qty(1.0),
                time_in_force=TimeInForce.IOC,
            )
            self.submit_order(order)
        elif self._quote_count == 2:
            order = self.order_factory.market(
                instrument_id=self.instrument_id,
                order_side=OrderSide.SELL,
                quantity=instrument.make_qty(1.0),
                time_in_force=TimeInForce.IOC,
                reduce_only=True,
            )
            self.submit_order(order)


@pytest.mark.nautilus
def test_market_order_lifecycle_can_open_and_reduce_to_flat() -> None:
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
        open_event = next(
            event for event in projected if event.phase is MarketPhase.OPEN_QUOTE
        )
        close_event = next(
            event for event in projected if event.phase is MarketPhase.CLOSE_QUOTE
        )
        quotes = [
            build_quote_tick(
                event,
                instrument=instrument,
                half_spread_ticks=1,
                displayed_size=10.0,
            )
            for event in (open_event, close_event)
        ]

        engine.add_data(quotes, sort=True)
        engine.add_strategy(_FlatLongFlatProbe(instrument.id))
        engine.run()

        assert engine.cache.positions_open(instrument_id=instrument.id) == []
        assert len(engine.cache.positions_closed(instrument_id=instrument.id)) == 1
        assert len(engine.cache.orders_closed(instrument_id=instrument.id)) == 2
    finally:
        engine.dispose()
