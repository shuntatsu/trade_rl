from __future__ import annotations

from decimal import Decimal

import pytest

pytest.importorskip("nautilus_trader")

from nautilus_trader.adapters.binance import BINANCE_VENUE
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.model import Money
from nautilus_trader.model.currencies import USDT
from nautilus_trader.model.enums import AccountType, OmsType, OrderSide, TimeInForce
from nautilus_trader.trading.strategy import Strategy

from trade_rl.integrations.nautilus.derivative_projection import (
    FundingPoint,
    build_funding_rate_update,
    build_mark_price_update,
)
from trade_rl.integrations.nautilus.event_projection import (
    MarketPhase,
    ProjectedMarketEvent,
)
from trade_rl.integrations.nautilus.instrument import (
    build_maintained_btcusdt_perpetual,
)
from trade_rl.integrations.nautilus.quote_projection import build_quote_tick


class _FundingProbe(Strategy):
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
            self.submit_order(
                self.order_factory.market(
                    instrument_id=self.instrument_id,
                    order_side=OrderSide.BUY,
                    quantity=instrument.make_qty(1.0),
                    time_in_force=TimeInForce.IOC,
                )
            )
        elif self._quote_count == 2:
            self.submit_order(
                self.order_factory.market(
                    instrument_id=self.instrument_id,
                    order_side=OrderSide.SELL,
                    quantity=instrument.make_qty(1.0),
                    time_in_force=TimeInForce.IOC,
                    reduce_only=True,
                )
            )


@pytest.mark.nautilus
def test_python_backtest_engine_1_230_does_not_dispatch_native_funding_settlement() -> None:
    """Lock the exact-wheel limitation that requires our canonical adapter.

    NautilusTrader v1.230.0 contains funding settlement in the Rust simulated
    exchange, but the Python low-level BacktestEngine data loop used here does not
    dispatch FundingRateUpdate/MarkPriceUpdate to that exchange. A future Nautilus
    upgrade that closes this gap should make this test fail and trigger a deliberate
    migration back to native settlement.
    """

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

        open_quote = build_quote_tick(
            ProjectedMarketEvent(MarketPhase.OPEN_QUOTE, 10, 100.0),
            instrument=instrument,
            half_spread_ticks=1,
            displayed_size=10.0,
        )
        funding = build_funding_rate_update(
            FundingPoint(
                rate=Decimal("0.0001"),
                observed_ns=100,
                next_funding_ns=120,
                interval_minutes=480,
            ),
            instrument=instrument,
        )
        mark = build_mark_price_update(
            ProjectedMarketEvent(MarketPhase.MARK, 119, 102.0),
            instrument=instrument,
        )
        close_quote = build_quote_tick(
            ProjectedMarketEvent(MarketPhase.CLOSE_QUOTE, 200, 105.0),
            instrument=instrument,
            half_spread_ticks=1,
            displayed_size=10.0,
        )

        engine.add_data([open_quote, close_quote], sort=True)
        engine.add_data([funding], sort=True)
        engine.add_data([mark], sort=True)
        engine.add_strategy(_FundingProbe(instrument.id))
        engine.run()

        closed = engine.cache.positions_closed(instrument_id=instrument.id)
        assert len(closed) == 1
        funding_adjustments = [
            adjustment
            for adjustment in closed[0].adjustments
            if str(adjustment.adjustment_type).endswith("FUNDING")
        ]
        assert funding_adjustments == []
    finally:
        engine.dispose()
