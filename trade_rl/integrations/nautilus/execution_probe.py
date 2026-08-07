"""Deterministic single-process execution probe for the pinned Nautilus runtime."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from decimal import Decimal

from trade_rl.integrations.nautilus.event_projection import (
    MarketPhase,
    ProjectedMarketEvent,
)
from trade_rl.integrations.nautilus.instrument import (
    build_maintained_btcusdt_perpetual,
)
from trade_rl.integrations.nautilus.quote_projection import build_quote_tick
from trade_rl.integrations.nautilus.runtime_identity import require_nautilus_runtime


@dataclass(frozen=True, slots=True)
class NautilusExecutionProbeResult:
    """Framework-neutral result of one flat-long-flat Nautilus run."""

    runtime_version: str
    orders_closed: int
    positions_closed: int
    open_positions: int
    avg_px_open: str
    avg_px_close: str
    realized_pnl: str
    commissions: tuple[str, ...]
    final_balance: str

    def digest(self) -> str:
        payload = json.dumps(
            asdict(self),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def run_flat_long_flat_execution_probe() -> NautilusExecutionProbeResult:
    """Run one deterministic Nautilus lifecycle; callers must isolate processes."""

    runtime = require_nautilus_runtime()

    from nautilus_trader.adapters.binance import BINANCE_VENUE
    from nautilus_trader.backtest.engine import BacktestEngine
    from nautilus_trader.model import Money
    from nautilus_trader.model.currencies import USDT
    from nautilus_trader.model.enums import (
        AccountType,
        OmsType,
        OrderSide,
        TimeInForce,
    )
    from nautilus_trader.trading.strategy import Strategy

    class ProbeStrategy(Strategy):
        def __init__(self, instrument_id: object) -> None:
            super().__init__()
            self.instrument_id = instrument_id
            self.quote_count = 0

        def on_start(self) -> None:
            self.subscribe_quote_ticks(self.instrument_id)

        def on_quote_tick(self, tick: object) -> None:
            instrument = self.cache.instrument(self.instrument_id)
            assert instrument is not None
            self.quote_count += 1
            if self.quote_count == 1:
                self.submit_order(
                    self.order_factory.market(
                        instrument_id=self.instrument_id,
                        order_side=OrderSide.BUY,
                        quantity=instrument.make_qty(1.0),
                        time_in_force=TimeInForce.IOC,
                    )
                )
            elif self.quote_count == 2:
                self.submit_order(
                    self.order_factory.market(
                        instrument_id=self.instrument_id,
                        order_side=OrderSide.SELL,
                        quantity=instrument.make_qty(1.0),
                        time_in_force=TimeInForce.IOC,
                        reduce_only=True,
                    )
                )

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
        events = (
            ProjectedMarketEvent(MarketPhase.OPEN_QUOTE, 10, 100.0),
            ProjectedMarketEvent(MarketPhase.CLOSE_QUOTE, 200, 105.0),
        )
        quotes = [
            build_quote_tick(
                event,
                instrument=instrument,
                half_spread_ticks=1,
                displayed_size=10.0,
            )
            for event in events
        ]
        engine.add_data(quotes, sort=True)
        engine.add_strategy(ProbeStrategy(instrument.id))
        engine.run()

        closed = engine.cache.positions_closed(instrument_id=instrument.id)
        if len(closed) != 1:
            raise RuntimeError(f"expected one closed position, got {len(closed)}")
        position = closed[0]
        account = engine.cache.account_for_venue(BINANCE_VENUE)
        if account is None:
            raise RuntimeError("expected Binance backtest account")
        balance = account.balance_total(USDT)
        if balance is None:
            raise RuntimeError("expected USDT account balance")
        realized = position.realized_pnl
        if realized is None:
            raise RuntimeError("expected realized PnL on closed position")

        commission_values = tuple(
            _decimal_text(value.as_decimal())
            for value in sorted(position.commissions(), key=lambda item: item.currency.code)
        )
        return NautilusExecutionProbeResult(
            runtime_version=runtime.package_version or "",
            orders_closed=len(engine.cache.orders_closed(instrument_id=instrument.id)),
            positions_closed=len(closed),
            open_positions=len(engine.cache.positions_open(instrument_id=instrument.id)),
            avg_px_open=_float_text(position.avg_px_open),
            avg_px_close=_float_text(position.avg_px_close),
            realized_pnl=_decimal_text(realized.as_decimal()),
            commissions=commission_values,
            final_balance=_decimal_text(balance.as_decimal()),
        )
    finally:
        engine.dispose()


def _decimal_text(value: Decimal) -> str:
    return format(value.normalize(), "f")


def _float_text(value: float) -> str:
    return format(value, ".16g")


__all__ = [
    "NautilusExecutionProbeResult",
    "run_flat_long_flat_execution_probe",
]
