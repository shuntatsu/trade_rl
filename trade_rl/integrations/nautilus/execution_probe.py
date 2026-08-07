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
from trade_rl.integrations.nautilus.instrument import build_maintained_btcusdt_perpetual
from trade_rl.integrations.nautilus.quote_projection import build_quote_tick
from trade_rl.integrations.nautilus.runtime_identity import require_nautilus_runtime
from trade_rl.integrations.nautilus.trace_adapter import (
    canonicalize_nautilus_fill_events,
)
from trade_rl.simulation.execution_canonicalization import (
    CanonicalEconomicClosure,
    CanonicalFillSignature,
)

_HOUR_NS = 60 * 60 * 1_000_000_000


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
    fills: tuple[CanonicalFillSignature, ...]
    economics: CanonicalEconomicClosure

    def digest(self) -> str:
        payload = json.dumps(
            asdict(self),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def run_flat_long_flat_execution_probe(
    *,
    starting_balance: Decimal = Decimal("100000"),
) -> NautilusExecutionProbeResult:
    """Run one deterministic Nautilus lifecycle; callers must isolate processes."""

    if not starting_balance.is_finite() or starting_balance <= 0:
        raise ValueError("starting_balance must be finite and positive")
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
            self.filled_events: list[object] = []

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

        def on_order_filled(self, event: object) -> None:
            self.filled_events.append(event)

    engine = BacktestEngine()
    try:
        engine.add_venue(
            venue=BINANCE_VENUE,
            oms_type=OmsType.NETTING,
            account_type=AccountType.MARGIN,
            base_currency=USDT,
            starting_balances=[Money(starting_balance, USDT)],
        )
        instrument = build_maintained_btcusdt_perpetual()
        engine.add_instrument(instrument)
        events = (
            ProjectedMarketEvent(MarketPhase.OPEN_QUOTE, _HOUR_NS, 100.0),
            ProjectedMarketEvent(MarketPhase.CLOSE_QUOTE, 2 * _HOUR_NS, 105.0),
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
        strategy = ProbeStrategy(instrument.id)
        engine.add_strategy(strategy)
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

        canonical = canonicalize_nautilus_fill_events(
            strategy.filled_events,
            price_tick=instrument.price_increment.as_decimal(),
            lot_size=instrument.size_increment.as_decimal(),
            currency_precision=USDT.precision,
        )
        scale = Decimal(10) ** USDT.precision
        realized_minor = _minor_units(
            realized.as_decimal(), scale=scale, name="realized_pnl"
        )
        final_balance_minor = _minor_units(
            balance.as_decimal(),
            scale=scale,
            name="final_balance",
        )
        open_orders = engine.cache.orders_open(instrument_id=instrument.id)
        economics = CanonicalEconomicClosure(
            fee_minor=canonical.fee_minor,
            funding_minor=0,
            realized_pnl_minor=realized_minor,
            final_equity_minor=final_balance_minor,
            terminal_position_lots=0,
            terminal_open_orders=len(open_orders),
        )

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
            fills=canonical.fills,
            economics=economics,
        )
    finally:
        engine.dispose()


def _minor_units(value: Decimal, *, scale: Decimal, name: str) -> int:
    scaled = value * scale
    integral = scaled.to_integral_value()
    if scaled != integral:
        raise RuntimeError(f"{name} is not aligned to settlement minor unit")
    return int(integral)


def _decimal_text(value: Decimal) -> str:
    return format(value.normalize(), "f")


def _float_text(value: float) -> str:
    return format(value, ".16g")


__all__ = [
    "NautilusExecutionProbeResult",
    "run_flat_long_flat_execution_probe",
]
