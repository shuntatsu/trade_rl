"""Isolated Nautilus execution probes for controller-approved child-order sequences."""

from __future__ import annotations

import math
from dataclasses import dataclass
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
from trade_rl.simulation.target_exposure_controller import (
    TargetExposureChildOrder,
)

_HOUR_NS = 60 * 60 * 1_000_000_000
_QUANTITY_TOLERANCE = 1e-12


@dataclass(frozen=True, slots=True)
class NautilusChildOrderProbeResult:
    """Canonical terminal evidence for one isolated child-order sequence."""

    runtime_version: str
    orders_closed: int
    positions_closed: int
    open_positions: int
    fills: tuple[CanonicalFillSignature, ...]
    economics: CanonicalEconomicClosure


def run_child_order_sequence_execution_probe(
    child_orders: tuple[TargetExposureChildOrder, ...],
    *,
    starting_balance: Decimal = Decimal("100000"),
) -> NautilusChildOrderProbeResult:
    """Execute safe market child orders in one fresh Nautilus BacktestEngine."""

    if not child_orders:
        raise ValueError("child_orders must be non-empty")
    if not starting_balance.is_finite() or starting_balance <= 0:
        raise ValueError("starting_balance must be finite and positive")
    _validate_child_orders(child_orders)
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

    class ChildOrderProbeStrategy(Strategy):
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
            if self.quote_count >= len(child_orders):
                return
            child_order = child_orders[self.quote_count]
            self.quote_count += 1
            side = OrderSide.BUY if child_order.quantity > 0.0 else OrderSide.SELL
            self.submit_order(
                self.order_factory.market(
                    instrument_id=self.instrument_id,
                    order_side=side,
                    quantity=instrument.make_qty(abs(child_order.quantity)),
                    time_in_force=TimeInForce.IOC,
                    reduce_only=child_order.reduce_only,
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
        quotes = [
            build_quote_tick(
                ProjectedMarketEvent(MarketPhase.OPEN_QUOTE, index * _HOUR_NS, 100.0),
                instrument=instrument,
                half_spread_ticks=1,
                displayed_size=10.0,
            )
            for index in range(1, len(child_orders) + 1)
        ]
        engine.add_data(quotes, sort=True)
        strategy = ChildOrderProbeStrategy(instrument.id)
        engine.add_strategy(strategy)
        engine.run()

        open_positions = engine.cache.positions_open(instrument_id=instrument.id)
        if open_positions:
            raise RuntimeError("child-order conformance probe must terminate flat")
        account = engine.cache.account_for_venue(BINANCE_VENUE)
        if account is None:
            raise RuntimeError("expected Binance backtest account")
        balance = account.balance_total(USDT)
        if balance is None:
            raise RuntimeError("expected USDT account balance")

        canonical = canonicalize_nautilus_fill_events(
            strategy.filled_events,
            price_tick=instrument.price_increment.as_decimal(),
            lot_size=instrument.size_increment.as_decimal(),
            currency_precision=USDT.precision,
        )
        if len(canonical.fills) != len(child_orders):
            raise RuntimeError(
                "child-order conformance probe must fill every deterministic IOC leg"
            )
        if canonical.fills[-1].position_lots != 0:
            raise RuntimeError("child-order conformance probe must end with zero lots")

        scale = Decimal(10) ** USDT.precision
        final_balance = balance.as_decimal()
        economics = CanonicalEconomicClosure(
            fee_minor=canonical.fee_minor,
            funding_minor=0,
            realized_pnl_minor=_minor_units(
                final_balance - starting_balance,
                scale=scale,
                name="realized_pnl",
            ),
            final_equity_minor=_minor_units(
                final_balance,
                scale=scale,
                name="final_balance",
            ),
            terminal_position_lots=0,
            terminal_open_orders=len(
                engine.cache.orders_open(instrument_id=instrument.id)
            ),
        )
        closed = engine.cache.positions_closed(instrument_id=instrument.id)
        return NautilusChildOrderProbeResult(
            runtime_version=runtime.package_version or "",
            orders_closed=len(engine.cache.orders_closed(instrument_id=instrument.id)),
            positions_closed=len(closed),
            open_positions=len(open_positions),
            fills=canonical.fills,
            economics=economics,
        )
    finally:
        engine.dispose()


def _validate_child_orders(child_orders: tuple[TargetExposureChildOrder, ...]) -> None:
    realized = 0.0
    for child_order in child_orders:
        quantity = float(child_order.quantity)
        if not math.isfinite(quantity) or abs(quantity) <= _QUANTITY_TOLERANCE:
            raise ValueError("child order quantity must be finite and non-zero")
        if not child_order.reduce_only:
            if abs(realized) > _QUANTITY_TOLERANCE and realized * quantity < 0.0:
                raise ValueError(
                    "non-reduce child order cannot oppose an open realized position"
                )
        else:
            if abs(realized) <= _QUANTITY_TOLERANCE:
                raise ValueError("reduce-only child order cannot start from flat")
            if realized * quantity >= 0.0:
                raise ValueError(
                    "reduce-only child order must oppose the realized position"
                )
            if abs(quantity) > abs(realized) + _QUANTITY_TOLERANCE:
                raise ValueError("reduce-only child order cannot cross through flat")
        realized += quantity
    if abs(realized) > _QUANTITY_TOLERANCE:
        raise ValueError("child-order conformance sequence must terminate flat")


def _minor_units(value: Decimal, *, scale: Decimal, name: str) -> int:
    scaled = value * scale
    integral = scaled.to_integral_value()
    if scaled != integral:
        raise RuntimeError(f"{name} is not aligned to settlement minor unit")
    return int(integral)


__all__ = [
    "NautilusChildOrderProbeResult",
    "run_child_order_sequence_execution_probe",
]
