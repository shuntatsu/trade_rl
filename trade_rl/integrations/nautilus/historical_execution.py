"""Execute factual target-exposure intervals through the pinned Nautilus engine."""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from trade_rl.integrations.nautilus.event_projection import (
    MarketPhase,
    SourceBar,
    project_bar_events,
)
from trade_rl.integrations.nautilus.instrument import build_maintained_btcusdt_perpetual
from trade_rl.integrations.nautilus.order_adapter import submit_target_exposure_plan
from trade_rl.integrations.nautilus.quote_projection import build_quote_tick
from trade_rl.integrations.nautilus.runtime_identity import require_nautilus_runtime
from trade_rl.integrations.nautilus.trace_adapter import canonicalize_nautilus_fill_events
from trade_rl.simulation.execution_canonicalization import CanonicalFillSignature
from trade_rl.simulation.target_exposure_controller import (
    TargetExposureController,
    TargetExposureInput,
)

_PRICE_PHASES = frozenset(
    {
        MarketPhase.OPEN_QUOTE,
        MarketPhase.HIGH,
        MarketPhase.LOW,
        MarketPhase.CLOSE_QUOTE,
    }
)


@dataclass(frozen=True, slots=True)
class NautilusHistoricalTargetInterval:
    """One factual decision target and the exact source bars it consumes."""

    sequence: int
    target_exposure: float
    allocated_equity: float
    source_bars: tuple[SourceBar, ...]


@dataclass(frozen=True, slots=True)
class NautilusHistoricalPositionSnapshot:
    """Actual signed candidate quantity observed at one requested boundary."""

    timestamp_ns: int
    signed_quantity: Decimal


@dataclass(frozen=True, slots=True)
class NautilusHistoricalExecutionResult:
    """Framework-neutral evidence emitted by one historical BacktestEngine run."""

    runtime_version: str
    fills: tuple[CanonicalFillSignature, ...]
    fee_minor: int
    final_balance_minor: int
    terminal_position_lots: int
    terminal_open_orders: int
    position_snapshots: tuple[NautilusHistoricalPositionSnapshot, ...]


def run_historical_target_intervals(
    intervals: tuple[NautilusHistoricalTargetInterval, ...],
    *,
    snapshot_timestamps_ns: tuple[int, ...] = (),
    starting_balance: Decimal = Decimal("100000"),
    no_trade_band: float = 0.05,
) -> NautilusHistoricalExecutionResult:
    """Replay target intervals after each interval's first open quote is observed."""

    _validate_intervals(intervals)
    if snapshot_timestamps_ns:
        raise NotImplementedError("historical position snapshots are not implemented yet")
    if not starting_balance.is_finite() or starting_balance <= 0:
        raise ValueError("starting_balance must be finite and positive")
    runtime = require_nautilus_runtime()
    controller = TargetExposureController(no_trade_band=no_trade_band)

    from nautilus_trader.adapters.binance import BINANCE_VENUE
    from nautilus_trader.backtest.engine import BacktestEngine
    from nautilus_trader.model import Money
    from nautilus_trader.model.currencies import USDT
    from nautilus_trader.model.enums import AccountType, OmsType
    from nautilus_trader.trading.strategy import Strategy

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

        quotes: list[Any] = []
        activation_by_quote_index: dict[int, NautilusHistoricalTargetInterval] = {}
        for interval in intervals:
            first_quote_index: int | None = None
            for bar in interval.source_bars:
                for event in project_bar_events(bar, activate_queued_target=False):
                    if event.phase not in _PRICE_PHASES:
                        continue
                    if first_quote_index is None and event.phase is MarketPhase.OPEN_QUOTE:
                        first_quote_index = len(quotes)
                    quotes.append(
                        build_quote_tick(
                            event,
                            instrument=instrument,
                            half_spread_ticks=1,
                            displayed_size=1_000_000.0,
                        )
                    )
            if first_quote_index is None:
                raise RuntimeError("historical interval did not project an open quote")
            activation_by_quote_index[first_quote_index] = interval

        class HistoricalTargetStrategy(Strategy):
            def __init__(self) -> None:
                super().__init__()
                self.quote_index = 0
                self.realized_quantity = 0.0
                self.filled_events: list[object] = []

            def on_start(self) -> None:
                self.subscribe_quote_ticks(instrument.id)

            def on_quote_tick(self, tick: object) -> None:
                interval = activation_by_quote_index.get(self.quote_index)
                self.quote_index += 1
                if interval is None:
                    return
                plan = controller.plan(
                    TargetExposureInput(
                        target_exposure=interval.target_exposure,
                        allocated_equity=interval.allocated_equity,
                        reference_price=interval.source_bars[0].open_price,
                        contract_multiplier=1.0,
                        realized_quantity=self.realized_quantity,
                        working_remaining_quantities=(),
                    )
                )
                submit_target_exposure_plan(
                    strategy=self,
                    instrument=instrument,
                    plan=plan,
                )

            def on_order_filled(self, event: object) -> None:
                self.filled_events.append(event)
                self.realized_quantity += _signed_fill_quantity(event)

        strategy = HistoricalTargetStrategy()
        engine.add_data(quotes, sort=False)
        engine.add_strategy(strategy)
        engine.run()

        canonical = canonicalize_nautilus_fill_events(
            strategy.filled_events,
            price_tick=instrument.price_increment.as_decimal(),
            lot_size=instrument.size_increment.as_decimal(),
            currency_precision=USDT.precision,
        )
        account = engine.cache.account_for_venue(BINANCE_VENUE)
        if account is None:
            raise RuntimeError("expected Binance backtest account")
        balance = account.balance_total(USDT)
        if balance is None:
            raise RuntimeError("expected USDT account balance")
        final_balance_minor = _minor_units(
            balance.as_decimal(),
            currency_precision=USDT.precision,
            name="final_balance",
        )
        terminal_position_lots = canonical.fills[-1].position_lots if canonical.fills else 0
        terminal_open_orders = len(engine.cache.orders_open(instrument_id=instrument.id))
        return NautilusHistoricalExecutionResult(
            runtime_version=runtime.package_version or "",
            fills=canonical.fills,
            fee_minor=canonical.fee_minor,
            final_balance_minor=final_balance_minor,
            terminal_position_lots=terminal_position_lots,
            terminal_open_orders=terminal_open_orders,
            position_snapshots=(),
        )
    finally:
        engine.dispose()


def _validate_intervals(intervals: tuple[NautilusHistoricalTargetInterval, ...]) -> None:
    if not intervals:
        raise ValueError("historical target intervals must be non-empty")
    for expected_sequence, interval in enumerate(intervals, start=1):
        if interval.sequence != expected_sequence:
            raise ValueError("historical target interval sequence must be contiguous")
        if not math.isfinite(interval.target_exposure) or not -1.0 <= interval.target_exposure <= 1.0:
            raise ValueError("historical target exposure must be finite and within [-1, 1]")
        if not math.isfinite(interval.allocated_equity) or interval.allocated_equity <= 0.0:
            raise ValueError("historical allocated equity must be finite and positive")
        if not interval.source_bars:
            raise ValueError("historical target interval must contain source bars")


def _signed_fill_quantity(event: object) -> float:
    quantity = _as_decimal(getattr(event, "last_qty", None), name="last_qty")
    side = getattr(event, "order_side", None)
    side_name = getattr(side, "name", str(side)).upper()
    if side_name.endswith("BUY"):
        return float(quantity)
    if side_name.endswith("SELL"):
        return -float(quantity)
    raise ValueError(f"unsupported Nautilus order side: {side!r}")


def _as_decimal(value: object, *, name: str) -> Decimal:
    if value is None:
        raise ValueError(f"{name} is missing")
    if hasattr(value, "as_decimal"):
        value = value.as_decimal()
    result = value if isinstance(value, Decimal) else Decimal(str(value))
    if not result.is_finite():
        raise ValueError(f"{name} must be finite")
    return result


def _minor_units(value: Decimal, *, currency_precision: int, name: str) -> int:
    scaled = value * (Decimal(10) ** currency_precision)
    integral = scaled.to_integral_value()
    if scaled != integral:
        raise RuntimeError(f"{name} is not aligned to settlement minor unit")
    return int(integral)


__all__ = [
    "NautilusHistoricalExecutionResult",
    "NautilusHistoricalPositionSnapshot",
    "NautilusHistoricalTargetInterval",
    "run_historical_target_intervals",
]
