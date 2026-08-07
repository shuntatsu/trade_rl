from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("nautilus_trader")

from nautilus_trader.adapters.binance import BINANCE_VENUE
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.model import Money
from nautilus_trader.model.currencies import USDT
from nautilus_trader.model.enums import (
    AccountType,
    OmsType,
    OrderSide,
    OrderStatus,
    TimeInForce,
)
from nautilus_trader.trading.strategy import Strategy

from trade_rl.integrations.nautilus.event_projection import (
    MarketPhase,
    ProjectedMarketEvent,
)
from trade_rl.integrations.nautilus.instrument import build_maintained_btcusdt_perpetual
from trade_rl.integrations.nautilus.order_adapter import submit_target_exposure_plan
from trade_rl.integrations.nautilus.quote_projection import build_quote_tick
from trade_rl.simulation.target_exposure_controller import (
    ControllerPhase,
    TargetExposureController,
    TargetExposureInput,
    TargetExposurePlan,
)

_HOUR_NS = 60 * 60 * 1_000_000_000


def _quantity(value: object) -> float:
    return float(str(value))


class _PartialFillCancelReplaceProbe(Strategy):
    def __init__(self, instrument_id: object) -> None:
        super().__init__()
        self.instrument_id = instrument_id
        self.controller = TargetExposureController(no_trade_band=0.0)
        self.quote_count = 0
        self.primary_order: Any | None = None
        self.replacement_order: Any | None = None
        self.close_order: Any | None = None
        self.cancel_plan: TargetExposurePlan | None = None
        self.replacement_plan: TargetExposurePlan | None = None
        self.close_plan: TargetExposurePlan | None = None
        self.cancel_observed = False
        self.lifecycle: list[tuple[str, float | None]] = []

    def on_start(self) -> None:
        self.subscribe_quote_ticks(self.instrument_id)

    def on_quote_tick(self, tick: object) -> None:
        instrument = self.cache.instrument(self.instrument_id)
        assert instrument is not None
        self.quote_count += 1

        if self.quote_count == 1:
            order = self.order_factory.limit(
                instrument_id=self.instrument_id,
                order_side=OrderSide.BUY,
                quantity=instrument.make_qty(1.0),
                price=instrument.make_price(100.0),
                time_in_force=TimeInForce.GTC,
                post_only=True,
            )
            self.primary_order = order
            self.submit_order(order)
            return

        if self.quote_count == 2:
            assert self.primary_order is not None
            assert self.primary_order.status is OrderStatus.PARTIALLY_FILLED
            realized = _quantity(self.primary_order.filled_qty)
            working = _quantity(self.primary_order.leaves_qty)
            plan = self.controller.plan(
                TargetExposureInput(
                    target_exposure=0.05,
                    allocated_equity=1_000.0,
                    reference_price=100.0,
                    contract_multiplier=1.0,
                    realized_quantity=realized,
                    working_remaining_quantities=(working,),
                )
            )
            self.cancel_plan = plan
            submit_target_exposure_plan(
                strategy=self,
                instrument=instrument,
                plan=plan,
            )
            return

        if self.quote_count == 3:
            if not self.cancel_observed:
                raise RuntimeError("replacement cannot precede terminal cancel evidence")
            assert self.primary_order is not None
            realized = _quantity(self.primary_order.filled_qty)
            plan = self.controller.plan(
                TargetExposureInput(
                    target_exposure=0.05,
                    allocated_equity=1_000.0,
                    reference_price=100.0,
                    contract_multiplier=1.0,
                    realized_quantity=realized,
                    working_remaining_quantities=(),
                )
            )
            self.replacement_plan = plan
            self.replacement_order = submit_target_exposure_plan(
                strategy=self,
                instrument=instrument,
                plan=plan,
            )
            return

        if self.quote_count == 4:
            assert self.primary_order is not None
            assert self.replacement_order is not None
            realized = _quantity(self.primary_order.filled_qty) + _quantity(
                self.replacement_order.filled_qty
            )
            plan = self.controller.plan(
                TargetExposureInput(
                    target_exposure=0.0,
                    allocated_equity=1_000.0,
                    reference_price=100.0,
                    contract_multiplier=1.0,
                    realized_quantity=realized,
                    working_remaining_quantities=(),
                )
            )
            self.close_plan = plan
            self.close_order = submit_target_exposure_plan(
                strategy=self,
                instrument=instrument,
                plan=plan,
            )

    def on_order_filled(self, event: Any) -> None:
        self.lifecycle.append(("fill", _quantity(event.last_qty)))

    def on_order_canceled(self, event: Any) -> None:
        self.cancel_observed = True
        self.lifecycle.append(("cancel", None))


@pytest.mark.nautilus
def test_partial_fill_is_canceled_before_replacement_and_finishes_flat() -> None:
    engine = BacktestEngine()
    try:
        engine.add_venue(
            venue=BINANCE_VENUE,
            oms_type=OmsType.NETTING,
            account_type=AccountType.MARGIN,
            base_currency=USDT,
            starting_balances=[Money(1_000, USDT)],
            liquidity_consumption=True,
        )
        instrument = build_maintained_btcusdt_perpetual()
        engine.add_instrument(instrument)
        quotes = [
            build_quote_tick(
                ProjectedMarketEvent(MarketPhase.OPEN_QUOTE, _HOUR_NS, 100.0),
                instrument=instrument,
                half_spread_ticks=1,
                displayed_size=10.0,
            ),
            build_quote_tick(
                ProjectedMarketEvent(MarketPhase.OPEN_QUOTE, 2 * _HOUR_NS, 99.8),
                instrument=instrument,
                half_spread_ticks=1,
                displayed_size=0.4,
            ),
            build_quote_tick(
                ProjectedMarketEvent(MarketPhase.OPEN_QUOTE, 3 * _HOUR_NS, 100.0),
                instrument=instrument,
                half_spread_ticks=1,
                displayed_size=10.0,
            ),
            build_quote_tick(
                ProjectedMarketEvent(MarketPhase.OPEN_QUOTE, 4 * _HOUR_NS, 100.0),
                instrument=instrument,
                half_spread_ticks=1,
                displayed_size=10.0,
            ),
        ]
        engine.add_data(quotes, sort=True)
        strategy = _PartialFillCancelReplaceProbe(instrument.id)
        engine.add_strategy(strategy)
        engine.run()

        assert strategy.primary_order is not None
        assert strategy.primary_order.status is OrderStatus.CANCELED
        assert _quantity(strategy.primary_order.filled_qty) == pytest.approx(0.4)
        assert _quantity(strategy.primary_order.leaves_qty) == pytest.approx(0.6)
        assert strategy.cancel_plan is not None
        assert strategy.cancel_plan.phase is ControllerPhase.CANCELING_STALE
        assert strategy.cancel_plan.cancel_working_orders is True
        assert strategy.cancel_plan.child_order is None
        assert strategy.cancel_observed is True

        assert strategy.replacement_plan is not None
        assert strategy.replacement_plan.phase is ControllerPhase.OPENING
        assert strategy.replacement_plan.child_order is not None
        assert strategy.replacement_plan.child_order.quantity == pytest.approx(0.1)
        assert strategy.replacement_plan.child_order.reduce_only is False

        assert strategy.close_plan is not None
        assert strategy.close_plan.phase is ControllerPhase.REDUCING
        assert strategy.close_plan.child_order is not None
        assert strategy.close_plan.child_order.quantity == pytest.approx(-0.5)
        assert strategy.close_plan.child_order.reduce_only is True

        assert [kind for kind, _ in strategy.lifecycle] == [
            "fill",
            "cancel",
            "fill",
            "fill",
        ]
        fill_quantities = [
            quantity
            for kind, quantity in strategy.lifecycle
            if kind == "fill" and quantity is not None
        ]
        assert fill_quantities == pytest.approx([0.4, 0.1, 0.5])
        assert engine.cache.orders_open(instrument_id=instrument.id) == []
        assert engine.cache.positions_open(instrument_id=instrument.id) == []
    finally:
        engine.dispose()
