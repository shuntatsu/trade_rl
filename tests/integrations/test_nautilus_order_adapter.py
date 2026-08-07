from __future__ import annotations

import pytest

pytest.importorskip("nautilus_trader")

from nautilus_trader.model.enums import OrderSide

from trade_rl.integrations.nautilus.instrument import build_maintained_btcusdt_perpetual
from trade_rl.integrations.nautilus.order_adapter import submit_target_exposure_plan
from trade_rl.simulation.target_exposure_controller import (
    ControllerPhase,
    TargetExposureChildOrder,
    TargetExposurePlan,
)


class _FakeOrderFactory:
    def __init__(self) -> None:
        self.kwargs: dict[str, object] | None = None

    def market(self, **kwargs: object) -> dict[str, object]:
        self.kwargs = kwargs
        return kwargs


class _FakeStrategy:
    def __init__(self) -> None:
        self.order_factory = _FakeOrderFactory()
        self.submitted: list[object] = []
        self.cancelled_instrument_ids: list[object] = []

    def submit_order(self, order: object) -> None:
        self.submitted.append(order)

    def cancel_all_orders(self, instrument_id: object) -> None:
        self.cancelled_instrument_ids.append(instrument_id)


def _plan(
    *,
    phase: ControllerPhase,
    child: TargetExposureChildOrder | None,
    cancel: bool = False,
) -> TargetExposurePlan:
    return TargetExposurePlan(
        phase=phase,
        raw_target_exposure=0.1,
        effective_target_exposure=0.1,
        desired_quantity=1.0,
        committed_quantity=0.0,
        cancel_working_orders=cancel,
        child_order=child,
    )


@pytest.mark.nautilus
def test_opening_child_maps_to_market_ioc_buy() -> None:
    strategy = _FakeStrategy()
    instrument = build_maintained_btcusdt_perpetual()

    order = submit_target_exposure_plan(
        strategy=strategy,
        instrument=instrument,
        plan=_plan(
            phase=ControllerPhase.OPENING,
            child=TargetExposureChildOrder(quantity=1.0, reduce_only=False),
        ),
    )

    assert order is not None
    assert strategy.submitted == [order]
    assert strategy.order_factory.kwargs is not None
    assert strategy.order_factory.kwargs["order_side"] is OrderSide.BUY
    assert str(strategy.order_factory.kwargs["quantity"]) == "1.000"
    assert strategy.order_factory.kwargs["reduce_only"] is False


@pytest.mark.nautilus
def test_reducing_child_maps_to_reduce_only_sell() -> None:
    strategy = _FakeStrategy()
    instrument = build_maintained_btcusdt_perpetual()

    submit_target_exposure_plan(
        strategy=strategy,
        instrument=instrument,
        plan=_plan(
            phase=ControllerPhase.REDUCING,
            child=TargetExposureChildOrder(quantity=-1.0, reduce_only=True),
        ),
    )

    assert strategy.order_factory.kwargs is not None
    assert strategy.order_factory.kwargs["order_side"] is OrderSide.SELL
    assert strategy.order_factory.kwargs["reduce_only"] is True


@pytest.mark.nautilus
def test_cancel_phase_cancels_without_submitting_replacement() -> None:
    strategy = _FakeStrategy()
    instrument = build_maintained_btcusdt_perpetual()

    order = submit_target_exposure_plan(
        strategy=strategy,
        instrument=instrument,
        plan=_plan(
            phase=ControllerPhase.CANCELING_STALE,
            child=None,
            cancel=True,
        ),
    )

    assert order is None
    assert strategy.cancelled_instrument_ids == [instrument.id]
    assert strategy.submitted == []
