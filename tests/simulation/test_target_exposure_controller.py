from __future__ import annotations

import pytest

from trade_rl.simulation.target_exposure_controller import (
    ControllerPhase,
    TargetExposureController,
    TargetExposureInput,
)


def _input(
    *,
    target: float,
    realized: float = 0.0,
    working: tuple[float, ...] = (),
    emergency_flatten: bool = False,
    halted: bool = False,
) -> TargetExposureInput:
    return TargetExposureInput(
        target_exposure=target,
        allocated_equity=1_000.0,
        reference_price=100.0,
        contract_multiplier=1.0,
        realized_quantity=realized,
        working_remaining_quantities=working,
        emergency_flatten=emergency_flatten,
        halted=halted,
    )


def test_matching_working_quantity_is_committed_and_not_double_submitted() -> None:
    controller = TargetExposureController(no_trade_band=0.05)

    plan = controller.plan(_input(target=0.5, realized=2.0, working=(3.0,)))

    assert plan.phase is ControllerPhase.TRACKING
    assert plan.desired_quantity == pytest.approx(5.0)
    assert plan.committed_quantity == pytest.approx(5.0)
    assert plan.cancel_working_orders is False
    assert plan.child_order is None


def test_changed_target_cancels_stale_working_order_before_replacement() -> None:
    controller = TargetExposureController(no_trade_band=0.05)

    plan = controller.plan(_input(target=0.4, realized=2.0, working=(3.0,)))

    assert plan.phase is ControllerPhase.CANCELING_STALE
    assert plan.cancel_working_orders is True
    assert plan.child_order is None


def test_same_side_change_emits_only_delta_quantity() -> None:
    controller = TargetExposureController(no_trade_band=0.05)

    plan = controller.plan(_input(target=0.6, realized=2.0))

    assert plan.phase is ControllerPhase.OPENING
    assert plan.child_order is not None
    assert plan.child_order.quantity == pytest.approx(4.0)
    assert plan.child_order.reduce_only is False


def test_sign_flip_reduces_to_flat_before_opposite_open() -> None:
    controller = TargetExposureController(no_trade_band=0.05)

    reducing = controller.plan(_input(target=-0.3, realized=2.0))

    assert reducing.phase is ControllerPhase.REDUCING
    assert reducing.child_order is not None
    assert reducing.child_order.quantity == pytest.approx(-2.0)
    assert reducing.child_order.reduce_only is True
    assert reducing.deferred_target_quantity == pytest.approx(-3.0)

    opening = controller.plan(_input(target=-0.3, realized=0.0))
    assert opening.phase is ControllerPhase.OPENING
    assert opening.child_order is not None
    assert opening.child_order.quantity == pytest.approx(-3.0)
    assert opening.child_order.reduce_only is False


def test_small_change_inside_no_trade_band_does_not_emit_order() -> None:
    controller = TargetExposureController(no_trade_band=0.05)

    plan = controller.plan(_input(target=0.52, realized=5.0))

    assert plan.phase is ControllerPhase.IDLE
    assert plan.raw_target_exposure == pytest.approx(0.52)
    assert plan.effective_target_exposure == pytest.approx(0.5)
    assert plan.child_order is None


def test_emergency_flatten_bypasses_band_and_never_opens() -> None:
    controller = TargetExposureController(no_trade_band=0.05)

    plan = controller.plan(
        _input(target=0.52, realized=5.0, emergency_flatten=True)
    )

    assert plan.effective_target_exposure == 0.0
    assert plan.phase is ControllerPhase.REDUCING
    assert plan.child_order is not None
    assert plan.child_order.quantity == pytest.approx(-5.0)
    assert plan.child_order.reduce_only is True


def test_halted_controller_fails_closed_without_order() -> None:
    controller = TargetExposureController(no_trade_band=0.05)

    plan = controller.plan(_input(target=0.8, halted=True))

    assert plan.phase is ControllerPhase.HALTED
    assert plan.child_order is None
    assert plan.cancel_working_orders is False


def test_invalid_target_or_market_state_fails_closed() -> None:
    controller = TargetExposureController(no_trade_band=0.05)

    with pytest.raises(ValueError, match="target_exposure"):
        controller.plan(_input(target=1.1))
    with pytest.raises(ValueError, match="reference_price"):
        controller.plan(
            TargetExposureInput(
                target_exposure=0.5,
                allocated_equity=1_000.0,
                reference_price=0.0,
                contract_multiplier=1.0,
                realized_quantity=0.0,
                working_remaining_quantities=(),
            )
        )
