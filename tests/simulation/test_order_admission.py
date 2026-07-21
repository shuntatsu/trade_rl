from __future__ import annotations

import numpy as np
import pytest

from trade_rl.simulation.accounting import BookState
from trade_rl.simulation.order_admission import (
    OrderAdmissionError,
    OrderAdmissionPolicy,
)
from trade_rl.simulation.orders import (
    OrderIntent,
    OrderType,
    TimeInForce,
)


def _book(*, quantity: float = 0.0, capital: float = 1_000.0) -> BookState:
    return BookState(
        quantities=np.array([quantity]),
        cash=capital - quantity * 100.0,
        mark_prices=np.array([100.0]),
        peak_value=capital,
        contract_multipliers=np.array([1.0]),
    )


def _intent(
    quantity: float = 1.0,
    *,
    dataset_id: str = "d" * 64,
    policy_digest: str = "e" * 64,
    eligible_index: int = 1,
    expiry_index: int | None = None,
) -> OrderIntent:
    return OrderIntent.create(
        dataset_id=dataset_id,
        target_identity=f"target-{quantity}",
        execution_policy_digest=policy_digest,
        symbol_index=0,
        requested_quantity=quantity,
        order_type=OrderType.MARKET,
        time_in_force=TimeInForce.GTC,
        limit_price=None,
        stop_price=None,
        submit_index=0,
        eligible_index=eligible_index,
        expiry_index=expiry_index,
        submission_reference_price=100.0,
        decision_equity=1_000.0,
    )


def _evaluate(
    intent: OrderIntent | None = None,
    *,
    book: BookState | None = None,
    processing_index: int = 1,
    asset_active: bool = True,
    tradable: bool = True,
    buy_allowed: bool = True,
    sell_allowed: bool = True,
    borrow_available: bool = True,
    tick_size: float = 0.01,
    lot_size: float = 0.0,
    minimum_notional: float = 0.0,
    max_leverage: float = 1.0,
):
    policy = OrderAdmissionPolicy(
        expected_dataset_id="d" * 64,
        expected_execution_policy_digest="e" * 64,
        allow_short=True,
        max_leverage=max_leverage,
    )
    return policy.evaluate(
        _intent() if intent is None else intent,
        book=_book() if book is None else book,
        processing_index=processing_index,
        asset_active=asset_active,
        tradable=tradable,
        buy_allowed=buy_allowed,
        sell_allowed=sell_allowed,
        borrow_available=borrow_available,
        tick_size=tick_size,
        lot_size=lot_size,
        minimum_notional=minimum_notional,
        reference_prices=np.array([100.0]),
    )


def test_valid_order_is_admitted_with_rounded_quantity() -> None:
    decision = _evaluate(_intent(1.27), lot_size=0.1)

    assert decision.accepted
    assert decision.reason is None
    assert decision.admitted_quantity == pytest.approx(1.2)
    assert decision.admitted_notional == pytest.approx(120.0)


@pytest.mark.parametrize(
    ("override", "reason"),
    [
        ({"asset_active": False}, "inactive_asset"),
        ({"tradable": False}, "non_tradable_market"),
        ({"buy_allowed": False}, "buy_disabled"),
    ],
)
def test_market_and_direction_blocks_are_explicit(
    override: dict[str, bool], reason: str
) -> None:
    decision = _evaluate(**override)
    assert not decision.accepted
    assert decision.reason == reason


def test_sell_direction_and_incremental_short_borrow_are_checked() -> None:
    sell_disabled = _evaluate(_intent(-1.0), sell_allowed=False)
    incremental_short = _evaluate(
        _intent(-2.0),
        book=_book(quantity=-1.0),
        borrow_available=False,
    )
    reducing_short = _evaluate(
        _intent(0.5),
        book=_book(quantity=-1.0),
        borrow_available=False,
    )

    assert sell_disabled.reason == "sell_disabled"
    assert incremental_short.reason == "borrow_unavailable"
    assert reducing_short.accepted


def test_identity_mismatch_and_ineligible_or_expired_state_fail_closed() -> None:
    wrong_dataset = _evaluate(_intent(dataset_id="x" * 64))
    wrong_policy = _evaluate(_intent(policy_digest="f" * 64))
    latency = _evaluate(_intent(eligible_index=3), processing_index=2)
    expired = _evaluate(_intent(expiry_index=2), processing_index=3)

    assert wrong_dataset.reason == "identity_mismatch"
    assert wrong_policy.reason == "identity_mismatch"
    assert latency.reason == "not_eligible"
    assert expired.reason == "expired"


def test_rule_validation_rounding_and_minimum_notional_are_explicit() -> None:
    invalid_rule = _evaluate(tick_size=-1.0)
    rounded_zero = _evaluate(_intent(0.05), lot_size=0.1)
    below_minimum = _evaluate(_intent(0.4), minimum_notional=50.0)

    assert invalid_rule.reason == "invalid_execution_rule"
    assert rounded_zero.reason == "zero_quantity_after_rounding"
    assert below_minimum.reason == "below_minimum_notional"


def test_short_policy_and_pretrade_leverage_gate() -> None:
    no_short_policy = OrderAdmissionPolicy(
        expected_dataset_id="d" * 64,
        expected_execution_policy_digest="e" * 64,
        allow_short=False,
        max_leverage=1.0,
    )
    no_short = no_short_policy.evaluate(
        _intent(-1.0),
        book=_book(),
        processing_index=1,
        asset_active=True,
        tradable=True,
        buy_allowed=True,
        sell_allowed=True,
        borrow_available=True,
        tick_size=0.01,
        lot_size=0.0,
        minimum_notional=0.0,
        reference_prices=np.array([100.0]),
    )
    leverage = _evaluate(_intent(11.0), max_leverage=1.0)

    assert no_short.reason == "shorting_disabled"
    assert leverage.reason == "pretrade_leverage_exceeded"


def test_invalid_reference_shape_and_policy_configuration_raise_domain_error() -> None:
    with pytest.raises(OrderAdmissionError, match="max_leverage"):
        OrderAdmissionPolicy(
            expected_dataset_id="d" * 64,
            expected_execution_policy_digest="e" * 64,
            allow_short=True,
            max_leverage=0.0,
        )
    policy = OrderAdmissionPolicy(
        expected_dataset_id="d" * 64,
        expected_execution_policy_digest="e" * 64,
        allow_short=True,
        max_leverage=1.0,
    )
    with pytest.raises(OrderAdmissionError, match="reference_prices"):
        policy.evaluate(
            _intent(),
            book=_book(),
            processing_index=1,
            asset_active=True,
            tradable=True,
            buy_allowed=True,
            sell_allowed=True,
            borrow_available=True,
            tick_size=0.01,
            lot_size=0.0,
            minimum_notional=0.0,
            reference_prices=np.array([100.0, 101.0]),
        )
