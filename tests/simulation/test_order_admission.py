from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest

from trade_rl.simulation.accounting import BookState
from trade_rl.simulation.orders.admission import (
    OrderAdmissionError,
    OrderAdmissionPolicy,
    price_on_tick_grid,
    snap_price_to_tick,
)
from trade_rl.simulation.orders.model import (
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


@pytest.mark.parametrize(
    ("open_price", "multiplier", "limit_price", "expected_reason"),
    [
        (1e-200, 1e-200, 1e200, "unrepresentable_notional"),
        (1e-160, 5e-164, 1e-159, None),
    ],
)
def test_nonmarketable_limit_open_reservation_requires_representable_notional(
    open_price: float,
    multiplier: float,
    limit_price: float,
    expected_reason: str | None,
) -> None:
    """Preflight open-risk value even when the limit notional is nonzero."""
    book = BookState.zero(1, 1_000.0, np.array([open_price]), np.array([multiplier]))
    intent = OrderIntent.create(
        dataset_id="d" * 64,
        target_identity="nonmarketable-sell-open-underflow",
        execution_policy_digest="e" * 64,
        symbol_index=0,
        requested_quantity=-1.0,
        order_type=OrderType.LIMIT,
        time_in_force=TimeInForce.GTC,
        limit_price=limit_price,
        stop_price=None,
        submit_index=0,
        eligible_index=1,
        expiry_index=None,
        submission_reference_price=open_price,
        decision_equity=1_000.0,
    )
    policy = OrderAdmissionPolicy(
        expected_dataset_id="d" * 64,
        expected_execution_policy_digest="e" * 64,
        allow_short=True,
        max_leverage=1.0,
    )
    with np.errstate(over="raise", under="raise", invalid="raise"):
        decision = policy.evaluate(
            intent,
            book=book,
            processing_index=1,
            asset_active=True,
            tradable=True,
            buy_allowed=True,
            sell_allowed=True,
            borrow_available=True,
            tick_size=0.0,
            lot_size=0.0,
            minimum_notional=0.0,
            reference_prices=np.array([open_price]),
        )
    assert decision.accepted is (expected_reason is None)
    assert decision.reason == expected_reason
    if expected_reason is None:
        assert decision.admitted_notional > 0.0
    assert book.quantities.tolist() == [0.0]
    assert book.cash == 1_000.0


@pytest.mark.parametrize(
    ("price", "tick_size"),
    [(1e308, 3e-200), (1e200, 3e-300)],
)
def test_tick_grid_quotient_overflow_uses_exact_decimal_projection(
    price: float, tick_size: float
) -> None:
    """A finite quote/tick pair must never reach round(inf) or int*float overflow."""
    assert math.isinf(price / tick_size)
    price_exact = Fraction(str(price))
    tick_exact = Fraction(str(tick_size))
    assert price_exact % tick_exact != 0  # Exercise the nontrivial grid check.
    nearest = round(price_exact / tick_exact) * tick_exact
    assert abs(price_exact - nearest) < Fraction.from_float(math.ulp(price))

    assert price_on_tick_grid(price, tick_size)
    assert snap_price_to_tick(price, tick_size, round_up=False) == price
    assert snap_price_to_tick(price, tick_size, round_up=True) == price


def test_large_limit_price_tiny_tick_is_admitted_without_numeric_exception() -> None:
    """Real order admission must not crash on a finite but huge tick quotient."""
    price = 1e308
    tick = 3e-200
    book = BookState.zero(1, price, np.array([price]), np.ones(1))
    intent = OrderIntent.create(
        dataset_id="d" * 64,
        target_identity="overflowing-tick-quotient",
        execution_policy_digest="e" * 64,
        symbol_index=0,
        requested_quantity=0.1,
        order_type=OrderType.LIMIT,
        time_in_force=TimeInForce.GTC,
        limit_price=price,
        stop_price=None,
        submit_index=0,
        eligible_index=1,
        expiry_index=None,
        submission_reference_price=price,
        decision_equity=price,
    )
    policy = OrderAdmissionPolicy(
        expected_dataset_id="d" * 64,
        expected_execution_policy_digest="e" * 64,
        allow_short=True,
        max_leverage=1.0,
    )
    with np.errstate(over="raise", under="raise", invalid="raise"):
        result = policy.evaluate(
            intent,
            book=book,
            processing_index=1,
            asset_active=True,
            tradable=True,
            buy_allowed=True,
            sell_allowed=True,
            borrow_available=True,
            tick_size=tick,
            lot_size=0.0,
            minimum_notional=0.0,
            reference_prices=np.array([price]),
        )
    assert result.accepted
    assert result.reason is None
    assert result.admitted_notional > 0.0
    assert book.quantities.tolist() == [0.0]
    assert book.cash == price


def test_finite_tick_quotient_with_overflowed_nearest_projection() -> None:
    """A finite ratio must not grant grid membership through inf <= inf."""
    price = float.fromhex("0x1.fffffffffffffp+1023")
    tick = 1e308
    ratio = price / tick
    assert math.isfinite(ratio)
    assert math.isinf(round(ratio) * tick)
    assert Fraction(str(price)) % Fraction(str(tick)) != 0

    with np.errstate(over="raise", under="raise", invalid="raise"):
        assert price_on_tick_grid(price, tick) is False
        assert snap_price_to_tick(price, tick, round_up=False) == tick
        with pytest.raises(ValueError, match="finite"):
            snap_price_to_tick(price, tick, round_up=True)


@pytest.mark.parametrize(
    ("price", "tick", "expected_ratio"),
    [(5e-324, 1e-323, 0.5), (5e-324, 1e308, 0.0)],
)
def test_positive_price_zero_nearest_tick_uses_conservative_direction(
    price: float, tick: float, expected_ratio: float
) -> None:
    """Near zero, upward snap uses the first tick and downward rejects zero."""
    assert price / tick == expected_ratio
    assert Fraction(str(price)) % Fraction(str(tick)) != 0
    with np.errstate(over="raise", under="raise", invalid="raise"):
        assert price_on_tick_grid(price, tick) is False
        assert snap_price_to_tick(price, tick, round_up=True) == tick
        with pytest.raises(ValueError, match="positive"):
            snap_price_to_tick(price, tick, round_up=False)


def test_valid_positive_subnormal_tick_and_float_tolerance_are_preserved() -> None:
    """Reject zero-nearest aliases, not genuine tiny grid prices or ULP noise."""
    tiny = 5e-324
    with np.errstate(over="raise", under="raise", invalid="raise"):
        assert price_on_tick_grid(tiny, tiny)
        assert snap_price_to_tick(tiny, tiny, round_up=False) == tiny
        assert snap_price_to_tick(tiny, tiny, round_up=True) == tiny
        price, tick = 99.0, 0.1 * 3.0
        assert price_on_tick_grid(price, tick)
        snapped = round(price / tick) * tick
        # Once this float price is accepted as an on-grid representation,
        # conservative DOWN snapping cannot move it upward by one ULP.
        assert snap_price_to_tick(price, tick, round_up=False) == price
        assert snap_price_to_tick(price, tick, round_up=True) == snapped
        assert price <= snapped


@pytest.mark.parametrize(
    ("price", "tick_size"),
    [
        (180000000000000.3, 0.3),
        (99.0, 0.3),
    ],
)
def test_exact_decimal_tick_snap_keeps_price_without_one_ulp_drift(
    price: float, tick_size: float
) -> None:
    """An already valid decimal grid point must never change when snapping."""
    assert Fraction(str(price)) % Fraction(str(tick_size)) == 0
    assert price_on_tick_grid(price, tick_size)
    with np.errstate(over="raise", under="raise", invalid="raise"):
        assert snap_price_to_tick(price, tick_size, round_up=True) == price
        assert snap_price_to_tick(price, tick_size, round_up=False) == price


def test_explicit_limit_rejects_offgrid_subnormal_under_strict_numpy() -> None:
    """Explicit LIMIT admission refuses an off-tick subnormal before mutation."""
    offgrid, tick = 5e-324, 1e-323
    book = BookState.zero(1, 1_000.0, np.array([100.0]), np.ones(1))
    intent = OrderIntent.create(
        dataset_id="d" * 64,
        target_identity="offgrid-subnormal-zero-nearest",
        execution_policy_digest="e" * 64,
        symbol_index=0,
        requested_quantity=1.0,
        order_type=OrderType.LIMIT,
        time_in_force=TimeInForce.GTC,
        limit_price=offgrid,
        stop_price=None,
        submit_index=0,
        eligible_index=1,
        expiry_index=None,
        submission_reference_price=100.0,
        decision_equity=1_000.0,
    )
    policy = OrderAdmissionPolicy(
        expected_dataset_id="d" * 64,
        expected_execution_policy_digest="e" * 64,
        allow_short=True,
        max_leverage=1.0,
    )
    with np.errstate(over="raise", under="raise", invalid="raise"):
        result = policy.evaluate(
            intent,
            book=book,
            processing_index=1,
            asset_active=True,
            tradable=True,
            buy_allowed=True,
            sell_allowed=True,
            borrow_available=True,
            tick_size=tick,
            lot_size=0.0,
            minimum_notional=0.0,
            reference_prices=np.array([100.0]),
        )
    assert not result.accepted
    assert result.reason == "price_not_on_tick"
    assert book.quantities.tolist() == [0.0]
    assert book.cash == 1_000.0


def test_positive_nearest_half_tick_cannot_override_conservative_snapping() -> None:
    """ULP tolerance cannot make a half-tick price on grid or round down up."""
    price, tick = 1.5e-323, 1e-323
    assert round(price / tick) == 2
    assert Fraction(str(price)) % Fraction(str(tick)) != 0
    with np.errstate(over="raise", under="raise", invalid="raise"):
        assert price_on_tick_grid(price, tick) is False
        assert snap_price_to_tick(price, tick, round_up=False) == tick
        assert snap_price_to_tick(price, tick, round_up=True) == 2 * tick


@pytest.mark.parametrize(
    ("price", "tick_size"),
    [
        (1.25, 3 * 2**-55),
        (1.75 + 3 * 2**-52, 3 * 2**-54),
    ],
)
def test_large_finite_tick_quotient_uses_decimal_count_and_projection(
    price: float, tick_size: float
) -> None:
    """Avoid one-tick quotient loss and nonconservative float projection."""
    ratio = price / tick_size
    assert math.isfinite(ratio) and ratio > 2**52
    tick_exact = Fraction(str(tick_size))
    price_exact = Fraction(str(price))
    exact_count = round(price_exact / tick_exact)
    assert float(exact_count * tick_exact) == price
    assert exact_count != round(ratio) or round(ratio) * tick_size != price

    with np.errstate(over="raise", under="raise", invalid="raise"):
        assert price_on_tick_grid(price, tick_size)
        assert snap_price_to_tick(price, tick_size, round_up=False) == price
        assert snap_price_to_tick(price, tick_size, round_up=True) == price


def test_subnormal_tick_decimal_grid_survives_float_quotient_loss() -> None:
    """Tiny tick and larger tiny price must use the same decimal grid authority."""
    price, tick = 1.3e-319, 1e-322
    assert tick > 8.0 * math.ulp(price)
    assert Fraction(str(price)) % Fraction(str(tick)) == 0
    assert price / tick != Fraction(str(price)) / Fraction(str(tick))

    with np.errstate(over="raise", under="raise", invalid="raise"):
        assert price_on_tick_grid(price, tick)
        assert snap_price_to_tick(price, tick, round_up=False) == price
        assert snap_price_to_tick(price, tick, round_up=True) == price
        neighbor = math.nextafter(price, math.inf)
        assert price_on_tick_grid(neighbor, tick)
        assert snap_price_to_tick(neighbor, tick, round_up=False) == price

        midpoint = 1.3005e-319
        assert Fraction(str(midpoint)) / Fraction(str(tick)) == Fraction(2601, 2)
        assert price_on_tick_grid(midpoint, tick) is False
        assert snap_price_to_tick(midpoint, tick, round_up=False) == price
        assert snap_price_to_tick(midpoint, tick, round_up=True) == 1.301e-319


def test_half_tick_float_rounding_never_reverses_conservative_snap() -> None:
    """An off-grid quote beyond half a tick must not snap up on round_down."""
    price = 1.0 + 37 * 2**-52
    tick = 65 * 2**-55
    exact_tick = Fraction(str(tick))
    ratio = Fraction(str(price)) / exact_tick
    nearest_count = round(price / tick)
    tolerance = 4.0 * max(math.ulp(price), math.ulp(nearest_count * tick))
    assert tick > 8.0 * math.ulp(price)
    assert tick > 2.0 * tolerance
    assert 2 * abs(Fraction(str(price)) - nearest_count * exact_tick) >= exact_tick

    with np.errstate(over="raise", under="raise", invalid="raise"):
        assert price_on_tick_grid(price, tick) is False
        downward = snap_price_to_tick(price, tick, round_up=False)
        upward = snap_price_to_tick(price, tick, round_up=True)

    assert downward == float(math.floor(ratio) * exact_tick)
    assert upward == float(math.ceil(ratio) * exact_tick)
    assert downward <= price <= upward
    assert price_on_tick_grid(downward, tick)
    assert price_on_tick_grid(upward, tick)


@pytest.mark.parametrize(
    ("price", "expected_down", "expected_up"),
    [
        (180000000000000.4, 180000000000000.3, 180000000000000.6),
        (180000000000000.5, 180000000000000.3, 180000000000000.6),
    ],
)
def test_wide_tick_projection_rejects_offgrid_and_preserves_snap_direction(
    price: float, expected_down: float, expected_up: float
) -> None:
    """Four ULPs must not admit a meaningful off-tick price at large scales."""
    tick = 0.3
    exact_tick = Fraction(str(tick))
    assert tick > 8 * math.ulp(price)
    assert Fraction(str(price)) % exact_tick != 0
    exact_ratio = Fraction.from_float(price) / exact_tick
    assert expected_down == float(math.floor(exact_ratio) * exact_tick)
    assert expected_up == float(math.ceil(exact_ratio) * exact_tick)

    with np.errstate(over="raise", under="raise", invalid="raise"):
        assert price_on_tick_grid(price, tick) is False
        down = snap_price_to_tick(price, tick, round_up=False)
        up = snap_price_to_tick(price, tick, round_up=True)

    assert down == expected_down
    assert up == expected_up
    assert down <= price <= up
    assert price_on_tick_grid(down, tick)
    assert price_on_tick_grid(up, tick)


@pytest.mark.parametrize(
    ("price", "tick_size"),
    [
        (1.5 + 3 * 2**-52, 2e-16),
        (1.5 + 2**-52, 8e-17),
    ],
)
def test_dense_tick_binary64_projection_is_recognized_as_a_valid_grid_price(
    price: float, tick_size: float
) -> None:
    """The shortest decimal float may be an exact half-tick even for a true image."""
    decimal_price = Fraction(str(price))
    decimal_tick = Fraction(str(tick_size))
    assert decimal_price % decimal_tick != 0
    decimal_count = round(decimal_price / decimal_tick)
    assert 2 * abs(decimal_price - decimal_count * decimal_tick) >= decimal_tick
    binary_count = round(Fraction.from_float(price) / decimal_tick)
    assert binary_count > 0
    assert float(binary_count * decimal_tick) == price

    with np.errstate(over="raise", under="raise", invalid="raise"):
        assert price_on_tick_grid(price, tick_size)
        down = snap_price_to_tick(price, tick_size, round_up=False)
        up = snap_price_to_tick(price, tick_size, round_up=True)
        assert down <= price <= up
        assert price_on_tick_grid(down, tick_size)
        assert price_on_tick_grid(up, tick_size)


def test_dense_tick_checks_both_neighbors_at_asymmetric_binary64_boundary() -> None:
    """An exact odd half-tick may round back from only its upper neighbor."""
    price = 1.0
    tick_size = 1.6777216e-16
    tick_exact = Fraction(str(tick_size))
    ratio = Fraction.from_float(price) / tick_exact
    assert ratio.denominator == 2
    lower = math.floor(ratio)
    upper = math.ceil(ratio)
    assert lower < upper
    assert float(lower * tick_exact) < price
    assert float(upper * tick_exact) == price

    with np.errstate(over="raise", under="raise", invalid="raise"):
        assert price_on_tick_grid(price, tick_size)
        down = snap_price_to_tick(price, tick_size, round_up=False)
        up = snap_price_to_tick(price, tick_size, round_up=True)
        assert down < price
        assert up == price
        assert price_on_tick_grid(down, tick_size)
        assert price_on_tick_grid(up, tick_size)


def test_dense_tick_nearest_projection_overflow_fails_closed() -> None:
    """A nearest tick beyond finite float64 must not raise from grid checking."""
    price = float.fromhex("0x1.fffffffffffffp+1023")
    tick_size = 3.732139401517567e292
    tick_exact = Fraction(str(tick_size))
    assert tick_size <= 2 * math.ulp(price)
    count = round(Fraction.from_float(price) / tick_exact)
    with pytest.raises(OverflowError):
        float(count * tick_exact)

    with np.errstate(over="raise", under="raise", invalid="raise"):
        assert price_on_tick_grid(price, tick_size) is False
        down = snap_price_to_tick(price, tick_size, round_up=False)
        assert math.isfinite(down)
        assert down <= price
        assert price_on_tick_grid(down, tick_size)
        with pytest.raises(ValueError, match="finite"):
            snap_price_to_tick(price, tick_size, round_up=True)


def test_nonmarketable_limit_minimum_notional_uses_limit_price() -> None:
    intent = OrderIntent.create(
        dataset_id="d" * 64,
        target_identity="sell-limit",
        execution_policy_digest="e" * 64,
        symbol_index=0,
        requested_quantity=-1.0,
        order_type=OrderType.LIMIT,
        time_in_force=TimeInForce.GTC,
        limit_price=110.0,
        stop_price=None,
        submit_index=0,
        eligible_index=1,
        expiry_index=None,
        submission_reference_price=90.0,
        decision_equity=1_000.0,
    )
    decision = OrderAdmissionPolicy(
        expected_dataset_id="d" * 64,
        expected_execution_policy_digest="e" * 64,
        allow_short=True,
        max_leverage=1.0,
    ).evaluate(
        intent,
        book=_book(quantity=1.0),
        processing_index=1,
        asset_active=True,
        tradable=True,
        buy_allowed=True,
        sell_allowed=True,
        borrow_available=True,
        tick_size=0.01,
        lot_size=0.0,
        minimum_notional=100.0,
        reference_prices=np.array([90.0]),
    )

    assert decision.accepted
    assert decision.reason is None
    assert decision.admitted_notional == pytest.approx(110.0)


@pytest.mark.parametrize(
    ("quantity", "open_price", "limit_price", "minimum_notional", "accepted"),
    [
        (-1.0, 120.0, 110.0, 115.0, True),
        (1.0, 90.0, 110.0, 100.0, False),
    ],
)
def test_marketable_limit_minimum_notional_uses_processing_open(
    quantity: float,
    open_price: float,
    limit_price: float,
    minimum_notional: float,
    accepted: bool,
) -> None:
    intent = OrderIntent.create(
        dataset_id="d" * 64,
        target_identity=f"marketable-limit-{quantity}",
        execution_policy_digest="e" * 64,
        symbol_index=0,
        requested_quantity=quantity,
        order_type=OrderType.LIMIT,
        time_in_force=TimeInForce.GTC,
        limit_price=limit_price,
        stop_price=None,
        submit_index=0,
        eligible_index=1,
        expiry_index=None,
        submission_reference_price=open_price,
        decision_equity=1_000.0,
    )
    decision = OrderAdmissionPolicy(
        expected_dataset_id="d" * 64,
        expected_execution_policy_digest="e" * 64,
        allow_short=True,
        max_leverage=1.0,
    ).evaluate(
        intent,
        book=_book(quantity=1.0 if quantity < 0.0 else 0.0),
        processing_index=1,
        asset_active=True,
        tradable=True,
        buy_allowed=True,
        sell_allowed=True,
        borrow_available=True,
        tick_size=0.01,
        lot_size=0.0,
        minimum_notional=minimum_notional,
        reference_prices=np.array([open_price]),
    )

    assert decision.accepted is accepted
    if accepted:
        assert decision.reason is None
        assert decision.admitted_notional == pytest.approx(abs(quantity) * open_price)
    else:
        assert decision.reason == "below_minimum_notional"


@pytest.mark.parametrize(
    ("order_type", "price_field"),
    [
        (OrderType.LIMIT, "limit_price"),
        (OrderType.STOP_MARKET, "stop_price"),
    ],
)
def test_nonmarket_order_price_must_align_to_tick_grid(
    order_type: OrderType,
    price_field: str,
) -> None:
    prices = {"limit_price": None, "stop_price": None}
    prices[price_field] = 100.26
    intent = OrderIntent.create(
        dataset_id="d" * 64,
        target_identity=f"off-tick-{order_type.value}",
        execution_policy_digest="e" * 64,
        symbol_index=0,
        requested_quantity=1.0,
        order_type=order_type,
        time_in_force=TimeInForce.GTC,
        limit_price=prices["limit_price"],
        stop_price=prices["stop_price"],
        submit_index=0,
        eligible_index=1,
        expiry_index=None,
        submission_reference_price=101.0,
        decision_equity=1_000.0,
    )

    decision = OrderAdmissionPolicy(
        expected_dataset_id="d" * 64,
        expected_execution_policy_digest="e" * 64,
        allow_short=True,
        max_leverage=1.0,
    ).evaluate(
        intent,
        book=_book(),
        processing_index=1,
        asset_active=True,
        tradable=True,
        buy_allowed=True,
        sell_allowed=True,
        borrow_available=True,
        tick_size=0.5,
        lot_size=0.0,
        minimum_notional=0.0,
        reference_prices=np.array([101.0]),
    )

    assert not decision.accepted
    assert decision.reason == "price_not_on_tick"


def test_aligned_nonmarket_order_price_is_admitted() -> None:
    intent = OrderIntent.create(
        dataset_id="d" * 64,
        target_identity="aligned-limit",
        execution_policy_digest="e" * 64,
        symbol_index=0,
        requested_quantity=1.0,
        order_type=OrderType.LIMIT,
        time_in_force=TimeInForce.GTC,
        limit_price=100.5,
        stop_price=None,
        submit_index=0,
        eligible_index=1,
        expiry_index=None,
        submission_reference_price=101.0,
        decision_equity=1_000.0,
    )

    decision = OrderAdmissionPolicy(
        expected_dataset_id="d" * 64,
        expected_execution_policy_digest="e" * 64,
        allow_short=True,
        max_leverage=1.0,
    ).evaluate(
        intent,
        book=_book(),
        processing_index=1,
        asset_active=True,
        tradable=True,
        buy_allowed=True,
        sell_allowed=True,
        borrow_available=True,
        tick_size=0.5,
        lot_size=0.0,
        minimum_notional=0.0,
        reference_prices=np.array([101.0]),
    )

    assert decision.accepted
    assert decision.reason is None


def test_tick_grid_accepts_float_projection_of_scaled_tick() -> None:
    intent = OrderIntent.create(
        dataset_id="d" * 64,
        target_identity="scaled-tick-limit",
        execution_policy_digest="e" * 64,
        symbol_index=0,
        requested_quantity=1.0,
        order_type=OrderType.LIMIT,
        time_in_force=TimeInForce.GTC,
        limit_price=99.0,
        stop_price=None,
        submit_index=0,
        eligible_index=1,
        expiry_index=None,
        submission_reference_price=101.0,
        decision_equity=1_000.0,
    )

    decision = OrderAdmissionPolicy(
        expected_dataset_id="d" * 64,
        expected_execution_policy_digest="e" * 64,
        allow_short=True,
        max_leverage=1.0,
    ).evaluate(
        intent,
        book=_book(),
        processing_index=1,
        asset_active=True,
        tradable=True,
        buy_allowed=True,
        sell_allowed=True,
        borrow_available=True,
        tick_size=0.1 * 3.0,
        lot_size=0.0,
        minimum_notional=0.0,
        reference_prices=np.array([101.0]),
    )

    assert decision.accepted
    assert decision.reason is None


def test_zero_tick_size_keeps_unconstrained_price_behavior() -> None:
    intent = OrderIntent.create(
        dataset_id="d" * 64,
        target_identity="zero-tick-limit",
        execution_policy_digest="e" * 64,
        symbol_index=0,
        requested_quantity=1.0,
        order_type=OrderType.LIMIT,
        time_in_force=TimeInForce.GTC,
        limit_price=100.26,
        stop_price=None,
        submit_index=0,
        eligible_index=1,
        expiry_index=None,
        submission_reference_price=101.0,
        decision_equity=1_000.0,
    )

    decision = OrderAdmissionPolicy(
        expected_dataset_id="d" * 64,
        expected_execution_policy_digest="e" * 64,
        allow_short=True,
        max_leverage=1.0,
    ).evaluate(
        intent,
        book=_book(),
        processing_index=1,
        asset_active=True,
        tradable=True,
        buy_allowed=True,
        sell_allowed=True,
        borrow_available=True,
        tick_size=0.0,
        lot_size=0.0,
        minimum_notional=0.0,
        reference_prices=np.array([101.0]),
    )

    assert decision.accepted
    assert decision.reason is None


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


def test_nonfinite_admitted_notional_is_rejected_before_order_execution() -> None:
    book = BookState.zero(1, 1e308, np.array([1e308]))
    intent = OrderIntent.create(
        dataset_id="d" * 64,
        target_identity="overflowed-notional",
        execution_policy_digest="e" * 64,
        symbol_index=0,
        requested_quantity=2.0,
        order_type=OrderType.MARKET,
        time_in_force=TimeInForce.GTC,
        limit_price=None,
        stop_price=None,
        submit_index=0,
        eligible_index=1,
        expiry_index=None,
        submission_reference_price=1e308,
        decision_equity=1e308,
    )
    policy = OrderAdmissionPolicy(
        expected_dataset_id="d" * 64,
        expected_execution_policy_digest="e" * 64,
        allow_short=True,
        max_leverage=2.0,
    )
    decision = policy.evaluate(
        intent,
        book=book,
        processing_index=1,
        asset_active=True,
        tradable=True,
        buy_allowed=True,
        sell_allowed=True,
        borrow_available=True,
        tick_size=0.0,
        lot_size=0.0,
        minimum_notional=0.0,
        reference_prices=np.array([1e308]),
    )
    assert not decision.accepted
    assert decision.reason == "pretrade_leverage_exceeded"


@pytest.mark.parametrize("quantity", [-1.0, 1.0])
@pytest.mark.parametrize(
    ("price", "multiplier", "must_reject"),
    [
        (1e-200, 1e-200, True),  # Mathematical notional is nonzero, float is 0.
        (1e-150, 1e-150, False),  # Tiny but representable.
        (1e-160, 5e-164, False),  # The smallest positive subnormal.
    ],
)
def test_order_admission_matches_bookstate_notional_representability(
    quantity: float, price: float, multiplier: float, must_reject: bool
) -> None:
    book = BookState.zero(1, 1_000.0, np.array([price]), np.array([multiplier]))
    intent = OrderIntent.create(
        dataset_id="d" * 64,
        target_identity=f"tiny-notional-{quantity}-{price}-{multiplier}",
        execution_policy_digest="e" * 64,
        symbol_index=0,
        requested_quantity=quantity,
        order_type=OrderType.MARKET,
        time_in_force=TimeInForce.GTC,
        limit_price=None,
        stop_price=None,
        submit_index=0,
        eligible_index=1,
        expiry_index=None,
        submission_reference_price=price,
        decision_equity=1_000.0,
    )
    with np.errstate(under="raise"):
        decision = OrderAdmissionPolicy(
            expected_dataset_id="d" * 64,
            expected_execution_policy_digest="e" * 64,
            allow_short=True,
            max_leverage=1.0,
        ).evaluate(
            intent,
            book=book,
            processing_index=1,
            asset_active=True,
            tradable=True,
            buy_allowed=True,
            sell_allowed=True,
            borrow_available=True,
            tick_size=0.0,
            lot_size=0.0,
            minimum_notional=0.0,
            reference_prices=np.array([price]),
        )
    actual_notional = abs(quantity) * price * multiplier
    if must_reject:
        assert actual_notional == 0.0
        assert not decision.accepted
        assert decision.reason == "unrepresentable_notional"
        # The old path admitted quantity != 0 with notional == 0, then failed here.
        with pytest.raises(ValueError, match="nonzero execution leg notional"):
            book.execute_fill(
                symbol_index=0,
                quantity=quantity,
                fill_prices=np.array([price]),
                cost_amount=0.0,
                turnover=0.0,
            )
        assert book.quantities[0] == 0.0
        assert book.cash == 1_000.0
    else:
        assert actual_notional > 0.0
        assert decision.accepted
        assert decision.admitted_notional == actual_notional
        book.execute_fill(
            symbol_index=0,
            quantity=decision.admitted_quantity,
            fill_prices=np.array([price]),
            cost_amount=0.0,
            turnover=0.0,
        )
        assert book.quantities[0] == quantity
