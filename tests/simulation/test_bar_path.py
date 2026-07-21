from __future__ import annotations

import pytest

from trade_rl.simulation.bar_path import (
    BarPathError,
    PathMode,
    TriggerSegment,
    evaluate_trigger,
    select_bar_path,
    volume_fraction_for_segment,
)
from trade_rl.simulation.orders import (
    OrderIntent,
    OrderStatus,
    OrderType,
    PendingOrder,
    TimeInForce,
)


def _order(
    *,
    quantity: float = 1.0,
    order_type: OrderType = OrderType.LIMIT,
    limit_price: float | None = 99.0,
    stop_price: float | None = None,
    status: OrderStatus = OrderStatus.ELIGIBLE,
) -> PendingOrder:
    intent = OrderIntent.create(
        dataset_id="d" * 64,
        target_identity=f"target-{quantity}-{order_type.value}",
        execution_policy_digest="e" * 64,
        symbol_index=0,
        requested_quantity=quantity,
        order_type=order_type,
        time_in_force=TimeInForce.GTC,
        limit_price=limit_price,
        stop_price=stop_price,
        submit_index=0,
        eligible_index=1,
        expiry_index=None,
        submission_reference_price=100.0,
        decision_equity=1_000.0,
    )
    pending = PendingOrder.from_intent(intent).mark_eligible(processing_index=1)
    if status is OrderStatus.TRIGGERED:
        return pending.mark_triggered(processing_index=1)
    return pending


def test_neutral_path_uses_closest_extreme_and_low_first_on_tie() -> None:
    closer_low = select_bar_path(
        open_price=100.0,
        high=110.0,
        low=98.0,
        close=105.0,
        mode=PathMode.NEUTRAL,
        active_directions=frozenset({1}),
    )
    tie = select_bar_path(
        open_price=100.0,
        high=105.0,
        low=95.0,
        close=101.0,
        mode=PathMode.NEUTRAL,
        active_directions=frozenset({-1}),
    )

    assert closer_low.points == (100.0, 98.0, 110.0, 105.0)
    assert tie.points == (100.0, 95.0, 105.0, 101.0)


def test_conservative_and_optimistic_paths_are_directional() -> None:
    conservative_buy = select_bar_path(
        open_price=100.0,
        high=110.0,
        low=90.0,
        close=101.0,
        mode=PathMode.CONSERVATIVE,
        active_directions=frozenset({1}),
    )
    optimistic_buy = select_bar_path(
        open_price=100.0,
        high=110.0,
        low=90.0,
        close=101.0,
        mode=PathMode.OPTIMISTIC,
        active_directions=frozenset({1}),
    )
    conservative_sell = select_bar_path(
        open_price=100.0,
        high=110.0,
        low=90.0,
        close=99.0,
        mode=PathMode.CONSERVATIVE,
        active_directions=frozenset({-1}),
    )

    assert conservative_buy.points == (100.0, 110.0, 90.0, 101.0)
    assert optimistic_buy.points == (100.0, 90.0, 110.0, 101.0)
    assert conservative_sell.points == (100.0, 90.0, 110.0, 99.0)


def test_mixed_directions_force_one_neutral_path() -> None:
    result = select_bar_path(
        open_price=100.0,
        high=109.0,
        low=98.0,
        close=104.0,
        mode=PathMode.CONSERVATIVE,
        active_directions=frozenset({-1, 1}),
    )

    assert result.mode is PathMode.NEUTRAL
    assert result.mixed_direction_fallback
    assert result.points == (100.0, 98.0, 109.0, 104.0)


def test_buy_limit_gap_executes_at_open_below_limit() -> None:
    path = select_bar_path(
        open_price=98.0,
        high=102.0,
        low=97.0,
        close=101.0,
        mode=PathMode.CONSERVATIVE,
        active_directions=frozenset({1}),
    )
    decision = evaluate_trigger(_order(limit_price=99.0), path)

    assert decision.executable
    assert not decision.triggered
    assert decision.execution_price == pytest.approx(98.0)
    assert decision.segment is TriggerSegment.OPEN
    assert decision.available_volume_fraction == pytest.approx(1.0)


def test_buy_and_sell_limits_report_touch_segment_and_fraction() -> None:
    buy_path = select_bar_path(
        open_price=100.0,
        high=105.0,
        low=95.0,
        close=101.0,
        mode=PathMode.CONSERVATIVE,
        active_directions=frozenset({1}),
    )
    sell_path = select_bar_path(
        open_price=100.0,
        high=105.0,
        low=95.0,
        close=99.0,
        mode=PathMode.CONSERVATIVE,
        active_directions=frozenset({-1}),
    )

    buy = evaluate_trigger(_order(limit_price=99.0), buy_path)
    sell = evaluate_trigger(
        _order(quantity=-1.0, limit_price=101.0),
        sell_path,
    )

    assert buy.segment is TriggerSegment.SECOND_EXTREME
    assert buy.available_volume_fraction == pytest.approx(0.25)
    assert buy.execution_price == pytest.approx(99.0)
    assert sell.segment is TriggerSegment.SECOND_EXTREME
    assert sell.available_volume_fraction == pytest.approx(0.25)
    assert sell.execution_price == pytest.approx(101.0)


def test_untouched_limit_is_explicit_no_fill() -> None:
    path = select_bar_path(
        open_price=100.0,
        high=104.0,
        low=99.5,
        close=102.0,
        mode=PathMode.NEUTRAL,
        active_directions=frozenset({1}),
    )
    decision = evaluate_trigger(_order(limit_price=99.0), path)

    assert not decision.executable
    assert decision.execution_price is None
    assert decision.reason == "not_touched"
    assert decision.available_volume_fraction == 0.0


def test_stop_gap_uses_open_and_conservative_intrabar_stop_uses_adverse_reachable_price() -> (
    None
):
    gap_path = select_bar_path(
        open_price=103.0,
        high=108.0,
        low=101.0,
        close=106.0,
        mode=PathMode.CONSERVATIVE,
        active_directions=frozenset({1}),
    )
    intrabar_path = select_bar_path(
        open_price=100.0,
        high=110.0,
        low=95.0,
        close=106.0,
        mode=PathMode.CONSERVATIVE,
        active_directions=frozenset({1}),
    )
    gap = evaluate_trigger(
        _order(order_type=OrderType.STOP_MARKET, limit_price=None, stop_price=102.0),
        gap_path,
    )
    intrabar = evaluate_trigger(
        _order(order_type=OrderType.STOP_MARKET, limit_price=None, stop_price=105.0),
        intrabar_path,
    )

    assert gap.triggered and gap.execution_price == pytest.approx(103.0)
    assert gap.segment is TriggerSegment.OPEN
    assert intrabar.triggered
    assert intrabar.segment is TriggerSegment.FIRST_EXTREME
    assert intrabar.execution_price == pytest.approx(110.0)
    assert intrabar.available_volume_fraction == pytest.approx(0.5)


def test_triggered_stop_persists_and_executes_as_market_at_next_open() -> None:
    path = select_bar_path(
        open_price=97.0,
        high=100.0,
        low=94.0,
        close=96.0,
        mode=PathMode.CONSERVATIVE,
        active_directions=frozenset({1}),
    )
    decision = evaluate_trigger(
        _order(
            order_type=OrderType.STOP_MARKET,
            limit_price=None,
            stop_price=105.0,
            status=OrderStatus.TRIGGERED,
        ),
        path,
    )

    assert decision.executable and decision.triggered
    assert decision.execution_price == pytest.approx(97.0)
    assert decision.segment is TriggerSegment.OPEN


def test_market_is_open_executable_and_close_segment_has_zero_volume() -> None:
    path = select_bar_path(
        open_price=100.0,
        high=102.0,
        low=98.0,
        close=101.0,
        mode=PathMode.NEUTRAL,
        active_directions=frozenset({1}),
    )
    market = evaluate_trigger(
        _order(order_type=OrderType.MARKET, limit_price=None),
        path,
    )

    assert market.segment is TriggerSegment.OPEN
    assert market.available_volume_fraction == 1.0
    assert volume_fraction_for_segment(TriggerSegment.CLOSE) == 0.0


def test_invalid_ohlc_fails_closed() -> None:
    with pytest.raises(BarPathError, match="OHLC"):
        select_bar_path(
            open_price=100.0,
            high=99.0,
            low=95.0,
            close=98.0,
            mode=PathMode.NEUTRAL,
            active_directions=frozenset(),
        )
