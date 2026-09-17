from __future__ import annotations

from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest

from tests.integrations.test_binance_market_order_profile import (
    _dataset,
    _profile,
    _snapshot,
)
from tests.simulation.test_order_admission import _book
from tests.simulation.test_stateful_execution import _intent
from trade_rl.simulation.accounting import BookState
from trade_rl.simulation.execution import (
    ExecutionCostConfig,
    ExecutionRuleStress,
    MarketExecutor,
)
from trade_rl.simulation.liquidity import (
    LiquidityPriority,
    LiquidityRequest,
    allocate_symbol_capacity,
)
from trade_rl.simulation.orders.model import (
    OrderBookState,
    OrderIntent,
    OrderType,
    PendingOrder,
)
from trade_rl.simulation.orders.reconciliation import reconcile_target
from trade_rl.simulation.quantities import project_quantity
from trade_rl.simulation.targets.execution import execute_target_statefully


def _executor(dataset=None, *, enabled=True, snapshot=None, stress=None, **cost):
    dataset = dataset or _dataset()
    config = replace(
        ExecutionCostConfig.zero(), processing_bar_volume_capacity=True, **cost
    )
    return MarketExecutor(
        dataset,
        config,
        rule_stress=stress,
        market_order_profile=_profile(dataset, snapshot, reduce_only_exits=enabled),
    )


def _run(executor, position, target=0.0, order_book=None):
    return execute_target_statefully(
        executor,
        _book(quantity=position),
        order_book or OrderBookState.empty(),
        np.array([target]),
        start_index=0,
        bars=1,
        target_identity="new",
    )


@pytest.mark.parametrize(
    "position,target,remaining",
    [(0.004, 0, 0), (-0.004, 0, 0), (0.008, 0.0004, 0.004), (-0.008, -0.0004, -0.004)],
)
def test_source_profile_only_exempts_actual_small_reductions(
    position, target, remaining
):
    result = _run(_executor(fee_rate=0.001), position, target)
    assert result.book.exact_quantities == (Fraction(str(remaining)),)
    assert result.fill_count == 1
    assert result.interval_cost == pytest.approx(0.0004)
    assert any(e.reduce_only and e.event_type == "filled" for e in result.order_events)
    ordinary = _run(_executor(enabled=False), position, target)
    assert ordinary.book.exact_quantities == (Fraction(str(position)),)
    assert any(e.reason == "below_minimum_notional" for e in ordinary.order_events)


@pytest.mark.parametrize(
    "position,target",
    [(0, 0.0004), (0.004, 0.0008), (0.004, -0.0004), (-0.004, 0.0004)],
)
def test_openings_increases_and_reversals_keep_ordinary_notional(position, target):
    result = _run(_executor(), position, target)
    assert result.fill_count == 0
    assert not any(e.reduce_only for e in result.order_events)
    assert any(e.reason == "below_minimum_notional" for e in result.order_events)


def test_runtime_floor_and_notional_stress_survive_reduction_exception():
    executor = _executor(
        minimum_notional=0.3, stress=ExecutionRuleStress(minimum_notional_factor=2)
    )
    result = _run(executor, 0.004)
    assert result.fill_count == 0
    assert any(e.reason == "below_minimum_notional" for e in result.order_events)


def test_lot_intersection_and_stress_bind_execution_identity():
    def mutate(p):
        p["symbols"][0]["filters"][2]["stepSize"] = "0.003"

    data = _dataset(lot=0.002)
    executor = _executor(data, snapshot=_snapshot(mutate=mutate), lot_size=0.003)
    assert executor.effective_rule_arrays(index=1)[1][0] == 0.006
    stressed = _executor(
        data,
        snapshot=_snapshot(mutate=mutate),
        lot_size=0.003,
        stress=ExecutionRuleStress(lot_size_factor=1.5),
    )
    assert stressed.effective_rule_arrays(index=1)[1][0] == 0.018
    assert (
        len(
            {
                executor.execution_policy_digest,
                stressed.execution_policy_digest,
                _executor(data, enabled=False).execution_policy_digest,
                MarketExecutor(data, executor.cost).execution_policy_digest,
            }
        )
        == 4
    )


def test_profile_burden_reports_actual_common_grid_stress_ratio():
    def mutate(payload):
        payload["symbols"][0]["filters"][2]["stepSize"] = "0.003"

    executor = _executor(
        _dataset(("BTCUSDT", "OTHER"), lot=0.002),
        snapshot=_snapshot(mutate=mutate),
        lot_size=0.003,
        stress=ExecutionRuleStress(lot_size_factor=1.5),
    )
    burden = executor.rule_burden_percentiles(start=1, stop=3)
    # Selected BTC grid tightens .006 -> .018; unselected OTHER keeps 1.5x.
    assert burden["lot_size_ratio"] == {"p50": 2.25, "p95": 3.0, "max": 3.0}


def test_profile_rejects_wrong_dataset_and_unsupported_execution_routes():
    data = _dataset()
    with pytest.raises(ValueError, match="dataset"):
        MarketExecutor(_dataset(minimum=40), market_order_profile=_profile(data))
    with pytest.raises(ValueError, match="MARKET"):
        _executor(order_type="limit")
    executor = _executor()
    with pytest.raises(ValueError, match="stateful"):
        executor.liquidate_at_close(_book(quantity=0.004), index=1)
    intent = _intent(executor, 1, order_type=OrderType.LIMIT, limit_price=100)
    result = executor.execute_orders(
        _book(), OrderBookState.empty(), (intent,), start_index=0, bars=1
    )
    assert result.fill_count == 0
    assert any(e.reason == "profile_requires_market_order" for e in result.order_events)


def test_source_quantity_bounds_apply_at_admission_and_after_capacity_clipping():
    def mutate(p):
        p["symbols"][0]["filters"][2].update(minQty="0.003", maxQty="0.01")

    snapshot = _snapshot(mutate=mutate)
    executor = _executor(snapshot=snapshot)
    for quantity, reason in [
        (0.002, "below_minimum_quantity"),
        (0.011, "above_maximum_quantity"),
    ]:
        result = _run(executor, quantity)
        assert result.fill_count == 0
        assert any(e.reason == reason for e in result.order_events)
    dataset = replace(
        _dataset(), volume=np.full((6, 1), 0.002), identity_payload_json=None
    ).with_content_identity()
    result = _run(
        _executor(dataset, snapshot=snapshot, max_participation_rate=1.0), 0.004
    )
    assert result.fill_count == 0
    assert any(e.reason == "below_minimum_quantity" for e in result.order_events)
    assert result.interval_cost == 0


def test_profile_exception_does_not_apply_to_unselected_symbols():
    data = _dataset(("BTCUSDT", "OTHER"))
    executor = _executor(data)
    book = BookState(
        quantities=np.array([0.004, 0.004]),
        cash=999.2,
        mark_prices=np.array([100.0, 100.0]),
        peak_value=1000,
        contract_multipliers=np.ones(2),
    )
    result = execute_target_statefully(
        executor,
        book,
        OrderBookState.empty(),
        np.zeros(2),
        start_index=0,
        bars=1,
        target_identity="mixed",
    )
    assert result.book.exact_quantities == (Fraction(0), Fraction("0.004"))
    assert any(
        e.symbol_index == 1
        and e.reason == "below_minimum_notional"
        and not e.reduce_only
        for e in result.order_events
    )


@pytest.mark.parametrize("old_enabled", [False, True])
def test_equal_residual_with_stale_profile_or_flag_is_replaced(old_enabled):
    executor = _executor()
    old = _executor(enabled=old_enabled, minimum_notional=0.1)
    intent = _intent(old, -0.004)
    values = intent.canonical_payload()
    values.pop("order_id")
    values["reduce_only"] = old_enabled
    intent = OrderIntent.create(**values)
    order_book = OrderBookState(
        active_orders=(PendingOrder.from_intent(intent),), terminal_orders=()
    )
    result = _run(executor, 0.004, order_book=order_book)
    assert result.book.exact_quantities == (Fraction(0),)
    assert any(
        e.event_type == "cancelled" and e.reason == "superseded"
        for e in result.order_events
    )


def test_profile_reconciliation_preserves_conservative_float_request_limit():
    executor = _executor()
    book = _book(capital=2e10)
    lots, step = 123456789, 0.123456789
    book.execute_fill(
        symbol_index=0,
        quantity=project_quantity(Fraction(lots) * Fraction(str(step))),
        fill_prices=np.array([100.0]),
        cost_amount=0,
        turnover=0,
        lot_size=step,
        lot_count=lots,
    )
    result = reconcile_target(
        dataset_id=executor.dataset.dataset_id,
        target_identity="close",
        execution_policy_digest=executor.execution_policy_digest,
        target_weights=np.zeros(1),
        book=book,
        order_book=OrderBookState.empty(),
        reference_prices=np.array([100.0]),
        decision_equity=book.portfolio_value,
        submit_index=0,
        latency_bars=1,
        order_type=OrderType.MARKET,
        time_in_force=_intent(executor, 1).time_in_force,
        expiry_index=None,
        limit_offset_rate=0,
        reduce_only_symbols=(0,),
    )
    requested = Fraction(str(result.new_intents[0].requested_quantity))
    assert result.new_intents[0].reduce_only
    assert abs(requested) < book.exact_quantities[0]
    assert abs(requested) // Fraction(str(step)) == lots - 1


def test_per_request_notional_floor_and_quantity_limits_survive_mixed_allocation():
    closing = LiquidityRequest(
        "a" * 64,
        -0.004,
        100,
        1,
        LiquidityPriority.MARKET,
        0,
        reduce_only=True,
        minimum_notional=0,
        minimum_quantity=0.001,
        maximum_quantity=0.01,
    )
    ordinary = LiquidityRequest("b" * 64, 0.004, 100, 1, LiquidityPriority.MARKET, 0)
    allocations, evidence = allocate_symbol_capacity(
        (closing, ordinary),
        processing_volume=1,
        price=100,
        contract_multiplier=1,
        participation_limit=1,
        lot_size=0.001,
        minimum_notional=50,
        initial_position=Fraction("0.004"),
    )
    assert [a.filled_quantity for a in allocations] == [-0.004, 0]
    assert allocations[1].no_fill_reason == "below_minimum_notional"
    assert evidence.consumed_capacity_notional == pytest.approx(0.4)
