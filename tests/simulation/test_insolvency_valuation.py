from __future__ import annotations

from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest

from tests.simulation.test_execution_sensitivity import _market
from trade_rl.artifacts.hashing import content_digest
from trade_rl.simulation.accounting import BookState
from trade_rl.simulation.execution import (
    ExecutionCostConfig,
    ExecutionRuleStress,
    MarketExecutor,
)
from trade_rl.simulation.orders.model import (
    OrderBookState,
    OrderIntent,
    OrderType,
    TimeInForce,
)
from trade_rl.simulation.targets.execution import execute_target_statefully


def market():
    data = _market()
    prices = np.array([100.0, 100.0, 400.0, 400.0])[:, None]
    return replace(
        data,
        open=prices,
        close=prices,
        high=prices,
        low=prices,
        mark_price=prices,
        index_price=prices,
    )


def cost():
    return replace(
        ExecutionCostConfig.zero(), fee_rate=0.002, max_participation_rate=1.0
    )


def short_entry(executor):
    book = BookState.zero(
        n_symbols=1, initial_capital=1000.0, initial_prices=executor.dataset.close[0]
    )
    return execute_target_statefully(
        executor,
        book,
        OrderBookState.empty(),
        np.array([-0.5]),
        start_index=0,
        bars=1,
        target_identity="synthetic-short-entry",
    )


@pytest.mark.parametrize(
    "mode, expected", [("floor_zero", 0.0), ("retain_debt", -501.0)]
)
def test_actual_canonical_short_gap_preserves_declared_terminal_valuation(
    mode, expected
):
    executor = MarketExecutor(market(), cost(), insolvency_valuation=mode)
    entry = short_entry(executor)
    assert entry.book.exact_quantities == (Fraction(-5),)
    assert entry.book.cash == 1499.0
    assert entry.book.total_cost == 1.0
    crashed = executor.execute_orders(
        entry.book, entry.order_book, (), start_index=1, bars=1
    )
    # Cash 1499 less the short liability 5*400 gives debt -501. No second fee.
    assert crashed.book.cash == expected
    assert crashed.book.portfolio_value == expected
    assert crashed.book.exact_quantities == (Fraction(0),)
    assert crashed.book.total_cost == 1.0
    assert crashed.book.fill_count == 1
    assert crashed.book.insolvent and crashed.book.termination_reason is not None
    assert crashed.book.margin_used == 0.0
    assert crashed.book.maintenance_margin == 0.0
    assert crashed.book.maintenance_requirement == 0.0
    # The existing net/log-return diagnostic keeps its legacy bounded semantics.
    assert crashed.interval_net_return > -1.0
    assert np.isfinite(crashed.interval_log_return)
    for _ in range(3):
        executor._update_margin(crashed.book)
        assert (
            crashed.book.cash == expected and crashed.book.portfolio_value == expected
        )


def test_default_and_explicit_floor_keep_captured_economic_digests():
    implicit = MarketExecutor(market(), cost())
    explicit = MarketExecutor(market(), cost(), insolvency_valuation="floor_zero")
    assert implicit.insolvency_valuation == "floor_zero"
    assert (
        implicit.execution_policy_digest
        == explicit.execution_policy_digest
        == ("4ea34eb7e769a941e0bb83c8a9cab0d393052d3b47a3b8dba66735e6fabb6102")
    )
    assert implicit.execution_policy_digest == implicit.cost.execution_policy_digest
    stressed = MarketExecutor(
        market(),
        cost(),
        rule_stress=ExecutionRuleStress(name="double", tick_size_factor=2.0),
    )
    assert stressed.execution_policy_digest == (
        "979a8ac04943591c4bda3d22a5d8ecb7dd224896c8df8dad49388f4ef6f092eb"
    )


@pytest.mark.parametrize("stressed", [False, True])
def test_debt_policy_wraps_resolved_economics_and_invalidates_digest_cache(stressed):
    stress = (
        ExecutionRuleStress(name="double", tick_size_factor=2.0) if stressed else None
    )
    executor = MarketExecutor(market(), cost(), rule_stress=stress)
    legacy = executor.execution_policy_digest
    executor.insolvency_valuation = "retain_debt"
    expected = content_digest(
        {
            "schema_version": "insolvency_execution_policy_v1",
            "base_policy_digest": legacy,
            "insolvency_valuation": "retain_debt",
        }
    )
    assert executor.execution_policy_digest == expected != legacy
    assert executor.execution_policy_digest == expected
    executor.insolvency_valuation = "floor_zero"
    assert executor.execution_policy_digest == legacy


@pytest.mark.parametrize(
    "source, target", [("floor_zero", "retain_debt"), ("retain_debt", "floor_zero")]
)
def test_opposite_policy_intent_is_rejected_before_filling(source, target):
    data = _market()
    source_executor = MarketExecutor(data, cost(), insolvency_valuation=source)
    target_executor = MarketExecutor(data, cost(), insolvency_valuation=target)
    intent = OrderIntent.create(
        dataset_id=data.dataset_id,
        target_identity="synthetic-policy-bound-order",
        execution_policy_digest=source_executor.execution_policy_digest,
        symbol_index=0,
        requested_quantity=-1.0,
        order_type=OrderType.MARKET,
        time_in_force=TimeInForce.GTC,
        limit_price=None,
        stop_price=None,
        submit_index=0,
        eligible_index=1,
        expiry_index=None,
        submission_reference_price=100.0,
        decision_equity=1000.0,
    )
    book = BookState.zero(
        n_symbols=1, initial_capital=1000.0, initial_prices=data.close[0]
    )
    result = target_executor.execute_orders(
        book, OrderBookState.empty(), (intent,), start_index=0, bars=1
    )
    assert result.book.exact_quantities == (Fraction(0),)
    assert result.book.cash == 1000.0 and result.book.total_cost == 0.0
    assert result.fill_count == 0
    assert result.order_book.terminal_orders[0].terminal_reason == "identity_mismatch"


@pytest.mark.parametrize(
    "value", [None, True, 0, "", "retain", "RETAIN_DEBT", ["retain_debt"]]
)
def test_unknown_or_untyped_valuation_options_reject(value):
    with pytest.raises(ValueError, match="insolvency_valuation"):
        MarketExecutor(market(), cost(), insolvency_valuation=value)


def test_mutating_policy_to_an_invalid_choice_cannot_enter_digest_cache():
    executor = MarketExecutor(market(), cost())
    original = executor.execution_policy_digest
    with pytest.raises(ValueError, match="insolvency_valuation"):
        executor.insolvency_valuation = True
    assert executor.execution_policy_digest == original


@pytest.mark.parametrize("terminal_cash_rate", [0.0, 1.0])
def test_retained_debt_stops_at_first_termination_without_later_bar_consumption(
    terminal_cash_rate,
):
    prices = np.array([100.0, 100.0, 400.0, 900.0])[:, None]
    data = replace(
        market(),
        open=prices,
        high=prices,
        low=prices,
        close=prices,
        mark_price=prices,
        index_price=prices,
        cash_rate=np.array([0.0, 0.0, terminal_cash_rate, 1.0]),
        funding_due=np.array([False, False, False, True])[:, None],
        funding_rate=np.array([0.0, 0.0, 0.0, 0.01])[:, None],
    )
    executor = MarketExecutor(data, cost(), insolvency_valuation="retain_debt")
    entry = short_entry(executor)
    crashed = executor.execute_orders(
        entry.book, entry.order_book, (), start_index=1, bars=2
    )

    assert crashed.book.cash == crashed.book.portfolio_value == -501.0
    assert crashed.next_index == 2
    assert crashed.bars_advanced == 1
    assert crashed.book.insolvent
    assert crashed.book.exact_quantities == (Fraction(0),)
    assert crashed.interval_cash_interest == 0.0
    assert crashed.interval_funding == 0.0
    assert crashed.funding_evidence == ()
    np.testing.assert_array_equal(crashed.book.mark_prices, np.array([400.0]))
    assert len(crashed.book.returns_history) == 2


@pytest.mark.parametrize("mode", [None, "floor_zero"])
def test_floor_zero_keeps_historical_full_requested_bar_path(mode):
    prices = np.array([100.0, 100.0, 400.0, 900.0])[:, None]
    data = replace(
        market(),
        open=prices,
        high=prices,
        low=prices,
        close=prices,
        mark_price=prices,
        index_price=prices,
        cash_rate=np.array([0.0, 0.0, 1.0, 1.0]),
        funding_due=np.array([False, False, False, True])[:, None],
        funding_rate=np.array([0.0, 0.0, 0.0, 0.01])[:, None],
    )
    executor = (
        MarketExecutor(data, cost())
        if mode is None
        else MarketExecutor(data, cost(), insolvency_valuation=mode)
    )
    entry = short_entry(executor)
    crashed = executor.execute_orders(
        entry.book, entry.order_book, (), start_index=1, bars=2
    )

    assert crashed.book.cash == crashed.book.portfolio_value == 0.0
    assert crashed.next_index == 3
    assert crashed.bars_advanced == 2
    assert crashed.interval_cash_interest == 0.0
    assert crashed.book.returns_history == pytest.approx(
        [-0.001, -1.0 + 1e-12, -1.0 + 1e-12]
    )
    np.testing.assert_array_equal(crashed.book.mark_prices, np.array([900.0]))
    assert len(crashed.funding_evidence) == 1
    assert crashed.funding_evidence[0].processing_index == 3


@pytest.mark.parametrize(
    "cause, expected, interest, dividend",
    [("dividend", -501.0, 0.0, -1500.0), ("interest", -500.0, -1499.0, 0.0)],
)
def test_retained_termination_during_cash_flow_stops_remaining_phases(
    cause, expected, interest, dividend
):
    prices = np.full((4, 1), 100.0)
    marks = prices.copy()
    marks[2:] = 900.0
    data = replace(
        market(),
        open=prices,
        high=prices,
        low=prices,
        close=prices,
        mark_price=marks,
        index_price=marks,
        dividend=np.array([0.0, 0.0, 300.0 if cause == "dividend" else 0.0, 0.0])[
            :, None
        ],
        cash_rate=np.array([0.0, 0.0, 1.0 if cause == "dividend" else -8760.0, 1.0]),
        funding_due=np.array([False, False, True, True])[:, None],
        funding_rate=np.array([0.0, 0.0, 0.1, 0.1])[:, None],
    )
    executor = MarketExecutor(data, cost(), insolvency_valuation="retain_debt")
    entry = short_entry(executor)
    crashed = executor.execute_orders(
        entry.book, entry.order_book, (), start_index=1, bars=2
    )

    assert crashed.book.cash == pytest.approx(expected)
    assert crashed.book.portfolio_value == pytest.approx(expected)
    assert crashed.interval_cash_interest == pytest.approx(interest)
    assert crashed.interval_dividend == dividend
    assert crashed.interval_funding == 0.0
    assert crashed.funding_evidence == ()
    assert crashed.next_index == 2 and crashed.bars_advanced == 1
    np.testing.assert_array_equal(crashed.book.mark_prices, np.array([100.0]))


def test_retained_gap_interest_termination_precedes_borrow_and_open_mark():
    data = replace(
        market(),
        calendar_kind="session_calendar",
        nominal_bar_hours=1.0,
        available_at=None,
        timestamps=np.datetime64("2026-01-01", "ns")
        + np.array([0, 1, 3, 4]) * np.timedelta64(1, "h"),
        cash_rate=np.array([0.0, 0.0, -8760.0, 1.0]),
        borrow_rate=np.array([0.0, 0.0, 8760.0, 8760.0])[:, None],
    )
    executor = MarketExecutor(data, cost(), insolvency_valuation="retain_debt")
    entry = short_entry(executor)
    crashed = executor.execute_orders(
        entry.book, entry.order_book, (), start_index=1, bars=2
    )

    assert crashed.book.cash == pytest.approx(-500.0)
    assert crashed.book.portfolio_value == pytest.approx(-500.0)
    assert crashed.interval_cash_interest == pytest.approx(-1499.0)
    assert crashed.interval_borrow_cost == 0.0
    assert crashed.next_index == 2 and crashed.bars_advanced == 1
    np.testing.assert_array_equal(crashed.book.mark_prices, np.array([100.0]))


@pytest.mark.parametrize("mode, fills", [("retain_debt", 1), ("floor_zero", 2)])
@pytest.mark.parametrize("time_in_force", [TimeInForce.GTC, TimeInForce.IOC])
def test_fee_termination_prevents_later_admitted_fills_only_for_retained_mode(
    mode, fills, time_in_force
):
    data = _market()
    executor = MarketExecutor(
        data, replace(cost(), fee_rate=3.0), insolvency_valuation=mode
    )
    book = BookState.zero(
        n_symbols=1, initial_capital=1000.0, initial_prices=data.close[0]
    )
    intents = tuple(
        OrderIntent.create(
            dataset_id=data.dataset_id,
            target_identity=f"synthetic-terminal-fee-{index}",
            execution_policy_digest=executor.execution_policy_digest,
            symbol_index=0,
            requested_quantity=5.0,
            order_type=OrderType.MARKET,
            time_in_force=time_in_force,
            limit_price=None,
            stop_price=None,
            submit_index=0,
            eligible_index=1,
            expiry_index=None,
            submission_reference_price=100.0,
            decision_equity=1000.0,
        )
        for index in range(2)
    )
    result = executor.execute_orders(
        book, OrderBookState.empty(), intents, start_index=0, bars=3
    )

    assert result.fill_count == result.book.fill_count == fills
    assert result.book.total_cost == result.interval_cost == 1500.0 * fills
    assert result.filled_notional == 500.0 * fills
    assert result.book.cash == (-500.0 if mode == "retain_debt" else 0.0)
    assert (
        result.next_index == result.bars_advanced == (1 if mode == "retain_debt" else 3)
    )
    assert result.capacity_evidence[0].consumed_capacity_notional == 500.0 * fills
    if mode == "retain_debt":
        assert any(
            order.terminal_reason == "economic_termination"
            for order in result.order_book.terminal_orders
        )
