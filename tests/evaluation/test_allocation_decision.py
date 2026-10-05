from __future__ import annotations

from dataclasses import fields, is_dataclass, replace
from fractions import Fraction

import numpy as np
import pytest

from tests.evaluation.test_forecast_allocation import market, pending, setup
from trade_rl.artifacts.hashing import content_digest
from trade_rl.evaluation.allocation_decision import (
    execute_allocation_action,
    prepare_allocation_decision,
)
from trade_rl.evaluation.forecast_allocation import (
    execute_forecast_proposal,
    propose_forecast_target,
)
from trade_rl.simulation.stateful.runtime import StatefulExecutionRuntime
from trade_rl.strategies.allocation import AfterCostTargetAllocator
from trade_rl.strategies.allocation_action import AllocationActionContract


def arguments(kwargs, *, mode="residual", scale=1.0):
    return dict(
        **kwargs,
        action_contract=AllocationActionContract(mode=mode, scale=scale),
        initial_capital=1000.0,
        remaining_steps=3,
        feature_indices=(0,),
    )


def plain(value):
    if is_dataclass(value):
        return {f.name: plain(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.datetime64):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Fraction):
        return str(value)
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    return value


def test_zero_residual_preserves_captured_legacy_identity_and_complete_trace():
    ex, book, orders, kwargs = setup()
    baseline = propose_forecast_target(ex, book, orders, **kwargs)
    old = execute_forecast_proposal(ex, book, orders, baseline, **kwargs)
    # Captured before the shared transition extraction, including order identity.
    assert baseline.decision_digest == (
        "52c3d0f11ca4b962aefebf9cf53afbf449d201d5f6c3eaad72af9af39698458d"
    )
    expected_trace = "7fc059b2fbbb8ae11702d0531f92cfeb60a1fefe4ec56a28f7e37daaf52d816d"
    assert content_digest(plain(old.execution)) == expected_trace
    args = arguments(kwargs)
    decision = prepare_allocation_decision(ex, book, orders, **args)
    result = execute_allocation_action(ex, book, orders, decision, np.int64(2), **args)
    assert result.proposal.action == "baseline"
    assert result.proposal.raw_action == 2
    assert result.decision == decision and result.action == 2
    assert result.proposal.target_weight == baseline.target_weight == 0.2
    assert content_digest(plain(result.execution)) == expected_trace
    np.testing.assert_array_equal(result.risk_target.weights, old.risk_target.weights)


@pytest.mark.parametrize(
    ("mode", "expected"),
    [("direct", [-0.4, 0.0, 0.5]), ("residual", [-0.1, 0.2, 0.4])],
)
def test_direct_and_residual_have_independent_bounded_arithmetic(mode, expected):
    ex, book, orders, kwargs = setup()
    kwargs["allocator"] = AfterCostTargetAllocator(
        lower_weight=-0.4, upper_weight=0.6, risk_aversion=1.0
    )
    decision = prepare_allocation_decision(
        ex, book, orders, **arguments(kwargs, mode=mode, scale=0.5)
    )
    assert decision.baseline.target_weight == 0.2
    assert [decision.propose(a).target_weight for a in (1, 2, 3)] == pytest.approx(
        expected
    )
    endpoints = replace(decision, action_contract=AllocationActionContract("residual"))
    assert endpoints.propose(1).target_weight == -0.4
    assert endpoints.propose(3).target_weight == 0.6


def test_turnover_mapping_preserves_requested_action_and_distinct_identity():
    ex, book, orders, kwargs = setup(quantity=7.0)
    kwargs["allocator"] = AfterCostTargetAllocator(
        lower_weight=-1.0, upper_weight=1.0, max_turnover=0.1, risk_aversion=1.0
    )
    decision = prepare_allocation_decision(
        ex, book, orders, **arguments(kwargs, mode="direct", scale=0.5)
    )
    proposals = [decision.propose(a) for a in (1, 2, 3)]
    assert [p.raw_target_weight for p in proposals] == [-0.5, 0.0, 0.5]
    assert [p.target_weight for p in proposals] == pytest.approx([0.6] * 3)
    assert len({p.digest for p in proposals}) == 3
    assert all(not p.is_hold for p in proposals)
    assert decision.propose(0).target_weight == 0.7
    assert decision.propose(0).is_hold


def test_hold_outside_allocator_bounds_preserves_units_and_cancels_pending():
    ex, book, orders, kwargs = setup(quantity=7.0, fee=0.002)
    prior = pending(ex, quantity=0.5)
    orders = orders.add(prior)
    kwargs["allocator"] = AfterCostTargetAllocator(lower_weight=0.0, upper_weight=0.2)
    args = arguments(kwargs)
    decision = prepare_allocation_decision(ex, book, orders, **args)
    assert decision.pending_gross == 0.05 and decision.pending_count == 1
    result = execute_allocation_action(ex, book, orders, decision, 0, **args)
    assert result.proposal.is_hold and result.proposal.target_weight == 0.7
    assert result.execution.book.exact_quantities == (Fraction(7),)
    assert result.execution.book.cash == 300.0
    assert result.execution.book.total_cost == 0.0
    assert result.execution.order_book.active_orders == ()
    assert result.execution.order_events[0].order_id == prior.order_id
    assert result.execution.order_events[0].event_type == "cancelled"


def test_hold_is_overridden_by_drawdown_once(monkeypatch):
    ex, book, orders, kwargs = setup(quantity=4.0, fee=0.0)
    book.peak_value, book.max_drawdown = 1000.0 / 0.85, 0.15
    args = arguments(kwargs)
    calls = []
    constrain = kwargs["pretrade_risk"].constrain

    def counted(weights, **arguments):
        calls.append(weights.copy())
        return constrain(weights, **arguments)

    monkeypatch.setattr(kwargs["pretrade_risk"], "constrain", counted)
    decision = prepare_allocation_decision(ex, book, orders, **args)
    result = execute_allocation_action(ex, book, orders, decision, 0, **args)
    assert result.proposal.target_weight == 0.4
    np.testing.assert_allclose(result.risk_target.weights, [0.2])
    assert result.execution.book.exact_quantities[0] == pytest.approx(2.0)
    assert result.execution.book.cash == pytest.approx(800.0)
    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0], [0.4])


def test_direct_action_has_independent_canonical_cash_fee_and_equity_oracle():
    ex, book, orders, kwargs = setup()
    args = arguments(kwargs, mode="direct", scale=0.5)
    decision = prepare_allocation_decision(ex, book, orders, **args)
    result = execute_allocation_action(ex, book, orders, decision, 3, **args)
    assert result.execution.book.exact_quantities == (Fraction(5),)
    assert result.execution.book.cash == 499.0
    assert result.execution.book.total_cost == 1.0
    assert result.execution.book.portfolio_value == 1049.0
    assert result.execution.interval_net_return == pytest.approx(0.049)


def test_partial_fill_then_hold_retains_actual_units_and_cancels_remainder():
    dataset = market()
    dataset = replace(dataset, volume=np.ones_like(dataset.volume))
    ex, book, orders, kwargs = setup(dataset, fee=0.0)
    args = arguments(kwargs, mode="direct", scale=0.5)
    decision = prepare_allocation_decision(ex, book, orders, **args)
    first = execute_allocation_action(ex, book, orders, decision, 3, **args).execution
    assert first.book.exact_quantities == (Fraction(1),)
    assert first.book.cash == 900.0
    assert len(first.order_book.active_orders) == 1
    assert first.order_book.active_orders[0].remaining_quantity == 4.0
    kwargs["start_index"] = first.next_index
    kwargs["estimates"] = replace(
        kwargs["estimates"],
        decision_time=dataset.timestamps[7],
        available_at=dataset.timestamps[7],
        horizon_end=dataset.timestamps[8],
    )
    args = arguments(kwargs, mode="direct", scale=0.5)
    next_decision = prepare_allocation_decision(
        ex, first.book, first.order_book, **args
    )
    assert next_decision.pending_gross == 0.44
    held = execute_allocation_action(
        ex, first.book, first.order_book, next_decision, 0, **args
    ).execution
    assert held.book.exact_quantities == (Fraction(1),)
    assert held.book.cash == 900.0 and held.book.total_cost == 0.0
    assert held.order_book.active_orders == ()


def test_partial_fill_retarget_sizes_from_actual_units_without_subtracting_remainder():
    dataset = market(next_close=100.0)
    dataset = replace(dataset, volume=np.ones_like(dataset.volume))
    ex, book, orders, kwargs = setup(dataset, fee=0.0)
    args = arguments(kwargs, mode="direct", scale=0.5)
    decision = prepare_allocation_decision(ex, book, orders, **args)
    first = execute_allocation_action(ex, book, orders, decision, 3, **args).execution
    previous = first.order_book.active_orders[0]
    assert first.book.exact_quantities == (Fraction(1),)
    assert previous.remaining_quantity == 4.0
    kwargs["start_index"] = first.next_index
    kwargs["estimates"] = replace(
        kwargs["estimates"],
        decision_time=dataset.timestamps[7],
        available_at=dataset.timestamps[7],
        horizon_end=dataset.timestamps[8],
    )
    args = arguments(kwargs, mode="direct", scale=0.4)
    decision = prepare_allocation_decision(ex, first.book, first.order_book, **args)
    second = execute_allocation_action(
        ex, first.book, first.order_book, decision, 3, **args
    ).execution
    # Target four units against the actual one: replace the old +4 by +3.
    # Capacity fills one, leaving two; the prior remainder is not cash/holdings.
    assert second.book.exact_quantities == (Fraction(2),)
    assert second.book.cash == 800.0
    assert second.order_book.active_orders[0].intent.requested_quantity == 3.0
    assert second.order_book.active_orders[0].remaining_quantity == 2.0
    assert any(
        e.event_type == "cancelled" and e.order_id == previous.order_id
        for e in second.order_events
    )


@pytest.mark.parametrize("split", [False, True])
def test_zero_residual_quantity_hold_keeps_exact_units_and_legacy_order_trace(split):
    ex, book, orders, kwargs = setup(market(split=split), quantity=1.0 / 3.0)
    book = replace(book, _exact_quantities=("1/3",))
    orders = orders.add(pending(ex))
    kwargs["allocator"] = AfterCostTargetAllocator(lower_weight=0.0, upper_weight=1.0)
    kwargs["estimates"] = replace(kwargs["estimates"], buy_cost=0.3, sell_cost=0.3)
    baseline = propose_forecast_target(ex, book, orders, **kwargs)
    legacy = execute_forecast_proposal(ex, book, orders, baseline, **kwargs)
    args = arguments(kwargs)
    decision = prepare_allocation_decision(ex, book, orders, **args)
    result = execute_allocation_action(ex, book, orders, decision, 2, **args)
    assert result.proposal.is_hold
    assert plain(result.execution) == plain(legacy.execution)
    assert result.execution.book.exact_quantities == (Fraction(2 if split else 1, 3),)
    assert result.execution.book.cash == book.cash
    assert result.execution.book.total_cost == 0.0


def test_pending_gross_uses_contract_multiplier_and_fixed_capital_not_current_equity():
    dataset = replace(market(), contract_multipliers=np.array([3.0]))
    ex, book, orders, kwargs = setup(dataset, quantity=1.0)
    book = replace(book, cash=700.0, contract_multipliers=np.array([3.0]))
    orders = orders.add(pending(ex))
    args = arguments(kwargs)
    args["initial_capital"] = 2000.0
    decision = prepare_allocation_decision(ex, book, orders, **args)
    assert decision.baseline.context.equity == 1000.0
    assert decision.pending_gross == 0.075  # 0.5 * 100 * 3 / 2000


@pytest.mark.parametrize(
    "drift",
    [
        "cost",
        "holdings",
        "orders",
        "clock",
        "mapping",
        "features",
        "capital",
        "remaining",
    ],
)
def test_reprepare_rejects_changed_decision_before_execution(monkeypatch, drift):
    ex, book, orders, kwargs = setup()
    args = arguments(kwargs)
    decision = prepare_allocation_decision(ex, book, orders, **args)
    if drift == "cost":
        args["estimates"] = replace(args["estimates"], buy_cost=0.04)
    elif drift == "holdings":
        book = replace(book, quantities=np.ones(1), cash=900.0, _exact_quantities=None)
    elif drift == "orders":
        orders = orders.add(pending(ex))
    elif drift == "clock":
        book.as_of_index = 5
    elif drift == "mapping":
        args["action_contract"] = AllocationActionContract("direct", 0.5)
    elif drift == "features":
        decision = replace(decision, feature_values=(2.0,))
    elif drift == "capital":
        args["initial_capital"] = 2000.0
    else:
        args["remaining_steps"] = 2

    def forbidden(*args, **kwargs):
        pytest.fail("stale decision reached the canonical executor")

    monkeypatch.setattr(StatefulExecutionRuntime, "create", classmethod(forbidden))
    with pytest.raises(ValueError):
        execute_allocation_action(ex, book, orders, decision, 0, **args)


@pytest.mark.parametrize(
    "action", [True, np.bool_(False), -1, 4, 1.0, float("nan"), [1, 2]]
)
def test_invalid_actions_are_rejected(action):
    ex, book, orders, kwargs = setup()
    args = arguments(kwargs)
    decision = prepare_allocation_decision(ex, book, orders, **args)
    with pytest.raises(ValueError, match="action"):
        execute_allocation_action(ex, book, orders, decision, action, **args)


@pytest.mark.parametrize(
    "mode, scale",
    [
        ("intent", 1),
        ("direct", True),
        ("residual", 0),
        ("direct", 1.1),
        ("direct", np.inf),
    ],
)
def test_invalid_action_contracts_fail(mode, scale):
    with pytest.raises(ValueError):
        AllocationActionContract(mode, scale)


@pytest.mark.parametrize(
    "field, value",
    [
        ("feature_names", ["signal"]),
        ("feature_names", ("",)),
        ("feature_values", [1.0]),
        ("feature_values", (np.nan,)),
        ("max_drawdown", 1.1),
        ("max_drawdown", True),
        ("initial_capital", 0),
        ("remaining_steps", True),
        ("remaining_steps", 0),
        ("pending_gross", -1),
        ("pending_count", -1),
    ],
)
def test_detached_decision_rejects_invalid_or_mutable_fields(field, value):
    ex, book, orders, kwargs = setup()
    decision = prepare_allocation_decision(ex, book, orders, **arguments(kwargs))
    with pytest.raises(ValueError):
        replace(decision, **{field: value})


def test_admitted_numpy_action_proposal_scalars_are_canonical_builtins():
    ex, book, orders, kwargs = setup()
    decision = prepare_allocation_decision(
        ex, book, orders, **arguments(kwargs, mode="direct", scale=np.float32(0.5))
    )
    proposal = replace(
        decision.propose(3),
        raw_action=np.int64(3),
        raw_target_weight=np.float32(0.5),
        target_weight=np.float32(0.5),
        is_hold=np.bool_(False),
    )
    assert type(proposal.raw_action) is int
    assert type(proposal.raw_target_weight) is float
    assert type(proposal.target_weight) is float
    assert type(proposal.is_hold) is bool
    assert proposal.digest == decision.propose(3).digest


def test_extra_rl_feature_is_admitted_only_when_available():
    dataset = market()
    dataset = replace(
        dataset,
        features=np.repeat(dataset.features, 2, axis=2),
        feature_available=np.repeat(dataset.feature_available, 2, axis=2),
        feature_staleness_hours=None,
        feature_missing_reason=None,
        feature_staleness=None,
        feature_names=("signal", "rl_signal"),
    )
    ex, book, orders, kwargs = setup(dataset)
    args = arguments(kwargs)
    args["feature_indices"] = (1,)
    decision = prepare_allocation_decision(ex, book, orders, **args)
    assert decision.feature_names == ("rl_signal",)
    unavailable = dataset.feature_available.copy()
    unavailable[6, 0, 1] = False
    unavailable_ex, book, orders, kwargs = setup(
        replace(dataset, feature_available=unavailable, feature_staleness=None)
    )
    args = arguments(kwargs)
    args["feature_indices"] = (1,)
    with pytest.raises(ValueError, match="features.*unavailable"):
        prepare_allocation_decision(unavailable_ex, book, orders, **args)


def test_opposing_pending_orders_do_not_disappear_from_gross_or_state_identity():
    ex, book, orders, kwargs = setup()
    args = arguments(kwargs)
    empty = prepare_allocation_decision(ex, book, orders, **args)
    orders = orders.add(pending(ex, quantity=0.5)).add(pending(ex, quantity=-0.5))
    decision = prepare_allocation_decision(ex, book, orders, **args)
    assert decision.baseline.context.pending_remaining == "0"
    assert decision.pending_count == 2 and decision.pending_gross == 0.1
    assert decision.decision_digest != empty.decision_digest


@pytest.mark.parametrize(
    "field, value",
    [("raw_target_weight", False), ("target_weight", False), ("is_hold", 1)],
)
def test_action_proposal_rejects_boolean_weights_and_numeric_hold_flags(field, value):
    ex, book, orders, kwargs = setup()
    decision = prepare_allocation_decision(
        ex, book, orders, **arguments(kwargs, mode="direct")
    )
    with pytest.raises(ValueError):
        replace(decision.propose(2), **{field: value})
