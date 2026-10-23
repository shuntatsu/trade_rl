from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
from typing import Callable

import numpy as np
import pytest

import tools.market_b4_oracle.expected_intent as EXPECTED_INTENT
import tools.market_b4_oracle.frozen_decision as DECISION
import tools.market_b4_oracle.frozen_forecast as FORECAST
from tools.market_b4_oracle.allocation_choice import AllocationAssumptions
from trade_rl.risk.pretrade import PreTradeRisk, PreTradeRiskConfig
from trade_rl.simulation.accounting import BookState
from trade_rl.simulation.orders.model import OrderBookState, OrderType, TimeInForce
from trade_rl.simulation.orders.reconciliation import reconcile_target
from trade_rl.strategies.allocation import (
    AfterCostTargetAllocator,
    AllocationContext,
    AllocationInputs,
)
from trade_rl.strategies.allocation_action import (
    AllocationActionContract,
    AllocationDecision,
)

BASE_NS = 1_768_000_000_000_000_000
HOUR_NS = 3_600_000_000_000
DATASET_ID = "a" * 64
POLICY_DIGEST = "b" * 64
TARGET_IDENTITY = "synthetic-target-identity-not-authenticated"
STATE_DIGEST = "c" * 64
FEATURE_NAMES = tuple(f"feature_{index}" for index in range(6))
FEATURE_VALUES = (0.1, 0.2, 0.5, 0.4, 0.7, 0.6)
MARKS = (113.17, 257.31)
MULTIPLIERS = (1.3, 0.7)
SYMBOLS = ("BTC", "ETH")
RISK_CONFIG = {
    "max_gross": 1.0,
    "max_abs_weight": 1.0,
    "max_turnover": None,
    "drawdown_start": 0.10,
    "drawdown_stop": 0.20,
    "emergency_turnover_override": True,
    "fail_closed_tolerance": 1e-10,
}


def _source(*, symbol_index: int = 0, decision_time_ns: int = BASE_NS):
    return FORECAST.OwnedDecisionSnapshot(
        dataset_id=DATASET_ID,
        symbol=SYMBOLS[symbol_index],
        decision_time_ns=decision_time_ns,
        source_available_at_ns=BASE_NS,
        information_available=True,
        feature_names=FEATURE_NAMES,
        feature_values=FEATURE_VALUES,
        feature_available=(True,) * len(FEATURE_NAMES),
        dataset_close=MARKS[symbol_index],
        book_mark=MARKS[symbol_index],
        expected_horizon_seconds=3600,
        observation_feature_indices=(0, 1),
    )


def _stream(source, *, expected_return: float, variance: float = 0.01):
    cutoff = BASE_NS - 2 * HOUR_NS
    vintage_digest = "d" * 64
    vintage = {
        "schema": "simple_return_ridge_vintage_v1",
        "return_unit": "expected_simple_return",
        "forecast_kind": "regularized_linear_projection",
        "valuation_basis": "same_close_price_return",
        "calibration_state": "uncalibrated",
        "variance_kind": "fit_prefix_marginal_variance",
        "block": {
            "fit_cutoff": cutoff,
            "model_fit_time": cutoff,
            "prediction_start": BASE_NS - HOUR_NS,
            "prediction_stop": BASE_NS + 2 * HOUR_NS,
            "inference_delay_seconds": 0,
        },
        "training": {
            "schema": "simple_return_training_scope_v1",
            "log_training": {
                "schema": "forecast_training_scope_v1",
                "feature_indices": [2, 4],
                "feature_names": [FEATURE_NAMES[2], FEATURE_NAMES[4]],
                "features": [[0.2, 0.3]],
                "labels": [0.01],
                "sample_weights": [1.0],
                "horizon_hours": 1,
                "fit_cutoff": cutoff,
                "trace": {},
            },
            "labels": [0.01],
            "return_unit": "expected_simple_return",
            "valuation_basis": "same_close_price_return",
            "variance_kind": "fit_prefix_marginal_variance",
            "fit_prefix_marginal_variance": variance,
        },
        "model": {
            "feature_indices": [2, 4],
            "feature_mean": [0.0, 0.0],
            "feature_scale": [1.0, 1.0],
            "coefficients": [0.0, 0.0],
            "intercept": 0.0,
            "horizon_hours": 1,
            "alpha": 1.0,
            "n_samples": 1,
            "fit_cutoff": cutoff,
            "return_unit": "expected_simple_return",
            "valuation_basis": "same_close_price_return",
            "variance_kind": "fit_prefix_marginal_variance",
            "fit_prefix_marginal_variance": variance,
        },
        "prediction_symbols": [source.symbol],
        "digest": vintage_digest,
    }
    packet = {
        "schema": "simple_return_packet_v1",
        "symbol": source.symbol,
        "as_of": BASE_NS,
        "source_available_at": BASE_NS,
        "feature_values": [source.feature_values[2], source.feature_values[4]],
        "decision_close": source.dataset_close,
        "snapshot_digest": "e" * 64,
        "forecast_available_at": BASE_NS,
        "horizon_end": BASE_NS + HOUR_NS,
        "horizon_seconds": 3600,
        "expected_simple_return": expected_return,
        "fit_prefix_marginal_variance": variance,
        "vintage_digest": vintage_digest,
        "return_unit": "expected_simple_return",
        "valuation_basis": "same_close_price_return",
        "variance_kind": "fit_prefix_marginal_variance",
        "forecast_kind": "regularized_linear_projection",
        "digest": "f" * 64,
    }
    return {
        "schema": "simple_return_ridge_stream_v1",
        "availability_semantics": "declared_simulation_v1",
        "dataset_id": DATASET_ID,
        "vintages": [vintage],
        "packets": [packet],
        "causal_scope_digest": "1" * 64,
    }


def _costs(
    *, buy: float = 0.01, sell: float = 0.01, exit_cost: float = 0.005
) -> AllocationAssumptions:
    return AllocationAssumptions(
        expected_simple_return=0.0,
        return_variance=0.25,
        buy_cost=buy,
        sell_cost=sell,
        exit_cost=exit_cost,
        funding_return=0.0,
        borrow_return=0.002,
        cash_return=0.0,
    )


def _account(symbol_index: int, *, selected_quantity: Fraction = Fraction(0)):
    quantities = [Fraction(0), Fraction(0)]
    quantities[symbol_index] = selected_quantity
    positions = tuple(
        float(quantity) * price * multiplier
        for quantity, price, multiplier in zip(
            quantities, MARKS, MULTIPLIERS, strict=True
        )
    )
    equity = 10_000.0
    cash = equity - sum(positions)
    weights = tuple(value / equity for value in positions)
    return {
        "quantities": tuple(quantities),
        "cash": cash,
        "equity": equity,
        "weights": weights,
    }


def _native_target(assumptions, source, account, *, bounds, drawdown, role, emergency):
    symbol_index = SYMBOLS.index(source.symbol)
    decision_time = np.datetime64(source.decision_time_ns, "ns")
    inputs = AllocationInputs(
        symbol=source.symbol,
        decision_time=decision_time,
        available_at=np.datetime64(source.source_available_at_ns, "ns"),
        horizon_end=decision_time + np.timedelta64(1, "h"),
        source_identity=DATASET_ID,
        expected_simple_return=assumptions.expected_simple_return,
        return_variance=assumptions.return_variance,
        buy_cost=assumptions.buy_cost,
        sell_cost=assumptions.sell_cost,
        exit_cost=assumptions.exit_cost,
        funding_return=assumptions.funding_return,
        borrow_return=assumptions.borrow_return,
        cash_return=assumptions.cash_return,
    )
    context = AllocationContext(
        account_id="synthetic-caller-owned-account",
        symbol=source.symbol,
        decision_time=decision_time,
        state_digest=STATE_DIGEST,
        current_weight=account["weights"][symbol_index],
        quantity=str(account["quantities"][symbol_index]),
        cash=account["cash"],
        equity=account["equity"],
        pending_remaining="0",
    )
    lower, upper = bounds
    baseline = AfterCostTargetAllocator(
        lower_weight=lower,
        upper_weight=upper,
        max_turnover=None,
        risk_aversion=0.0,
    ).propose(inputs, context)
    mode, code = ("residual", 2) if role == "NONRL" else ("direct", 0)
    decision = AllocationDecision(
        baseline=baseline,
        action_contract=AllocationActionContract(mode=mode, scale=1.0),
        feature_names=("synthetic-feature",),
        feature_values=(0.0,),
        max_drawdown=drawdown,
        initial_capital=account["equity"],
        remaining_steps=1,
        pending_gross=0.0,
        pending_count=0,
    )
    action = decision.propose(code)
    targets = np.asarray(account["weights"], dtype=np.float64).copy()
    targets[symbol_index] = action.target_weight
    emergency_mask = np.zeros(2, dtype=np.bool_)
    emergency_mask[symbol_index] = emergency
    risk = PreTradeRisk(PreTradeRiskConfig(**RISK_CONFIG)).constrain(
        targets,
        current=np.asarray(account["weights"], dtype=np.float64),
        drawdown=drawdown,
        emergency_flatten_mask=emergency_mask,
    )
    return baseline, action, risk


def _native_reconcile(account, final_weights):
    quantities = np.asarray(
        [EXPECTED_INTENT.project_quantity(value) for value in account["quantities"]],
        dtype=np.float64,
    )
    marks = np.asarray(MARKS, dtype=np.float64)
    multipliers = np.asarray(MULTIPLIERS, dtype=np.float64)
    book = BookState(
        quantities=quantities,
        cash=account["cash"],
        mark_prices=marks,
        peak_value=account["equity"],
        contract_multipliers=multipliers,
        _exact_quantities=tuple(str(value) for value in account["quantities"]),
    )
    return reconcile_target(
        dataset_id=DATASET_ID,
        target_identity=TARGET_IDENTITY,
        execution_policy_digest=POLICY_DIGEST,
        target_weights=np.asarray(final_weights, dtype=np.float64),
        book=book,
        order_book=OrderBookState(active_orders=(), terminal_orders=()),
        reference_prices=marks,
        decision_equity=account["equity"],
        submit_index=7,
        latency_bars=1,
        order_type=OrderType.MARKET,
        time_in_force=TimeInForce.GTC,
        expiry_index=None,
        limit_offset_rate=0.0,
        maximum_gross=1.0,
        valuation_prices=marks,
    )


def _run_pipeline(
    *,
    symbol_index: int = 0,
    expected_return: float = 0.08,
    costs: AllocationAssumptions | None = None,
    selected_quantity: Fraction = Fraction(0),
    drawdown: float = 0.0,
    role: str = "NONRL",
    bounds: tuple[float, float] = (-1.0, 1.0),
    emergency: bool = False,
    source_override=None,
    intent_stage: Callable[[], None] | None = None,
):
    source = _source(symbol_index=symbol_index)
    payload = _stream(source, expected_return=expected_return)
    if source_override is not None:
        source = source_override(source)
    caller_costs = costs or _costs()
    assumptions = FORECAST.bind_frozen_forecast_assumptions(
        payload, source=source, cost_assumptions=caller_costs
    )
    account = _account(symbol_index, selected_quantity=selected_quantity)
    target = DECISION.compose_frozen_allocation_target(
        assumptions,
        current_weight=account["weights"][symbol_index],
        lower_weight=bounds[0],
        upper_weight=bounds[1],
        drawdown=drawdown,
        role=role,
        emergency_flatten=emergency,
    )
    native_baseline, native_action, native_risk = _native_target(
        assumptions,
        source,
        account,
        bounds=bounds,
        drawdown=drawdown,
        role=role,
        emergency=emergency,
    )
    assert target.baseline_weight == native_baseline.target_weight
    assert target.action_target_weight == native_action.target_weight
    assert target.final_target_weight == native_risk.weights[symbol_index]
    final_weights = list(account["weights"])
    final_weights[symbol_index] = target.final_target_weight
    if intent_stage is not None:
        intent_stage()
    expected = EXPECTED_INTENT.reconcile_expected(
        exact_quantities=account["quantities"],
        current_weights=account["weights"],
        final_risk_weights=tuple(final_weights),
        decision_equity=account["equity"],
        valuation_prices=MARKS,
        multipliers=MULTIPLIERS,
        reference_prices=MARKS,
        dataset_id=DATASET_ID,
        target_identity=TARGET_IDENTITY,
        execution_policy_digest=POLICY_DIGEST,
        submit_index=7,
        latency_bars=1,
        pending_orders=(),
    )
    native = _native_reconcile(account, tuple(final_weights))
    assert len(expected.new_intents) == len(native.new_intents)
    assert tuple(item.canonical_payload() for item in expected.new_intents) == tuple(
        item.canonical_payload() for item in native.new_intents
    )
    assert tuple(item.order_id for item in expected.cancelled_orders) == tuple(
        item.order_id for item in native.cancelled_orders
    )
    if tuple(final_weights) == account["weights"]:
        assert expected.desired_quantities is None
    else:
        assert expected.desired_quantities == tuple(native.desired_quantities)
    return source, assumptions, account, target, expected, native


@pytest.mark.parametrize("symbol_index", [0, 1], ids=["BTC-slot", "ETH-slot"])
def test_positive_forecast_flows_to_native_fractional_market_intent(symbol_index):
    _, assumptions, _, target, expected, native = _run_pipeline(
        symbol_index=symbol_index,
        expected_return=0.08,
        costs=_costs(buy=0.01, exit_cost=0.005),
    )
    assert assumptions.expected_simple_return == 0.08
    assert target.final_target_weight == 1.0
    assert len(expected.new_intents) == 1
    wanted = expected.new_intents[0]
    actual = native.new_intents[0]
    assert wanted.symbol_index == symbol_index
    assert wanted.requested_quantity != int(wanted.requested_quantity)
    assert wanted.order_id == actual.order_id
    assert wanted.canonical_payload() == actual.canonical_payload()
    assert wanted.reduce_only is False


def test_negative_forecast_and_sell_borrow_costs_flow_to_short_intent():
    _, assumptions, _, target, expected, native = _run_pipeline(
        symbol_index=1,
        expected_return=-0.08,
        costs=_costs(sell=0.01, exit_cost=0.005),
    )
    assert assumptions.expected_simple_return == -0.08
    assert target.final_target_weight == -1.0
    assert len(expected.new_intents) == len(native.new_intents) == 1
    assert expected.new_intents[0].requested_quantity < 0.0
    assert (
        expected.new_intents[0].canonical_payload()
        == native.new_intents[0].canonical_payload()
    )


def test_exact_fee_edge_prefers_hold_and_reconciles_no_order():
    _, _, _, target, expected, native = _run_pipeline(
        expected_return=0.02,
        costs=_costs(buy=0.01, exit_cost=0.01),
    )
    assert target.baseline_weight == 0.0
    assert target.is_hold
    assert expected.new_intents == ()
    assert native.new_intents == ()


def test_cash_hold_bypasses_alpha_bounds_but_still_passes_risk_and_makes_no_order():
    _, _, account, target, expected, native = _run_pipeline(
        expected_return=0.4,
        selected_quantity=Fraction(20),
        role="CASH",
        bounds=(-0.05, 0.05),
    )
    assert account["weights"][0] > 0.05
    assert target.action_code == 0
    assert (
        target.action_target_weight
        == target.final_target_weight
        == account["weights"][0]
    )
    assert expected.new_intents == native.new_intents == ()


@pytest.mark.parametrize(
    ("drawdown", "expected_weight"),
    [(0.10, 1.0), (0.15, 0.5000000000000001), (0.20, 0.0)],
)
def test_drawdown_stages_match_q_native_risk_and_intent_api(drawdown, expected_weight):
    _, _, _, target, expected, native = _run_pipeline(drawdown=drawdown)
    assert target.final_target_weight == expected_weight
    assert len(expected.new_intents) == len(native.new_intents)
    if expected_weight == 0.0:
        assert expected.new_intents == native.new_intents == ()
    else:
        assert (
            expected.new_intents[0].canonical_payload()
            == native.new_intents[0].canonical_payload()
        )


def test_emergency_flatten_reconciles_reduce_only_close_without_lot_rounding():
    _, _, account, target, expected, native = _run_pipeline(
        selected_quantity=Fraction(25), emergency=True
    )
    assert account["weights"][0] > 0.0
    assert target.final_target_weight == 0.0
    assert len(expected.new_intents) == len(native.new_intents) == 1
    expected_order = expected.new_intents[0]
    native_order = native.new_intents[0]
    assert expected_order.reduce_only is True
    assert expected_order.requested_quantity == -25.0
    assert expected_order.order_id == native_order.order_id
    assert expected_order.canonical_payload() == native_order.canonical_payload()


@pytest.mark.parametrize("corruption", ["clock", "feature"])
def test_bad_forecast_source_stops_before_expected_intent_stage(corruption):
    calls = 0

    def before_intent():
        nonlocal calls
        calls += 1

    if corruption == "clock":

        def corrupt(source):
            return replace(source, decision_time_ns=source.decision_time_ns + HOUR_NS)
    else:

        def corrupt(source):
            values = list(source.feature_values)
            values[2] += 0.01
            return replace(source, feature_values=tuple(values))

    with pytest.raises(ValueError):
        _run_pipeline(source_override=corrupt, intent_stage=before_intent)
    assert calls == 0
