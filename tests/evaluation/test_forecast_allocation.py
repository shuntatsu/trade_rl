from __future__ import annotations

from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest

from trade_rl.data.contracts import VolumeUnit
from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.forecast_allocation import (
    HorizonCostEstimates,
    execute_forecast_proposal,
    propose_forecast_target,
)
from trade_rl.risk import PreTradeRisk, PreTradeRiskConfig
from trade_rl.simulation import BookState, MarketExecutor
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.simulation.orders.model import (
    OrderBookState,
    OrderIntent,
    OrderType,
    PendingOrder,
    TimeInForce,
)
from trade_rl.simulation.stateful.runtime import StatefulExecutionRuntime
from trade_rl.strategies.allocation import AfterCostTargetAllocator
from trade_rl.strategies.forecasts.prequential import fit_prequential_ridge
from trade_rl.strategies.forecasts.simple_prequential import (
    fit_prequential_simple_ridge,
)
from trade_rl.strategies.forecasts.stream import ForecastBlock


def market(*, next_open=100.0, next_close=110.0, split=False):
    # The four mature labels are 1, -0.5, 1, -0.5. The endpoint at hour 5
    # is excluded by cutoff 5, so future prices cannot change this prefix fit.
    close = np.array([100, 200, 100, 200, 100] + [100] * 7, dtype=float)[:, None]
    close[7:] = 50.0 if split else next_close
    opens = close.copy()
    opens[7] = 50.0 if split else next_open
    shape = close.shape
    splits = np.ones(shape)
    if split:
        splits[7] = 2.0
    return MarketDataset(
        dataset_id="d" * 64,
        symbols=("S0",),
        timestamps=np.datetime64("2026-01-01", "ns")
        + np.arange(len(close)) * np.timedelta64(1, "h"),
        features=np.ones((len(close), 1, 1), dtype=np.float32),
        global_features=np.zeros((len(close), 1), dtype=np.float32),
        open=opens,
        high=np.maximum(opens, close),
        low=np.minimum(opens, close),
        close=close,
        volume=np.full(shape, 100_000.0),
        volume_units=(VolumeUnit.BASE_ASSET,),
        funding_rate=np.zeros(shape),
        tradable=np.ones(shape, dtype=np.bool_),
        feature_available=np.ones((len(close), 1, 1), dtype=np.bool_),
        feature_names=("signal",),
        global_feature_names=("regime",),
        periods_per_year=8760,
        split_factor=splits,
    )


def fitted(dataset, *, delay=0, legacy=False):
    times = dataset.timestamps
    block = ForecastBlock(
        times[5], times[5] + np.timedelta64(15, "m"), times[6], times[10], delay
    )
    fit = fit_prequential_ridge if legacy else fit_prequential_simple_ridge
    return fit(dataset, blocks=(block,), feature_indices=(0,), horizon_hours=1)


def setup(dataset=None, *, quantity=0.0, fee=0.002, buy_cost=0.025):
    dataset = market() if dataset is None else dataset
    executor = MarketExecutor(
        dataset,
        replace(ExecutionCostConfig.zero(), fee_rate=fee, max_participation_rate=1.0),
    )
    book = BookState(
        quantities=np.array([quantity]),
        cash=1000.0 - quantity * 100.0,
        mark_prices=dataset.close[6],
        peak_value=1000.0,
        as_of_index=6,
        as_of_dataset_id=dataset.dataset_id,
    )
    decision = dataset.timestamps[6]
    kwargs = dict(
        account_id="independent-S0",
        stream=fitted(dataset),
        estimates=HorizonCostEstimates(
            symbol="S0",
            decision_time=decision,
            available_at=decision,
            horizon_end=decision + np.timedelta64(1, "h"),
            source_identity="synthetic-declared-costs",
            buy_cost=buy_cost,
        ),
        allocator=AfterCostTargetAllocator(
            lower_weight=0.0, upper_weight=1.0, risk_aversion=1.0
        ),
        pretrade_risk=PreTradeRisk(
            PreTradeRiskConfig(max_abs_weight=1.0, max_turnover=None)
        ),
        symbol_index=0,
        start_index=6,
        expected_horizon_seconds=3600,
    )
    return executor, book, OrderBookState.empty(), kwargs


def pending(executor, *, quantity=0.5):
    return PendingOrder.from_intent(
        OrderIntent.create(
            dataset_id=executor.dataset.dataset_id,
            target_identity="prior-target",
            execution_policy_digest=executor.execution_policy_digest,
            symbol_index=0,
            requested_quantity=quantity,
            order_type=OrderType.MARKET,
            time_in_force=TimeInForce.GTC,
            limit_price=None,
            stop_price=None,
            submit_index=6,
            eligible_index=7,
            expiry_index=None,
            submission_reference_price=100.0,
            decision_equity=1000.0,
        )
    )


def test_returned_account_cannot_replay_a_paid_dividend_at_the_old_decision():
    dataset = market(next_close=100.0)
    dividend = np.zeros_like(dataset.close)
    dividend[7, 0] = 2.0
    executor, book, orders, kwargs = setup(
        replace(dataset, dividend=dividend), quantity=1.0, fee=0.0, buy_cost=0.3
    )
    kwargs["allocator"] = AfterCostTargetAllocator(lower_weight=0.0, upper_weight=1.0)
    kwargs["estimates"] = replace(kwargs["estimates"], sell_cost=0.3)
    proposal = propose_forecast_target(executor, book, orders, **kwargs)
    assert proposal.is_hold
    result = execute_forecast_proposal(executor, book, orders, proposal, **kwargs)
    assert result.execution.next_index == 7
    assert result.execution.book.cash == 902.0
    with pytest.raises(ValueError, match="account.*clock|processing.*index"):
        propose_forecast_target(
            executor, result.execution.book, result.execution.order_book, **kwargs
        )
    # Carry the actual book, pending orders and processing index together.
    kwargs["start_index"] = result.execution.next_index
    kwargs["estimates"] = replace(
        kwargs["estimates"],
        decision_time=dataset.timestamps[7],
        available_at=dataset.timestamps[7],
        horizon_end=dataset.timestamps[8],
    )
    next_proposal = propose_forecast_target(
        executor, result.execution.book, result.execution.order_book, **kwargs
    )
    assert next_proposal.is_hold
    continued = execute_forecast_proposal(
        executor,
        result.execution.book,
        result.execution.order_book,
        next_proposal,
        **kwargs,
    )
    assert continued.execution.next_index == 8
    assert continued.execution.book.as_of_index == 8
    assert continued.execution.book.cash == 902.0
    assert continued.execution.book.fill_count == 0


def test_forecast_admission_requires_an_explicit_account_processing_clock():
    executor, book, orders, kwargs = setup()
    book.as_of_index = None
    book.as_of_dataset_id = None
    with pytest.raises(ValueError, match="account processing clock is required"):
        propose_forecast_target(executor, book, orders, **kwargs)


def test_direct_simple_projection_drives_analytical_target_and_canonical_net_book():
    executor, book, orders, kwargs = setup()
    proposal = propose_forecast_target(executor, book, orders, **kwargs)
    # E[r]=0.25, Var[r]=0.5625, lambda=1, declared buy rate=0.025.
    # Maximize (0.25-0.025)*w - 0.5625*w^2 => w=0.2.
    assert proposal.inputs.expected_simple_return == pytest.approx(0.25)
    assert proposal.inputs.return_variance == pytest.approx(0.5625)
    assert proposal.target_weight == pytest.approx(0.2)
    assert proposal.objective_value == pytest.approx(0.0225)
    assert proposal.inputs.decision_time == executor.dataset.timestamps[6]
    assert proposal.inputs.decision_time != executor.dataset.timestamps[5]
    assert proposal.context.quantity == "0"
    assert proposal.context.cash == 1000.0
    result = execute_forecast_proposal(executor, book, orders, proposal, **kwargs)
    # The declared forecast rate is separate from the explicit executor fee.
    # Buy 2 at the next open 100; pay 200*0.002=0.4; mark 2 at 110.
    realized = result.execution.book
    assert result.execution.next_index == 7
    assert realized.exact_quantities == (Fraction(2),)
    assert realized.cash == pytest.approx(799.6)
    assert realized.total_cost == pytest.approx(0.4)
    assert realized.portfolio_value == pytest.approx(1019.6)
    assert result.execution.interval_net_return == pytest.approx(0.0196)
    assert realized.fill_count == 1
    np.testing.assert_allclose(result.risk_target.weights, [0.2])
    # Execution is detached; the original account remains the decision snapshot.
    assert book.cash == 1000.0 and book.exact_quantities == (Fraction(0),)


def test_next_open_gap_has_independent_realized_return_not_raw_close_surrogate():
    executor, book, orders, kwargs = setup(market(next_open=120.0))
    proposal = propose_forecast_target(executor, book, orders, **kwargs)
    result = execute_forecast_proposal(executor, book, orders, proposal, **kwargs)
    # Desired quantity still uses decision close 100: 0.2*1000/100=2.
    # At open 120, cash=1000-2*120-240*0.002=759.52; close value=979.52.
    assert result.execution.book.exact_quantities == (Fraction(2),)
    assert result.execution.book.cash == pytest.approx(759.52)
    assert result.execution.book.total_cost == pytest.approx(0.48)
    assert result.execution.book.portfolio_value == pytest.approx(979.52)
    assert result.execution.interval_net_return == pytest.approx(-0.02048)
    raw_close_surrogate = 0.2 * (110.0 / 100.0 - 1.0)
    assert raw_close_surrogate == pytest.approx(0.02)
    assert result.execution.interval_net_return != pytest.approx(raw_close_surrogate)


@pytest.mark.parametrize("split", [False, True])
def test_forecast_hold_cancels_pending_and_canonical_split_preserves_wealth(split):
    executor, book, orders, kwargs = setup(market(split=split), quantity=1.0 / 3.0)
    book = replace(book, _exact_quantities=("1/3",))
    prior = pending(executor)
    orders = orders.add(prior)
    kwargs["allocator"] = AfterCostTargetAllocator(lower_weight=0.0, upper_weight=1.0)
    kwargs["estimates"] = replace(kwargs["estimates"], buy_cost=0.3, sell_cost=0.3)
    proposal = propose_forecast_target(executor, book, orders, **kwargs)
    assert proposal.is_hold
    assert proposal.context.quantity == "1/3"
    assert proposal.context.pending_remaining == "1/2"
    # A future split neither suppresses fitting nor removes the current packet.
    assert proposal.inputs.expected_simple_return == pytest.approx(0.25)
    assert len(kwargs["stream"].packets) == 4
    result = execute_forecast_proposal(executor, book, orders, proposal, **kwargs)
    assert result.execution.book.exact_quantities == (Fraction(2 if split else 1, 3),)
    assert result.execution.book.cash == book.cash
    assert result.execution.book.total_cost == 0.0
    assert result.execution.book.fill_count == 0
    assert result.execution.order_book.active_orders == ()
    assert len(result.execution.order_events) == 1
    event = result.execution.order_events[0]
    assert event.event_type == "cancelled" and event.order_id == prior.order_id
    expected_equity = book.cash + (2.0 / 3.0 * 50 if split else 1.0 / 3.0 * 110)
    assert result.execution.book.portfolio_value == pytest.approx(expected_equity)
    if split:
        assert result.execution.interval_net_return == pytest.approx(0.0)


def test_hard_drawdown_risk_overrides_forecast_hold_via_same_execution_book():
    executor, book, orders, kwargs = setup(quantity=3.0, fee=0.0)
    book.peak_value, book.max_drawdown = 1250.0, 0.2
    kwargs["allocator"] = AfterCostTargetAllocator(lower_weight=0.0, upper_weight=1.0)
    kwargs["estimates"] = replace(kwargs["estimates"], buy_cost=0.3, sell_cost=0.3)
    proposal = propose_forecast_target(executor, book, orders, **kwargs)
    assert proposal.is_hold
    result = execute_forecast_proposal(executor, book, orders, proposal, **kwargs)
    np.testing.assert_array_equal(result.risk_target.weights, [0.0])
    assert "drawdown_deleveraging" in result.risk_target.reasons
    assert result.execution.book.exact_quantities == (Fraction(0),)
    assert result.execution.book.cash == pytest.approx(1000.0)
    assert result.execution.book.portfolio_value == pytest.approx(1000.0)
    assert result.execution.book.fill_count == 1


@pytest.mark.parametrize("drift", ["cost", "forecast", "holdings", "pending"])
def test_execution_rebinds_and_rejects_changed_inputs_before_executor(
    monkeypatch, drift
):
    executor, book, orders, kwargs = setup()
    proposal = propose_forecast_target(executor, book, orders, **kwargs)
    if drift == "cost":
        kwargs["estimates"] = replace(kwargs["estimates"], buy_cost=0.035)
    elif drift == "forecast":
        packet = kwargs["stream"].packets[0]
        changed = replace(
            packet, source_available_at=packet.as_of - np.timedelta64(1, "h")
        )
        kwargs["stream"] = replace(
            kwargs["stream"], packets=(changed, *kwargs["stream"].packets[1:])
        )
    elif drift == "holdings":
        book = replace(
            book, quantities=np.array([1.0]), cash=900.0, _exact_quantities=None
        )
    else:
        orders = orders.add(pending(executor))

    def forbidden(*args, **kwargs):
        pytest.fail("changed forecast/cost/account reached canonical execution")

    monkeypatch.setattr(StatefulExecutionRuntime, "create", classmethod(forbidden))
    with pytest.raises(ValueError):
        execute_forecast_proposal(executor, book, orders, proposal, **kwargs)


@pytest.mark.parametrize(
    "invalid",
    [
        "stale_packet",
        "delayed_packet",
        "source_clock",
        "features",
        "feature_names",
        "unavailable_feature",
        "symbol",
        "dataset",
        "horizon",
        "cost_horizon",
        "cost_decision",
        "cost_symbol",
        "close",
        "mark",
        "book_mark",
        "legacy_log_stream",
    ],
)
def test_proposal_rejects_incompatible_current_snapshot_and_semantics(invalid):
    executor, book, orders, kwargs = setup()
    dataset = executor.dataset
    stream = kwargs["stream"]
    if invalid == "stale_packet":
        kwargs["stream"] = replace(stream, packets=stream.packets[1:])
    elif invalid == "delayed_packet":
        kwargs["stream"] = fitted(dataset, delay=1)
    elif invalid == "source_clock":
        changed = replace(
            stream.packets[0],
            source_available_at=dataset.timestamps[6] - np.timedelta64(1, "h"),
        )
        kwargs["stream"] = replace(stream, packets=(changed, *stream.packets[1:]))
    elif invalid == "features":
        features = dataset.features.copy()
        features[6, 0, 0] = 2.0
        dataset = replace(dataset, features=features)
    elif invalid == "feature_names":
        dataset = replace(dataset, feature_names=("renamed_signal",))
    elif invalid == "unavailable_feature":
        available = dataset.feature_available.copy()
        available[6, 0, 0] = False
        staleness = dataset.resolved_array("feature_staleness").copy()
        staleness[6, 0, 0] = 1.0
        dataset = replace(
            dataset, feature_available=available, feature_staleness=staleness
        )
    elif invalid == "symbol":
        dataset = replace(dataset, symbols=("S1",))
    elif invalid == "dataset":
        dataset = replace(dataset, dataset_id="e" * 64)
    elif invalid == "horizon":
        kwargs["expected_horizon_seconds"] = 7200
    elif invalid == "cost_horizon":
        kwargs["estimates"] = replace(
            kwargs["estimates"], horizon_end=dataset.timestamps[8]
        )
    elif invalid == "cost_decision":
        kwargs["estimates"] = replace(
            kwargs["estimates"],
            decision_time=dataset.timestamps[5],
            available_at=dataset.timestamps[5],
        )
    elif invalid == "cost_symbol":
        kwargs["estimates"] = replace(kwargs["estimates"], symbol="S1")
    elif invalid == "close":
        changed = replace(stream.packets[0], decision_close=99.0)
        kwargs["stream"] = replace(stream, packets=(changed, *stream.packets[1:]))
    elif invalid == "mark":
        marks = dataset.resolved_array("mark_price").copy()
        marks[6, 0] = 99.0
        dataset = replace(dataset, mark_price=marks)
        book = replace(book, mark_prices=np.array([99.0]))
    elif invalid == "book_mark":
        book = replace(book, mark_prices=np.array([99.0]))
    elif invalid == "legacy_log_stream":
        kwargs["stream"] = fitted(dataset, legacy=True)
    executor = MarketExecutor(dataset, executor.cost)
    with pytest.raises(ValueError):
        propose_forecast_target(executor, book, orders, **kwargs)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("buy_cost", float("nan")),
        ("sell_cost", True),
        ("exit_cost", -0.01),
        ("funding_return", float("inf")),
        ("borrow_return", -0.01),
        ("cash_return", False),
        ("available_at", np.datetime64("2026-01-01T07", "ns")),
        ("decision_time", np.datetime64("NaT")),
        ("valuation_basis", "next_open_wealth_return"),
    ],
)
def test_cost_estimates_require_available_finite_declared_rates_and_same_close_basis(
    field, value
):
    _, _, _, kwargs = setup()
    estimates = kwargs["estimates"]
    assert estimates.buy_cost == 0.025
    assert estimates.valuation_basis == "same_close_price_return"
    with pytest.raises(ValueError):
        replace(estimates, **{field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("return_unit", "expected_log_return"),
        ("valuation_basis", "next_open_wealth_return"),
        ("horizon_seconds", True),
        ("expected_simple_return", float("nan")),
    ],
)
def test_forecast_packet_rejects_unit_basis_and_nonfinite_or_boolean_contracts(
    field, value
):
    _, _, _, kwargs = setup()
    with pytest.raises(ValueError):
        replace(kwargs["stream"].packets[0], **{field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("start_index", True),
        ("start_index", -1),
        ("symbol_index", True),
        ("expected_horizon_seconds", True),
        ("expected_horizon_seconds", 0),
    ],
)
def test_forecast_admission_rejects_invalid_indices_and_horizon(field, value):
    executor, book, orders, kwargs = setup()
    kwargs[field] = value
    with pytest.raises(ValueError):
        propose_forecast_target(executor, book, orders, **kwargs)
