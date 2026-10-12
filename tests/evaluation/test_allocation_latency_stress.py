"""Literal latency oracles on fixed synthetic sources; no fit or market read."""

from dataclasses import fields, replace
from fractions import Fraction

import numpy as np
import pytest

from tests.evaluation.test_after_cost_allocation import (
    apply,
    book,
    executor,
    inputs,
    market,
    propose,
    risk,
)
from tests.evaluation.test_allocation_nonrl_walk_forward import fixture
from tests.evaluation.test_allocation_terminal_execution import plans
from trade_rl.artifacts import content_digest
from trade_rl.evaluation.allocation import propose_nonrl_target
from trade_rl.evaluation.allocation_nonrl_walk_forward import _carriers
from trade_rl.evaluation.allocation_terminal_execution import (
    run_terminal_allocation_execution,
)
from trade_rl.evaluation.rl_allocation.env import AllocationTradingEnv
from trade_rl.simulation import MarketExecutor
from trade_rl.simulation.orders.model import OrderBookState
from trade_rl.strategies.allocation import AfterCostTargetAllocator
from trade_rl.strategies.forecasts.simple_return import SimpleReturnTrainingSet
from trade_rl.strategies.forecasts.simple_stream import FrozenSimpleReturnStream
from trade_rl.strategies.forecasts.training_trace import ForecastTrainingTrace
from trade_rl.strategies.rl.allocation_policy import AllocationRuntimeProfile
from trade_rl.strategies.rl.allocation_recipe_v2 import allocation_recipe_payload_v2


def carrier(
    *,
    latency=2,
    stop=8,
    kind="nonrl",
    source_bars=12,
    penalty=False,
    upper=0.5,
    closing_capacity=100000,
):
    """Rebind an existing native carrier to a fully declared two-hour forecast."""
    folds, (old,) = fixture(
        ranges=((6, stop),),
        signals=(1, 0.5, 0, 0),
        mode="direct" if kind == "cash" else "residual",
        fee=0.01 if penalty else 0,
    )
    prices = np.full((12, 1), 100.0)
    dataset = replace(
        old.dataset,
        open=prices,
        close=prices,
        high=prices,
        low=prices,
        mark_price=prices,
    )
    volume = dataset.volume.copy()
    volume[9:] = closing_capacity
    dataset = replace(dataset, volume=volume)
    if source_bars < 12:
        sliced = {
            f.name: getattr(dataset, f.name)[:source_bars]
            for f in fields(dataset)
            if f.init
            and isinstance(getattr(dataset, f.name), np.ndarray)
            and getattr(dataset, f.name).ndim
            and getattr(dataset, f.name).shape[0] == 12
        }
        dataset = replace(dataset, **sliced)
    times = old.dataset.timestamps
    old_vintage = old.stream.vintages[0]
    returns = np.array([-np.sqrt(0.02), np.sqrt(0.02)]) if penalty else np.zeros(2)
    trace = ForecastTrainingTrace(
        ("S0", "S0"),
        times[:2],
        times[2:4],
        times[:2],
        times[2:4],
        np.full(2, 100.0),
        100.0 * (1.0 + returns),
    )
    training = SimpleReturnTrainingSet(
        replace(
            old_vintage.training.log_training,
            features=np.zeros((2, 1)),
            labels=np.log1p(returns),
            label_end_times=times[2:4],
            sample_weights=np.ones(2),
            horizon_hours=2,
            trace=trace,
        )
    )
    model = replace(
        old_vintage.model,
        horizon_hours=2,
        n_samples=2,
        coefficients=np.array([0.03 if penalty else 0.1]),
        fit_prefix_marginal_variance=training.fit_prefix_marginal_variance,
    )
    vintage = replace(old_vintage, training=training, model=model)
    stream = FrozenSimpleReturnStream(
        dataset.dataset_id,
        (vintage,),
        tuple(
            replace(
                p,
                horizon_end=p.as_of + np.timedelta64(2, "h"),
                horizon_seconds=7200,
                vintage_digest=vintage.digest,
                decision_close=100.0,
                expected_simple_return=model.predict_selected(
                    np.asarray(p.feature_values)
                ),
                fit_prefix_marginal_variance=training.fit_prefix_marginal_variance,
            )
            for p in old.stream.packets
        ),
    )
    estimates = tuple(
        replace(
            e,
            horizon_end=e.decision_time + np.timedelta64(2, "h"),
            buy_cost=0.01 if penalty else 0,
            sell_cost=0.01 if penalty else 0,
            exit_cost=0.01 if penalty else 0,
        )
        for e in old._estimates.values()
    )
    cost = replace(old.execution_cost, order_latency_bars=latency)
    economics = MarketExecutor(
        dataset, cost, insolvency_valuation="retain_debt"
    ).execution_policy_digest
    allocator = replace(
        old.allocator, risk_aversion=1.0 if penalty else 0, upper_weight=upper
    )
    profile = AllocationRuntimeProfile(
        economics,
        content_digest(old.risk_config),
        1000.0,
        "USD",
        3600,
        (stop - 6) * 3600,
    )
    recipe = allocation_recipe_payload_v2(
        old.action_contract,
        ("signal",),
        allocator=allocator,
        expected_horizon_seconds=7200,
        observation_schema=old.observation_schema,
        runtime_profile=profile,
    )
    bound = replace(
        old.bound,
        objective=replace(
            old.bound.objective,
            deployment_recipe_digest=content_digest(recipe),
            economics_digest=economics,
        ),
    )
    return folds, AllocationTradingEnv(
        dataset,
        stream=stream,
        estimates=estimates,
        bound=bound,
        action_contract=old.action_contract,
        allocator=allocator,
        execution_cost=cost,
        risk_config=old.risk_config,
        feature_indices=(0,),
        symbol_index=0,
        start_index=6,
        stop_index=stop,
        account_id=old.account_id,
        observation_schema=old.observation_schema,
        expected_horizon_seconds=7200,
    )


def test_l2_carrier_admitted_without_reset(monkeypatch):
    _, env = carrier()
    monkeypatch.setattr(env, "reset", lambda **_: pytest.fail("preflight reset"))
    assert len(_carriers((env,))) == 64


@pytest.mark.parametrize("latency", [0, 2])
def test_matching_pending_retains_identity_and_first_eligible_fill(latency):
    data = market()
    ex = executor(data, order_latency_bars=latency)
    bk, orders = book(data), OrderBookState.empty()
    first = apply(ex, bk, orders, propose(ex, bk, orders, signal=0.1, upper=0.5))
    if latency == 0:
        assert first.execution.book.exact_quantities == (Fraction(5),)
        assert not first.execution.order_book.active_orders
        return
    assert first.execution.book.cash == 1000
    assert first.execution.book.exact_quantities == (Fraction(0),)
    pending = first.execution.order_book.active_orders[0]
    assert pending.intent.submit_index == 0 and pending.intent.eligible_index == 2
    assert pending.last_processed_index == 1
    second = apply(
        ex,
        first.execution.book,
        first.execution.order_book,
        propose(
            ex,
            first.execution.book,
            first.execution.order_book,
            signal=0.1,
            index=1,
            upper=0.5,
        ),
        index=1,
    )
    assert second.execution.book.exact_quantities == (Fraction(5),)
    assert second.execution.book.cash == 500
    final = second.execution.order_book.terminal_orders[-1]
    assert final.order_id == pending.order_id and final.intent.eligible_index == 2


@pytest.mark.parametrize("changed", [False, True])
def test_waiting_opening_order_can_cancel_or_replace_before_eligibility(changed):
    data = market()
    ex = executor(data, order_latency_bars=2)
    first = apply(
        ex,
        book(data),
        OrderBookState.empty(),
        propose(ex, book(data), OrderBookState.empty(), signal=0.1, upper=0.5),
    )
    bk, orders = first.execution.book, first.execution.order_book
    old = orders.active_orders[0]
    second = apply(
        ex,
        bk,
        orders,
        propose(
            ex,
            bk,
            orders,
            signal=0.1 if changed else 0,
            index=1,
            upper=0.4 if changed else 0.5,
        ),
        index=1,
    )
    assert second.execution.book.exact_quantities == (Fraction(0),)
    assert old.order_id in {
        o.order_id for o in second.execution.order_book.terminal_orders
    }
    if changed:
        replacement = second.execution.order_book.active_orders[0]
        assert replacement.order_id != old.order_id
        assert (
            replacement.intent.submit_index == 1
            and replacement.intent.eligible_index == 3
        )
        assert replacement.remaining_quantity == 4
        third = apply(
            ex,
            second.execution.book,
            second.execution.order_book,
            propose(
                ex,
                second.execution.book,
                second.execution.order_book,
                signal=0.1,
                index=2,
                upper=0.4,
            ),
            index=2,
        )
        assert third.execution.book.exact_quantities == (Fraction(4),)
    else:
        assert not second.execution.order_book.active_orders
        assert (
            second.execution.order_book.terminal_orders[-1].terminal_reason
            == "quantity_hold"
        )


@pytest.mark.parametrize("latency", [0, 2])
def test_capacity_is_used_on_actual_processing_bar(latency):
    data = market()
    volume = data.volume.copy()
    volume[1:] = 0
    data = replace(data, volume=volume)
    ex = executor(
        data, order_latency_bars=latency, processing_bar_volume_capacity=False
    )
    first = apply(
        ex,
        book(data),
        OrderBookState.empty(),
        propose(ex, book(data), OrderBookState.empty(), signal=0.1, upper=0.5),
    )
    if latency == 0:
        assert first.execution.book.exact_quantities == (Fraction(5),)
    else:
        second = apply(
            ex,
            first.execution.book,
            first.execution.order_book,
            propose(
                ex,
                first.execution.book,
                first.execution.order_book,
                signal=0.1,
                index=1,
                upper=0.5,
            ),
            index=1,
        )
        assert second.execution.book.exact_quantities == (Fraction(0),)
        assert second.execution.order_book.active_orders[0].remaining_quantity == 5
        assert second.execution.book.total_cost == 0


def test_closed_loop_same_beliefs_need_not_produce_same_orders():
    from trade_rl.evaluation import allocation_terminal_execution as terminal

    records = []
    contexts = []
    for latency in (0, 2):
        folds, env = carrier(latency=latency, penalty=True)
        plan = plans(terminal, env, folds, stop=10)
        contexts.append(
            (env.stream.digest, tuple(e.payload() for e in env._estimates.values()))
        )
        result = run_terminal_allocation_execution(folds, env, plan)
        raw = result.payload
        assert raw["settlement_complete"] and raw["risk_eligible"]
        assert raw["final_active_orders"] == []
        assert raw["final_book"]["exact_quantities"] == ["0"]
        assert len(raw["closing_rows"]) == 2
        records.append(raw)
    assert contexts[0] == contexts[1]
    assert records[0]["final_book"]["cash"] == pytest.approx(995, abs=1e-9)
    assert records[0]["final_book"]["total_cost"] == pytest.approx(5, abs=1e-9)
    assert records[1]["final_book"]["cash"] == 1000
    assert records[1]["final_book"]["total_cost"] == 0
    assert records[1]["native_receipt"]["rows"][0]["native"]["book"][
        "exact_quantities"
    ] == ["0"]


@pytest.mark.parametrize(
    "reserve,settled,quantity",
    [(1, False, "4"), (2, False, "2"), (3, True, "0")],
)
def test_finite_reserve_partial_close_does_not_invent_settlement(
    reserve, settled, quantity
):
    # NONRL buys four at8. Closing submitted8 waits9, capacity2 fills10/11.
    from trade_rl.evaluation import allocation_terminal_execution as terminal

    folds, env = carrier(stop=8, upper=0.4, closing_capacity=2)
    plan = plans(terminal, env, folds, stop=8 + reserve)
    raw = run_terminal_allocation_execution(folds, env, plan).payload
    assert len(raw["closing_rows"]) == reserve
    assert raw["settlement_complete"] is settled
    assert raw["settled_profit_rate"] == (0.0 if settled else None)
    assert raw["final_book"]["exact_quantities"] == [quantity]


def test_l2_horizon_and_source_are_rejected_before_whole_carrier_reset(monkeypatch):
    _, (short_horizon, _) = fixture(cost_changes={"order_latency_bars": 2})
    _, good = carrier(stop=10)
    _, short_source = carrier(stop=10, source_bars=11)
    for roster in ((short_horizon,), (good, short_source)):
        for env in roster:
            monkeypatch.setattr(
                env, "reset", lambda **_: pytest.fail("reset before rejection")
            )
        with pytest.raises(ValueError, match="horizon|eligible|source"):
            _carriers(roster)


@pytest.mark.parametrize("clock", ["late_eligible", "future_processed"])
def test_l2_returned_pending_clocks_remain_strict(clock):
    data = market()
    ex = executor(data, order_latency_bars=2)
    first = apply(
        ex,
        book(data),
        OrderBookState.empty(),
        propose(ex, book(data), OrderBookState.empty(), signal=0.1, upper=0.5),
    )
    original = first.execution.order_book.active_orders[0]
    if clock == "late_eligible":
        forged = replace(original, intent=replace(original.intent, eligible_index=3))
    else:
        forged = replace(original, last_processed_index=2)
    with pytest.raises(ValueError, match="clock"):
        propose(
            ex,
            first.execution.book,
            OrderBookState.empty().add(forged),
            signal=0.1,
            index=1,
            upper=0.5,
        )


@pytest.mark.parametrize("latency", [False, True, -1, 1, 3, 2.0])
def test_scalar_admission_rejects_noninteger_or_unregistered_latency(latency):
    data = market()
    ex = executor(data)
    # Configuration construction already rejects malformed types; prove the
    # admission itself cannot be bypassed by caller mutation of its public config.
    object.__setattr__(ex.cost, "order_latency_bars", latency)
    with pytest.raises(ValueError, match="latency"):
        propose(ex, book(data), OrderBookState.empty())


def test_l2_scalar_horizon_equal_boundary_and_missing_source():
    data = market()
    ex = executor(data, order_latency_bars=2)
    arguments = dict(
        account_id="independent-S0",
        allocator=AfterCostTargetAllocator(),
        pretrade_risk=risk(),
        symbol_index=0,
        start_index=0,
    )
    with pytest.raises(ValueError, match="horizon"):
        propose_nonrl_target(
            ex,
            book(data),
            OrderBookState.empty(),
            inputs=replace(inputs(ex), horizon_end=data.timestamps[1]),
            **arguments,
        )
    propose_nonrl_target(
        ex,
        book(data),
        OrderBookState.empty(),
        inputs=replace(inputs(ex), horizon_end=data.timestamps[2]),
        **arguments,
    )
    with pytest.raises(ValueError, match="eligible|source"):
        propose_nonrl_target(
            ex,
            book(data),
            OrderBookState.empty(),
            inputs=inputs(ex, index=4),
            **(arguments | {"start_index": 4}),
        )
