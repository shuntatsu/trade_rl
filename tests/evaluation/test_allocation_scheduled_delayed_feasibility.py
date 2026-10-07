"""Native feasibility must pass before any delayed actor model construction."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from tests.evaluation.allocation_scheduled_delayed_fixture import delayed_env
from trade_rl.simulation import MarketExecutor
from trade_rl.simulation.targets.execution import execute_target_statefully


def scripted(
    cue,
    *,
    first_entry=6,
    code=None,
    latency=0,
    payoff=True,
    processing=True,
    persistent=False,
):
    if latency:
        # The allocation decision adapter rejects every extra latency. Native
        # executor timing controls therefore use its existing target path only.
        env = delayed_env(1, cue)
        env.reset(seed=7)
        executor = MarketExecutor(
            env.dataset,
            replace(env.execution_cost, order_latency_bars=latency),
            insolvency_valuation="retain_debt",
        )
        book, orders = env.book, env.order_book
        costs, fills = [], []
        for index in range(6, 22):
            result = execute_target_statefully(
                executor,
                book,
                orders,
                np.array([cue * 0.5]),
                start_index=index,
                bars=1,
                target_identity=f"timing-control-{cue}",
            )
            book, orders = result.book, result.order_book
            costs.append(result.interval_cost)
            fills.append(result.filled_notional)
        return SimpleNamespace(book=book, order_book=orders), costs, fills
    env = delayed_env(1, cue, latency=latency, payoff=payoff, processing=processing)
    env.reset(seed=7)
    entry = (3 if cue == 1 else 1) if code is None else code
    costs, fills = [], []
    for index in range(6, 22):
        action = (
            entry if index == first_entry or persistent and index >= first_entry else 0
        )
        _, _, done, truncated, info = env.step(action)
        assert not truncated and done is (index == 21)
        costs.append(info["execution"].interval_cost)
        fills.append(info["execution"].filled_notional)
    return env, costs, fills


@pytest.mark.parametrize("cue", [1, -1])
@pytest.mark.parametrize("latency", [0, 1])
def test_early_paid_quantity_literal_marked_profit(cue, latency):
    env, costs, fills = scripted(cue, latency=latency)
    assert env.book.quantities.tolist() == [cue * 4.0]
    assert env.book.cash == (511 if cue == 1 else 1535)
    assert costs == [1.0] + [0.0] * 15
    assert fills == [512.0] + [0.0] * 15
    assert env.book.portfolio_value == pytest.approx(1063.96, rel=0, abs=1e-10)
    assert (env.book.portfolio_value - 1024) / 1024 == pytest.approx(
        0.0390234375, rel=0, abs=1e-14
    )
    assert env.book.as_of_index == 22


@pytest.mark.parametrize("cue", [1, -1])
@pytest.mark.parametrize("first_entry", range(7, 22))
def test_all_late_entries_including_pending_through_payoff_cannot_fill(
    cue, first_entry
):
    env, costs, fills = scripted(cue, first_entry=first_entry, persistent=True)
    assert fills == costs == [0.0] * 16
    assert env.book.quantities.tolist() == [0.0]
    assert env.book.portfolio_value == env.book.cash == 1024
    assert env.order_book.active_orders  # Repeated direct action retains pending GTC.


@pytest.mark.parametrize("cue", [1, -1])
def test_discriminating_latency_two_misses_sole_capacity(cue):
    env, costs, fills = scripted(cue, latency=2, persistent=True)
    assert costs == fills == [0.0] * 16
    assert env.book.portfolio_value == 1024
    assert env.order_book.active_orders


@pytest.mark.parametrize("latency", [1, 2])
def test_allocation_adapter_explicitly_rejects_extra_latency(latency):
    env = delayed_env(1, 1, latency=latency)
    with pytest.raises(ValueError, match="zero extra latency"):
        env.reset()


@pytest.mark.parametrize("cue", [1, -1])
def test_cash_wrong_sign_fee_only_and_previous_volume_falsifiers(cue):
    cash, _, _ = scripted(cue, first_entry=99)
    wrong, costs, _ = scripted(cue, code=1 if cue == 1 else 3)
    fee, fee_costs, _ = scripted(cue, payoff=False)
    invalid, _, invalid_fills = scripted(cue, first_entry=7, processing=False)
    assert cash.book.portfolio_value == 1024
    assert wrong.book.portfolio_value == pytest.approx(982.04, abs=1e-10, rel=0)
    assert costs == fee_costs == [1.0] + [0.0] * 15
    assert fee.book.portfolio_value == 1023
    assert invalid_fills[1] == 512
    assert invalid.book.portfolio_value == pytest.approx(1063.96, abs=1e-10, rel=0)
