"""Core-owned spot decision reserve and native cash-account regression."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from tests.strategies.test_portfolio_policy_adapter import SYMBOLS, _dataset, _snapshot
from trade_rl.evaluation.replay import run_shared_portfolio_replay
from trade_rl.risk import PreTradeRisk, PreTradeRiskConfig
from trade_rl.risk.spot_cash import spot_cash_target_factor
from trade_rl.simulation import ExecutionCostConfig
from trade_rl.strategies.allocation.buy_and_hold import BuyAndHoldPortfolioPolicy
from trade_rl.strategies.allocation.spot_cash import SpotCashBudgetPolicy
from trade_rl.strategies.portfolio import PortfolioTargetProposal


def test_full_spot_entry_reserves_fees_in_native_book():
    cost = ExecutionCostConfig(
        fee_rate=0.001,
        spread_rate=0,
        impact_rate=0,
        allow_short=False,
        partial_fill_carry=False,
    )
    data = replace(_dataset(), volume=np.full((3, 3), 1e8))
    risk = PreTradeRisk(
        PreTradeRiskConfig(
            max_gross=1,
            max_abs_weight=1,
            max_turnover=None,
            drawdown_start=1,
            drawdown_stop=1,
        )
    )

    def replay(policy):
        return run_shared_portfolio_replay(
            data,
            policy,
            start_index=0,
            stop_index=2,
            initial_capital=1000,
            execution_cost=cost,
            risk=risk,
            capture_ledger_evidence=True,
        )

    unreserved = replay(BuyAndHoldPortfolioPolicy(symbols=SYMBOLS, gross_budget=1))
    assert unreserved.ledger_evidence.intervals.intervals[0].cash_after < 0
    wrapped = SpotCashBudgetPolicy(
        BuyAndHoldPortfolioPolicy(symbols=SYMBOLS, gross_budget=1), reserve_rate=0.001
    )
    result = replay(wrapped)
    rows = result.ledger_evidence.intervals.intervals
    assert min(row.cash_after for row in rows) >= -1e-9
    assert result.book.fill_count == 3
    assert rows[0].exact_quantities_after == rows[1].exact_quantities_after
    assert sum(result.decisions[0].proposal_weights) == pytest.approx(1 / 1.001)
    assert result.book.total_cost == pytest.approx(1000 * 0.001 / 1.001)


def test_rotation_cannot_count_unexecuted_sell_proceeds():
    current = (0.0, 0.0, 0.6)
    target = (0.3, 0.3, 0.0)
    factor = spot_cash_target_factor(
        current, target, cash_fraction=0.4, gross_limit=1.0, reserve_rate=0.002
    )
    assert factor == pytest.approx(0.4 / (0.6 * 1.002))
    assert 0.6 * factor * 1.002 <= 0.4 + 1e-15


def test_projection_randomized_budget_and_gross_oracles():
    rng = np.random.default_rng(53)
    for _ in range(300):
        current = tuple(rng.dirichlet(np.ones(5)) * rng.uniform(0, 0.99))
        target = tuple(rng.dirichlet(np.ones(5)) * rng.uniform(0, 2))
        cash = max(0, 1 - sum(current))
        rate = float(rng.uniform(0, 0.05))
        cap = float(rng.uniform(0.1, 1))
        f = spot_cash_target_factor(
            current, target, cash_fraction=cash, gross_limit=cap, reserve_rate=rate
        )
        buys = sum(max(f * t - c, 0) for c, t in zip(current, target, strict=True))
        assert 0 <= f <= 1
        assert buys * (1 + rate) <= cash + 1e-14
        assert (
            f * sum(target) <= max(0, (cap - rate * sum(current)) / (1 + rate)) + 1e-14
        )


@pytest.mark.parametrize(
    "current,target,kwargs",
    [
        ((), (), {}),
        ((0.0,), (1.0, 0.0), {}),
        ((-0.1,), (1.0,), {}),
        ((0.0,), (float("nan"),), {}),
        ((0.0,), (True,), {}),
        ((0.0,), (1.0,), {"cash_fraction": float("inf")}),
        ((0.0,), (1.0,), {"reserve_rate": 1.0}),
        ((0.0,), (1.0,), {"gross_limit": True}),
    ],
)
def test_projection_rejects_invalid_inputs(current, target, kwargs):
    params = {"cash_fraction": 1.0, "reserve_rate": 0.001, "gross_limit": 1.0}
    params.update(kwargs)
    with pytest.raises((TypeError, ValueError)):
        spot_cash_target_factor(current, target, **params)


def test_policy_preserves_holds_and_rejects_short_spot():
    s = _snapshot()
    s = replace(
        s,
        account=replace(
            s.account,
            weights=(0.2, 0.0, 0.0),
            quantities=(2.0, 0.0, 0.0),
            exact_quantities=("2", "0", "0"),
            cash=800.0,
            portfolio_value=1000.0,
            peak_value=1000.0,
            max_drawdown=0.0,
        ),
    )

    class Fixed:
        def __init__(self, p):
            self.p = p

        def decide_portfolio(self, snapshot):
            return self.p

    hold = PortfolioTargetProposal(s.account.weights, (0, 1, 2))
    assert (
        SpotCashBudgetPolicy(Fixed(hold), reserve_rate=0.001).decide_portfolio(s)
        is hold
    )
    with pytest.raises(ValueError, match="nonnegative"):
        SpotCashBudgetPolicy(
            Fixed(PortfolioTargetProposal((0.0, -0.1, 0.0))), reserve_rate=0.001
        ).decide_portfolio(s)


def test_projection_preserves_zero_cash_sells_and_rejects_overflow():
    assert (
        spot_cash_target_factor(
            (0.8, 0.0),
            (0.0, 0.0),
            cash_fraction=-0.02,
            gross_limit=1.0,
            reserve_rate=0.001,
        )
        == 1.0
    )
    assert (
        spot_cash_target_factor(
            (0.8, 0.0),
            (0.0, 0.8),
            cash_fraction=0.0,
            gross_limit=1.0,
            reserve_rate=0.001,
        )
        == 0.0
    )
    with pytest.raises(ValueError, match="gross"):
        spot_cash_target_factor(
            (0.0, 0.0),
            (1.7e308, 1.7e308),
            cash_fraction=1.0,
            gross_limit=1.0,
            reserve_rate=0.001,
        )


def test_spot_wrapper_uses_native_drawdown_scale_once_and_rejects_inconsistent_hold():
    s = _snapshot()
    s = replace(
        s,
        risk=replace(
            s.risk,
            max_gross=1.0,
            max_abs_weight=1.0,
            drawdown_start=0.1,
            drawdown_stop=0.3,
            max_turnover=None,
        ),
        account=replace(
            s.account,
            weights=(0.2, 0.0, 0.0),
            quantities=(2.0, 0.0, 0.0),
            exact_quantities=("2", "0", "0"),
            cash=800.0,
            portfolio_value=1000.0,
            peak_value=1250.0,
            max_drawdown=0.2,
        ),
    )

    class Fixed:
        def __init__(self, p):
            self.p = p

        def decide_portfolio(self, snapshot):
            return self.p

    p = PortfolioTargetProposal((1.0, 1.0, 1.0))
    wrapped = SpotCashBudgetPolicy(Fixed(p), reserve_rate=0.001)
    target = wrapped.decide_portfolio(s)
    risk = PreTradeRisk(
        PreTradeRiskConfig(
            max_gross=1,
            max_abs_weight=1,
            max_turnover=None,
            drawdown_start=0.1,
            drawdown_stop=0.3,
        )
    )
    native = risk.constrain(
        np.array(target.target_weights),
        current=np.array(s.account.weights),
        drawdown=0.2,
    )
    assert 0 < sum(native.weights) <= 0.5
    with pytest.raises(ValueError, match="held spot target"):
        SpotCashBudgetPolicy(
            Fixed(PortfolioTargetProposal((0.3, 0.0, 0.0), (0,))), reserve_rate=0.001
        ).decide_portfolio(s)
