from __future__ import annotations

import numpy as np
import pytest

from tests.strategies.test_cost_aware_allocator import (
    CAPITAL,
    N_DAYS,
    SYMBOLS,
    _dataset,
    _noise,
)
from trade_rl.evaluation.replay import run_shared_portfolio_replay
from trade_rl.simulation import ExecutionCostConfig
from trade_rl.strategies.allocation.buy_and_hold import BuyAndHoldPortfolioPolicy


def _run(
    policy: BuyAndHoldPortfolioPolicy, *, days: int = N_DAYS, settle: bool = False
):
    dataset = _dataset(_noise(seed=2))
    return dataset, run_shared_portfolio_replay(
        dataset,
        policy,
        start_index=0,
        stop_index=days * 24,
        initial_capital=CAPITAL,
        execution_cost=ExecutionCostConfig(),
        settle_terminal_position=settle,
    )


def test_buy_and_hold_trades_once_then_holds_exact_quantities() -> None:
    policy = BuyAndHoldPortfolioPolicy(symbols=SYMBOLS)
    dataset, result = _run(policy)
    assert result.book.fill_count == len(SYMBOLS)
    assert result.book.rebalance_events == len(SYMBOLS)
    first = result.decisions[0]
    np.testing.assert_allclose(first.proposal_weights, [0.99 / 3] * 3)
    for decision in result.decisions[1:]:
        assert decision.requested_hold_symbols == (0, 1, 2)
    initial_price = dataset.close[1]
    quantities = np.asarray(result.book.quantities)
    assert np.all(quantities > 0.0)
    np.testing.assert_allclose(
        quantities * dataset.close[1] / CAPITAL, [0.99 / 3] * 3, rtol=2e-3
    )
    assert initial_price.shape == (3,)


def test_buy_and_hold_pays_fees_once_and_tracks_equal_weight_index() -> None:
    policy = BuyAndHoldPortfolioPolicy(symbols=SYMBOLS)
    dataset, result = _run(policy)
    assert result.book.total_cost > 0.0
    growth = float(np.mean(dataset.close[N_DAYS * 24] / dataset.close[1]))
    expected = CAPITAL * (1.0 + 0.99 * (growth - 1.0))
    assert float(result.book.portfolio_value) == pytest.approx(expected, rel=5e-3)
    entry_cost_bound = CAPITAL * 0.99 * 0.002  # one entry at a few bp, never repeated
    assert result.book.total_cost < entry_cost_bound


def test_terminal_settlement_closes_the_position_with_one_extra_trade() -> None:
    policy = BuyAndHoldPortfolioPolicy(symbols=SYMBOLS)
    _, result = _run(policy, settle=True)
    assert np.allclose(result.book.quantities, 0.0)
    assert result.book.rebalance_events == 2 * len(SYMBOLS)


def test_optional_calendar_rebalance_reissues_equal_weights() -> None:
    policy = BuyAndHoldPortfolioPolicy(symbols=SYMBOLS, rebalance_every_bars=24 * 30)
    _, result = _run(policy)
    assert result.book.fill_count > len(SYMBOLS)
    never = BuyAndHoldPortfolioPolicy(symbols=SYMBOLS)
    _, base = _run(never)
    assert result.book.turnover_total > base.book.turnover_total


@pytest.mark.parametrize(
    "overrides",
    [
        {"symbols": ()},
        {"symbols": ("A", "A")},
        {"symbols": ["A"]},
        {"gross_budget": 0.0},
        {"gross_budget": 1.5},
        {"gross_budget": float("nan")},
        {"gross_budget": True},
        {"rebalance_every_bars": 0},
        {"rebalance_every_bars": 1.5},
    ],
)
def test_constructor_validation(overrides: dict[str, object]) -> None:
    kwargs: dict[str, object] = {"symbols": SYMBOLS}
    kwargs.update(overrides)
    with pytest.raises((TypeError, ValueError)):
        BuyAndHoldPortfolioPolicy(**kwargs)  # type: ignore[arg-type]


def test_roster_mismatch_and_non_increasing_time_are_rejected() -> None:
    dataset = _dataset(_noise(seed=2))
    wrong = BuyAndHoldPortfolioPolicy(symbols=("X", "Y", "Z"))
    with pytest.raises(ValueError, match="roster"):
        run_shared_portfolio_replay(
            dataset, wrong, start_index=0, stop_index=48, initial_capital=CAPITAL
        )
    policy = BuyAndHoldPortfolioPolicy(symbols=SYMBOLS)
    run_shared_portfolio_replay(
        dataset, policy, start_index=0, stop_index=48, initial_capital=CAPITAL
    )
    with pytest.raises(ValueError, match="increase"):
        run_shared_portfolio_replay(
            dataset, policy, start_index=0, stop_index=48, initial_capital=CAPITAL
        )
