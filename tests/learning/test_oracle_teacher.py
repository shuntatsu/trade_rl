from __future__ import annotations

import numpy as np
import pytest

from trade_rl.data.market import MarketDataset
from trade_rl.learning.oracle_teacher import (
    OracleTeacherConfig,
    oracle_target_path,
)
from trade_rl.simulation.execution import ExecutionCostConfig


def _market(close_values: np.ndarray) -> MarketDataset:
    close = np.asarray(close_values, dtype=np.float64)
    if close.ndim == 1:
        close = close[:, None]
    n_bars, n_symbols = close.shape
    open_price = np.vstack([close[0], close[:-1]])
    return MarketDataset(
        dataset_id="a" * 64,
        symbols=tuple(f"S{index}" for index in range(n_symbols)),
        timestamps=np.datetime64("2026-01-01", "ns")
        + np.arange(n_bars) * np.timedelta64(15, "m"),
        features=np.zeros((n_bars, n_symbols, 1), dtype=np.float32),
        global_features=np.zeros((n_bars, 1), dtype=np.float32),
        open=open_price,
        high=np.maximum(open_price, close) * 1.001,
        low=np.minimum(open_price, close) * 0.999,
        close=close,
        volume=np.full_like(close, 1_000_000.0),
        funding_rate=np.zeros_like(close),
        tradable=np.ones_like(close, dtype=np.bool_),
        feature_available=np.ones((n_bars, n_symbols, 1), dtype=np.bool_),
        feature_names=("return",),
        global_feature_names=("regime",),
        periods_per_year=35_040,
    )


def test_oracle_path_is_deterministic_bounded_and_train_range_only() -> None:
    market = _market(
        np.column_stack(
            [
                100.0 * np.exp(np.arange(9) * 0.02),
                100.0 * np.exp(np.arange(9) * -0.02),
            ]
        )
    )
    config = OracleTeacherConfig(execution_cost=ExecutionCostConfig.zero())

    first = oracle_target_path(market, (1, 8), config)
    second = oracle_target_path(market, (1, 8), config)
    changed_close = market.close.copy()
    changed_close[8] *= 100.0
    changed_future = _market(changed_close)

    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(
        first,
        oracle_target_path(changed_future, (1, 8), config),
    )
    assert first.shape == (6, 2)
    assert np.max(np.abs(first)) <= 0.5
    assert np.max(np.abs(first).sum(axis=1)) <= 1.0
    assert np.mean(first[:, 0]) > 0.0
    assert np.mean(first[:, 1]) < 0.0


def test_execution_cost_creates_flat_or_hold_regions() -> None:
    market = _market(np.array([100.0, 100.1, 100.0, 100.1, 100.0, 100.1]))
    free = OracleTeacherConfig(execution_cost=ExecutionCostConfig.zero())
    expensive = OracleTeacherConfig(
        execution_cost=ExecutionCostConfig(
            fee_rate=0.01,
            spread_rate=0.01,
            impact_rate=0.0,
            max_participation_rate=1.0,
        )
    )

    free_path = oracle_target_path(market, (0, 6), free)
    expensive_path = oracle_target_path(market, (0, 6), expensive)

    assert np.count_nonzero(np.diff(free_path[:, 0])) > 0
    assert np.count_nonzero(expensive_path) == 0


@pytest.mark.parametrize("train_range", [(-1, 4), (0, 1), (3, 3), (0, 7)])
def test_oracle_rejects_ranges_outside_dataset(
    train_range: tuple[int, int],
) -> None:
    market = _market(np.linspace(100.0, 105.0, 6))

    with pytest.raises(ValueError, match="training range"):
        oracle_target_path(market, train_range, OracleTeacherConfig())


def test_oracle_labels_only_fully_executable_portfolio_targets() -> None:
    from dataclasses import replace

    close = np.column_stack(
        [
            100.0 * np.exp(np.arange(8) * 0.05),
            100.0 * np.exp(np.arange(8) * 0.01),
        ]
    )
    market = _market(close)
    tiny_volume = market.volume.copy()
    tiny_volume[:, 0] = 0.001
    constrained = replace(
        market,
        volume=tiny_volume,
        max_participation_rate=np.full_like(close, 0.01),
        minimum_notional=np.full_like(close, 5.0),
    )
    config = OracleTeacherConfig(
        execution_cost=ExecutionCostConfig.zero(),
        reference_portfolio_value=100_000.0,
    )

    targets = oracle_target_path(constrained, (0, 8), config)

    assert np.any(targets[:, 0] > 0.0)
    assert np.max(np.abs(targets).sum(axis=1)) <= config.max_gross


def test_oracle_respects_point_in_time_tradability() -> None:
    from dataclasses import replace

    market = _market(100.0 * np.exp(np.arange(8) * 0.03))
    tradable = market.tradable.copy()
    tradable[1:, 0] = False
    constrained = replace(market, tradable=tradable)

    targets = oracle_target_path(
        constrained,
        (0, 8),
        OracleTeacherConfig(execution_cost=ExecutionCostConfig.zero()),
    )

    assert np.count_nonzero(targets[:, 0]) == 0


def test_oracle_accounts_for_weight_drift_and_direction_permissions() -> None:
    """A blocked reduction is a no-fill hold, matching the real executor."""

    from dataclasses import replace

    market = _market(np.array([100.0, 200.0, 200.0]))
    sell_allowed = np.ones_like(market.close, dtype=np.bool_)
    sell_allowed[2, 0] = False
    constrained = replace(market, sell_allowed=sell_allowed)

    targets = oracle_target_path(
        constrained,
        (0, 3),
        OracleTeacherConfig(execution_cost=ExecutionCostConfig.zero()),
    )

    assert targets[0, 0] > 0.0
    assert np.all(targets[:, 0] >= 0.0)


def test_oracle_one_bar_transition_matches_deterministic_executor() -> None:
    from trade_rl.learning.oracle_teacher import (
        _open_state_matrix,
        _transition_matrices,
    )
    from trade_rl.simulation.accounting import BookState
    from trade_rl.simulation.execution import MarketExecutor

    market = _market(np.array([100.0, 110.0, 110.0]))
    cost = ExecutionCostConfig(
        fee_rate=0.001,
        spread_rate=0.002,
        impact_rate=0.0,
        max_participation_rate=1.0,
        maintenance_margin_rate=0.0,
    )
    config = OracleTeacherConfig(
        execution_cost=cost,
        reference_portfolio_value=100_000.0,
    )
    target = np.array([[0.45]], dtype=np.float64)
    gap, open_weights, open_equity, valid_prior = _open_state_matrix(
        market,
        close_index=0,
        prior_close_weights=np.zeros((1, 1), dtype=np.float64),
        prior_scores=np.zeros(1, dtype=np.float64),
        reference_portfolio_value=config.reference_portfolio_value,
    )
    valid, close_factor, close_weights, effective_targets = _transition_matrices(
        market,
        config,
        close_index=0,
        current_weights=open_weights,
        open_equity=open_equity,
        targets=target,
    )
    result = MarketExecutor(market, cost).execute_interval(
        BookState.zero(
            1,
            config.reference_portfolio_value,
            market.close[0],
            contract_multipliers=market.resolved_array("contract_multipliers"),
        ),
        target[0],
        start_index=0,
        bars=1,
    )

    assert valid_prior[0]
    assert valid[0, 0]
    np.testing.assert_allclose(effective_targets[0, 0], target[0])
    assert (
        result.book.portfolio_value / config.reference_portfolio_value
        == pytest.approx(gap[0] * close_factor[0, 0])
    )
    np.testing.assert_allclose(result.book.weights, close_weights[0, 0], atol=1e-12)


def test_oracle_holds_profitable_position_when_rebalance_is_suppressed() -> None:
    market = _market(np.array([100.0, 110.0, 111.0, 112.0, 113.0]))
    config = OracleTeacherConfig(
        execution_cost=ExecutionCostConfig.zero(),
        no_trade_band=0.05,
        entry_threshold=0.10,
        exit_threshold=0.03,
    )

    targets = oracle_target_path(market, (0, 5), config)

    assert np.all(targets[:, 0] > 0.0)


def test_oracle_partial_fill_matches_executor_instead_of_invalidating_transition() -> (
    None
):
    from dataclasses import replace

    from trade_rl.learning.oracle_teacher import (
        _open_state_matrix,
        _transition_matrices,
    )
    from trade_rl.simulation.accounting import BookState
    from trade_rl.simulation.execution import MarketExecutor

    market = _market(np.array([100.0, 110.0, 110.0]))
    constrained = replace(
        market,
        volume=np.full_like(market.volume, 10.0),
        max_participation_rate=np.full_like(market.close, 0.01),
        minimum_notional=np.zeros_like(market.close),
    )
    cost = ExecutionCostConfig(
        fee_rate=0.0,
        spread_rate=0.0,
        impact_rate=0.0,
        max_participation_rate=0.01,
        maintenance_margin_rate=0.0,
    )
    config = OracleTeacherConfig(
        execution_cost=cost,
        reference_portfolio_value=1_000.0,
    )
    target = np.array([[0.45]], dtype=np.float64)
    _, open_weights, open_equity, _ = _open_state_matrix(
        constrained,
        close_index=0,
        prior_close_weights=np.zeros((1, 1), dtype=np.float64),
        prior_scores=np.zeros(1, dtype=np.float64),
        reference_portfolio_value=config.reference_portfolio_value,
    )
    valid, _, _, effective_targets = _transition_matrices(
        constrained,
        config,
        close_index=0,
        current_weights=open_weights,
        open_equity=open_equity,
        targets=target,
    )
    result = MarketExecutor(constrained, cost).execute_interval(
        BookState.zero(
            1,
            config.reference_portfolio_value,
            constrained.close[0],
            contract_multipliers=constrained.resolved_array("contract_multipliers"),
        ),
        target[0],
        start_index=0,
        bars=1,
    )

    assert valid[0, 0]
    expected_open_weight = (
        result.filled_notional_by_symbol[0] / config.reference_portfolio_value
    )
    assert 0.0 < effective_targets[0, 0, 0] < target[0, 0]
    assert effective_targets[0, 0, 0] == pytest.approx(expected_open_weight)


def test_oracle_below_minimum_notional_is_an_executable_noop() -> None:
    from dataclasses import replace

    from trade_rl.learning.oracle_teacher import (
        _open_state_matrix,
        _transition_matrices,
    )

    market = _market(np.array([100.0, 101.0, 101.0]))
    constrained = replace(
        market,
        minimum_notional=np.full_like(market.close, 500.0),
    )
    config = OracleTeacherConfig(
        execution_cost=ExecutionCostConfig.zero(),
        reference_portfolio_value=1_000.0,
    )
    _, open_weights, open_equity, _ = _open_state_matrix(
        constrained,
        close_index=0,
        prior_close_weights=np.zeros((1, 1), dtype=np.float64),
        prior_scores=np.zeros(1, dtype=np.float64),
        reference_portfolio_value=config.reference_portfolio_value,
    )
    valid, _, _, effective_targets = _transition_matrices(
        constrained,
        config,
        close_index=0,
        current_weights=open_weights,
        open_equity=open_equity,
        targets=np.array([[0.45]], dtype=np.float64),
    )

    assert valid[0, 0]
    np.testing.assert_array_equal(effective_targets[0, 0], open_weights[0])


def test_delayed_oracle_returns_submitted_actions_and_discards_terminal_pending_action() -> (
    None
):
    market = _market(100.0 * np.exp(np.arange(8) * 0.04))
    config = OracleTeacherConfig(
        execution_cost=ExecutionCostConfig.zero(),
        signal_delay_decisions=1,
    )

    targets = oracle_target_path(market, (0, 8), config)

    assert targets.shape == (7, 1)
    assert np.any(targets[:-1, 0] > 0.0)
    np.testing.assert_array_equal(targets[-1], np.zeros(1, dtype=np.float32))
    assert "approximate" in config.schema_version
