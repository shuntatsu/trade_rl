from __future__ import annotations

from dataclasses import replace

import numpy as np

from trade_rl.data.market import MarketDataset
from trade_rl.learning.oracle_market_tape import build_oracle_market_tape
from trade_rl.learning.oracle_teacher import (
    OracleTeacherConfig,
    _open_state_matrix,
    _transition_matrices,
)
from trade_rl.learning.oracle_transition_numpy import (
    FillClassification,
    numpy_transition_step,
)
from trade_rl.simulation.execution import ExecutionCostConfig


def _market() -> MarketDataset:
    close = np.array(
        [
            [100.0, 50.0],
            [105.0, 49.0],
            [106.0, 48.0],
            [107.0, 47.0],
        ],
        dtype=np.float64,
    )
    open_price = np.vstack([close[0], close[:-1]])
    return MarketDataset(
        dataset_id="a" * 64,
        symbols=("A", "B"),
        timestamps=np.datetime64("2026-01-01", "ns")
        + np.arange(4) * np.timedelta64(15, "m"),
        features=np.zeros((4, 2, 1), dtype=np.float32),
        global_features=np.zeros((4, 1), dtype=np.float32),
        open=open_price,
        high=np.maximum(open_price, close) * 1.001,
        low=np.minimum(open_price, close) * 0.999,
        close=close,
        volume=np.full_like(close, 10_000.0),
        funding_rate=np.array([[0.0, 0.0], [0.001, -0.001], [0.0, 0.0], [0.0, 0.0]]),
        tradable=np.ones_like(close, dtype=np.bool_),
        feature_available=np.ones((4, 2, 1), dtype=np.bool_),
        feature_names=("return",),
        global_feature_names=("regime",),
        periods_per_year=35_040,
        fee_rate=np.full_like(close, 0.0002),
        taker_fee_rate=np.full_like(close, 0.0003),
        spread_rate=np.full_like(close, 0.0004),
        max_participation_rate=np.full_like(close, 0.25),
        minimum_notional=np.zeros_like(close),
        borrow_rate=np.full_like(close, 0.01),
        funding_due=np.ones_like(close, dtype=np.bool_),
        mark_price=close * 1.0005,
        cash_rate=np.full(4, 0.02),
    )


def _config() -> OracleTeacherConfig:
    return OracleTeacherConfig(
        execution_cost=ExecutionCostConfig(
            fee_rate=0.0005,
            taker_fee_rate=0.0006,
            spread_rate=0.0007,
            impact_rate=0.001,
            max_participation_rate=0.2,
            maintenance_margin_rate=0.05,
        ),
        reference_portfolio_value=100_000.0,
    )


def test_batched_transition_matches_legacy_helpers() -> None:
    market = _market()
    config = _config()
    tape = build_oracle_market_tape(market, (0, 3), config.bellman_parameters)
    prior_scores = np.array([0.0, -0.1], dtype=np.float64)
    prior_weights = np.array([[0.0, 0.0], [0.2, -0.1]], dtype=np.float64)
    targets = np.array([[0.0, 0.0], [0.3, -0.2]], dtype=np.float64)

    gap, open_weights, open_equity, valid_prior = _open_state_matrix(
        market,
        close_index=0,
        prior_close_weights=prior_weights,
        prior_scores=prior_scores,
        reference_portfolio_value=config.reference_portfolio_value,
    )
    valid, close_factor, close_weights, effective_targets = _transition_matrices(
        market,
        config,
        close_index=0,
        current_weights=open_weights,
        open_equity=open_equity,
        targets=targets,
    )

    result = numpy_transition_step(
        tape=tape,
        step=0,
        prior_scores=prior_scores[None, :],
        prior_close_weights=prior_weights[None, :, :],
        targets=targets,
        parameters=config.bellman_parameters,
    )

    np.testing.assert_allclose(result.gap_factor[0], gap, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(
        result.open_weights[0], open_weights, rtol=1e-10, atol=1e-12
    )
    np.testing.assert_allclose(
        result.open_equity[0], open_equity, rtol=1e-10, atol=1e-12
    )
    np.testing.assert_array_equal(result.valid_prior[0], valid_prior)
    np.testing.assert_array_equal(result.valid[0], valid)
    np.testing.assert_allclose(
        result.close_factor[0], close_factor, rtol=1e-10, atol=1e-12
    )
    np.testing.assert_allclose(
        result.close_weights[0], close_weights, rtol=1e-10, atol=1e-12
    )
    np.testing.assert_allclose(
        result.effective_targets[0],
        effective_targets,
        rtol=1e-10,
        atol=1e-12,
    )


def test_transition_batches_independent_episodes() -> None:
    market = _market()
    config = _config()
    tape = build_oracle_market_tape(market, (0, 3), config.bellman_parameters)
    scores = np.array([[0.0], [-0.2]], dtype=np.float64)
    weights = np.array([[[0.0, 0.0]], [[0.1, -0.1]]], dtype=np.float64)
    targets = np.array([[0.0, 0.0], [0.3, -0.2]], dtype=np.float64)

    batched = numpy_transition_step(
        tape=tape,
        step=0,
        prior_scores=scores,
        prior_close_weights=weights,
        targets=targets,
        parameters=config.bellman_parameters,
    )

    assert batched.valid.shape == (2, 1, 2)
    assert batched.close_weights.shape == (2, 1, 2, 2)
    for batch_index in range(2):
        single = numpy_transition_step(
            tape=tape,
            step=0,
            prior_scores=scores[batch_index : batch_index + 1],
            prior_close_weights=weights[batch_index : batch_index + 1],
            targets=targets,
            parameters=config.bellman_parameters,
        )
        np.testing.assert_array_equal(batched.valid[batch_index], single.valid[0])
        np.testing.assert_allclose(
            batched.close_weights[batch_index], single.close_weights[0]
        )


def test_minimum_notional_noop_is_classified_and_valid() -> None:
    market = _market()
    constrained = replace(
        market,
        minimum_notional=np.full_like(market.close, 1_000_000.0),
    )
    config = _config()
    tape = build_oracle_market_tape(
        constrained,
        (0, 3),
        config.bellman_parameters,
    )

    result = numpy_transition_step(
        tape=tape,
        step=0,
        prior_scores=np.zeros((1, 1), dtype=np.float64),
        prior_close_weights=np.zeros((1, 1, 2), dtype=np.float64),
        targets=np.array([[0.3, 0.0]], dtype=np.float64),
        parameters=config.bellman_parameters,
    )

    assert result.valid[0, 0, 0]
    assert (
        result.fill_classification[0, 0, 0] == FillClassification.MINIMUM_NOTIONAL_NOOP
    )
    np.testing.assert_array_equal(result.effective_targets[0, 0, 0], np.zeros(2))
