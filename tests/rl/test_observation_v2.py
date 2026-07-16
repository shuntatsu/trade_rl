from __future__ import annotations

import numpy as np

from trade_rl.data.market import MarketDataset
from trade_rl.rl.normalization import ObservationNormalizer
from trade_rl.rl.observations import (
    build_observation,
    observation_layout,
    observation_passthrough_indices,
)
from trade_rl.simulation.accounting import BookState
from trade_rl.strategies.trend import TrendTargets


def market() -> MarketDataset:
    n = 6
    close = np.column_stack([100.0 + np.arange(n), 200.0 + np.arange(n)])
    open_price = np.vstack([close[0], close[:-1]])
    available = np.ones((n, 2, 2), dtype=np.bool_)
    available[2, 0, 1] = False
    return MarketDataset(
        dataset_id="d" * 64,
        symbols=("A", "B"),
        timestamps=np.datetime64("2026-01-01", "ns")
        + np.arange(n) * np.timedelta64(1, "h"),
        features=np.arange(n * 4, dtype=np.float32).reshape(n, 2, 2),
        global_features=np.ones((n, 1), dtype=np.float32),
        open=open_price,
        high=np.maximum(open_price, close) + 1.0,
        low=np.minimum(open_price, close) - 1.0,
        close=close,
        volume=np.full_like(close, 1_000.0),
        funding_rate=np.zeros_like(close),
        tradable=np.ones_like(close, dtype=np.bool_),
        feature_available=available,
        feature_staleness_hours=np.ones((n, 2, 2)),
        feature_names=("ret", "rsi"),
        global_feature_names=("regime",),
        periods_per_year=8_760,
        mark_price=close * 1.001,
        index_price=close,
    )


def observation() -> tuple[MarketDataset, np.ndarray]:
    dataset = market()
    book = BookState.zero(2, 1_000.0, dataset.close[2])
    result = build_observation(
        dataset=dataset,
        index=2,
        trends=TrendTargets(
            fast=np.array([0.4, -0.4]),
            base=np.array([0.3, -0.3]),
            slow=np.array([0.2, -0.2]),
        ),
        alpha=np.zeros(2),
        factor_basis=np.array([[0.5, -0.5]]),
        hybrid=book,
        shadow=book.clone(),
        start_index=0,
        end_index=5,
        hybrid_risk_scale=1.0,
        shadow_risk_scale=1.0,
        previous_action=np.zeros(4),
        action_size=4,
    )
    return dataset, result


def test_observation_v3_contains_factor_and_execution_market_contracts() -> None:
    dataset, result = observation()
    layout = observation_layout(dataset, action_size=4, n_factors=1)
    assert result.shape == (layout.size,)
    rows = result[: dataset.n_symbols * layout.per_symbol_width].reshape(
        dataset.n_symbols, layout.per_symbol_width
    )
    assert rows[0, -1] > 0.0  # mark/index premium


def test_normalizer_is_fitted_only_on_train_range_and_preserves_masks() -> None:
    dataset, one = observation()
    observations = np.vstack([one, one + 1.0, one + 100.0])
    passthrough = observation_passthrough_indices(dataset, action_size=4, n_factors=1)
    normalizer = ObservationNormalizer.fit(
        observations,
        train_start=0,
        train_end=2,
        passthrough_indices=passthrough,
        dataset_id=dataset.dataset_id,
    )
    assert normalizer.mean[0] < 10.0
    transformed = normalizer.transform(one)
    np.testing.assert_array_equal(
        transformed[list(passthrough)], one[list(passthrough)]
    )


def test_observation_v4_semantically_scales_age_basis_and_equity_state() -> None:
    from trade_rl.rl.observations import ObservationExecutionState

    dataset = market()
    book = BookState.zero(2, 1_000.0, dataset.close[2])
    book.peak_value = 1_250.0
    result = build_observation(
        dataset=dataset,
        index=2,
        trends=TrendTargets(fast=np.zeros(2), base=np.zeros(2), slow=np.zeros(2)),
        alpha=np.zeros(2),
        hybrid=book,
        shadow=book.clone(),
        start_index=0,
        end_index=5,
        hybrid_risk_scale=1.0,
        shadow_risk_scale=1.0,
        execution_state=ObservationExecutionState(
            requested_weights=np.zeros(2),
            fill_ratio=np.ones(2),
            unfilled_turnover=np.zeros(2),
            participation=np.zeros(2),
            execution_cost=np.zeros(2),
            position_age=np.array([24.0, 48.0]),
        ),
        previous_action=np.zeros(2),
        action_size=2,
    )
    layout = observation_layout(dataset, action_size=2)
    rows = result[: dataset.n_symbols * layout.per_symbol_width].reshape(
        dataset.n_symbols, layout.per_symbol_width
    )
    offset = 4 * dataset.n_features
    np.testing.assert_allclose(rows[0, offset + 14], np.log1p(1.0), rtol=0.0, atol=1e-7)
    assert abs(rows[0, -1]) < 0.2
    global_values = result[dataset.n_symbols * layout.per_symbol_width :]
    endogenous = 4 * len(dataset.global_feature_names)
    np.testing.assert_allclose(
        global_values[endogenous], np.log(0.8), rtol=0.0, atol=1e-7
    )
