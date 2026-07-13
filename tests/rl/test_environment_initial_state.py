from __future__ import annotations

import numpy as np

from trade_rl.data.market import MarketDataset
from trade_rl.rl.environment import ResidualMarketEnv, ResidualMarketEnvConfig
from trade_rl.simulation.accounting import BookState
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.strategies.trend import TrendConfig, TrendStrategy


def market() -> MarketDataset:
    n_bars = 80
    elapsed = np.arange(n_bars, dtype=np.float64)
    close = np.column_stack(
        [
            np.exp(elapsed * 0.001),
            np.exp(-elapsed * 0.001),
        ]
    )
    open_price = np.vstack([close[0], close[:-1]])
    sampling = np.zeros(n_bars, dtype=np.float64)
    sampling[30] = 1.0
    warmup = np.ones(n_bars, dtype=np.bool_)
    warmup[:20] = False
    return MarketDataset(
        dataset_id="f" * 64,
        symbols=("UP", "DOWN"),
        timestamps=np.datetime64("2026-01-01T01:00:00", "ns")
        + np.arange(n_bars) * np.timedelta64(1, "h"),
        features=np.zeros((n_bars, 2, 1), dtype=np.float32),
        global_features=np.zeros((n_bars, 1), dtype=np.float32),
        open=open_price,
        high=np.maximum(open_price, close) * 1.001,
        low=np.minimum(open_price, close) * 0.999,
        close=close,
        volume=np.full((n_bars, 2), 1_000_000.0),
        funding_rate=np.zeros((n_bars, 2)),
        tradable=np.ones((n_bars, 2), dtype=np.bool_),
        feature_available=np.ones((n_bars, 2, 1), dtype=np.bool_),
        feature_names=("ret",),
        global_feature_names=("regime",),
        periods_per_year=8_760,
        episode_sampling_weight=sampling,
        warmup_complete=warmup,
    )


def environment(*, random_gross: float = 0.0) -> ResidualMarketEnv:
    return ResidualMarketEnv(
        market(),
        trend_strategy=TrendStrategy(
            TrendConfig(fast_lookback=4, base_lookback=8, slow_lookback=16)
        ),
        config=ResidualMarketEnvConfig(
            episode_bars=8,
            decision_every=4,
            random_initial_inventory_gross=random_gross,
            execution_cost=ExecutionCostConfig.zero(),
        ),
    )


def test_reset_uses_configured_episode_sampling_weights() -> None:
    env = environment()

    _, info = env.reset(seed=3)

    assert info["start_index"] == 30


def test_random_initial_inventory_is_reproducible_and_shared_by_books() -> None:
    first = environment(random_gross=0.5)
    second = environment(random_gross=0.5)

    first.reset(seed=11)
    second.reset(seed=11)

    np.testing.assert_allclose(first.hybrid.weights, second.hybrid.weights)
    np.testing.assert_allclose(first.hybrid.weights, first.shadow.weights)
    assert np.abs(first.hybrid.weights).sum() == 0.5


def test_explicit_initial_book_is_carried_into_episode() -> None:
    env = environment()
    initial = BookState.from_weights(
        weights=np.array([0.25, -0.25]),
        capital=2.0,
        prices=env.dataset.mark_prices[30],
    )

    _, info = env.reset(
        seed=0,
        options={"start_idx": 30, "initial_hybrid_book": initial},
    )

    np.testing.assert_allclose(env.hybrid.weights, initial.weights)
    np.testing.assert_allclose(env.shadow.weights, initial.weights)
    assert info["hybrid_state_digest"] == info["shadow_state_digest"]
