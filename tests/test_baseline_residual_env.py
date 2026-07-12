import numpy as np

from mars_lite.env.baseline_residual_env import BaselineResidualTradingEnv
from mars_lite.features.feature_pipeline import FeatureSet
from mars_lite.trading.post_processor import make_legacy_processor
from mars_lite.trading.trend_family import TrendFamily, TrendFamilyConfig


def _feature_set(n_bars: int = 220) -> FeatureSet:
    timestamps = np.datetime64("2026-01-01T00:00:00", "ns") + np.arange(
        n_bars
    ) * np.timedelta64(1, "h")
    up = np.exp(np.arange(n_bars, dtype=np.float64) * 0.001)
    down = np.exp(-np.arange(n_bars, dtype=np.float64) * 0.001)
    close = np.column_stack([up, down])
    return FeatureSet(
        symbols=["UP", "DOWN"],
        timestamps=timestamps,
        features=np.zeros((n_bars, 2, 1), dtype=np.float32),
        global_features=np.zeros((n_bars, 1), dtype=np.float32),
        close=close,
        open_next=close.copy(),
        funding_rate=np.zeros_like(close),
        feature_names=["dummy"],
        global_feature_names=["dummy_global"],
    )


def _env(decision_every: int = 4) -> BaselineResidualTradingEnv:
    return BaselineResidualTradingEnv(
        _feature_set(),
        trend_family=TrendFamily(
            TrendFamilyConfig(
                fast_lookback=12,
                base_lookback=24,
                slow_lookback=48,
                rebalance_every=12,
            )
        ),
        decision_every=decision_every,
        episode_bars=48,
        post_processor=make_legacy_processor(0.0),
        min_trade_delta=0.0,
        fee_rate=0.0,
        spread_rate=0.0,
        impact_rate=0.0,
    )


def test_identity_action_matches_shadow_book_exactly() -> None:
    env = _env()
    env.reset(options={"start_idx": 72})

    for _ in range(6):
        _, reward, terminated, truncated, info = env.step(np.zeros(2))
        assert reward == 0.0
        np.testing.assert_allclose(env.hybrid.weights, env.shadow.weights, atol=1e-12)
        assert env.hybrid.portfolio_value == env.shadow.portfolio_value
        assert info["bars_advanced"] == 4
        if terminated or truncated:
            break


def test_one_action_returns_one_aggregated_reward() -> None:
    env = _env(decision_every=8)
    env.reset(options={"start_idx": 72})

    _, _, _, _, info = env.step(np.array([1.0, 0.0]))

    assert info["bars_advanced"] == 8
    assert env.t == 80
    assert info["decision_step_index"] == 1


def test_action_space_is_two_dimensional() -> None:
    env = _env()
    assert env.action_space.shape == (2,)
    obs, _ = env.reset(options={"start_idx": 72})
    assert obs.shape == env.observation_space.shape
    assert np.all(np.isfinite(obs))
