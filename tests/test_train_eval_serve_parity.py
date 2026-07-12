from types import SimpleNamespace

import numpy as np

from mars_lite.env.baseline_residual_env import BaselineResidualTradingEnv
from mars_lite.features.feature_pipeline import FeatureSet
from mars_lite.serving.contracts import InferenceState
from mars_lite.serving.residual_serving import build_residual_serving_adapters
from mars_lite.serving.runtime import FeatureSnapshot
from mars_lite.trading.pipeline import DecisionPipeline
from mars_lite.trading.post_processor import make_legacy_processor
from mars_lite.trading.residual_alpha import FrozenResidualAlpha
from mars_lite.trading.trend_family import TrendFamily, TrendFamilyConfig


def _feature_set(n_bars: int = 160) -> FeatureSet:
    timestamps = np.datetime64("2026-01-01T00:00:00", "ns") + np.arange(
        n_bars
    ) * np.timedelta64(1, "h")
    base = np.exp(np.arange(n_bars, dtype=np.float64) * 0.001)
    close = np.column_stack([base, base[::-1]])
    features = np.zeros((n_bars, 2, 1), dtype=np.float32)
    return FeatureSet(
        symbols=["UP", "DOWN"],
        timestamps=timestamps,
        features=features,
        global_features=np.zeros((n_bars, 1), dtype=np.float32),
        close=close,
        open_next=close.copy(),
        funding_rate=np.zeros_like(close),
        feature_names=["dummy"],
        global_feature_names=["global"],
    )


def test_train_and_serve_generate_identical_target(tmp_path) -> None:
    fs = _feature_set()
    t = 96
    action = np.array([0.5, 0.0], dtype=np.float32)
    trend_config = TrendFamilyConfig(
        fast_lookback=12,
        base_lookback=24,
        slow_lookback=48,
        rebalance_every=12,
    )
    trend_family = TrendFamily(trend_config)
    alpha = FrozenResidualAlpha.fit(
        fs.slice(0, 80),
        horizon=4,
        model="ridge",
        gate_result={"passed": False},
    )
    alpha_path = alpha.save(tmp_path / "residual_alpha.json")
    post_processor = make_legacy_processor(0.0)

    env = BaselineResidualTradingEnv(
        fs,
        trend_family=trend_family,
        alpha_provider=alpha,
        alpha_enabled=False,
        decision_every=4,
        episode_bars=8,
        post_processor=post_processor,
        min_trade_delta=0.0,
        fee_rate=0.0,
        spread_rate=0.0,
        impact_rate=0.0,
    )
    env.reset(options={"start_idx": t})
    env.step(action)
    training_target = env.hybrid.weights.copy()

    bundle = SimpleNamespace(
        root=tmp_path,
        metadata={
            "symbols": fs.symbols,
            "run_config": {"base_timeframe": "1h"},
            "trend_family": {
                "fast_lookback": trend_config.fast_lookback,
                "base_lookback": trend_config.base_lookback,
                "slow_lookback": trend_config.slow_lookback,
                "rebalance_every": trend_config.rebalance_every,
                "momentum_scale": trend_config.momentum_scale,
            },
            "composer": {"alpha_budget_max": 0.30, "max_gross": 1.0},
            "residual_alpha_file": alpha_path.name,
        },
        preprocessing={"feature_names": fs.feature_names},
    )
    pipeline = DecisionPipeline(
        post_processor=make_legacy_processor(0.0), min_trade_delta=0.0
    )
    augment, decide = build_residual_serving_adapters(bundle, pipeline)
    snapshot = FeatureSnapshot(
        snapshot_id="snapshot",
        symbols=tuple(fs.symbols),
        feature_names=tuple(fs.feature_names),
        global_feature_names=tuple(fs.global_feature_names),
        features=fs.features[: t + 1],
        global_features=fs.global_features[: t + 1],
        close_history=fs.close[: t + 1],
        data_age_hours=0.0,
        timestamps=fs.timestamps[: t + 1],
    )
    _, context = augment(snapshot, fs.features[t])
    state = InferenceState(
        current_weights={symbol: 0.0 for symbol in fs.symbols},
        portfolio_value=1.0,
        day_start_value=1.0,
        peak_value=1.0,
        consecutive_losses=0,
        turnover_mean=0.0,
        turnover_std=0.0,
    )
    serving_target, _ = decide(action, state, None, None, context)

    np.testing.assert_allclose(serving_target, training_target, atol=1e-12)
