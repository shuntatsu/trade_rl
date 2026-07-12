import numpy as np

from mars_lite.env.baseline_residual_env import BaselineResidualTradingEnv
from mars_lite.eval.relative_evaluation import evaluate_relative_agent
from mars_lite.features.feature_pipeline import FeatureSet
from mars_lite.trading.post_processor import make_legacy_processor
from mars_lite.trading.trend_family import TrendFamily, TrendFamilyConfig


class IdentityAgent:
    def predict(self, obs, deterministic=True):
        return np.zeros(2, dtype=np.float32), None


def _feature_set(n_bars: int = 180) -> FeatureSet:
    timestamps = np.datetime64("2026-01-01T00:00:00", "ns") + np.arange(
        n_bars
    ) * np.timedelta64(1, "h")
    base = np.exp(np.arange(n_bars, dtype=np.float64) * 0.001)
    close = np.column_stack([base, base[::-1]])
    return FeatureSet(
        symbols=["UP", "DOWN"],
        timestamps=timestamps,
        features=np.zeros((n_bars, 2, 1), dtype=np.float32),
        global_features=np.zeros((n_bars, 1), dtype=np.float32),
        close=close,
        open_next=close.copy(),
        funding_rate=np.zeros_like(close),
        feature_names=["dummy"],
        global_feature_names=["global"],
    )


def test_identity_agent_has_zero_excess_and_complete_report() -> None:
    fs = _feature_set()
    result = evaluate_relative_agent(
        IdentityAgent(),
        fs,
        env_kwargs={
            "trend_family": TrendFamily(
                TrendFamilyConfig(
                    fast_lookback=12,
                    base_lookback=24,
                    slow_lookback=48,
                    rebalance_every=12,
                )
            ),
            "decision_every": 4,
            "post_processor": make_legacy_processor(0.0),
            "min_trade_delta": 0.0,
            "fee_rate": 0.0,
            "spread_rate": 0.0,
            "impact_rate": 0.0,
        },
        start_idx=60,
    )

    assert result["hybrid"]["total_return"] == result["shadow"]["total_return"]
    assert result["paired"]["excess_log_return"] == 0.0
    assert result["paired"]["p_value"] == 1.0
    assert result["identity"]["action_schema"] == "baseline_residual_v1"
    assert result["execution"]["decision_every"] == 4
    assert result["actions"]["count"] > 0
