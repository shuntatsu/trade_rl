from __future__ import annotations

import numpy as np
import pytest

from mars_lite.features.feature_pipeline import FeatureSet
from mars_lite.learning.baselines import simulate_strategy


def _feature_set(returns: np.ndarray) -> FeatureSet:
    close = np.concatenate([[1.0], np.cumprod(1.0 + returns)])[:, None]
    timestamps = np.datetime64("2026-01-01T00:00:00", "ns") + np.arange(
        len(close)
    ) * np.timedelta64(4, "h")
    return FeatureSet(
        symbols=["ONE"],
        timestamps=timestamps,
        features=np.zeros((len(close), 1, 1), dtype=np.float32),
        global_features=np.zeros((len(close), 1), dtype=np.float32),
        close=close,
        open_next=close.copy(),
        funding_rate=np.zeros_like(close),
        feature_names=["dummy"],
        global_feature_names=["global"],
    )


def _long_only(fs, t, weights):
    return np.ones(fs.n_symbols, dtype=np.float64)


@pytest.mark.parametrize("bars_per_year", [2_190, 365])
def test_baseline_sharpe_uses_effective_annualization(bars_per_year: int) -> None:
    returns = np.array([0.01, -0.005, 0.002, 0.004], dtype=np.float64)
    result = simulate_strategy(
        _feature_set(returns),
        _long_only,
        fee_rate=0.0,
        spread_rate=0.0,
        impact_rate=0.0,
        min_trade_delta=0.0,
        bars_per_year=bars_per_year,
    )
    expected = float(returns.mean() / returns.std() * np.sqrt(bars_per_year))

    assert result.sharpe == pytest.approx(expected)
