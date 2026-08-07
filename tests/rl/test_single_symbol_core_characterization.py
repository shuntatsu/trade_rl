from __future__ import annotations

import numpy as np

from trade_rl.data.market import MarketDataset
from trade_rl.rl.actions import ActionMode, ActionSpec, TargetWeightAction


def _single_symbol_dataset() -> MarketDataset:
    n_bars = 8
    timestamps = np.datetime64("2026-01-01T01:00:00", "ns") + np.arange(
        n_bars
    ) * np.timedelta64(1, "h")
    close = (100.0 + np.arange(n_bars, dtype=np.float64))[:, None]
    open_price = np.vstack([close[0], close[:-1]])
    return MarketDataset(
        dataset_id="a" * 64,
        symbols=("BTCUSDT",),
        timestamps=timestamps,
        features=np.zeros((n_bars, 1, 2), dtype=np.float32),
        global_features=np.zeros((n_bars, 1), dtype=np.float32),
        open=open_price,
        high=np.maximum(open_price, close) + 1.0,
        low=np.minimum(open_price, close) - 1.0,
        close=close,
        volume=np.full((n_bars, 1), 1_000.0),
        funding_rate=np.zeros_like(close),
        tradable=np.ones((n_bars, 1), dtype=np.bool_),
        feature_available=np.ones((n_bars, 1, 2), dtype=np.bool_),
        feature_names=("ret", "rsi"),
        global_feature_names=("regime",),
        periods_per_year=8_760,
    )


def test_market_dataset_accepts_exactly_one_symbol() -> None:
    dataset = _single_symbol_dataset()

    assert dataset.symbols == ("BTCUSDT",)
    assert dataset.n_symbols == 1
    assert dataset.features.shape == (8, 1, 2)
    assert dataset.close.shape == (8, 1)


def test_target_weight_action_supports_one_symbol() -> None:
    spec = ActionSpec(
        mode=ActionMode.TARGET_WEIGHT,
        alpha_enabled=False,
        risk_tilt_enabled=False,
        n_factors=0,
        target_weight_count=1,
    )

    assert spec.size == 1
    assert spec.names_for_symbols(("BTCUSDT",)) == ("target_weight:BTCUSDT",)
    parsed = spec.parse(np.array([0.75], dtype=np.float32))
    assert isinstance(parsed, TargetWeightAction)
    np.testing.assert_allclose(
        parsed.as_array(),
        np.array([0.75], dtype=np.float32),
    )
