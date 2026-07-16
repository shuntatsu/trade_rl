from __future__ import annotations

from dataclasses import replace

import numpy as np

from trade_rl.data.market import MarketDataset


def market() -> MarketDataset:
    n_bars = 10
    shape = (n_bars, 2)
    tradable = np.ones(shape, dtype=np.bool_)
    tradable[4, 0] = False
    features = np.ones((n_bars, 2, 1), dtype=np.float32)
    feature_available = np.ones_like(features, dtype=np.bool_)
    feature_available[5, 1, 0] = False
    timestamps = np.datetime64("2026-01-01", "ns") + np.arange(n_bars) * np.timedelta64(
        1, "h"
    )
    return MarketDataset(
        dataset_id="a" * 64,
        symbols=("BTC", "ETH"),
        timestamps=timestamps,
        features=features,
        global_features=np.ones((n_bars, 1), dtype=np.float32),
        open=np.full(shape, 100.0),
        high=np.full(shape, 101.0),
        low=np.full(shape, 99.0),
        close=np.full(shape, 100.0),
        volume=np.full(shape, 1_000_000.0),
        funding_rate=np.zeros(shape),
        tradable=tradable,
        feature_available=feature_available,
        feature_names=("ret",),
        global_feature_names=("regime",),
        periods_per_year=8_760,
    )


def test_immediate_information_eligibility_uses_prefix_contract() -> None:
    dataset = market()

    assert dataset._information_is_immediate is True
    np.testing.assert_array_equal(
        dataset.eligibility_mask(6, lookback=3),
        np.array([False, True]),
    )
    np.testing.assert_array_equal(
        dataset.eligibility_mask(6, lookback=3, require_features=True),
        np.array([False, False]),
    )
    np.testing.assert_array_equal(
        dataset.eligibility_mask(9, lookback=2, require_features=True),
        np.array([True, True]),
    )


def test_delayed_information_preserves_decision_time_semantics() -> None:
    base = market()
    available_at = base.available_at.copy()
    available_at[4, 1] = base.timestamps[7]
    dataset = replace(
        base,
        available_at=available_at,
        information_available=None,
    )

    assert dataset._information_is_immediate is False
    np.testing.assert_array_equal(
        dataset.eligibility_mask(6, lookback=3),
        np.array([False, False]),
    )
    np.testing.assert_array_equal(
        dataset.eligibility_mask(7, lookback=3),
        np.array([False, True]),
    )
