from __future__ import annotations

import numpy as np

from trade_rl.data.market import MarketDataset
from trade_rl.rl.normalization import ObservationNormalizer
from trade_rl.rl.observations import (
    observation_market_matrix,
    observation_passthrough_indices,
)


def _dataset() -> MarketDataset:
    n = 5
    prices = np.full((n, 1), 100.0)
    return MarketDataset(
        dataset_id="a" * 64,
        symbols=("BTC",),
        timestamps=np.datetime64("2026-01-01", "ns")
        + np.arange(n) * np.timedelta64(1, "h"),
        features=np.arange(n, dtype=np.float32).reshape(n, 1, 1),
        global_features=(10 + np.arange(n, dtype=np.float32)).reshape(n, 1),
        open=prices,
        high=prices,
        low=prices,
        close=prices,
        volume=np.ones((n, 1)),
        funding_rate=np.zeros((n, 1)),
        tradable=np.ones((n, 1), dtype=np.bool_),
        feature_available=np.ones((n, 1, 1), dtype=np.bool_),
        feature_names=("ret",),
        global_feature_names=("regime",),
        periods_per_year=8_760,
        feature_staleness_hours=np.arange(n, dtype=np.float32).reshape(n, 1, 1),
        global_feature_staleness_hours=np.arange(n, dtype=np.float32).reshape(n, 1),
    )


def test_market_matrix_fits_only_exogenous_observation_fields() -> None:
    dataset = _dataset()
    matrix = observation_market_matrix(
        dataset, start=1, stop=5, action_size=3, n_factors=2
    )
    passthrough = observation_passthrough_indices(dataset, action_size=3, n_factors=2)
    normalizer = ObservationNormalizer.fit(
        matrix,
        train_start=0,
        train_end=matrix.shape[0],
        passthrough_indices=passthrough,
        dataset_id=dataset.dataset_id,
    )

    assert normalizer.dataset_id == dataset.dataset_id
    assert np.all(normalizer.mean[np.asarray(passthrough)] == 0.0)
    assert np.all(normalizer.scale[np.asarray(passthrough)] == 1.0)
    assert any(
        index not in passthrough and normalizer.scale[index] != 1.0
        for index in range(normalizer.size)
    )
