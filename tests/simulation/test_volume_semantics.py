from __future__ import annotations

import numpy as np

from trade_rl.data.contracts import VolumeUnit
from trade_rl.data.market import MarketDataset


def test_market_notional_respects_volume_units_and_contract_multiplier() -> None:
    n_bars = 4
    prices = np.tile(np.array([10.0, 20.0, 30.0]), (n_bars, 1))
    dataset = MarketDataset(
        dataset_id="b" * 64,
        symbols=("BASE", "QUOTE", "CONTRACT"),
        timestamps=np.datetime64("2026-01-01T00:00:00", "ns")
        + np.arange(n_bars) * np.timedelta64(1, "h"),
        features=np.zeros((n_bars, 3, 1), dtype=np.float32),
        global_features=np.zeros((n_bars, 1), dtype=np.float32),
        open=prices,
        high=prices,
        low=prices,
        close=prices,
        volume=np.tile(np.array([2.0, 1_000.0, 5.0]), (n_bars, 1)),
        funding_rate=np.zeros((n_bars, 3)),
        tradable=np.ones((n_bars, 3), dtype=np.bool_),
        feature_available=np.ones((n_bars, 3, 1), dtype=np.bool_),
        feature_names=("dummy",),
        global_feature_names=("dummy_global",),
        periods_per_year=8_760,
        volume_units=(
            VolumeUnit.BASE_ASSET,
            VolumeUnit.QUOTE_NOTIONAL,
            VolumeUnit.CONTRACTS,
        ),
        contract_multipliers=np.array([1.0, 1.0, 0.1]),
    )

    np.testing.assert_allclose(dataset.market_notional(1), [20.0, 1_000.0, 15.0])
