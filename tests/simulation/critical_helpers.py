from __future__ import annotations

import numpy as np

from trade_rl.data.market import MarketDataset


def market(n_symbols: int = 1, **overrides: object) -> MarketDataset:
    n_bars = 5
    shape = (n_bars, n_symbols)
    close = np.full(shape, 100.0)
    values: dict[str, object] = {
        "dataset_id": "b" * 64,
        "symbols": tuple(f"S{index}" for index in range(n_symbols)),
        "timestamps": np.datetime64("2026-01-01", "ns")
        + np.arange(n_bars) * np.timedelta64(1, "h"),
        "features": np.zeros((n_bars, n_symbols, 1), dtype=np.float32),
        "global_features": np.zeros((n_bars, 1), dtype=np.float32),
        "open": close.copy(),
        "high": close + 1.0,
        "low": close - 1.0,
        "close": close,
        "volume": np.full(shape, 1_000_000.0),
        "funding_rate": np.zeros(shape),
        "tradable": np.ones(shape, dtype=np.bool_),
        "feature_available": np.ones((n_bars, n_symbols, 1), dtype=np.bool_),
        "feature_names": ("ret",),
        "global_feature_names": ("regime",),
        "periods_per_year": 8_760,
    }
    values.update(overrides)
    return MarketDataset(**values)
