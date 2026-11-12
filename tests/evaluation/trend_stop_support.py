"""Synthetic multi-symbol bars for the trend-stop study tests (no market data)."""

from __future__ import annotations

import numpy as np

from trade_rl.data.market import MarketDataset

BAR_HOURS = {"4h": 4, "1d": 24}
PERIODS_PER_YEAR = {"4h": 2190, "1d": 365}
AUTHORIZATION_URL = "https://github.com/shuntatsu/trade_rl/issues/1#issuecomment-1"


def synthetic_dataset(
    timeframe: str,
    symbols: tuple[str, ...],
    *,
    first_close: str,
    last_close: str,
    seed: int = 7,
    drift: float = 0.0004,
    volatility: float = 0.02,
    dataset_id: str | None = None,
) -> MarketDataset:
    """Seeded geometric random walks; bar closes on a regular grid (inclusive)."""

    step = np.timedelta64(BAR_HOURS[timeframe], "h")
    start = np.datetime64(first_close, "ns")
    stop = np.datetime64(last_close, "ns")
    timestamps = np.arange(start, stop + step, step)
    n_rows, n_symbols = timestamps.size, len(symbols)
    rng = np.random.default_rng(seed)
    log_returns = rng.normal(drift, volatility, size=(n_rows, n_symbols))
    close = 100.0 * np.exp(np.cumsum(log_returns, axis=0))
    open_ = np.vstack([close[:1], close[:-1]]) * np.exp(
        rng.normal(0.0, volatility / 4.0, size=(n_rows, n_symbols))
    )
    spread = np.abs(rng.normal(0.0, volatility / 2.0, size=(2, n_rows, n_symbols)))
    high = np.maximum(open_, close) * (1.0 + spread[0])
    low = np.minimum(open_, close) * (1.0 - spread[1])
    features = np.zeros((n_rows, n_symbols, 1), dtype=np.float64)
    shape = (n_rows, n_symbols)
    digest = dataset_id or ("e" * 63 + timeframe[0])
    return MarketDataset(
        dataset_id=digest,
        symbols=symbols,
        timestamps=timestamps,
        features=features,
        global_features=np.zeros((n_rows, 1), dtype=np.float64),
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=np.full(shape, 1.0e12, dtype=np.float64),
        funding_rate=np.zeros(shape, dtype=np.float64),
        tradable=np.ones(shape, dtype=np.bool_),
        feature_available=np.ones_like(features, dtype=np.bool_),
        feature_names=("unused",),
        global_feature_names=("unused_global",),
        periods_per_year=PERIODS_PER_YEAR[timeframe],
    )
