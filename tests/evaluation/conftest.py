from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from trade_rl.data.market import MarketDataset


@pytest.fixture
def market_dataset_factory() -> Callable[..., MarketDataset]:
    def build(
        *,
        n_bars: int = 1_024,
        symbols: tuple[str, ...] = ("BTCUSDT", "ETHUSDT", "BNBUSDT"),
        seed_offset: float = 0.0,
    ) -> MarketDataset:
        n_symbols = len(symbols)
        timestamps = np.datetime64("2025-01-01T00:00:00", "ns") + np.arange(
            n_bars, dtype=np.int64
        ) * np.timedelta64(15, "m")
        t = np.arange(n_bars, dtype=np.float64)[:, None]
        asset = np.arange(n_symbols, dtype=np.float64)[None, :]
        close = 100.0 + seed_offset + 15.0 * asset + 0.03 * t
        close = close * (1.0 + 0.002 * np.sin(t / 17.0 + asset))
        open_price = close * (1.0 + 0.0005 * np.cos(t / 11.0 + asset))
        high = np.maximum(open_price, close) * 1.002
        low = np.minimum(open_price, close) * 0.998
        volume = 1_000.0 + 20.0 * asset + (t % 97.0)
        features = np.stack(
            (
                np.broadcast_to(np.sin(t / 13.0), (n_bars, n_symbols)),
                np.broadcast_to(np.cos(t / 29.0), (n_bars, n_symbols)),
            ),
            axis=2,
        ).astype(np.float32)
        global_features = np.column_stack(
            (np.sin(np.arange(n_bars) / 31.0), np.cos(np.arange(n_bars) / 43.0))
        ).astype(np.float32)
        funding_rate = np.broadcast_to(
            0.00001 * np.sin(t / 19.0 + asset), (n_bars, n_symbols)
        ).copy()
        tradable = np.ones((n_bars, n_symbols), dtype=np.bool_)
        feature_available = np.ones_like(features, dtype=np.bool_)
        spread_rate = np.broadcast_to(
            0.0002 + asset * 0.00001 + (t % 5.0) * 0.000001,
            (n_bars, n_symbols),
        ).copy()
        funding_due = np.zeros((n_bars, n_symbols), dtype=np.bool_)
        funding_due[::32] = True
        borrow_available = np.ones((n_bars, n_symbols), dtype=np.bool_)
        asset_active = np.ones((n_bars, n_symbols), dtype=np.bool_)
        buy_allowed = np.ones((n_bars, n_symbols), dtype=np.bool_)
        sell_allowed = np.ones((n_bars, n_symbols), dtype=np.bool_)
        delay_seconds = (
            np.arange(n_bars, dtype=np.int64)[:, None]
            + np.arange(n_symbols, dtype=np.int64)[None, :]
        ) % 4
        available_at = timestamps[:, None] + delay_seconds.astype("timedelta64[s]")
        return MarketDataset(
            dataset_id="a" * 64,
            symbols=symbols,
            timestamps=timestamps,
            features=features,
            global_features=global_features,
            open=open_price,
            high=high,
            low=low,
            close=close,
            volume=volume,
            funding_rate=funding_rate,
            tradable=tradable,
            feature_available=feature_available,
            feature_names=("feature_a", "feature_b"),
            global_feature_names=("global_a", "global_b"),
            periods_per_year=35_040,
            nominal_bar_hours=0.25,
            spread_rate=spread_rate,
            funding_due=funding_due,
            borrow_available=borrow_available,
            asset_active=asset_active,
            buy_allowed=buy_allowed,
            sell_allowed=sell_allowed,
            available_at=available_at,
        )

    return build
