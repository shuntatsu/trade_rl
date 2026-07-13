from __future__ import annotations

import numpy as np
import pytest

from trade_rl.data.market import MarketCalendarKind, MarketDataset


def kwargs(n_bars: int = 8, n_symbols: int = 2) -> dict[str, object]:
    timestamps = np.datetime64("2026-01-01T01:00:00", "ns") + np.arange(
        n_bars
    ) * np.timedelta64(1, "h")
    base = 100.0 + np.arange(n_bars, dtype=np.float64)
    close = np.column_stack([base + 10.0 * i for i in range(n_symbols)])
    open_price = np.vstack([close[0], close[:-1]])
    return {
        "dataset_id": "a" * 64,
        "symbols": tuple(f"S{i}" for i in range(n_symbols)),
        "timestamps": timestamps,
        "features": np.zeros((n_bars, n_symbols, 2), dtype=np.float32),
        "global_features": np.zeros((n_bars, 1), dtype=np.float32),
        "open": open_price,
        "high": np.maximum(open_price, close) + 1.0,
        "low": np.minimum(open_price, close) - 1.0,
        "close": close,
        "volume": np.full((n_bars, n_symbols), 1_000.0),
        "funding_rate": np.zeros_like(close),
        "tradable": np.ones((n_bars, n_symbols), dtype=np.bool_),
        "feature_available": np.ones((n_bars, n_symbols, 2), dtype=np.bool_),
        "feature_names": ("ret", "rsi"),
        "global_feature_names": ("regime",),
        "periods_per_year": 8_760,
    }


def test_continuous_market_requires_regular_cadence() -> None:
    values = kwargs()
    timestamps = np.asarray(values["timestamps"]).copy()
    timestamps[4:] += np.timedelta64(16, "h")
    values["timestamps"] = timestamps
    with pytest.raises(ValueError, match="regular"):
        MarketDataset(**values)


def test_session_calendar_accepts_gaps_and_uses_wall_clock_helpers() -> None:
    values = kwargs()
    timestamps = np.asarray(values["timestamps"]).copy()
    timestamps[4:] += np.timedelta64(16, "h")
    values.update(
        timestamps=timestamps,
        calendar_kind=MarketCalendarKind.SESSION,
        nominal_bar_hours=1.0,
        periods_per_year=1_638,
    )
    market = MarketDataset(**values)
    assert market.regular_cadence is False
    assert market.elapsed_hours(3, 4) == pytest.approx(17.0)
    assert market.lookback_index(4, 8.0) == 3
    assert market.forward_index(3, 8.0) == 4


def test_execution_and_missingness_metadata_are_immutable_and_shaped() -> None:
    values = kwargs()
    values["feature_available"] = np.zeros((8, 2, 2), dtype=np.bool_)
    values["feature_staleness_hours"] = np.full((8, 2, 2), 3.0)
    values["borrow_available"] = np.zeros((8, 2), dtype=np.bool_)
    values["mark_price"] = np.asarray(values["close"]) * 1.001
    values["index_price"] = np.asarray(values["close"])
    market = MarketDataset(**values)
    assert market.feature_staleness_hours.shape == (8, 2, 2)
    assert market.borrow_available.sum() == 0
    assert np.all(market.mark_price > market.index_price)
    with pytest.raises(ValueError):
        market.close[0, 0] = 0.0
