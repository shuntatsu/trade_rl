from __future__ import annotations

import numpy as np
import pytest

from trade_rl.data.market import MarketDataset
from trade_rl.strategies.forecasts.supervised import (
    validated_feature_indices,
    validated_symbol_indices,
)


def _market() -> MarketDataset:
    n_bars = 3
    n_symbols = 2
    close = np.full((n_bars, n_symbols), 100.0, dtype=np.float64)
    return MarketDataset(
        dataset_id="a" * 64,
        symbols=("BTCUSDT", "ETHUSDT"),
        timestamps=np.datetime64("2026-01-01T00:00:00", "ns")
        + np.arange(n_bars) * np.timedelta64(1, "h"),
        features=np.zeros((n_bars, n_symbols, 2), dtype=np.float32),
        global_features=np.zeros((n_bars, 1), dtype=np.float32),
        open=close.copy(),
        high=close.copy(),
        low=close.copy(),
        close=close,
        volume=np.full((n_bars, n_symbols), 1_000_000.0),
        funding_rate=np.zeros((n_bars, n_symbols), dtype=np.float64),
        tradable=np.ones((n_bars, n_symbols), dtype=np.bool_),
        feature_available=np.ones((n_bars, n_symbols, 2), dtype=np.bool_),
        feature_names=("linear", "quadratic"),
        global_feature_names=("regime",),
        periods_per_year=8_760,
    )


def test_dataset_bound_feature_indices_preserve_order() -> None:
    assert validated_feature_indices(_market(), (1, 0)) == (1, 0)


@pytest.mark.parametrize(
    "indices, message",
    (
        ((), "non-empty and unique"),
        ((0, 0), "non-empty and unique"),
        ((-1,), "non-negative integers"),
        ((True,), "non-negative integers"),
        ((2,), "outside dataset features"),
    ),
)
def test_dataset_bound_feature_indices_fail_closed(
    indices: tuple[int, ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        validated_feature_indices(_market(), indices)


def test_none_symbol_scope_resolves_all_dataset_symbols() -> None:
    assert validated_symbol_indices(_market(), None) == (0, 1)


def test_explicit_symbol_scope_preserves_order() -> None:
    assert validated_symbol_indices(_market(), (1, 0)) == (1, 0)


@pytest.mark.parametrize(
    "indices, message",
    (
        ((), "non-empty and unique"),
        ((0, 0), "non-empty and unique"),
        ((-1,), "non-negative integers"),
        ((True,), "non-negative integers"),
        ((2,), "outside dataset symbols"),
    ),
)
def test_explicit_symbol_scope_fails_closed(
    indices: tuple[int, ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        validated_symbol_indices(_market(), indices)
