from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from trade_rl.data.builder import MarketDatasetBuilder
from trade_rl.data.contracts import (
    FeatureKind,
    FeatureSpec,
    InstrumentContract,
    MarketBuildConfig,
)
from trade_rl.data.source import InMemoryMarketDataSource, RawMarketSeries

_CUTOFF = 30
_N_BARS = 72


def _config() -> MarketBuildConfig:
    return MarketBuildConfig(
        base_timeframe="1h",
        features=(
            FeatureSpec(name="ret", kind=FeatureKind.LOG_RETURN),
            FeatureSpec(
                name="volume_z",
                kind=FeatureKind.VOLUME_ZSCORE,
                lookback=8,
                min_periods=4,
            ),
            FeatureSpec(
                name="funding_bps",
                kind=FeatureKind.FUNDING_BPS,
                max_staleness_hours=8.0,
            ),
        ),
    )


def _raw(
    *,
    scale: float,
    mutate_future: bool,
    remove_future_rows: bool = False,
) -> RawMarketSeries:
    timestamps = np.datetime64("2026-01-01T00:00:00", "ns") + np.arange(
        _N_BARS
    ) * np.timedelta64(1, "h")
    close = scale * np.exp(np.arange(_N_BARS, dtype=np.float64) * 0.002)
    open_price = np.concatenate([close[:1], close[:-1]])
    volume = 100.0 + np.arange(_N_BARS, dtype=np.float64)
    funding = np.where(np.arange(_N_BARS) % 8 == 0, 0.0001, 0.0)
    tradable = np.ones(_N_BARS, dtype=np.bool_)
    available_at = timestamps.copy()

    if mutate_future:
        volume[_CUTOFF + 1 :] *= 100.0
        funding[_CUTOFF + 1 :] += 0.01
        tradable[_CUTOFF + 1 :: 3] = False
        available_at[_CUTOFF + 1 :] += np.timedelta64(6, "h")

    keep = np.ones(_N_BARS, dtype=np.bool_)
    if remove_future_rows:
        keep[[_CUTOFF + 5, _CUTOFF + 11, _CUTOFF + 19]] = False

    return RawMarketSeries(
        timestamps=timestamps[keep],
        available_at=available_at[keep],
        open=open_price[keep],
        high=np.maximum(open_price, close)[keep] * 1.001,
        low=np.minimum(open_price, close)[keep] * 0.999,
        close=close[keep],
        volume=volume[keep],
        funding_rate=funding[keep],
        tradable=tradable[keep],
    )


def _contracts(*, mutate_future: bool) -> tuple[InstrumentContract, ...]:
    future_delisting = (
        datetime(2026, 1, 3, 6, tzinfo=timezone.utc) if mutate_future else None
    )
    return (
        InstrumentContract(
            symbol="A",
            listed_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        ),
        InstrumentContract(
            symbol="B",
            listed_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            delisted_at=future_delisting,
        ),
    )


def _dataset(*, mutate_future: bool, remove_future_rows: bool = False):
    source = InMemoryMarketDataSource(
        {
            "A": _raw(
                scale=1.0,
                mutate_future=mutate_future,
                remove_future_rows=remove_future_rows,
            ),
            "B": _raw(
                scale=2.0,
                mutate_future=mutate_future,
                remove_future_rows=remove_future_rows,
            ),
        }
    )
    return MarketDatasetBuilder(_config()).build(
        source,
        _contracts(mutate_future=mutate_future),
    )


def _assert_equal_prefix(left, right) -> None:
    stop = _CUTOFF + 1
    for field_name in (
        "timestamps",
        "available_at",
        "features",
        "global_features",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "funding_rate",
        "tradable",
        "symbol_active",
        "information_available",
        "feature_available",
        "feature_staleness",
    ):
        left_value = getattr(left, field_name)
        right_value = getattr(right, field_name)
        assert left_value is not None
        assert right_value is not None
        np.testing.assert_array_equal(left_value[:stop], right_value[:stop])


def test_future_volume_funding_availability_and_universe_do_not_change_prefix() -> None:
    baseline = _dataset(mutate_future=False)
    mutated = _dataset(mutate_future=True)

    _assert_equal_prefix(baseline, mutated)
    assert baseline.dataset_id != mutated.dataset_id


def test_future_missing_rows_do_not_change_resolved_prefix() -> None:
    baseline = _dataset(mutate_future=False)
    missing = _dataset(mutate_future=True, remove_future_rows=True)

    _assert_equal_prefix(baseline, missing)
    assert baseline.dataset_id != missing.dataset_id
