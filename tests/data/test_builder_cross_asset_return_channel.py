"""Cross-asset builder features use a private unnormalized one-bar return.

Regression for an audit witness: the builder used to pick any public
``log_return`` lookback-1 spec as the cross-asset return channel, including a
rolling z-scored one, so relative return, correlation, beta, momentum rank and
dispersion were computed from z-scores instead of returns.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

from trade_rl.data.build.builder import MarketDatasetBuilder
from trade_rl.data.contracts import (
    FeatureKind,
    FeatureSpec,
    InstrumentContract,
    MarketBuildConfig,
    NormalizationMode,
)
from trade_rl.data.market import MarketDataset
from trade_rl.data.source import InMemoryMarketDataSource, RawMarketSeries

_T0 = datetime(2024, 1, 1, tzinfo=UTC)
_N = 80
_SYMBOLS = ("BTC", "ETH", "SOL")


def _raw(seed: int) -> RawMarketSeries:
    rng = np.random.default_rng(seed)
    timestamps = np.datetime64(_T0.replace(tzinfo=None), "ns") + np.arange(
        _N
    ) * np.timedelta64(1, "h")
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, _N)))
    open_price = close * np.exp(rng.normal(0.0, 0.002, _N))
    return RawMarketSeries(
        timestamps=timestamps,
        available_at=timestamps.copy(),
        open=open_price,
        high=np.maximum(open_price, close) * 1.003,
        low=np.minimum(open_price, close) * 0.997,
        close=close,
        volume=rng.lognormal(5.0, 0.5, _N),
        funding_rate=np.zeros(_N),
        tradable=np.ones(_N, dtype=np.bool_),
    )


def _cross_specs() -> tuple[FeatureSpec, ...]:
    return (
        FeatureSpec("rel", FeatureKind.RELATIVE_RETURN_TO_BTC),
        FeatureSpec(
            "corr", FeatureKind.ROLLING_CORRELATION_TO_BTC, lookback=12, min_periods=6
        ),
        FeatureSpec(
            "beta", FeatureKind.ROLLING_BETA_TO_BTC, lookback=12, min_periods=6
        ),
        FeatureSpec(
            "rank",
            FeatureKind.CROSS_SECTIONAL_MOMENTUM_RANK,
            lookback=6,
            min_periods=3,
        ),
        FeatureSpec("disp", FeatureKind.CROSS_ASSET_DISPERSION),
    )


def _build(public: tuple[FeatureSpec, ...]) -> MarketDataset:
    config = MarketBuildConfig(
        base_timeframe="1h",
        features=(*public, *_cross_specs()),
        cross_asset_reference_symbol="BTC",
    )
    source = InMemoryMarketDataSource(
        {symbol: _raw(seed) for seed, symbol in enumerate(_SYMBOLS, start=1)}
    )
    contracts = tuple(InstrumentContract(symbol, listed_at=_T0) for symbol in _SYMBOLS)
    return MarketDatasetBuilder(config).build(source, contracts)


def _cross(dataset: MarketDataset) -> tuple[np.ndarray, np.ndarray]:
    names = [spec.name for spec in _cross_specs()]
    columns = [dataset.feature_names.index(name) for name in names]
    assert dataset.feature_available is not None
    return dataset.features[..., columns], dataset.feature_available[..., columns]


def _true_returns(dataset: MarketDataset) -> np.ndarray:
    returns = np.zeros_like(dataset.close)
    returns[1:] = np.log(dataset.close[1:] / dataset.close[:-1])
    return returns


def test_relative_return_and_dispersion_use_true_returns_under_zscore_spec() -> None:
    zscored = _build(
        (
            FeatureSpec(
                "ret1",
                FeatureKind.LOG_RETURN,
                normalization=NormalizationMode.ROLLING_ZSCORE,
                normalization_window=20,
                min_periods=5,
            ),
        )
    )
    returns = _true_returns(zscored)
    rel = zscored.feature_names.index("rel")
    disp = zscored.feature_names.index("disp")
    for row in (10, 30, 50, 79):
        for symbol in range(len(_SYMBOLS)):
            assert zscored.features[row, symbol, rel] == pytest.approx(
                returns[row, symbol] - returns[row, 0], abs=1e-6
            )
            assert zscored.features[row, symbol, disp] == pytest.approx(
                float(np.std(returns[row])), abs=1e-6
            )


@pytest.mark.parametrize(
    "public",
    [
        (),
        (
            FeatureSpec(
                "ret1",
                FeatureKind.LOG_RETURN,
                normalization=NormalizationMode.ROLLING_ZSCORE,
                normalization_window=20,
                min_periods=5,
            ),
        ),
        (
            FeatureSpec("ret_a", FeatureKind.LOG_RETURN, max_staleness_hours=2.0),
            FeatureSpec("ret_b", FeatureKind.LOG_RETURN),
        ),
    ],
    ids=["no_public_return", "zscored_public_return", "two_public_returns"],
)
def test_cross_asset_columns_are_independent_of_public_return_specs(
    public: tuple[FeatureSpec, ...],
) -> None:
    reference = _build((FeatureSpec("ret1", FeatureKind.LOG_RETURN),))
    other = _build(public)

    reference_values, reference_available = _cross(reference)
    other_values, other_available = _cross(other)
    np.testing.assert_array_equal(other_available, reference_available)
    np.testing.assert_array_equal(other_values, reference_values)
    assert other.feature_names == tuple(spec.name for spec in public) + tuple(
        spec.name for spec in _cross_specs()
    )


@pytest.mark.parametrize("native", [False, True])
def test_cross_asset_consumes_new_close_even_when_age_is_unchanged(
    native: bool,
) -> None:
    import math

    from tests.data.test_multitimeframe_builder import (
        MemoryMultiTimeframeSource,
        _series,
    )

    # Both delayed ETH selections at consecutive decisions have the same age,
    # but are distinct close events. BTC is on time. The statistic consumes
    # simultaneous newly selected events at the decision, not an age proxy.
    base_times = [f"2026-01-01T{i:02d}:00:00" for i in range(6)]
    btc = _series(base_times, [100.0, 101.0, 103.0, 107.0, 111.0, 113.0])
    if native:
        times = [f"2026-01-01T{i // 4:02d}:{(i % 4) * 15:02d}:00" for i in range(21)]
        delayed = [str(np.datetime64(t, "ns") + np.timedelta64(30, "m")) for t in times]
        btc_native = _series(times, [100.0 + i for i in range(21)])
        eth_close = [100.0 + i * i for i in range(21)]
        eth_native = _series(times, eth_close, available_at=delayed)
        source = MemoryMultiTimeframeSource(
            {
                ("BTC", "1h"): btc,
                ("ETH", "1h"): btc,
                ("BTC", "15m"): btc_native,
                ("ETH", "15m"): eth_native,
            }
        )
        expected = math.log(eth_close[6] / eth_close[5]) - math.log(108.0 / 107.0)
        row = 2
        expected_age = 0.5
        timeframe = "15m"
    else:
        eth = _series(
            base_times,
            [100.0, 101.0, 104.0, 109.0, 116.0, 125.0],
            available_at=[
                base_times[0],
                base_times[1],
                *base_times[3:],
                "2026-01-01T06:00:00",
            ],
        )
        source = MemoryMultiTimeframeSource({("BTC", "1h"): btc, ("ETH", "1h"): eth})
        expected = math.log(104.0 / 101.0) - math.log(107.0 / 103.0)
        row = 3
        expected_age = 1.0
        timeframe = None

    config = MarketBuildConfig(
        base_timeframe="1h",
        features=(
            FeatureSpec(
                "relative",
                FeatureKind.RELATIVE_RETURN_TO_BTC,
                timeframe=timeframe,
                max_staleness_hours=4.0,
            ),
        ),
        cross_asset_reference_symbol="BTC",
    )
    dataset = MarketDatasetBuilder(config).build(
        source,
        tuple(InstrumentContract(symbol, listed_at=_T0) for symbol in ("BTC", "ETH")),
    )
    assert dataset.feature_available[row, 1, 0]
    assert dataset.features[row, 1, 0] == pytest.approx(expected, abs=1e-8)
    assert dataset.feature_staleness_hours[row, 1, 0] == expected_age
