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
