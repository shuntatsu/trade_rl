from __future__ import annotations

import numpy as np

from trade_rl.data.build import MarketDatasetBuilder
from trade_rl.data.contracts import FeatureKind, FeatureSpec, InstrumentContract, MarketBuildConfig
from trade_rl.data.source import InMemoryMarketDataSource, RawMarketSeries


def _series(*, taker: np.ndarray | None) -> RawMarketSeries:
    n = 30
    timestamps = np.datetime64("2022-01-01T00:00:00", "ns") + np.arange(n) * np.timedelta64(1, "h")
    close = 100.0 + np.arange(n, dtype=np.float64)
    return RawMarketSeries(
        timestamps=timestamps,
        available_at=timestamps,
        open=np.concatenate((close[:1], close[:-1])),
        high=close + 1.0,
        low=close - 1.0,
        close=close,
        volume=np.full(n, 10.0, dtype=np.float64),
        taker_buy_quote_volume=taker,
        funding_rate=np.zeros(n, dtype=np.float64),
        tradable=np.ones(n, dtype=np.bool_),
    )


def test_optional_taker_source_does_not_change_dataset_when_feature_is_omitted() -> None:
    config = MarketBuildConfig(
        base_timeframe="1h",
        features=(FeatureSpec(name="return_1bar", kind=FeatureKind.LOG_RETURN, lookback=1),),
    )
    builder = MarketDatasetBuilder(config)
    instruments = (InstrumentContract(symbol="BTCUSDT"),)

    legacy = builder.build(
        InMemoryMarketDataSource({"BTCUSDT": _series(taker=None)}),
        instruments,
    )
    enriched = builder.build(
        InMemoryMarketDataSource(
            {"BTCUSDT": _series(taker=np.full(30, 6.0, dtype=np.float64))}
        ),
        instruments,
    )

    assert enriched.dataset_id == legacy.dataset_id
    assert enriched.feature_config_digest == legacy.feature_config_digest
    assert enriched.normalization_digest == legacy.normalization_digest
    np.testing.assert_array_equal(enriched.features, legacy.features)
    np.testing.assert_array_equal(enriched.feature_available, legacy.feature_available)
    assert enriched.identity_arrays().keys() == legacy.identity_arrays().keys()
    for name in legacy.identity_arrays():
        np.testing.assert_array_equal(
            enriched.identity_arrays()[name], legacy.identity_arrays()[name]
        )
