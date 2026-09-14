from __future__ import annotations

import numpy as np
import pytest

from trade_rl.data.build import MarketDatasetBuilder
from trade_rl.data.contracts import (
    FeatureKind,
    FeatureSpec,
    InstrumentContract,
    MarketBuildConfig,
    VolumeUnit,
)
from trade_rl.data.source import InMemoryMarketDataSource, RawMarketSeries


def _series(
    *, taker: np.ndarray | None, tradable: np.ndarray | None = None
) -> RawMarketSeries:
    n = 30
    timestamps = np.datetime64("2022-01-01T00:00:00", "ns") + np.arange(
        n
    ) * np.timedelta64(1, "h")
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
        tradable=(np.ones(n, dtype=np.bool_) if tradable is None else tradable),
    )


def test_optional_taker_source_does_not_change_dataset_when_feature_is_omitted() -> (
    None
):
    config = MarketBuildConfig(
        base_timeframe="1h",
        features=(
            FeatureSpec(name="return_1bar", kind=FeatureKind.LOG_RETURN, lookback=1),
        ),
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


def _signed_flow_builder() -> MarketDatasetBuilder:
    return MarketDatasetBuilder(
        MarketBuildConfig(
            base_timeframe="1h",
            features=(
                FeatureSpec(
                    name="1h__signed_taker_quote_flow_24bar",
                    kind=FeatureKind.SIGNED_TAKER_QUOTE_FLOW,
                    lookback=24,
                ),
            ),
        )
    )


def test_builder_materializes_signed_taker_flow_only_when_raw_field_exists() -> None:
    builder = _signed_flow_builder()
    instruments = (
        InstrumentContract(
            symbol="BTCUSDT",
            volume_unit=VolumeUnit.QUOTE_NOTIONAL,
        ),
    )

    enriched = builder.build(
        InMemoryMarketDataSource(
            {"BTCUSDT": _series(taker=np.full(30, 6.0, dtype=np.float64))}
        ),
        instruments,
    )
    missing = builder.build(
        InMemoryMarketDataSource({"BTCUSDT": _series(taker=None)}),
        instruments,
    )

    assert enriched.feature_available[23, 0, 0]
    assert enriched.features[23, 0, 0] == np.float32(0.2)
    assert not np.any(missing.feature_available[:, 0, 0])
    assert not np.any(missing.features[:, 0, 0])


def test_builder_does_not_carry_signed_flow_across_invalid_current_window() -> None:
    tradable = np.ones(30, dtype=np.bool_)
    tradable[24] = False
    dataset = _signed_flow_builder().build(
        InMemoryMarketDataSource(
            {
                "BTCUSDT": _series(
                    taker=np.full(30, 6.0, dtype=np.float64),
                    tradable=tradable,
                )
            }
        ),
        (
            InstrumentContract(
                symbol="BTCUSDT",
                volume_unit=VolumeUnit.QUOTE_NOTIONAL,
            ),
        ),
    )

    assert dataset.feature_available[23, 0, 0]
    assert not dataset.feature_available[24, 0, 0]
    assert dataset.features[24, 0, 0] == 0.0


def test_builder_rejects_non_quote_volume_semantics_for_signed_flow() -> None:
    with pytest.raises(ValueError, match="quote.*notional"):
        _signed_flow_builder().build(
            InMemoryMarketDataSource(
                {"BTCUSDT": _series(taker=np.full(30, 6.0, dtype=np.float64))}
            ),
            (InstrumentContract(symbol="BTCUSDT"),),
        )


def test_signed_flow_market_config_rejects_non_1h_base_clock() -> None:
    with pytest.raises(ValueError, match="1h base decision clock"):
        MarketBuildConfig(
            base_timeframe="15m",
            features=(
                FeatureSpec(
                    name="1h__signed_taker_quote_flow_24bar",
                    kind=FeatureKind.SIGNED_TAKER_QUOTE_FLOW,
                    lookback=24,
                    timeframe="1h",
                ),
            ),
        )


def test_builder_signed_flow_respects_delayed_row_at_each_decision_time() -> None:
    n = 30
    timestamps = np.datetime64("2022-01-01T00:00:00", "ns") + np.arange(
        n
    ) * np.timedelta64(1, "h")
    close = 100.0 + np.arange(n, dtype=np.float64)
    available_at = timestamps.copy()
    available_at[5] = timestamps[24]
    raw = RawMarketSeries(
        timestamps=timestamps,
        available_at=available_at,
        open=np.concatenate((close[:1], close[:-1])),
        high=close + 1.0,
        low=close - 1.0,
        close=close,
        volume=np.full(n, 10.0, dtype=np.float64),
        taker_buy_quote_volume=np.full(n, 6.0, dtype=np.float64),
        funding_rate=np.zeros(n, dtype=np.float64),
        tradable=np.ones(n, dtype=np.bool_),
    )
    dataset = _signed_flow_builder().build(
        InMemoryMarketDataSource({"BTCUSDT": raw}),
        (
            InstrumentContract(
                symbol="BTCUSDT",
                volume_unit=VolumeUnit.QUOTE_NOTIONAL,
            ),
        ),
    )

    assert not dataset.feature_available[23, 0, 0]
    assert dataset.features[23, 0, 0] == 0.0
    assert dataset.feature_available[24, 0, 0]
    assert dataset.features[24, 0, 0] == np.float32(0.2)
