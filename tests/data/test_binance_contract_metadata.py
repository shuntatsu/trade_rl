from __future__ import annotations

from datetime import UTC, datetime

import numpy as np

from trade_rl.data.builder import MarketDatasetBuilder
from trade_rl.data.contracts import (
    FeatureKind,
    FeatureSpec,
    InstrumentContract,
    MarketBuildConfig,
    VolumeUnit,
)
from trade_rl.data.source import InMemoryMarketDataSource, RawMarketSeries


def test_builder_broadcasts_static_execution_metadata() -> None:
    timestamps = np.arange(
        np.datetime64("2026-06-01T01:00:00", "ns"),
        np.datetime64("2026-06-01T05:00:00", "ns"),
        np.timedelta64(1, "h"),
    )
    source = InMemoryMarketDataSource(
        {
            "BTCUSDT": RawMarketSeries(
                timestamps=timestamps,
                open=np.array([100.0, 101.0, 102.0, 103.0]),
                high=np.array([101.0, 102.0, 103.0, 104.0]),
                low=np.array([99.0, 100.0, 101.0, 102.0]),
                close=np.array([100.5, 101.5, 102.5, 103.5]),
                volume=np.array([1_000_000.0] * 4),
                funding_rate=np.zeros(4),
                funding_available=np.zeros(4, dtype=np.bool_),
                tradable=np.ones(4, dtype=np.bool_),
            )
        }
    )
    config = MarketBuildConfig(
        base_timeframe="1h",
        features=(
            FeatureSpec(
                name="return_1h",
                kind=FeatureKind.LOG_RETURN,
                lookback=1,
                max_staleness_hours=2.0,
            ),
        ),
    )
    contract = InstrumentContract(
        symbol="BTCUSDT",
        listed_at=datetime(2026, 6, 1, tzinfo=UTC),
        volume_unit=VolumeUnit.QUOTE_NOTIONAL,
        tick_size=0.1,
        lot_size=0.001,
        minimum_notional=5.0,
    )

    dataset = MarketDatasetBuilder(config).build(source, (contract,))

    np.testing.assert_allclose(dataset.resolved_array("tick_size"), 0.1)
    np.testing.assert_allclose(dataset.resolved_array("lot_size"), 0.001)
    np.testing.assert_allclose(dataset.resolved_array("minimum_notional"), 5.0)
    assert dataset.volume_units == (VolumeUnit.QUOTE_NOTIONAL,)


def test_execution_metadata_participates_in_instrument_identity() -> None:
    base = InstrumentContract(symbol="BTCUSDT")
    changed = InstrumentContract(symbol="BTCUSDT", tick_size=0.1)

    assert base.canonical_payload() != changed.canonical_payload()
