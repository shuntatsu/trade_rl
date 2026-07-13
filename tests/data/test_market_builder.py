from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from trade_rl.data.builder import MarketDatasetBuilder
from trade_rl.data.contracts import (
    FeatureKind,
    FeatureSpec,
    InstrumentContract,
    MarketBuildConfig,
    NormalizationMode,
    VolumeUnit,
)
from trade_rl.data.source import (
    CsvMarketDataSource,
    InMemoryMarketDataSource,
    RawMarketSeries,
)


def raw_series(n_bars: int, *, scale: float = 1.0) -> RawMarketSeries:
    timestamps = np.datetime64("2026-01-01T00:00:00", "ns") + np.arange(
        n_bars
    ) * np.timedelta64(1, "h")
    close = scale * np.exp(np.arange(n_bars, dtype=np.float64) * 0.002)
    open_price = np.concatenate([close[:1], close[:-1]])
    return RawMarketSeries(
        timestamps=timestamps,
        open=open_price,
        high=np.maximum(open_price, close) * 1.001,
        low=np.minimum(open_price, close) * 0.999,
        close=close,
        volume=100.0 + np.arange(n_bars, dtype=np.float64),
        funding_rate=np.where(np.arange(n_bars) % 8 == 0, 0.0001, 0.0),
        tradable=np.ones(n_bars, dtype=np.bool_),
    )


def config(*, normalization_window: int = 24) -> MarketBuildConfig:
    return MarketBuildConfig(
        base_timeframe="1h",
        features=(
            FeatureSpec(
                name="ret_1",
                kind=FeatureKind.LOG_RETURN,
                lookback=1,
                normalization=NormalizationMode.NONE,
            ),
            FeatureSpec(
                name="ret_4_z",
                kind=FeatureKind.LOG_RETURN,
                lookback=4,
                normalization=NormalizationMode.ROLLING_ZSCORE,
                normalization_window=normalization_window,
                min_periods=8,
            ),
            FeatureSpec(
                name="vol_12",
                kind=FeatureKind.REALIZED_VOLATILITY,
                lookback=12,
            ),
            FeatureSpec(
                name="volume_z",
                kind=FeatureKind.VOLUME_ZSCORE,
                lookback=12,
                min_periods=6,
            ),
            FeatureSpec(
                name="funding_bps",
                kind=FeatureKind.FUNDING_BPS,
                max_staleness_hours=8.0,
            ),
        ),
    )


def instruments() -> tuple[InstrumentContract, ...]:
    return (
        InstrumentContract(
            symbol="BTCUSDT",
            listed_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            volume_unit=VolumeUnit.BASE_ASSET,
        ),
        InstrumentContract(
            symbol="ETHUSDT",
            listed_at=datetime(2026, 1, 1, 10, tzinfo=timezone.utc),
            delisted_at=datetime(2026, 1, 3, 2, tzinfo=timezone.utc),
            volume_unit=VolumeUnit.CONTRACTS,
            contract_multiplier=0.01,
        ),
    )


def test_builder_is_prefix_invariant() -> None:
    builder = MarketDatasetBuilder(config())
    prefix_source = InMemoryMarketDataSource(
        {"BTCUSDT": raw_series(60), "ETHUSDT": raw_series(60, scale=2.0)}
    )
    full_source = InMemoryMarketDataSource(
        {"BTCUSDT": raw_series(80), "ETHUSDT": raw_series(80, scale=2.0)}
    )

    prefix = builder.build(prefix_source, instruments())
    full = builder.build(full_source, instruments())

    n = prefix.n_bars
    np.testing.assert_array_equal(prefix.timestamps, full.timestamps[:n])
    np.testing.assert_allclose(
        prefix.features,
        full.features[:n],
        atol=0.0,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        prefix.global_features, full.global_features[:n], atol=0.0, rtol=0.0
    )
    np.testing.assert_array_equal(prefix.feature_available, full.feature_available[:n])
    np.testing.assert_allclose(
        prefix.feature_staleness, full.feature_staleness[:n], atol=0.0, rtol=0.0
    )
    np.testing.assert_array_equal(prefix.symbol_active, full.symbol_active[:n])
    np.testing.assert_array_equal(prefix.tradable, full.tradable[:n])


def test_builder_uses_point_in_time_universe() -> None:
    dataset = MarketDatasetBuilder(config()).build(
        InMemoryMarketDataSource(
            {"BTCUSDT": raw_series(72), "ETHUSDT": raw_series(72, scale=2.0)}
        ),
        instruments(),
    )

    eth = 1
    assert dataset.symbol_active is not None
    assert dataset.feature_staleness is not None
    assert not dataset.symbol_active[9, eth]
    assert dataset.symbol_active[10, eth]
    assert not dataset.symbol_active[50, eth]
    assert not dataset.tradable[9, eth]
    assert not dataset.feature_available[9, eth].any()
    assert np.all(dataset.feature_staleness[9, eth] == 1.0)


def test_dataset_identity_binds_order_config_and_contracts() -> None:
    source = InMemoryMarketDataSource(
        {"BTCUSDT": raw_series(72), "ETHUSDT": raw_series(72, scale=2.0)}
    )
    base = MarketDatasetBuilder(config()).build(source, instruments())
    reordered = MarketDatasetBuilder(config()).build(
        source, tuple(reversed(instruments()))
    )
    changed_norm = MarketDatasetBuilder(config(normalization_window=36)).build(
        source, instruments()
    )
    changed_contracts = list(instruments())
    changed_contracts[1] = InstrumentContract(
        symbol="ETHUSDT",
        listed_at=changed_contracts[1].listed_at,
        delisted_at=changed_contracts[1].delisted_at,
        volume_unit=VolumeUnit.CONTRACTS,
        contract_multiplier=0.02,
    )
    changed_contract = MarketDatasetBuilder(config()).build(
        source, tuple(changed_contracts)
    )

    identities = {
        base.dataset_id,
        reordered.dataset_id,
        changed_norm.dataset_id,
        changed_contract.dataset_id,
    }
    assert len(identities) == 4


def test_csv_source_builds_market_dataset(tmp_path: Path) -> None:
    for symbol, scale in (("BTCUSDT", 1.0), ("ETHUSDT", 2.0)):
        rows = ["timestamp,open,high,low,close,volume,funding_rate,tradable"]
        series = raw_series(36, scale=scale)
        for index, timestamp in enumerate(series.timestamps):
            rows.append(
                f"{timestamp},{series.open[index]},{series.high[index]},"
                f"{series.low[index]},{series.close[index]},{series.volume[index]},"
                f"{series.funding_rate[index]},true"
            )
        (tmp_path / f"{symbol}.csv").write_text(
            "\n".join(rows) + "\n",
            encoding="utf-8",
        )

    dataset = MarketDatasetBuilder(config()).build(
        CsvMarketDataSource(tmp_path), instruments()
    )

    assert dataset.n_bars == 36
    assert dataset.symbols == ("BTCUSDT", "ETHUSDT")
    assert dataset.feature_names == tuple(spec.name for spec in config().features)
