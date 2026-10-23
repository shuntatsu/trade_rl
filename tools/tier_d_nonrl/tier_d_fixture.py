"""Deterministic artificial fixture; it contains no market data."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from trade_rl.data import write_market_dataset_files
from trade_rl.data.build.builder import MarketDatasetBuilder
from trade_rl.data.build.economics import ExecutionEconomicsProfile
from trade_rl.data.contracts import (
    FeatureKind,
    FeatureSpec,
    InstrumentContract,
    MarketBuildConfig,
    VolumeUnit,
)
from trade_rl.data.market import MarketDataset
from trade_rl.data.source import InMemoryMarketDataSource, RawMarketSeries


def write_artificial_dataset(root: Path) -> MarketDataset:
    times = np.datetime64("2025-01-01T00:00:00", "ns") + np.arange(
        256
    ) * np.timedelta64(1, "h")
    values = {}
    for i, symbol in enumerate(("ARTIFICIAL_A", "ARTIFICIAL_B")):
        close = (100.0 + i * 25) * np.exp(np.arange(256) * 0.001)
        # A fixed actor/reserve plateau permits an unchanged latency2 entry
        # to reach eligibility. The rising fit prefix stays wholly before128.
        close[128:] = close[128]
        counts = (np.arange(256) % 8 == 0).astype(np.int32)
        funding_rates = counts * 0.0001
        values[symbol] = RawMarketSeries(
            timestamps=times,
            open=close,
            high=close * 1.001,
            low=close * 0.999,
            close=close,
            volume=np.full(256, 10000000.0),
            funding_rate=funding_rates,
            tradable=np.ones(256, dtype=bool),
            funding_available=counts > 0,
            funding_event_count=counts,
            funding_price_rate=funding_rates * close,
        )
    specs = (
        FeatureSpec("artificial_return_1", FeatureKind.LOG_RETURN),
        FeatureSpec("artificial_return_24", FeatureKind.LOG_RETURN, lookback=24),
        FeatureSpec("artificial_body", FeatureKind.BODY_RETURN),
        FeatureSpec("artificial_range", FeatureKind.HIGH_LOW_RANGE),
        FeatureSpec(
            "artificial_volatility", FeatureKind.REALIZED_VOLATILITY, lookback=24
        ),
        FeatureSpec(
            "artificial_volume_z",
            FeatureKind.VOLUME_ZSCORE,
            lookback=24,
            min_periods=24,
        ),
    )
    dataset = MarketDatasetBuilder(
        MarketBuildConfig(base_timeframe="1h", features=specs)
    ).build(
        InMemoryMarketDataSource(values),
        tuple(
            InstrumentContract(
                symbol,
                datetime(2024, 1, 1, tzinfo=UTC),
                volume_unit=VolumeUnit.QUOTE_NOTIONAL,
            )
            for symbol in values
        ),
        execution_economics=ExecutionEconomicsProfile(
            name="artificial Tier D assumptions",
            fee_rate=0.0005,
            spread_rate=0.0002,
            max_participation_rate=0.05,
            borrow_available=True,
            borrow_rate=0.0876,
        ),
    )
    write_market_dataset_files(root, dataset)
    return dataset
