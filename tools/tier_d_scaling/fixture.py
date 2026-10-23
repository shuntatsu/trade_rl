"""Deterministic artificial Dataset for bounded scaling checks."""

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

_ARTIFICIAL_BARS = 9_504
_ARTIFICIAL_SYMBOLS = ("ARTIFICIAL_A", "ARTIFICIAL_B")
_ACTOR_START = 2_903


def write_full_artificial_dataset(root: Path) -> MarketDataset:
    """Build and write a fixed 9,504-bar synthetic Dataset to a fresh root."""

    destination = Path(root)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"artificial Dataset root must be fresh: {destination}")

    bar_indices = np.arange(_ARTIFICIAL_BARS)
    timestamps = np.datetime64(
        "2000-01-01T00:00:00", "ns"
    ) + bar_indices * np.timedelta64(1, "h")
    source_by_symbol: dict[str, RawMarketSeries] = {}
    for symbol_index, symbol in enumerate(_ARTIFICIAL_SYMBOLS):
        close = (100.0 + 25.0 * symbol_index) * np.exp(bar_indices * 0.001)
        close[_ACTOR_START:] = close[_ACTOR_START]
        funding_event_count = (bar_indices % 8 == 0).astype(np.int32)
        funding_rate = funding_event_count.astype(np.float64) * 0.0001
        source_by_symbol[symbol] = RawMarketSeries(
            timestamps=timestamps,
            open=close,
            high=close * 1.001,
            low=close * 0.999,
            close=close,
            volume=np.full(_ARTIFICIAL_BARS, 1e7, dtype=np.float64),
            funding_rate=funding_rate,
            tradable=np.ones(_ARTIFICIAL_BARS, dtype=np.bool_),
            funding_available=funding_event_count > 0,
            funding_event_count=funding_event_count,
            funding_price_rate=funding_rate * close,
        )

    features = (
        FeatureSpec("artificial_return_1", FeatureKind.LOG_RETURN),
        FeatureSpec("artificial_return_24", FeatureKind.LOG_RETURN, lookback=24),
        FeatureSpec("artificial_body", FeatureKind.BODY_RETURN),
        FeatureSpec("artificial_range", FeatureKind.HIGH_LOW_RANGE),
        FeatureSpec(
            "artificial_volatility",
            FeatureKind.REALIZED_VOLATILITY,
            lookback=24,
        ),
        FeatureSpec(
            "artificial_volume_z",
            FeatureKind.VOLUME_ZSCORE,
            lookback=24,
            min_periods=24,
        ),
    )
    instruments = tuple(
        InstrumentContract(
            symbol=symbol,
            listed_at=datetime(2000, 1, 1, tzinfo=UTC),
            volume_unit=VolumeUnit.QUOTE_NOTIONAL,
        )
        for symbol in _ARTIFICIAL_SYMBOLS
    )
    economics = ExecutionEconomicsProfile(
        name="artificial scaling assumptions",
        fee_rate=0.0005,
        spread_rate=0.0002,
        max_participation_rate=0.05,
        borrow_available=True,
        borrow_rate=0.0,
    )
    dataset = MarketDatasetBuilder(
        MarketBuildConfig(base_timeframe="1h", features=features)
    ).build(
        InMemoryMarketDataSource(source_by_symbol),
        instruments,
        execution_economics=economics,
    )
    write_market_dataset_files(destination, dataset)
    return dataset
