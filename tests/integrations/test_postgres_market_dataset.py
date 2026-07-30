from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from trade_rl.data.builder import MarketDatasetBuilder
from trade_rl.data.contracts import (
    FeatureKind,
    FeatureSpec,
    InstrumentContract,
    InstrumentExecutionRule,
    MarketBuildConfig,
    VolumeUnit,
)
from trade_rl.data.source import InMemoryMarketDataSource, RawMarketSeries
from trade_rl.integrations.binance import binance_multitimeframe_feature_specs
from trade_rl.integrations.postgres_indicator_artifacts import (
    NativeIndicatorArtifact,
    NativeIndicatorArtifactBundle,
)
from trade_rl.integrations.postgres_market_dataset import (
    NATIVE_TIMEFRAMES,
    POLICY_ASSET_IDENTITY_MODE,
    build_postgres_market_dataset,
)

_ECONOMIC_FIELDS = (
    "symbol_active",
    "asset_active",
    "tradable",
    "information_available",
    "available_at",
    "fee_rate",
    "maker_fee_rate",
    "taker_fee_rate",
    "spread_rate",
    "max_participation_rate",
    "minimum_notional",
    "lot_size",
    "tick_size",
    "borrow_available",
    "borrow_rate",
    "funding_due",
    "buy_allowed",
    "sell_allowed",
    "mark_price",
    "index_price",
)


class _Cursor:
    def __init__(self, database: _Database) -> None:
        self.database = database
        self.rows: list[tuple[object, ...]] = []

    def __enter__(self) -> _Cursor:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def execute(self, query: str, params: object = None) -> None:
        assert isinstance(params, tuple)
        symbol = str(params[0])
        self.rows = (
            self.database.klines[symbol]
            if "klines" in query
            else self.database.funding[symbol]
        )

    def fetchall(self) -> list[tuple[object, ...]]:
        return self.rows


class _Database:
    def __init__(self, symbols: tuple[str, ...], start_ms: int) -> None:
        self.klines: dict[str, list[tuple[object, ...]]] = {}
        self.funding: dict[str, list[tuple[object, ...]]] = {}
        for symbol_index, symbol in enumerate(symbols):
            base = 10.0 + symbol_index
            self.klines[symbol] = [
                (
                    start_ms + row * 900_000,
                    base + row,
                    base + row + 1.0,
                    base + row - 1.0,
                    base + row + 0.5,
                    1_000.0 + row,
                )
                for row in range(4)
            ]
            self.funding[symbol] = [(start_ms + 1_800_000, 0.0001)]

    def cursor(self) -> _Cursor:
        return _Cursor(self)


def _bundle(symbols: tuple[str, ...], start_ms: int) -> NativeIndicatorArtifactBundle:
    specs = binance_multitimeframe_feature_specs(
        base_timeframe="15m", feature_timeframes=("1h", "4h", "1d")
    )
    artifacts: list[NativeIndicatorArtifact] = []
    event_time = start_ms + np.arange(1, 5, dtype=np.int64) * 900_000
    for symbol_index, symbol in enumerate(symbols):
        for timeframe in NATIVE_TIMEFRAMES:
            names = tuple(
                spec.name for spec in specs if spec.name.startswith(f"{timeframe}__")
            )
            values = np.full((4, len(names)), float(symbol_index + 1), dtype=np.float32)
            available = np.ones(values.shape, dtype=np.bool_)
            artifacts.append(
                NativeIndicatorArtifact(
                    symbol=symbol,
                    timeframe=timeframe,
                    feature_names=names,
                    event_time_ms=event_time.copy(),
                    values=values,
                    available=available,
                    payload_schema=f"npz_native_indicator_v1:{'1' * 64}",
                    payload_sha256=f"{symbol_index + 1:064x}",
                )
            )
    return NativeIndicatorArtifactBundle(
        cache_id="cache",
        market="usds-m",
        symbols=symbols,
        timeframes=NATIVE_TIMEFRAMES,
        start_time=datetime(2021, 1, 1, tzinfo=UTC),
        end_time=datetime(2026, 7, 1, tzinfo=UTC),
        feature_config_digest="2" * 64,
        artifacts=tuple(artifacts),
    )


def _metadata(
    symbols: tuple[str, ...],
    start: datetime,
    *,
    first_listing_delay: timedelta = timedelta(0),
) -> dict[str, dict[str, object]]:
    return {
        symbol: {
            "listed_at": (
                start + (first_listing_delay if index == 0 else timedelta(0))
            ).isoformat(),
            "tick_size": 0.1 + index * 0.01,
            "lot_size": 0.001 + index * 0.0001,
            "minimum_notional": 5.0 + index,
        }
        for index, symbol in enumerate(symbols)
    }


def _rule_histories(
    symbols: tuple[str, ...],
    start: datetime,
    metadata: dict[str, dict[str, object]],
) -> dict[str, tuple[InstrumentExecutionRule, ...]]:
    histories: dict[str, tuple[InstrumentExecutionRule, ...]] = {}
    for index, symbol in enumerate(symbols):
        entry = metadata[symbol]
        base = InstrumentExecutionRule(
            effective_at=start,
            tick_size=float(entry["tick_size"]),
            lot_size=float(entry["lot_size"]),
            minimum_notional=float(entry["minimum_notional"]),
        )
        histories[symbol] = (
            base,
            *(
                (
                    InstrumentExecutionRule(
                        effective_at=start + timedelta(minutes=45),
                        tick_size=0.25,
                        lot_size=0.0025,
                        minimum_notional=12.0,
                    ),
                )
                if index == 0
                else ()
            ),
        )
    return histories


def _raw_source(symbols: tuple[str, ...], start: datetime) -> InMemoryMarketDataSource:
    timestamps = np.datetime64(start.replace(tzinfo=None), "ns") + np.arange(
        1, 5
    ) * np.timedelta64(15, "m")
    values: dict[str, RawMarketSeries] = {}
    for index, symbol in enumerate(symbols):
        base = 10.0 + index
        rows = np.arange(4, dtype=np.float64)
        funding_count = np.array([0, 1, 0, 0], dtype=np.int32)
        values[symbol] = RawMarketSeries(
            timestamps=timestamps,
            open=base + rows,
            high=base + rows + 1.0,
            low=base + rows - 1.0,
            close=base + rows + 0.5,
            volume=1_000.0 + rows,
            funding_rate=np.array([0.0, 0.0001, 0.0, 0.0]),
            tradable=np.ones(4, dtype=np.bool_),
            funding_available=funding_count > 0,
            funding_event_count=funding_count,
        )
    return InMemoryMarketDataSource(values)


def test_builds_btc_free_triplet_with_identity_free_policy_features() -> None:
    symbols = ("SOLUSDT", "ETHUSDT", "BNBUSDT")
    vocabulary = (
        "BTCUSDT",
        "ETHUSDT",
        "BNBUSDT",
        "SOLUSDT",
        "XRPUSDT",
    )
    start = datetime(2024, 1, 1, tzinfo=UTC)
    end = start + timedelta(hours=1)
    start_ms = int(start.timestamp() * 1000)
    metadata = _metadata(symbols, start, first_listing_delay=timedelta(minutes=30))

    dataset = build_postgres_market_dataset(
        _Database(symbols, start_ms),
        symbols=symbols,
        symbol_vocabulary=vocabulary,
        start_time=start,
        end_time=end,
        metadata=metadata,
        metadata_evidence_digest="3" * 64,
        indicator_bundle=_bundle(symbols, start_ms),
        slot_symbols=("SLOT0", "SLOT1", "SLOT2"),
    )

    assert POLICY_ASSET_IDENTITY_MODE == "identity_free_v1"
    assert dataset.symbols == ("SLOT0", "SLOT1", "SLOT2")
    assert dataset.n_features == 226
    assert dataset.identity_verified
    assert not any("symbol_id" in name for name in dataset.feature_names)
    np.testing.assert_array_equal(
        dataset.symbol_active[:, 0], [False, True, True, True]
    )
    assert not dataset.feature_available[0, 0].any()
    assert dataset.funding_event_count[:, 0].tolist() == [0, 1, 0, 0]
    assert dataset.funding_rate[1, 0] == 0.0001


def test_postgres_metadata_is_required_instead_of_defaulting_to_zero() -> None:
    symbols = ("SOLUSDT", "ETHUSDT", "BNBUSDT")
    start = datetime(2024, 1, 1, tzinfo=UTC)
    end = start + timedelta(hours=1)
    metadata = _metadata(symbols, start)
    del metadata[symbols[0]]["tick_size"]

    with pytest.raises(ValueError, match="SOLUSDT.tick_size"):
        build_postgres_market_dataset(
            _Database(symbols, int(start.timestamp() * 1000)),
            symbols=symbols,
            symbol_vocabulary=symbols,
            start_time=start,
            end_time=end,
            metadata=metadata,
            metadata_evidence_digest="3" * 64,
            indicator_bundle=_bundle(symbols, int(start.timestamp() * 1000)),
        )


def test_postgres_and_builder_paths_have_identical_economic_arrays() -> None:
    symbols = ("SOLUSDT", "ETHUSDT", "BNBUSDT")
    start = datetime(2024, 1, 1, tzinfo=UTC)
    end = start + timedelta(hours=1)
    start_ms = int(start.timestamp() * 1000)
    metadata = _metadata(symbols, start)
    histories = _rule_histories(symbols, start, metadata)
    postgres = build_postgres_market_dataset(
        _Database(symbols, start_ms),
        symbols=symbols,
        symbol_vocabulary=symbols,
        start_time=start,
        end_time=end,
        metadata=metadata,
        metadata_evidence_digest="3" * 64,
        execution_rule_histories=histories,
        indicator_bundle=_bundle(symbols, start_ms),
    )
    instruments = tuple(
        InstrumentContract(
            symbol=symbol,
            listed_at=start,
            volume_unit=VolumeUnit.QUOTE_NOTIONAL,
            tick_size=float(metadata[symbol]["tick_size"]),
            lot_size=float(metadata[symbol]["lot_size"]),
            minimum_notional=float(metadata[symbol]["minimum_notional"]),
            execution_rules=histories[symbol],
        )
        for symbol in symbols
    )
    builder = MarketDatasetBuilder(
        MarketBuildConfig(
            base_timeframe="15m",
            features=(
                FeatureSpec(
                    name="ret_1",
                    kind=FeatureKind.LOG_RETURN,
                    lookback=1,
                ),
            ),
        )
    ).build(_raw_source(symbols, start), instruments)

    for field in _ECONOMIC_FIELDS:
        np.testing.assert_array_equal(
            getattr(postgres, field),
            getattr(builder, field),
            err_msg=field,
        )
