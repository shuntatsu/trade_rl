from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np

from trade_rl.integrations.binance import binance_multitimeframe_feature_specs
from trade_rl.integrations.postgres_indicator_artifacts import (
    NativeIndicatorArtifact,
    NativeIndicatorArtifactBundle,
)
from trade_rl.integrations.postgres_market_dataset import (
    NATIVE_TIMEFRAMES,
    build_postgres_market_dataset,
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
        if "klines" in query:
            self.rows = self.database.klines[symbol]
        else:
            self.rows = self.database.funding[symbol]

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


def test_builds_btc_free_triplet_with_stable_symbol_identity() -> None:
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

    dataset = build_postgres_market_dataset(
        _Database(symbols, start_ms),
        symbols=symbols,
        symbol_vocabulary=vocabulary,
        start_time=start,
        end_time=end,
        metadata={symbol: {} for symbol in symbols},
        metadata_evidence_digest="3" * 64,
        indicator_bundle=_bundle(symbols, start_ms),
        slot_symbols=("SLOT0", "SLOT1", "SLOT2"),
    )

    assert dataset.symbols == ("SLOT0", "SLOT1", "SLOT2")
    assert dataset.n_features == 226 + len(vocabulary)
    assert dataset.identity_verified
    assert dataset.feature_names[-5:] == tuple(
        f"15m__symbol_id_{symbol}" for symbol in vocabulary
    )
    identity = dataset.features[:, :, -5:]
    np.testing.assert_array_equal(identity[0, 0], [0, 0, 0, 1, 0])
    np.testing.assert_array_equal(identity[0, 1], [0, 1, 0, 0, 0])
    np.testing.assert_array_equal(identity[0, 2], [0, 0, 1, 0, 0])
    assert dataset.funding_event_count[:, 0].tolist() == [0, 1, 0, 0]
    assert dataset.funding_rate[1, 0] == 0.0001
