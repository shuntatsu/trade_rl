from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

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
        self.rows = (
            self.database.klines[symbol]
            if "klines" in query
            else self.database.funding[symbol]
        )

    def fetchall(self) -> list[tuple[object, ...]]:
        return self.rows


class _Database:
    def __init__(self, symbols: tuple[str, ...], start_ms: int) -> None:
        self.klines = {
            symbol: [
                (
                    start_ms + row * 900_000,
                    100.0 + row,
                    101.0 + row,
                    99.0 + row,
                    100.5 + row,
                    1_000.0 + row,
                )
                for row in range(4)
            ]
            for symbol in symbols
        }
        self.funding = {
            symbol: [(start_ms + 1_800_000, 0.0001)] for symbol in symbols
        }

    def cursor(self) -> _Cursor:
        return _Cursor(self)


def _bundle(symbols: tuple[str, ...], start_ms: int) -> NativeIndicatorArtifactBundle:
    specs = binance_multitimeframe_feature_specs(
        base_timeframe="15m",
        feature_timeframes=("1h", "4h", "1d"),
    )
    event_time = start_ms + np.arange(1, 5, dtype=np.int64) * 900_000
    artifacts: list[NativeIndicatorArtifact] = []
    for symbol in symbols:
        for timeframe in NATIVE_TIMEFRAMES:
            names = tuple(
                spec.name for spec in specs if spec.name.startswith(f"{timeframe}__")
            )
            values = np.ones((4, len(names)), dtype=np.float32)
            artifacts.append(
                NativeIndicatorArtifact(
                    symbol=symbol,
                    timeframe=timeframe,
                    feature_names=names,
                    event_time_ms=event_time.copy(),
                    values=values,
                    available=np.ones(values.shape, dtype=np.bool_),
                    payload_schema=f"npz_native_indicator_v1:{'1' * 64}",
                    payload_sha256="1" * 64,
                )
            )
    return NativeIndicatorArtifactBundle(
        cache_id="single-symbol-cache",
        market="usds-m",
        symbols=symbols,
        timeframes=NATIVE_TIMEFRAMES,
        start_time=datetime(2021, 1, 1, tzinfo=UTC),
        end_time=datetime(2026, 7, 1, tzinfo=UTC),
        feature_config_digest="2" * 64,
        artifacts=tuple(artifacts),
    )


def _metadata(symbols: tuple[str, ...], start: datetime) -> dict[str, dict[str, object]]:
    return {
        symbol: {
            "listed_at": start.isoformat(),
            "tick_size": 0.1,
            "lot_size": 0.001,
            "minimum_notional": 5.0,
        }
        for symbol in symbols
    }


def test_builds_one_btc_symbol_with_existing_feature_contract() -> None:
    symbols = ("BTCUSDT",)
    start = datetime(2024, 1, 1, tzinfo=UTC)
    end = start + timedelta(hours=1)
    start_ms = int(start.timestamp() * 1000)

    dataset = build_postgres_market_dataset(
        _Database(symbols, start_ms),
        symbols=symbols,
        symbol_vocabulary=symbols,
        start_time=start,
        end_time=end,
        metadata=_metadata(symbols, start),
        metadata_evidence_digest="3" * 64,
        indicator_bundle=_bundle(symbols, start_ms),
    )

    assert dataset.symbols == ("BTCUSDT",)
    assert dataset.n_symbols == 1
    assert dataset.n_features == 226
    assert dataset.features.shape == (4, 1, 226)
    assert dataset.close.shape == (4, 1)
    assert dataset.identity_verified
    assert dataset.funding_event_count[:, 0].tolist() == [0, 1, 0, 0]


def test_rejects_unsupported_two_symbol_dataset() -> None:
    symbols = ("BTCUSDT", "ETHUSDT")
    start = datetime(2024, 1, 1, tzinfo=UTC)

    with pytest.raises(ValueError, match="one or three symbols"):
        build_postgres_market_dataset(
            _Database(symbols, int(start.timestamp() * 1000)),
            symbols=symbols,
            symbol_vocabulary=symbols,
            start_time=start,
            end_time=start + timedelta(hours=1),
            metadata=_metadata(symbols, start),
            metadata_evidence_digest="3" * 64,
            indicator_bundle=_bundle(symbols, int(start.timestamp() * 1000)),
        )
