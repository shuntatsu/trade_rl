from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from trade_rl.data.contracts import InstrumentExecutionRule
from trade_rl.evaluation.walk_forward.folds import IndexRange, WalkForwardFold
from trade_rl.integrations.binance import binance_multitimeframe_feature_specs
from trade_rl.integrations.postgres_indicator_artifacts import (
    NativeIndicatorArtifact,
    NativeIndicatorArtifactBundle,
)
from trade_rl.integrations.postgres_market_dataset import NATIVE_TIMEFRAMES
from trade_rl.workflows.stage_a_postgres_evaluation_dataset import (
    build_stage_a_postgres_evaluation_datasets,
)
from trade_rl.workflows.symbol_disjoint_manifest import build_symbol_disjoint_manifest
from trade_rl.workflows.symbol_disjoint_triplet_manifest import (
    build_symbol_disjoint_triplet_manifest,
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
    def __init__(self, symbols: tuple[str, ...], start_ms: int, rows: int) -> None:
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
                for row in range(rows)
            ]
            self.funding[symbol] = [
                (start_ms + row * 900_000, 0.0001) for row in range(8, rows, 32)
            ]

    def cursor(self) -> _Cursor:
        return _Cursor(self)


def _bundle(
    symbols: tuple[str, ...], start: datetime, rows: int
) -> NativeIndicatorArtifactBundle:
    specs = binance_multitimeframe_feature_specs(
        base_timeframe="15m", feature_timeframes=("1h", "4h", "1d")
    )
    event_time = (
        int(start.timestamp() * 1000) + np.arange(1, rows + 1, dtype=np.int64) * 900_000
    )
    artifacts: list[NativeIndicatorArtifact] = []
    for symbol_index, symbol in enumerate(symbols):
        for timeframe in NATIVE_TIMEFRAMES:
            names = tuple(
                spec.name for spec in specs if spec.name.startswith(f"{timeframe}__")
            )
            values = np.full(
                (rows, len(names)), float(symbol_index + 1), dtype=np.float32
            )
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
        cache_id="cache-stage-a",
        market="usds-m",
        symbols=symbols,
        timeframes=NATIVE_TIMEFRAMES,
        start_time=start,
        end_time=start + timedelta(minutes=15 * rows),
        feature_config_digest="2" * 64,
        artifacts=tuple(artifacts),
    )


def _folds() -> tuple[WalkForwardFold, ...]:
    return (
        WalkForwardFold(
            fold_index=0,
            train=IndexRange(0, 10),
            checkpoint_validation=IndexRange(11, 20),
            configuration_selection=IndexRange(21, 30),
            test=IndexRange(31, 40),
            purge_bars=1,
        ),
        WalkForwardFold(
            fold_index=1,
            train=IndexRange(0, 30),
            checkpoint_validation=IndexRange(31, 40),
            configuration_selection=IndexRange(41, 50),
            test=IndexRange(51, 60),
            purge_bars=1,
        ),
    )


def _metadata(
    symbols: tuple[str, ...], start: datetime
) -> dict[str, dict[str, object]]:
    return {
        symbol: {
            "listed_at": start.isoformat(),
            "tick_size": 0.1,
            "lot_size": 0.001,
            "minimum_notional": 5.0,
        }
        for symbol in symbols
    }


def _histories(
    symbols: tuple[str, ...], start: datetime
) -> dict[str, tuple[InstrumentExecutionRule, ...]]:
    return {
        symbol: (
            InstrumentExecutionRule(
                effective_at=start,
                tick_size=0.1,
                lot_size=0.001,
                minimum_notional=5.0,
            ),
        )
        for symbol in symbols
    }


def _triplet_manifest():
    symbols = tuple(f"ASSET-{index:02d}" for index in range(15))
    source = build_symbol_disjoint_manifest(
        symbols,
        seed=20260801,
        validation_count=3,
        test_count=3,
    )
    return build_symbol_disjoint_triplet_manifest(source)


def test_builds_exact_postgres_datasets_and_manifest_from_declared_triplets() -> None:
    triplets = _triplet_manifest()
    evaluation_symbols = tuple(
        dict.fromkeys(
            symbol
            for split in ("validation", "test")
            for slot in triplets.slots_for(split)
            for symbol in slot.symbols
        )
    )
    start = datetime(2024, 1, 1, tzinfo=UTC)
    rows = 96
    end = start + timedelta(minutes=15 * rows)

    result = build_stage_a_postgres_evaluation_datasets(
        _Database(evaluation_symbols, int(start.timestamp() * 1000), rows),
        triplet_manifest=triplets,
        folds=_folds(),
        start_time=start,
        end_time=end,
        metadata=_metadata(evaluation_symbols, start),
        metadata_evidence_digest="3" * 64,
        execution_rule_histories=_histories(evaluation_symbols, start),
        indicator_bundle=_bundle(evaluation_symbols, start, rows),
    )

    assert (
        result.manifest.symbol_disjoint_manifest_digest
        == triplets.source_manifest_digest
    )
    assert result.manifest.symbol_disjoint_triplet_manifest_digest == triplets.digest
    assert result.manifest.folds_declared == (0, 1)
    assert result.manifest.range_for("validation", 0) == IndexRange(21, 30)
    assert result.manifest.range_for("test", 1) == IndexRange(51, 60)
    assert len(result.datasets) == 2

    timestamps = [dataset.timestamps for _, dataset in result.datasets]
    np.testing.assert_array_equal(timestamps[0], timestamps[1])
    for split in ("validation", "test"):
        slot = triplets.slots_for(split)[0]
        dataset = result.dataset_for(split, slot.triplet_id)
        binding = result.manifest.triplet_for(split, slot.triplet_id)
        assert binding.symbols == slot.symbols
        assert binding.dataset_id == dataset.dataset_id
        assert dataset.n_bars == rows
        assert dataset.symbols == ("SLOT0", "SLOT1", "SLOT2")


def test_rejects_fold_range_outside_common_postgres_timeline() -> None:
    triplets = _triplet_manifest()
    evaluation_symbols = tuple(
        dict.fromkeys(
            symbol
            for split in ("validation", "test")
            for slot in triplets.slots_for(split)
            for symbol in slot.symbols
        )
    )
    start = datetime(2024, 1, 1, tzinfo=UTC)
    rows = 64
    end = start + timedelta(minutes=15 * rows)
    invalid = (
        _folds()[0],
        WalkForwardFold(
            fold_index=1,
            train=IndexRange(0, 30),
            checkpoint_validation=IndexRange(31, 40),
            configuration_selection=IndexRange(41, 50),
            test=IndexRange(60, 70),
            purge_bars=1,
        ),
    )

    with pytest.raises(ValueError, match="range exceeds the common timeline"):
        build_stage_a_postgres_evaluation_datasets(
            _Database(evaluation_symbols, int(start.timestamp() * 1000), rows),
            triplet_manifest=triplets,
            folds=invalid,
            start_time=start,
            end_time=end,
            metadata=_metadata(evaluation_symbols, start),
            metadata_evidence_digest="3" * 64,
            execution_rule_histories=_histories(evaluation_symbols, start),
            indicator_bundle=_bundle(evaluation_symbols, start, rows),
        )
