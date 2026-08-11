from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import numpy as np
import pytest

from trade_rl.artifacts.hashing import content_digest
from trade_rl.workflows.binance_metadata_modes import (
    BinanceMetadataMode,
    BinanceMetadataResolution,
)


def _digest(label: str) -> str:
    return content_digest(label)


class _IndicatorBundle:
    def __init__(self, symbols: tuple[str, ...]) -> None:
        self.symbols = symbols

    def subset(self, symbols: tuple[str, ...]) -> _IndicatorBundle:
        assert set(symbols) <= set(self.symbols)
        return _IndicatorBundle(symbols)


def _dataset(symbol: str) -> SimpleNamespace:
    timestamps = np.arange(12, dtype=np.int64).astype("datetime64[m]")
    features = np.zeros((12, 1, 4), dtype=np.float32)
    available = np.ones_like(features, dtype=np.bool_)
    return SimpleNamespace(
        dataset_id=_digest(f"dataset:{symbol}"),
        symbols=(symbol,),
        n_symbols=1,
        timestamps=timestamps,
        features=features,
        feature_available=available,
        feature_names=(
            "15m__a",
            "1h__b",
            "4h__c",
            "1d__d",
        ),
        n_features=4,
        n_bars=12,
        bar_hours=0.25,
        nominal_bar_hours=0.25,
        global_feature_names=("g0",),
    )


def _bundle() -> SimpleNamespace:
    partition = SimpleNamespace(
        train_symbols=("AAAUSDT", "BBBUSDT"),
        validation_symbols=("VALIDUSDT",),
        test_symbols=("TESTUSDT",),
        digest=_digest("partition"),
        symbol_disjoint_manifest_digest=_digest("split"),
    )
    catalog = SimpleNamespace(
        digest=_digest("catalog"),
        research_start=datetime(2024, 1, 1, tzinfo=UTC),
        research_end=datetime(2024, 2, 1, tzinfo=UTC),
        eligible_symbols=("AAAUSDT", "BBBUSDT", "VALIDUSDT", "TESTUSDT"),
        per_symbol_metadata_digests=(
            ("AAAUSDT", _digest("meta:A")),
            ("BBBUSDT", _digest("meta:B")),
            ("VALIDUSDT", _digest("meta:V")),
            ("TESTUSDT", _digest("meta:T")),
        ),
    )
    return SimpleNamespace(partition=partition, catalog=catalog)


def _metadata() -> BinanceMetadataResolution:
    return BinanceMetadataResolution(
        mode=BinanceMetadataMode.FROZEN_SNAPSHOT,
        metadata={
            "AAAUSDT": {
                "listed_at": "2020-01-01T00:00:00+00:00",
                "tick_size": 0.1,
                "lot_size": 0.001,
                "minimum_notional": 5.0,
            },
            "BBBUSDT": {
                "listed_at": "2020-01-01T00:00:00+00:00",
                "tick_size": 0.01,
                "lot_size": 0.01,
                "minimum_notional": 5.0,
            },
            "VALIDUSDT": {
                "listed_at": "2020-01-01T00:00:00+00:00",
                "tick_size": 0.01,
                "lot_size": 0.01,
                "minimum_notional": 5.0,
            },
            "TESTUSDT": {
                "listed_at": "2020-01-01T00:00:00+00:00",
                "tick_size": 0.01,
                "lot_size": 0.01,
                "minimum_notional": 5.0,
            },
        },
        execution_rule_histories=None,
        identity_evidence={"schema_version": "test"},
        evidence_digest=_digest("metadata-resolution"),
        source_uri="test://metadata",
    )


def test_materialize_universal_train_datasets_never_builds_validation_or_test() -> None:
    from trade_rl.workflows.universal_training import (
        materialize_universal_train_datasets,
    )

    indicator_calls: list[tuple[str, ...]] = []
    dataset_calls: list[str] = []

    def indicator_loader(*_: object, **kwargs: object) -> _IndicatorBundle:
        symbols = tuple(kwargs["symbols"])  # type: ignore[arg-type]
        indicator_calls.append(symbols)
        return _IndicatorBundle(symbols)

    def dataset_builder(*_: object, **kwargs: object) -> SimpleNamespace:
        symbols = tuple(kwargs["symbols"])  # type: ignore[arg-type]
        assert len(symbols) == 1
        symbol = str(symbols[0])
        dataset_calls.append(symbol)
        metadata = kwargs["metadata"]
        assert isinstance(metadata, dict)
        assert set(metadata) == {symbol}
        return _dataset(symbol)

    datasets = materialize_universal_train_datasets(
        object(),
        instrument_bundle=_bundle(),
        metadata_resolution=_metadata(),
        feature_specs=(object(), object(), object(), object()),
        indicator_loader=indicator_loader,
        dataset_builder=dataset_builder,
    )

    assert indicator_calls == [("AAAUSDT", "BBBUSDT")]
    assert dataset_calls == ["AAAUSDT", "BBBUSDT"]
    assert tuple(datasets) == ("AAAUSDT", "BBBUSDT")
    assert "VALIDUSDT" not in dataset_calls
    assert "TESTUSDT" not in dataset_calls


def test_materialize_universal_train_datasets_requires_aligned_feature_contract() -> (
    None
):
    from trade_rl.workflows.universal_training import (
        materialize_universal_train_datasets,
    )

    def dataset_builder(*_: object, **kwargs: object) -> SimpleNamespace:
        symbol = str(tuple(kwargs["symbols"])[0])  # type: ignore[arg-type]
        dataset = _dataset(symbol)
        if symbol == "BBBUSDT":
            dataset.feature_names = ("15m__a", "1h__DIFFERENT", "4h__c", "1d__d")
        return dataset

    with pytest.raises(ValueError, match="feature order"):
        materialize_universal_train_datasets(
            object(),
            instrument_bundle=_bundle(),
            metadata_resolution=_metadata(),
            feature_specs=(object(), object(), object(), object()),
            indicator_loader=lambda *_args, **_kwargs: _IndicatorBundle(
                ("AAAUSDT", "BBBUSDT")
            ),
            dataset_builder=dataset_builder,
        )


def test_fit_universal_shared_normalizer_uses_only_explicit_fold_range() -> None:
    from trade_rl.workflows.universal_training import fit_universal_shared_normalizer

    first = _dataset("AAAUSDT")
    second = _dataset("BBBUSDT")
    first.features[:, 0, :] = np.arange(48, dtype=np.float32).reshape(12, 4)
    second.features[:, 0, :] = 100.0 + np.arange(48, dtype=np.float32).reshape(12, 4)

    normalizer = fit_universal_shared_normalizer(
        {"AAAUSDT": first, "BBBUSDT": second},
        train_symbols=("AAAUSDT", "BBBUSDT"),
        catalog_digest=_digest("catalog"),
        split_manifest_digest=_digest("split"),
        fold_train_range=(2, 6),
        max_samples_per_symbol=4,
    )

    assert normalizer.fold_train_range == (2, 6)
    assert normalizer.train_symbols == ("AAAUSDT", "BBBUSDT")
    assert normalizer.sample_count_per_symbol == 4
    assert normalizer.feature_schema_digest == content_digest(
        {
            "feature_names": first.feature_names,
            "profile": "binance_universal_target_local_v1",
            "schema_version": "universal_feature_schema_v1",
        }
    )
