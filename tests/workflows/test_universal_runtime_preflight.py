from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

from trade_rl.integrations.postgres_universal_source import MAINTAINED_SYMBOLS
from trade_rl.workflows import universal_runtime_preflight as module


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def test_preflight_materializes_only_train_symbols(
    tmp_path: Path, monkeypatch
) -> None:
    train = MAINTAINED_SYMBOLS[:9]
    validation = MAINTAINED_SYMBOLS[9:12]
    test = MAINTAINED_SYMBOLS[12:]
    partition = SimpleNamespace(
        train_symbols=train,
        validation_symbols=validation,
        test_symbols=test,
        digest=_digest("partition"),
        symbol_disjoint_manifest_digest=_digest("split"),
    )
    catalog = SimpleNamespace(
        digest=_digest("catalog"),
        source_manifest_digest=_digest("source"),
    )
    instrument_bundle = SimpleNamespace(partition=partition, catalog=catalog)
    resolution = SimpleNamespace(
        metadata={symbol: {"symbol": symbol} for symbol in MAINTAINED_SYMBOLS},
        evidence_digest=_digest("metadata"),
    )
    datasets = {
        symbol: SimpleNamespace(
            symbols=(symbol,),
            dataset_id=_digest(f"dataset:{symbol}"),
            n_bars=50_000,
        )
        for symbol in train
    }
    observed: dict[str, object] = {}

    monkeypatch.setattr(module, "resolve_frozen_snapshot", lambda **kwargs: resolution)

    def materialize_instruments(connection, **kwargs):
        del connection
        observed["instrument_kwargs"] = kwargs

    monkeypatch.setattr(
        module,
        "materialize_postgres_universal_instrument_artifacts",
        materialize_instruments,
    )
    monkeypatch.setattr(
        module,
        "load_universal_instrument_artifact_bundle",
        lambda root: instrument_bundle,
    )

    def materialize_datasets(connection, **kwargs):
        del connection
        observed["dataset_kwargs"] = kwargs
        return datasets

    monkeypatch.setattr(module, "materialize_universal_train_datasets", materialize_datasets)
    normalizer = SimpleNamespace(
        statistics_digest=_digest("statistics"),
        feature_schema_digest=_digest("features"),
    )
    monkeypatch.setattr(module, "fit_universal_shared_normalizer", lambda *a, **k: normalizer)
    monkeypatch.setattr(
        module,
        "write_universal_shared_normalizer",
        lambda root, value: Path(root),
    )
    monkeypatch.setattr(
        module, "load_universal_shared_normalizer", lambda root: normalizer
    )

    def publish(datasets_arg, *, train_symbols, artifact_root):
        assert tuple(datasets_arg) == train
        assert tuple(train_symbols) == train
        return {symbol: Path(artifact_root) / symbol for symbol in train}

    monkeypatch.setattr(module, "publish_universal_train_dataset_artifacts", publish)
    by_path = {str(tmp_path / "datasets" / symbol): datasets[symbol] for symbol in train}
    monkeypatch.setattr(
        module, "load_market_dataset_artifact", lambda path: by_path[str(path)]
    )

    manifest = module.materialize_universal_runtime_inputs(
        connection=object(),
        frozen_metadata_root=tmp_path / "frozen",
        instrument_artifact_root=tmp_path / "instruments",
        dataset_artifact_root=tmp_path / "datasets",
        normalizer_artifact_root=tmp_path / "normalizer",
        runtime_manifest_path=tmp_path / "runtime.json",
    )

    assert manifest.train_symbols == train
    assert manifest.validation_symbols == validation
    assert manifest.test_symbols == test
    assert manifest.fold_train_range == (0, 50_000)
    assert tuple(symbol for symbol, _ in manifest.dataset_digests) == train
    assert set(observed["dataset_kwargs"]["instrument_bundle"].partition.train_symbols) == set(train)  # type: ignore[index,union-attr]
