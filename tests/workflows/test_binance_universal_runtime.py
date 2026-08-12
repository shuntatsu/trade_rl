from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from trade_rl.artifacts.hashing import content_digest
from trade_rl.integrations.postgres_market_tables import (
    UNIVERSAL_202411_202607_CACHE_ID,
    UNIVERSAL_202411_202607_TABLES,
)
from trade_rl.integrations.postgres_universal_source import MAINTAINED_SYMBOLS
from trade_rl.rl.training_run_config import TrainingRunConfig
from trade_rl.workflows.universal_research import FullResearchAlgorithm
from trade_rl.workflows.universal_runtime_manifest import (
    UniversalRuntimeManifest,
    write_universal_runtime_manifest,
)


def _digest(label: str) -> str:
    return content_digest(label)


def _manifest(path: Path) -> UniversalRuntimeManifest:
    train = MAINTAINED_SYMBOLS[:9]
    manifest = UniversalRuntimeManifest(
        cache_id=UNIVERSAL_202411_202607_CACHE_ID,
        tables=UNIVERSAL_202411_202607_TABLES,
        research_start=datetime(2024, 11, 13, tzinfo=UTC),
        research_end=datetime(2026, 7, 5, tzinfo=UTC),
        instrument_artifact_relpath=Path("instruments"),
        dataset_artifact_relpath=Path("datasets"),
        normalizer_artifact_relpath=Path("normalizer"),
        train_symbols=train,
        validation_symbols=MAINTAINED_SYMBOLS[9:12],
        test_symbols=MAINTAINED_SYMBOLS[12:],
        fold_train_range=(0, 100),
        shared_complete_row_count=100,
        catalog_digest=_digest("catalog"),
        partition_digest=_digest("partition"),
        split_manifest_digest=_digest("split"),
        feature_schema_digest=_digest("features"),
        statistics_digest=_digest("statistics"),
        metadata_evidence_digest=_digest("metadata"),
        source_manifest_digest=_digest("source"),
        dataset_digests=tuple(
            (symbol, _digest(f"dataset:{symbol}")) for symbol in train
        ),
    )
    write_universal_runtime_manifest(path, manifest)
    return manifest


def _configs() -> dict[FullResearchAlgorithm, TrainingRunConfig]:
    root = Path("examples/binance-multitimeframe")
    common = TrainingRunConfig.from_json(root / "universal-u6-ppo.json")
    return {
        FullResearchAlgorithm.PPO: common,
        FullResearchAlgorithm.LAGRANGIAN: replace(
            common,
            training=TrainingRunConfig.from_json(
                root / "universal-u6-lagrangian.json"
            ).training,
        ),
        FullResearchAlgorithm.DISCOUNTED: replace(
            common,
            training=TrainingRunConfig.from_json(
                root / "universal-u6-discounted.json"
            ).training,
        ),
    }


def test_concrete_factory_returns_runtime_for_all_algorithms_with_shared_static_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import trade_rl.workflows.binance_universal_runtime as module
    from trade_rl.workflows.universal_full_research_entrypoint import (
        UniversalRuntimeFactoryContext,
    )

    manifest = _manifest(tmp_path / "runtime.json")
    context = UniversalRuntimeFactoryContext(
        runtime_manifest_path=tmp_path / "runtime.json",
        frozen_metadata_root=tmp_path / "frozen",
    )
    catalog = SimpleNamespace(digest=manifest.catalog_digest)
    partition = SimpleNamespace(
        train_symbols=manifest.train_symbols,
        validation_symbols=manifest.validation_symbols,
        test_symbols=manifest.test_symbols,
        digest=manifest.partition_digest,
        symbol_disjoint_manifest_digest=manifest.split_manifest_digest,
    )
    monkeypatch.setattr(
        module,
        "load_universal_instrument_artifact_bundle",
        lambda _root: SimpleNamespace(catalog=catalog, partition=partition),
    )
    monkeypatch.setattr(
        module,
        "load_market_dataset_artifact",
        lambda path: SimpleNamespace(
            symbols=(Path(path).name,), dataset_id=_digest(f"dataset:{Path(path).name}")
        ),
    )
    monkeypatch.setattr(
        module,
        "load_universal_shared_normalizer",
        lambda _root: SimpleNamespace(
            statistics_digest=manifest.statistics_digest,
            feature_schema_digest=manifest.feature_schema_digest,
            catalog_digest=manifest.catalog_digest,
            split_manifest_digest=manifest.split_manifest_digest,
            fold_train_range=manifest.fold_train_range,
            train_symbols=manifest.train_symbols,
        ),
    )

    def resolve(**kwargs):
        assert tuple(kwargs["symbols"]) == MAINTAINED_SYMBOLS
        return SimpleNamespace(evidence_digest=manifest.metadata_evidence_digest)

    monkeypatch.setattr(module, "resolve_frozen_snapshot", resolve)
    monkeypatch.setattr(
        module,
        "build_universal_instrument_contracts",
        lambda _resolution, *, train_symbols: {
            symbol: object() for symbol in train_symbols
        },
    )
    monkeypatch.setattr(
        module,
        "build_universal_bindings",
        lambda **_kwargs: tuple(
            SimpleNamespace(concrete_symbol=symbol, split="train")
            for symbol in manifest.train_symbols
        ),
    )
    monkeypatch.setattr(
        module,
        "CausalInstrumentContextProvider",
        lambda **_kwargs: SimpleNamespace(schema_digest=_digest("context")),
    )
    monkeypatch.setattr(
        module, "bind_universal_normalizers", lambda *a, **k: (object(), object())
    )

    runtimes = [
        module.build_runtime(algorithm=algorithm, run_config=config, context=context)
        for algorithm, config in _configs().items()
    ]

    assert len({runtime.catalog_digest for runtime in runtimes}) == 1
    assert len({runtime.statistics_digest for runtime in runtimes}) == 1
    assert len({runtime.feature_schema_digest for runtime in runtimes}) == 1
    assert len({runtime.training_contract_digest for runtime in runtimes}) == 3


def test_concrete_factory_rejects_dataset_identity_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import trade_rl.workflows.binance_universal_runtime as module
    from trade_rl.workflows.universal_full_research_entrypoint import (
        UniversalRuntimeFactoryContext,
    )

    manifest = _manifest(tmp_path / "runtime.json")
    context = UniversalRuntimeFactoryContext(
        runtime_manifest_path=tmp_path / "runtime.json",
        frozen_metadata_root=tmp_path / "frozen",
    )
    monkeypatch.setattr(
        module,
        "load_market_dataset_artifact",
        lambda path: SimpleNamespace(
            symbols=(Path(path).name,), dataset_id=_digest("drift")
        ),
    )
    monkeypatch.setattr(
        module,
        "load_universal_instrument_artifact_bundle",
        lambda _root: SimpleNamespace(
            catalog=SimpleNamespace(digest=manifest.catalog_digest),
            partition=SimpleNamespace(
                train_symbols=manifest.train_symbols,
                validation_symbols=manifest.validation_symbols,
                test_symbols=manifest.test_symbols,
                digest=manifest.partition_digest,
                symbol_disjoint_manifest_digest=manifest.split_manifest_digest,
            ),
        ),
    )

    with pytest.raises(ValueError, match="dataset artifact identity"):
        module.build_runtime(
            algorithm=FullResearchAlgorithm.PPO,
            run_config=_configs()[FullResearchAlgorithm.PPO],
            context=context,
        )
