from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from trade_rl.artifacts.hashing import content_digest
from trade_rl.data.v4_context import (
    CROSS_MARKET_CORE_NAMES,
    GLOBAL_MARKET_CORE_NAMES,
    V4ContextBlock,
    V4TargetContext,
)
from trade_rl.data.v4_context_artifact import write_v4_target_context_artifact
from trade_rl.integrations.postgres_market_tables import (
    UNIVERSAL_202411_202607_CACHE_ID,
    UNIVERSAL_202411_202607_TABLES,
)
from trade_rl.integrations.postgres_universal_source import MAINTAINED_SYMBOLS
from trade_rl.rl.training_run_config import TrainingRunConfig
from trade_rl.workflows.universal_causal_alpha_v4_manifest import (
    CausalAlphaV4ContextManifest,
    write_causal_alpha_v4_context_manifest,
)
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


def _v4_context(symbol: str) -> V4TargetContext:
    decisions = np.asarray([0, 1], dtype=np.int64)
    local_values = np.zeros((2, len(CROSS_MARKET_CORE_NAMES)), dtype=np.float64)
    global_values = np.zeros((2, len(GLOBAL_MARKET_CORE_NAMES)), dtype=np.float64)
    local = V4ContextBlock(
        feature_names=CROSS_MARKET_CORE_NAMES,
        decision_indices=decisions,
        values=local_values,
        available=np.ones(local_values.shape, dtype=np.bool_),
        staleness_hours=np.zeros(local_values.shape, dtype=np.float64),
        source_digest=_digest(f"v4-local:{symbol}"),
    )
    global_market = V4ContextBlock(
        feature_names=GLOBAL_MARKET_CORE_NAMES,
        decision_indices=decisions,
        values=global_values,
        available=np.ones(global_values.shape, dtype=np.bool_),
        staleness_hours=np.zeros(global_values.shape, dtype=np.float64),
        source_digest=_digest("v4-global"),
    )
    beta = np.ones(2, dtype=np.float64)
    if symbol != "BTCUSDT":
        beta[:] = 0.8
    return V4TargetContext(
        symbol=symbol,
        local=local,
        global_market=global_market,
        beta=beta,
        beta_available=np.ones(2, dtype=np.bool_),
        beta_source_digest=_digest(f"v4-beta:{symbol}"),
        profile_name="cross_market_core_v1",
    )


def _v4_schema_digest(*, kind: str, names: tuple[str, ...]) -> str:
    return content_digest(
        {
            "feature_names": names,
            "kind": kind,
            "schema_version": "causal_alpha_v4_context_feature_schema_v1",
        }
    )


def _write_v4_generation(
    path: Path,
    *,
    base: UniversalRuntimeManifest,
    base_digest: str | None = None,
) -> tuple[CausalAlphaV4ContextManifest, dict[str, V4TargetContext]]:
    contexts = {symbol: _v4_context(symbol) for symbol in MAINTAINED_SYMBOLS}
    root = path.parent / "contexts"
    for symbol, context in contexts.items():
        write_v4_target_context_artifact(root / symbol, context)
    manifest = CausalAlphaV4ContextManifest(
        base_runtime_manifest_digest=(
            base.manifest_digest if base_digest is None else base_digest
        ),
        profile_name="cross_market_core_v1",
        context_artifact_relpath=Path("contexts"),
        context_digests=tuple(
            (symbol, contexts[symbol].digest) for symbol in MAINTAINED_SYMBOLS
        ),
        local_schema_digest=_v4_schema_digest(
            kind="local_cross_market", names=CROSS_MARKET_CORE_NAMES
        ),
        global_schema_digest=_v4_schema_digest(
            kind="global_market", names=GLOBAL_MARKET_CORE_NAMES
        ),
        pit_flow_profile=None,
        source_capability_digest=_digest("v4-capability"),
    )
    write_causal_alpha_v4_context_manifest(path, manifest)
    return manifest, contexts


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


def _patch_static_runtime_inputs(
    module: object,
    monkeypatch: pytest.MonkeyPatch,
    manifest: UniversalRuntimeManifest,
) -> None:
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
        module,
        "bind_universal_normalizers",
        lambda *_args, **_kwargs: (object(), object()),
    )


def test_runtime_factory_context_loads_matching_v4_manifest(tmp_path: Path) -> None:
    from trade_rl.workflows.universal_full_research_entrypoint import (
        UniversalRuntimeFactoryContext,
    )

    base = _manifest(tmp_path / "runtime.json")
    v4_manifest, _ = _write_v4_generation(tmp_path / "v4" / "manifest.json", base=base)

    context = UniversalRuntimeFactoryContext(
        runtime_manifest_path=tmp_path / "runtime.json",
        frozen_metadata_root=tmp_path / "frozen",
        v4_context_manifest_path=tmp_path / "v4" / "manifest.json",
    )

    assert context.v4_context_manifest is not None
    assert context.v4_context_manifest.manifest_digest == v4_manifest.manifest_digest
    assert context.v4_context_manifest_path == tmp_path / "v4" / "manifest.json"


def test_runtime_factory_context_rejects_v4_base_manifest_mismatch(
    tmp_path: Path,
) -> None:
    from trade_rl.workflows.universal_full_research_entrypoint import (
        UniversalRuntimeFactoryContext,
    )

    base = _manifest(tmp_path / "runtime.json")
    _write_v4_generation(
        tmp_path / "v4" / "manifest.json",
        base=base,
        base_digest=_digest("wrong-base"),
    )

    with pytest.raises(ValueError, match="base runtime"):
        UniversalRuntimeFactoryContext(
            runtime_manifest_path=tmp_path / "runtime.json",
            frozen_metadata_root=tmp_path / "frozen",
            v4_context_manifest_path=tmp_path / "v4" / "manifest.json",
        )


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
    _patch_static_runtime_inputs(module, monkeypatch, manifest)

    runtimes = [
        module.build_runtime(algorithm=algorithm, run_config=config, context=context)
        for algorithm, config in _configs().items()
    ]

    assert len({runtime.catalog_digest for runtime in runtimes}) == 1
    assert len({runtime.statistics_digest for runtime in runtimes}) == 1
    assert len({runtime.feature_schema_digest for runtime in runtimes}) == 1
    assert len({runtime.training_contract_digest for runtime in runtimes}) == 3
    assert all(
        runtime.routed_environment_factory.v4_context_provider is None
        for runtime in runtimes
    )


def test_concrete_factory_binds_verified_v4_context_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import trade_rl.workflows.binance_universal_runtime as module
    from trade_rl.workflows.universal_full_research_entrypoint import (
        UniversalRuntimeFactoryContext,
    )

    manifest = _manifest(tmp_path / "runtime.json")
    v4_manifest, contexts = _write_v4_generation(
        tmp_path / "v4" / "manifest.json", base=manifest
    )
    context = UniversalRuntimeFactoryContext(
        runtime_manifest_path=tmp_path / "runtime.json",
        frozen_metadata_root=tmp_path / "frozen",
        v4_context_manifest_path=tmp_path / "v4" / "manifest.json",
    )
    _patch_static_runtime_inputs(module, monkeypatch, manifest)

    runtime = module.build_runtime(
        algorithm=FullResearchAlgorithm.PPO,
        run_config=_configs()[FullResearchAlgorithm.PPO],
        context=context,
    )

    provider = runtime.routed_environment_factory.v4_context_provider
    assert provider is not None
    assert tuple(provider.contexts) == MAINTAINED_SYMBOLS
    assert provider.contexts["BTCUSDT"].digest == contexts["BTCUSDT"].digest
    assert provider.contexts["ETHUSDT"].digest == contexts["ETHUSDT"].digest
    assert runtime.routed_environment_factory.v4_context_manifest_digest == (
        v4_manifest.manifest_digest
    )


def test_concrete_factory_rejects_tampered_v4_context_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import trade_rl.workflows.binance_universal_runtime as module
    from trade_rl.workflows.universal_full_research_entrypoint import (
        UniversalRuntimeFactoryContext,
    )

    manifest = _manifest(tmp_path / "runtime.json")
    _write_v4_generation(tmp_path / "v4" / "manifest.json", base=manifest)
    context = UniversalRuntimeFactoryContext(
        runtime_manifest_path=tmp_path / "runtime.json",
        frozen_metadata_root=tmp_path / "frozen",
        v4_context_manifest_path=tmp_path / "v4" / "manifest.json",
    )
    _patch_static_runtime_inputs(module, monkeypatch, manifest)
    arrays = tmp_path / "v4" / "contexts" / "BTCUSDT" / "arrays.npz"
    payload = bytearray(arrays.read_bytes())
    payload[-1] ^= 1
    arrays.write_bytes(payload)

    with pytest.raises(ValueError, match="arrays digest|context"):
        module.build_runtime(
            algorithm=FullResearchAlgorithm.PPO,
            run_config=_configs()[FullResearchAlgorithm.PPO],
            context=context,
        )


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
