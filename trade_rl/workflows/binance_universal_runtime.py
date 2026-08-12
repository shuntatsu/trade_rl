"""Concrete, artifact-verified Binance Universal training runtime."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from trade_rl.artifacts.hashing import content_digest
from trade_rl.data import load_market_dataset_artifact
from trade_rl.integrations.binance import BinanceMarket
from trade_rl.integrations.frozen_binance_metadata import (
    FrozenBinanceExchangeInfoTransport,
)
from trade_rl.integrations.postgres_universal_source import MAINTAINED_SYMBOLS
from trade_rl.rl.training_run_config import TrainingRunConfig
from trade_rl.rl.universal_instrument_context import CausalInstrumentContextProvider
from trade_rl.workflows.binance_metadata_modes import resolve_frozen_snapshot
from trade_rl.workflows.universal_full_research_entrypoint import (
    UniversalRuntimeFactoryContext,
)
from trade_rl.workflows.universal_instrument_artifacts import (
    load_universal_instrument_artifact_bundle,
)
from trade_rl.workflows.universal_normalizer_artifact import (
    load_universal_shared_normalizer,
)
from trade_rl.workflows.universal_research import FullResearchAlgorithm
from trade_rl.workflows.universal_training import bind_universal_normalizers
from trade_rl.workflows.universal_training_runner import (
    UniversalDatasetArtifactEnvironmentFactory,
    UniversalRoutedEnvironmentFactory,
    UniversalTrainingRuntime,
    build_universal_bindings,
    build_universal_instrument_contracts,
    build_universal_training_runtime,
    concrete_action_spec_digest,
    validate_universal_training_config,
)


def _require_static_artifact_closure(
    context: UniversalRuntimeFactoryContext,
) -> tuple[Any, dict[str, Any], dict[str, Path], Any]:
    manifest = context.manifest
    bundle = load_universal_instrument_artifact_bundle(
        context.resolved_instrument_artifact_root
    )
    partition = bundle.partition
    catalog = bundle.catalog
    if (
        catalog.digest != manifest.catalog_digest
        or getattr(catalog, "source_manifest_digest", manifest.source_manifest_digest)
        != manifest.source_manifest_digest
        or partition.digest != manifest.partition_digest
        or partition.symbol_disjoint_manifest_digest != manifest.split_manifest_digest
        or tuple(partition.train_symbols) != manifest.train_symbols
        or tuple(partition.validation_symbols) != manifest.validation_symbols
        or tuple(partition.test_symbols) != manifest.test_symbols
    ):
        raise ValueError("Universal instrument artifact identity mismatch")

    expected_datasets = dict(manifest.dataset_digests)
    paths = {
        symbol: context.resolved_dataset_artifact_root / symbol
        for symbol in manifest.train_symbols
    }
    datasets = {}
    for symbol, path in paths.items():
        dataset = load_market_dataset_artifact(path)
        if (
            tuple(getattr(dataset, "symbols", ())) != (symbol,)
            or getattr(dataset, "dataset_id", None) != expected_datasets[symbol]
        ):
            raise ValueError("Universal dataset artifact identity mismatch")
        datasets[symbol] = dataset

    shared = load_universal_shared_normalizer(context.normalizer_artifact_root)
    if (
        shared.statistics_digest != manifest.statistics_digest
        or shared.feature_schema_digest != manifest.feature_schema_digest
        or shared.catalog_digest != manifest.catalog_digest
        or shared.split_manifest_digest != manifest.split_manifest_digest
        or shared.fold_train_range != manifest.fold_train_range
        or tuple(shared.train_symbols) != manifest.train_symbols
    ):
        raise ValueError("Universal normalizer artifact identity mismatch")
    return bundle, datasets, paths, shared


def build_runtime(
    *,
    algorithm: FullResearchAlgorithm | str,
    run_config: TrainingRunConfig,
    context: UniversalRuntimeFactoryContext,
) -> UniversalTrainingRuntime:
    """Recompose one candidate runtime only after all artifact identities close."""

    FullResearchAlgorithm(algorithm)
    if not isinstance(run_config, TrainingRunConfig):
        raise TypeError("run_config must be a TrainingRunConfig")
    if not isinstance(context, UniversalRuntimeFactoryContext):
        raise TypeError("context must be a UniversalRuntimeFactoryContext")
    validate_universal_training_config(run_config)
    manifest = context.manifest
    if set(
        (*manifest.train_symbols, *manifest.validation_symbols, *manifest.test_symbols)
    ) != set(MAINTAINED_SYMBOLS):
        raise ValueError(
            "Universal runtime manifest maintained symbol identity mismatch"
        )
    bundle, datasets, dataset_paths, shared = _require_static_artifact_closure(context)
    resolution = resolve_frozen_snapshot(
        transport=FrozenBinanceExchangeInfoTransport(context.frozen_metadata_root),
        market=BinanceMarket.USDS_M,
        symbols=MAINTAINED_SYMBOLS,
        start_time=manifest.research_start,
        end_time=manifest.research_end,
    )
    if resolution.evidence_digest != manifest.metadata_evidence_digest:
        raise ValueError("Universal frozen metadata evidence identity mismatch")
    contracts = build_universal_instrument_contracts(
        resolution, train_symbols=manifest.train_symbols
    )
    bindings = build_universal_bindings(
        datasets=datasets,
        contracts=contracts,
        catalog=bundle.catalog,
        train_symbols=manifest.train_symbols,
    )
    provider = CausalInstrumentContextProvider(contracts=contracts)
    normalizers = {
        symbol: bind_universal_normalizers(
            datasets[symbol],
            shared=shared,
            action_spec_digest=concrete_action_spec_digest(run_config.action, symbol),
            action_size=1,
            n_factors=0,
            finite_horizon=True,
            candidate_config_digest=content_digest(
                run_config.candidate_digest_payload()
            ),
        )
        for symbol in manifest.train_symbols
    }
    concrete = UniversalDatasetArtifactEnvironmentFactory(
        dataset_artifact_paths=dataset_paths,
        run_config=run_config,
        normalizers=normalizers,
    )
    routed = UniversalRoutedEnvironmentFactory(
        train_symbols=manifest.train_symbols,
        partition_digest=manifest.partition_digest,
        bindings=bindings,
        concrete_environment_factory=concrete,
        instrument_context_provider=provider,
        training_contract_digest=content_digest({"phase": "runtime-rebind"}),
        run_seed=min(run_config.training.seeds),
    )
    return build_universal_training_runtime(
        train_symbols=manifest.train_symbols,
        catalog_digest=manifest.catalog_digest,
        partition_digest=manifest.partition_digest,
        split_manifest_digest=manifest.split_manifest_digest,
        feature_schema_digest=manifest.feature_schema_digest,
        statistics_digest=manifest.statistics_digest,
        instrument_context_schema_digest=provider.schema_digest,
        routed_environment_factory=routed,
        training=run_config.training,
    )


__all__ = ["build_runtime"]
