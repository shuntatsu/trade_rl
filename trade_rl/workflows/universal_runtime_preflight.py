"""Materialize and close every static input for Universal U3-U6 training."""

from __future__ import annotations

from datetime import UTC, datetime
from functools import partial
from pathlib import Path

from trade_rl.artifacts.hashing import content_digest
from trade_rl.data import load_market_dataset_artifact
from trade_rl.integrations.binance import BinanceMarket
from trade_rl.integrations.binance_universal import binance_universal_feature_specs
from trade_rl.integrations.frozen_binance_metadata import (
    FrozenBinanceExchangeInfoTransport,
)
from trade_rl.integrations.postgres_indicator_artifacts import (
    IndicatorArtifactConnection,
    load_postgres_indicator_artifacts,
)
from trade_rl.integrations.postgres_market_dataset import build_postgres_market_dataset
from trade_rl.integrations.postgres_market_tables import (
    UNIVERSAL_202411_202607_CACHE_ID,
    UNIVERSAL_202411_202607_TABLES,
)
from trade_rl.integrations.postgres_universal_source import MAINTAINED_SYMBOLS
from trade_rl.workflows.binance_metadata_modes import resolve_frozen_snapshot
from trade_rl.workflows.postgres_universal_instrument_artifacts import (
    materialize_postgres_universal_instrument_artifacts,
)
from trade_rl.workflows.universal_instrument_artifacts import (
    load_universal_instrument_artifact_bundle,
)
from trade_rl.workflows.universal_normalizer_artifact import (
    load_universal_shared_normalizer,
    write_universal_shared_normalizer,
)
from trade_rl.workflows.universal_runtime_manifest import (
    UniversalRuntimeManifest,
    load_universal_runtime_manifest,
    write_universal_runtime_manifest,
)
from trade_rl.workflows.universal_training import (
    fit_universal_shared_normalizer,
    materialize_universal_train_datasets,
)
from trade_rl.workflows.universal_training_runner import (
    publish_universal_train_dataset_artifacts,
)

RESEARCH_START = datetime(2024, 11, 13, tzinfo=UTC)
RESEARCH_END = datetime(2026, 7, 5, tzinfo=UTC)
PARTITION_SEED = 17


def _relative_to_manifest(path: Path, *, manifest_path: Path, field: str) -> Path:
    base = manifest_path.parent.resolve()
    target = path.resolve()
    try:
        return target.relative_to(base)
    except ValueError as error:
        raise ValueError(f"{field} must be below the runtime manifest directory") from error


def materialize_universal_runtime_inputs(
    *,
    connection: IndicatorArtifactConnection,
    frozen_metadata_root: str | Path,
    instrument_artifact_root: str | Path,
    dataset_artifact_root: str | Path,
    normalizer_artifact_root: str | Path,
    runtime_manifest_path: str | Path,
) -> UniversalRuntimeManifest:
    """Build, publish, reload, and bind all train-only runtime inputs."""

    instrument_root = Path(instrument_artifact_root)
    dataset_root = Path(dataset_artifact_root)
    normalizer_root = Path(normalizer_artifact_root)
    manifest_path = Path(runtime_manifest_path)
    resolution = resolve_frozen_snapshot(
        transport=FrozenBinanceExchangeInfoTransport(Path(frozen_metadata_root)),
        market=BinanceMarket.USDS_M,
        symbols=MAINTAINED_SYMBOLS,
        start_time=RESEARCH_START,
        end_time=RESEARCH_END,
    )
    metadata_digests = {
        symbol: content_digest(dict(resolution.metadata[symbol]))
        for symbol in MAINTAINED_SYMBOLS
    }
    materialize_postgres_universal_instrument_artifacts(
        connection,
        output_dir=instrument_root,
        research_start=RESEARCH_START,
        research_end=RESEARCH_END,
        metadata_digests=metadata_digests,
        seed=PARTITION_SEED,
        cache_id=UNIVERSAL_202411_202607_CACHE_ID,
        tables=UNIVERSAL_202411_202607_TABLES,
    )
    instrument_bundle = load_universal_instrument_artifact_bundle(instrument_root)
    partition = instrument_bundle.partition
    catalog = instrument_bundle.catalog
    feature_specs = binance_universal_feature_specs(
        base_timeframe="15m", feature_timeframes=("1h", "4h", "1d")
    )
    datasets = materialize_universal_train_datasets(
        connection,
        instrument_bundle=instrument_bundle,
        metadata_resolution=resolution,
        feature_specs=feature_specs,
        indicator_loader=partial(
            load_postgres_indicator_artifacts,
            cache_id=UNIVERSAL_202411_202607_CACHE_ID,
            tables=UNIVERSAL_202411_202607_TABLES,
        ),
        dataset_builder=partial(
            build_postgres_market_dataset,
            tables=UNIVERSAL_202411_202607_TABLES,
        ),
    )
    train_symbols = tuple(partition.train_symbols)
    if tuple(datasets) != train_symbols:
        raise ValueError("preflight datasets must follow train_symbols exactly")
    shared_count = min(int(dataset.n_bars) for dataset in datasets.values())
    fold_range = (0, shared_count)
    normalizer = fit_universal_shared_normalizer(
        datasets,
        train_symbols=train_symbols,
        catalog_digest=catalog.digest,
        split_manifest_digest=partition.symbol_disjoint_manifest_digest,
        fold_train_range=fold_range,
    )
    dataset_paths = publish_universal_train_dataset_artifacts(
        datasets,
        train_symbols=train_symbols,
        artifact_root=dataset_root,
    )
    dataset_digests: list[tuple[str, str]] = []
    for symbol in train_symbols:
        loaded = load_market_dataset_artifact(dataset_paths[symbol])
        dataset_id = getattr(loaded, "dataset_id", None)
        if tuple(getattr(loaded, "symbols", ())) != (symbol,) or dataset_id != getattr(
            datasets[symbol], "dataset_id", None
        ):
            raise ValueError("preflight dataset artifact identity mismatch")
        if not isinstance(dataset_id, str):
            raise ValueError("preflight dataset artifact digest is unavailable")
        dataset_digests.append((symbol, dataset_id))
    write_universal_shared_normalizer(normalizer_root, normalizer)
    loaded_normalizer = load_universal_shared_normalizer(normalizer_root)
    if (
        loaded_normalizer.statistics_digest != normalizer.statistics_digest
        or loaded_normalizer.feature_schema_digest != normalizer.feature_schema_digest
    ):
        raise ValueError("preflight shared normalizer identity mismatch")
    manifest = UniversalRuntimeManifest(
        cache_id=UNIVERSAL_202411_202607_CACHE_ID,
        tables=UNIVERSAL_202411_202607_TABLES,
        research_start=RESEARCH_START,
        research_end=RESEARCH_END,
        instrument_artifact_relpath=_relative_to_manifest(
            instrument_root,
            manifest_path=manifest_path,
            field="instrument_artifact_root",
        ),
        dataset_artifact_relpath=_relative_to_manifest(
            dataset_root,
            manifest_path=manifest_path,
            field="dataset_artifact_root",
        ),
        normalizer_artifact_relpath=_relative_to_manifest(
            normalizer_root,
            manifest_path=manifest_path,
            field="normalizer_artifact_root",
        ),
        train_symbols=train_symbols,
        validation_symbols=tuple(partition.validation_symbols),
        test_symbols=tuple(partition.test_symbols),
        fold_train_range=fold_range,
        shared_complete_row_count=shared_count,
        catalog_digest=catalog.digest,
        partition_digest=partition.digest,
        split_manifest_digest=partition.symbol_disjoint_manifest_digest,
        feature_schema_digest=normalizer.feature_schema_digest,
        statistics_digest=normalizer.statistics_digest,
        metadata_evidence_digest=resolution.evidence_digest,
        source_manifest_digest=catalog.source_manifest_digest,
        dataset_digests=tuple(dataset_digests),
    )
    write_universal_runtime_manifest(manifest_path, manifest)
    loaded_manifest = load_universal_runtime_manifest(manifest_path)
    if loaded_manifest != manifest:
        raise ValueError("preflight runtime manifest round-trip mismatch")
    return loaded_manifest


__all__ = [
    "PARTITION_SEED",
    "RESEARCH_END",
    "RESEARCH_START",
    "materialize_universal_runtime_inputs",
]
