"""PostgreSQL-backed Stage A evaluation dataset manifest construction."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import cast

from trade_rl.artifacts.hashing import content_digest
from trade_rl.data.contracts import InstrumentExecutionRule
from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.walk_forward.folds import WalkForwardFold
from trade_rl.integrations.postgres_indicator_artifacts import (
    IndicatorArtifactConnection,
    NativeIndicatorArtifactBundle,
    load_postgres_indicator_artifacts,
)
from trade_rl.integrations.postgres_market_dataset import (
    FUNDING_TABLE,
    KLINE_TABLE,
    NATIVE_TIMEFRAMES,
    build_postgres_market_dataset,
)
from trade_rl.workflows.stage_a_evaluation_dataset_manifest import (
    StageAEvaluationDatasetFold,
    StageAEvaluationDatasetManifest,
    StageAEvaluationDatasetSplit,
    StageAEvaluationDatasetTriplet,
)
from trade_rl.workflows.symbol_disjoint_triplet_manifest import (
    SymbolDisjointTripletManifest,
)


@dataclass(frozen=True, slots=True)
class StageAPostgresEvaluationDatasets:
    """Verified datasets and their immutable Stage A evaluation manifest."""

    manifest: StageAEvaluationDatasetManifest
    datasets: tuple[tuple[str, MarketDataset], ...]

    def __post_init__(self) -> None:
        by_id = dict(self.datasets)
        if len(by_id) != len(self.datasets):
            raise ValueError("Stage A PostgreSQL evaluation dataset IDs must be unique")
        expected_ids = {item.triplet_id for item in self.manifest.triplets}
        if set(by_id) != expected_ids:
            raise ValueError("Stage A PostgreSQL evaluation triplet closure mismatch")
        for item in self.manifest.triplets:
            dataset = by_id[item.triplet_id]
            if dataset.dataset_id != item.dataset_id:
                raise ValueError(
                    "Stage A PostgreSQL evaluation dataset identity mismatch"
                )
            if dataset.n_bars != self.manifest.n_bars:
                raise ValueError("Stage A PostgreSQL evaluation timeline mismatch")

    def dataset_for(
        self, split: StageAEvaluationDatasetSplit, triplet_id: str
    ) -> MarketDataset:
        self.manifest.triplet_for(split, triplet_id)
        for resolved_id, dataset in self.datasets:
            if resolved_id == triplet_id:
                return dataset
        raise ValueError("Stage A PostgreSQL evaluation triplet is not available")


def _evaluation_slots(manifest: SymbolDisjointTripletManifest):
    return tuple(
        slot for split in ("validation", "test") for slot in manifest.slots_for(split)
    )


def _execution_histories_for(
    symbols: tuple[str, ...],
    histories: Mapping[str, Sequence[InstrumentExecutionRule]] | None,
) -> Mapping[str, Sequence[InstrumentExecutionRule]] | None:
    if histories is None:
        return None
    missing = set(symbols) - set(histories)
    if missing:
        raise ValueError(
            f"Stage A PostgreSQL execution histories are missing symbols: {sorted(missing)}"
        )
    return {symbol: tuple(histories[symbol]) for symbol in symbols}


def build_stage_a_postgres_evaluation_datasets(
    connection: IndicatorArtifactConnection,
    *,
    triplet_manifest: SymbolDisjointTripletManifest,
    folds: Sequence[WalkForwardFold],
    start_time: datetime,
    end_time: datetime,
    metadata: Mapping[str, Mapping[str, object]],
    metadata_evidence_digest: str,
    execution_rule_histories: Mapping[str, Sequence[InstrumentExecutionRule]]
    | None = None,
    indicator_bundle: NativeIndicatorArtifactBundle | None = None,
) -> StageAPostgresEvaluationDatasets:
    """Build every declared validation/test triplet over one common timeline."""

    slots = _evaluation_slots(triplet_manifest)
    if not slots:
        raise ValueError("Stage A PostgreSQL evaluation requires declared triplets")
    evaluation_symbols = tuple(
        dict.fromkeys(symbol for slot in slots for symbol in slot.symbols)
    )
    bundle = indicator_bundle or load_postgres_indicator_artifacts(
        connection,
        symbols=evaluation_symbols,
        timeframes=NATIVE_TIMEFRAMES,
    )
    if set(bundle.symbols) != set(evaluation_symbols):
        raise ValueError("Stage A PostgreSQL indicator symbol closure mismatch")

    datasets: list[tuple[str, MarketDataset]] = []
    bindings: list[StageAEvaluationDatasetTriplet] = []
    feature_identity: str | None = None
    for slot in slots:
        split = cast(StageAEvaluationDatasetSplit, slot.split)
        symbols = cast(tuple[str, str, str], slot.symbols)
        dataset = build_postgres_market_dataset(
            connection,
            symbols=symbols,
            symbol_vocabulary=triplet_manifest.universe,
            start_time=start_time,
            end_time=end_time,
            metadata=metadata,
            metadata_evidence_digest=metadata_evidence_digest,
            execution_rule_histories=_execution_histories_for(
                symbols, execution_rule_histories
            ),
            indicator_bundle=bundle.subset(symbols),
            slot_symbols=("SLOT0", "SLOT1", "SLOT2"),
            symbol_triplet_provenance={
                "schema_version": "stage_a_evaluation_triplet_binding_v1",
                "split": split,
                "symbols": symbols,
                "triplet_id": slot.triplet_id,
                "triplet_manifest_digest": triplet_manifest.digest,
            },
        )
        if feature_identity is None:
            feature_identity = dataset.feature_config_digest
        elif dataset.feature_config_digest != feature_identity:
            raise ValueError("Stage A PostgreSQL feature identity drift")
        datasets.append((slot.triplet_id, dataset))
        bindings.append(
            StageAEvaluationDatasetTriplet(
                split=split,
                triplet_id=slot.triplet_id,
                symbols=symbols,
                dataset_id=dataset.dataset_id,
            )
        )
    assert feature_identity is not None

    fold_bindings = tuple(
        StageAEvaluationDatasetFold(
            fold=fold.fold_index,
            configuration_selection=fold.configuration_selection,
            test=fold.test,
        )
        for fold in folds
    )
    source_closure_digest = content_digest(
        {
            "feature_identity": feature_identity,
            "funding_table": FUNDING_TABLE,
            "indicator_cache_id": bundle.cache_id,
            "kline_table": KLINE_TABLE,
            "metadata_evidence_digest": metadata_evidence_digest,
            "range": (start_time.isoformat(), end_time.isoformat()),
            "schema_version": "stage_a_postgres_evaluation_source_closure_v1",
            "triplet_manifest_digest": triplet_manifest.digest,
        }
    )
    manifest = StageAEvaluationDatasetManifest(
        symbol_disjoint_manifest_digest=triplet_manifest.source_manifest_digest,
        symbol_disjoint_triplet_manifest_digest=triplet_manifest.digest,
        source_closure_digest=source_closure_digest,
        source_metadata_evidence_digest=metadata_evidence_digest,
        indicator_cache_id=bundle.cache_id,
        feature_identity=feature_identity,
        timeline_start_time=start_time,
        timeline_end_time=end_time,
        triplets=tuple(bindings),
        folds=fold_bindings,
    )
    return StageAPostgresEvaluationDatasets(
        manifest=manifest,
        datasets=tuple(datasets),
    )


__all__ = [
    "StageAPostgresEvaluationDatasets",
    "build_stage_a_postgres_evaluation_datasets",
]
