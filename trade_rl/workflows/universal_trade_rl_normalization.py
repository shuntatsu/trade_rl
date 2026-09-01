"""U1 normalization orchestration above the RL statistics layer."""

from __future__ import annotations

from collections.abc import Sequence

from trade_rl.data import (
    inspect_published_market_dataset_artifact,
    load_market_dataset_artifact,
)
from trade_rl.data.market import MarketDataset
from trade_rl.rl.universal_normalization import (
    UniversalTradePublishedSource,
    UniversalTradeSequenceNormalizer,
    build_universal_trade_sequence_normalizer,
)
from trade_rl.rl.universal_trade_contract import UniversalTradePolicyContract
from trade_rl.workflows.universal_trade_rl_data_provenance import (
    UniversalTradeRLFitPurpose,
    build_universal_trade_rl_fit_provenance,
)
from trade_rl.workflows.universal_trade_rl_universe_access import (
    UniversalTradeRLAccessPhase,
    UniversalTradeRLUniverseAccess,
)
from trade_rl.workflows.universal_trade_rl_universe_manifest import (
    UniversalTradeRLUniverseManifest,
)


def _canonical_sources(
    sources: Sequence[UniversalTradePublishedSource],
) -> tuple[UniversalTradePublishedSource, ...]:
    resolved = tuple(sources)
    if not resolved:
        raise ValueError("Universal Trade normalization sources must not be empty")
    if any(not isinstance(source, UniversalTradePublishedSource) for source in resolved):
        raise TypeError("Universal Trade normalization source contract is invalid")
    ordered = tuple(sorted(resolved, key=lambda source: source.symbol))
    symbols = tuple(source.symbol for source in ordered)
    if len(set(symbols)) != len(symbols):
        raise ValueError("Universal Trade normalization source symbols must be unique")
    return ordered


def fit_universal_trade_sequence_normalizer(
    *,
    manifest: UniversalTradeRLUniverseManifest,
    access: UniversalTradeRLUniverseAccess,
    sources: Sequence[UniversalTradePublishedSource],
    contract: UniversalTradePolicyContract,
    knowledge_cutoff_ns: int,
) -> UniversalTradeSequenceNormalizer:
    """Fit one U0-bound U1 normalizer from the complete Train universe only."""

    if access.phase is not UniversalTradeRLAccessPhase.TRAIN:
        raise PermissionError("Universal Trade RL normalization fitting is Train-only")

    ordered_sources = _canonical_sources(sources)
    source_symbols = tuple(source.symbol for source in ordered_sources)
    scope = access.require_normalization_scope(source_symbols)
    expected_train_symbols = tuple(sorted(access.train_symbols))
    if scope != expected_train_symbols:
        raise ValueError(
            "Universal Trade RL normalization sources must exactly match U0 Train symbols"
        )

    provenance = build_universal_trade_rl_fit_provenance(
        manifest=manifest,
        access=access,
        purpose=UniversalTradeRLFitPurpose.FEATURE_NORMALIZATION,
        source_symbols=scope,
        knowledge_cutoff=knowledge_cutoff_ns,
    )
    expected_digests = dict(provenance.source_dataset_digests)
    expected_feature_names = tuple(spec.name for spec in contract.feature_specs)
    symbol_datasets: dict[str, MarketDataset] = {}
    for source in ordered_sources:
        artifact = inspect_published_market_dataset_artifact(source.artifact_root)
        if artifact.artifact_digest != expected_digests[source.symbol]:
            raise ValueError(
                "published market dataset artifact digest mismatch for "
                f"{source.symbol}"
            )
        dataset = load_market_dataset_artifact(source.artifact_root)
        if dataset.n_symbols != 1 or dataset.symbols != (source.symbol,):
            raise ValueError(
                "Universal Trade RL normalization requires one matching symbol per "
                "published dataset"
            )
        if dataset.feature_names != expected_feature_names:
            raise ValueError(
                "Universal Trade RL normalization feature order does not match U1 "
                "contract"
            )
        symbol_datasets[source.symbol] = dataset

    return build_universal_trade_sequence_normalizer(
        symbol_datasets=symbol_datasets,
        contract=contract,
        source_dataset_digests=provenance.source_dataset_digests,
        knowledge_cutoff_ns=provenance.knowledge_cutoff,
        universe_manifest_digest=manifest.digest,
        provenance_digest=provenance.digest,
    )


__all__ = ["fit_universal_trade_sequence_normalizer"]
