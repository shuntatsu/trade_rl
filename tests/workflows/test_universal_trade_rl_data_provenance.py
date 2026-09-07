from __future__ import annotations

from dataclasses import replace

import pytest

from trade_rl.domain.universal_trade_rl_universe import UniversalTradeRLUniverseConfig
from trade_rl.workflows.universal_trade_rl_data_provenance import (
    UniversalTradeRLFitProvenance,
    UniversalTradeRLFitPurpose,
    build_universal_trade_rl_fit_provenance,
    require_universal_trade_rl_train_only_provenance,
)
from trade_rl.workflows.universal_trade_rl_universe_access import (
    UniversalTradeRLAccessPhase,
    UniversalTradeRLAdmissionAuthorization,
    UniversalTradeRLUniverseAccess,
)
from trade_rl.workflows.universal_trade_rl_universe_config import (
    UniversalTradeRLSymbolSource,
)
from trade_rl.workflows.universal_trade_rl_universe_manifest import (
    UniversalTradeRLUniverseManifest,
    build_universal_trade_rl_universe_manifest,
)


def _manifest(*, btc_digest: str = "b" * 64) -> UniversalTradeRLUniverseManifest:
    config = UniversalTradeRLUniverseConfig(
        train_symbols=("BTCUSDT", "ETHUSDT"),
        development_symbols=("LINKUSDT",),
        admission_symbols=("AVAXUSDT",),
    )
    sources = (
        UniversalTradeRLSymbolSource(
            symbol="AVAXUSDT",
            dataset_digest="a" * 64,
            first_timestamp_ns=1,
            last_timestamp_ns=100,
            row_count=100,
        ),
        UniversalTradeRLSymbolSource(
            symbol="BTCUSDT",
            dataset_digest=btc_digest,
            first_timestamp_ns=1,
            last_timestamp_ns=100,
            row_count=100,
        ),
        UniversalTradeRLSymbolSource(
            symbol="ETHUSDT",
            dataset_digest="c" * 64,
            first_timestamp_ns=1,
            last_timestamp_ns=100,
            row_count=100,
        ),
        UniversalTradeRLSymbolSource(
            symbol="LINKUSDT",
            dataset_digest="d" * 64,
            first_timestamp_ns=1,
            last_timestamp_ns=100,
            row_count=100,
        ),
    )
    return build_universal_trade_rl_universe_manifest(config=config, sources=sources)


def _access(
    manifest: UniversalTradeRLUniverseManifest,
    phase: UniversalTradeRLAccessPhase,
) -> UniversalTradeRLUniverseAccess:
    if phase is not UniversalTradeRLAccessPhase.ADMISSION:
        return UniversalTradeRLUniverseAccess.for_phase(
            manifest=manifest,
            phase=phase,
        )
    authorization = UniversalTradeRLAdmissionAuthorization(
        universe_manifest_digest=manifest.digest,
        frozen_generation_digest="f" * 64,
        selection_evidence_digest="e" * 64,
    )
    return UniversalTradeRLUniverseAccess.for_phase(
        manifest=manifest,
        phase=phase,
        authorization=authorization,
        frozen_generation_digest="f" * 64,
        selection_evidence_digest="e" * 64,
    )


def test_normalization_provenance_is_train_only_and_source_bound() -> None:
    manifest = _manifest()
    evidence = build_universal_trade_rl_fit_provenance(
        manifest=manifest,
        access=_access(manifest, UniversalTradeRLAccessPhase.TRAIN),
        purpose=UniversalTradeRLFitPurpose.FEATURE_NORMALIZATION,
        source_symbols=("BTCUSDT", "ETHUSDT"),
        knowledge_cutoff=10_000,
    )

    assert evidence.source_symbols == ("BTCUSDT", "ETHUSDT")
    assert evidence.source_dataset_digests == (
        ("BTCUSDT", "b" * 64),
        ("ETHUSDT", "c" * 64),
    )
    assert evidence.universe_manifest_digest == manifest.digest
    assert evidence.knowledge_cutoff == 10_000
    assert len(evidence.digest) == 64
    require_universal_trade_rl_train_only_provenance(evidence, manifest=manifest)


def test_calibration_rejects_development_symbol() -> None:
    manifest = _manifest()
    with pytest.raises(PermissionError, match="Train-only"):
        build_universal_trade_rl_fit_provenance(
            manifest=manifest,
            access=_access(manifest, UniversalTradeRLAccessPhase.DEVELOPMENT),
            purpose=UniversalTradeRLFitPurpose.CALIBRATION,
            source_symbols=("BTCUSDT", "LINKUSDT"),
            knowledge_cutoff=10_000,
        )


def test_fit_scope_is_checked_before_manifest_source_resolution() -> None:
    manifest = _manifest()
    with pytest.raises(PermissionError, match="Train-only"):
        build_universal_trade_rl_fit_provenance(
            manifest=manifest,
            access=_access(manifest, UniversalTradeRLAccessPhase.DEVELOPMENT),
            purpose=UniversalTradeRLFitPurpose.FORECAST_FIT,
            source_symbols=("ZZZUSDT",),
            knowledge_cutoff=10_000,
        )


@pytest.mark.parametrize("purpose", tuple(UniversalTradeRLFitPurpose))
def test_admission_rejects_every_fit_purpose(
    purpose: UniversalTradeRLFitPurpose,
) -> None:
    manifest = _manifest()
    with pytest.raises(PermissionError, match="Train-only"):
        build_universal_trade_rl_fit_provenance(
            manifest=manifest,
            access=_access(manifest, UniversalTradeRLAccessPhase.ADMISSION),
            purpose=purpose,
            source_symbols=("BTCUSDT",),
            knowledge_cutoff=10_000,
        )


def test_provenance_from_another_generation_is_rejected_even_for_same_symbols() -> None:
    manifest_a = _manifest(btc_digest="b" * 64)
    manifest_b = _manifest(btc_digest="f" * 64)
    evidence = build_universal_trade_rl_fit_provenance(
        manifest=manifest_a,
        access=_access(manifest_a, UniversalTradeRLAccessPhase.TRAIN),
        purpose=UniversalTradeRLFitPurpose.RL_TRAINING,
        source_symbols=("BTCUSDT", "ETHUSDT"),
        knowledge_cutoff=10_000,
    )

    with pytest.raises(ValueError, match="universe|source"):
        require_universal_trade_rl_train_only_provenance(
            evidence,
            manifest=manifest_b,
        )


def test_source_dataset_drift_changes_provenance_digest() -> None:
    manifest_a = _manifest(btc_digest="b" * 64)
    manifest_b = _manifest(btc_digest="f" * 64)
    evidence_a = build_universal_trade_rl_fit_provenance(
        manifest=manifest_a,
        access=_access(manifest_a, UniversalTradeRLAccessPhase.TRAIN),
        purpose=UniversalTradeRLFitPurpose.FORECAST_FIT,
        source_symbols=("BTCUSDT", "ETHUSDT"),
        knowledge_cutoff=10_000,
    )
    evidence_b = build_universal_trade_rl_fit_provenance(
        manifest=manifest_b,
        access=_access(manifest_b, UniversalTradeRLAccessPhase.TRAIN),
        purpose=UniversalTradeRLFitPurpose.FORECAST_FIT,
        source_symbols=("BTCUSDT", "ETHUSDT"),
        knowledge_cutoff=10_000,
    )

    assert evidence_a.digest != evidence_b.digest
    assert evidence_a.source_dataset_digests != evidence_b.source_dataset_digests


def test_validator_reconstructs_source_identity_instead_of_trusting_claims() -> None:
    manifest = _manifest()
    evidence = build_universal_trade_rl_fit_provenance(
        manifest=manifest,
        access=_access(manifest, UniversalTradeRLAccessPhase.TRAIN),
        purpose=UniversalTradeRLFitPurpose.POPULATION_THRESHOLD_FIT,
        source_symbols=("BTCUSDT",),
        knowledge_cutoff=10_000,
    )
    forged = replace(
        evidence,
        source_dataset_digests=(("BTCUSDT", "f" * 64),),
        digest="",
    )

    with pytest.raises(ValueError, match="source"):
        require_universal_trade_rl_train_only_provenance(
            forged,
            manifest=manifest,
        )


def test_builder_rejects_manifest_access_generation_mismatch() -> None:
    manifest_a = _manifest(btc_digest="b" * 64)
    manifest_b = _manifest(btc_digest="f" * 64)

    with pytest.raises(ValueError, match="universe"):
        build_universal_trade_rl_fit_provenance(
            manifest=manifest_b,
            access=_access(manifest_a, UniversalTradeRLAccessPhase.TRAIN),
            purpose=UniversalTradeRLFitPurpose.REWARD_COEFFICIENT_FIT,
            source_symbols=("BTCUSDT",),
            knowledge_cutoff=10_000,
        )


@pytest.mark.parametrize("knowledge_cutoff", (0, -1, True))
def test_provenance_rejects_non_positive_or_boolean_knowledge_cutoff(
    knowledge_cutoff: object,
) -> None:
    manifest = _manifest()

    with pytest.raises(ValueError, match="knowledge cutoff"):
        build_universal_trade_rl_fit_provenance(
            manifest=manifest,
            access=_access(manifest, UniversalTradeRLAccessPhase.TRAIN),
            purpose=UniversalTradeRLFitPurpose.RL_TRAINING,
            source_symbols=("BTCUSDT",),
            knowledge_cutoff=knowledge_cutoff,  # type: ignore[arg-type]
        )


def test_provenance_rejects_unsorted_or_duplicate_sources() -> None:
    manifest = _manifest()
    access = _access(manifest, UniversalTradeRLAccessPhase.TRAIN)

    with pytest.raises(ValueError, match="sorted"):
        build_universal_trade_rl_fit_provenance(
            manifest=manifest,
            access=access,
            purpose=UniversalTradeRLFitPurpose.RL_TRAINING,
            source_symbols=("ETHUSDT", "BTCUSDT"),
            knowledge_cutoff=10_000,
        )
    with pytest.raises(ValueError, match="unique"):
        build_universal_trade_rl_fit_provenance(
            manifest=manifest,
            access=access,
            purpose=UniversalTradeRLFitPurpose.RL_TRAINING,
            source_symbols=("BTCUSDT", "BTCUSDT"),
            knowledge_cutoff=10_000,
        )


def test_provenance_schema_and_direct_contract_are_strict() -> None:
    with pytest.raises(ValueError, match="source"):
        UniversalTradeRLFitProvenance(
            purpose=UniversalTradeRLFitPurpose.RL_TRAINING,
            universe_manifest_digest="a" * 64,
            source_symbols=("BTCUSDT",),
            source_dataset_digests=(("ETHUSDT", "b" * 64),),
            knowledge_cutoff=10_000,
        )
