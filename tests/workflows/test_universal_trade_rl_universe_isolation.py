from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest
from hypothesis import given, settings, strategies as st

from trade_rl.domain.universal_trade_rl_universe import (
    UniversalTradeRLSymbolExclusion,
    UniversalTradeRLSymbolRole,
    UniversalTradeRLUniverseConfig,
)
from trade_rl.workflows.universal_trade_rl_data_provenance import (
    UniversalTradeRLFitPurpose,
    build_universal_trade_rl_fit_provenance,
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

_FROZEN_GENERATION_DIGEST = hashlib.sha256(b"frozen-generation").hexdigest()
_SELECTION_EVIDENCE_DIGEST = hashlib.sha256(b"selection-evidence").hexdigest()
_RESERVED = {
    "AVAXUSDT",
    "BTCUSDT",
    "ETHUSDT",
    "LINKUSDT",
    "LUNA2USDT",
}
_SYMBOL_POOL = (
    "ADAUSDT",
    "ATOMUSDT",
    "BNBUSDT",
    "DOGEUSDT",
    "DOTUSDT",
    "NEARUSDT",
    "SOLUSDT",
    "TRXUSDT",
    "UNIUSDT",
    "XRPUSDT",
)
_SYMBOL = st.sampled_from(_SYMBOL_POOL)
_SYMBOL_PAIR = st.lists(_SYMBOL, min_size=2, max_size=2, unique=True).map(tuple)
_PURPOSE = st.sampled_from(tuple(UniversalTradeRLFitPurpose))


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _source(symbol: str, *, generation: str = "base") -> UniversalTradeRLSymbolSource:
    return UniversalTradeRLSymbolSource(
        symbol=symbol,
        dataset_digest=_digest(f"{symbol}:{generation}"),
        first_timestamp_ns=1,
        last_timestamp_ns=100,
        row_count=100,
    )


def _config(
    *,
    development_symbols: tuple[str, ...] = ("LINKUSDT",),
    admission_symbols: tuple[str, ...] = ("AVAXUSDT",),
) -> UniversalTradeRLUniverseConfig:
    return UniversalTradeRLUniverseConfig(
        train_symbols=("BTCUSDT", "ETHUSDT"),
        development_symbols=tuple(sorted(development_symbols)),
        admission_symbols=tuple(sorted(admission_symbols)),
        exclusions=(
            UniversalTradeRLSymbolExclusion(
                symbol="LUNA2USDT",
                reason="insufficient_contiguous_history",
            ),
        ),
    )


def _sources_for(
    config: UniversalTradeRLUniverseConfig,
    *,
    extra_symbols: tuple[str, ...] = (),
) -> tuple[UniversalTradeRLSymbolSource, ...]:
    symbols = {
        *config.train_symbols,
        *config.development_symbols,
        *config.admission_symbols,
        *(item.symbol for item in config.exclusions),
        *extra_symbols,
    }
    return tuple(_source(symbol) for symbol in sorted(symbols))


def _manifest(
    *,
    development_symbols: tuple[str, ...] = ("LINKUSDT",),
    admission_symbols: tuple[str, ...] = ("AVAXUSDT",),
) -> UniversalTradeRLUniverseManifest:
    config = _config(
        development_symbols=development_symbols,
        admission_symbols=admission_symbols,
    )
    return build_universal_trade_rl_universe_manifest(
        config=config,
        sources=_sources_for(config),
    )


def _development_access(
    manifest: UniversalTradeRLUniverseManifest,
) -> UniversalTradeRLUniverseAccess:
    return UniversalTradeRLUniverseAccess.for_phase(
        manifest=manifest,
        phase=UniversalTradeRLAccessPhase.DEVELOPMENT,
    )


def _admission_access(
    manifest: UniversalTradeRLUniverseManifest,
) -> UniversalTradeRLUniverseAccess:
    authorization = UniversalTradeRLAdmissionAuthorization(
        universe_manifest_digest=manifest.digest,
        frozen_generation_digest=_FROZEN_GENERATION_DIGEST,
        selection_evidence_digest=_SELECTION_EVIDENCE_DIGEST,
    )
    return UniversalTradeRLUniverseAccess.for_phase(
        manifest=manifest,
        phase=UniversalTradeRLAccessPhase.ADMISSION,
        authorization=authorization,
        frozen_generation_digest=_FROZEN_GENERATION_DIGEST,
        selection_evidence_digest=_SELECTION_EVIDENCE_DIGEST,
    )


def test_admission_metadata_can_be_bound_but_not_fit() -> None:
    manifest = _manifest()
    admission = manifest.entry_for("AVAXUSDT")
    access = _development_access(manifest)

    assert admission.role is UniversalTradeRLSymbolRole.ADMISSION
    assert len(admission.dataset_digest) == 64
    assert admission.row_count == 100
    with pytest.raises(PermissionError, match="Train-only"):
        build_universal_trade_rl_fit_provenance(
            manifest=manifest,
            access=access,
            purpose=UniversalTradeRLFitPurpose.FEATURE_NORMALIZATION,
            source_symbols=(admission.symbol,),
            knowledge_cutoff=50,
        )


def test_existing_fit_boundary_receives_only_train_symbols() -> None:
    manifest = _manifest()
    access = _development_access(manifest)
    observed: list[tuple[str, ...]] = []

    def fit_spy(*, train_symbols: tuple[str, ...]) -> None:
        observed.append(train_symbols)

    fit_spy(train_symbols=access.fit_symbols)

    assert observed == [("BTCUSDT", "ETHUSDT")]
    assert "LINKUSDT" not in observed[0]
    assert "AVAXUSDT" not in observed[0]


@settings(max_examples=100, deadline=None)
@given(shared=_SYMBOL)
def test_role_overlap_is_always_rejected(shared: str) -> None:
    assert shared not in _RESERVED
    with pytest.raises(ValueError, match="pairwise disjoint"):
        UniversalTradeRLUniverseConfig(
            train_symbols=tuple(sorted(("BTCUSDT", shared))),
            development_symbols=(shared,),
            admission_symbols=("AVAXUSDT",),
        )


@settings(max_examples=100, deadline=None)
@given(extra=_SYMBOL)
def test_unassigned_available_source_is_always_rejected(extra: str) -> None:
    assert extra not in _RESERVED
    config = _config()

    with pytest.raises(ValueError, match="unassigned available symbol"):
        build_universal_trade_rl_universe_manifest(
            config=config,
            sources=_sources_for(config, extra_symbols=(extra,)),
        )


@settings(max_examples=100, deadline=None)
@given(symbols=_SYMBOL_PAIR)
def test_role_reassignment_changes_only_role_bound_identity(
    symbols: tuple[str, ...],
) -> None:
    development, admission = symbols
    baseline_config = _config(
        development_symbols=(development,),
        admission_symbols=(admission,),
    )
    swapped_config = _config(
        development_symbols=(admission,),
        admission_symbols=(development,),
    )
    sources = _sources_for(baseline_config)
    baseline = build_universal_trade_rl_universe_manifest(
        config=baseline_config,
        sources=sources,
    )
    swapped = build_universal_trade_rl_universe_manifest(
        config=swapped_config,
        sources=sources,
    )

    assert baseline.source_catalog_digest == swapped.source_catalog_digest
    assert baseline.config_digest != swapped.config_digest
    assert baseline.digest != swapped.digest


@settings(max_examples=100, deadline=None)
@given(symbol=st.sampled_from(tuple(sorted(_RESERVED))))
def test_source_identity_mutation_changes_only_source_bound_identity(symbol: str) -> None:
    config = _config()
    sources = _sources_for(config)
    mutated_sources = tuple(
        replace(source, dataset_digest=_digest(f"{source.symbol}:mutated"))
        if source.symbol == symbol
        else source
        for source in sources
    )
    baseline = build_universal_trade_rl_universe_manifest(
        config=config,
        sources=sources,
    )
    mutated = build_universal_trade_rl_universe_manifest(
        config=config,
        sources=mutated_sources,
    )

    assert baseline.config_digest == mutated.config_digest
    assert baseline.source_catalog_digest != mutated.source_catalog_digest
    assert baseline.digest != mutated.digest


@settings(max_examples=100, deadline=None)
@given(admission_symbol=_SYMBOL, purpose=_PURPOSE)
def test_admission_can_never_enter_any_fit_purpose(
    admission_symbol: str,
    purpose: UniversalTradeRLFitPurpose,
) -> None:
    assert admission_symbol not in _RESERVED
    manifest = _manifest(admission_symbols=(admission_symbol,))
    access = _admission_access(manifest)

    with pytest.raises(PermissionError, match="Train-only"):
        access.require_fit_scope((admission_symbol,))
    with pytest.raises(PermissionError, match="Train-only"):
        build_universal_trade_rl_fit_provenance(
            manifest=manifest,
            access=access,
            purpose=purpose,
            source_symbols=(admission_symbol,),
            knowledge_cutoff=50,
        )
