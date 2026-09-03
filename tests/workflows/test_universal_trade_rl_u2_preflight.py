from __future__ import annotations

import importlib
from dataclasses import dataclass, replace

import numpy as np
import pytest

from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.universal_trade_rl_universe import (
    UniversalTradeRLSymbolRole,
    UniversalTradeRLUniverseConfig,
)
from trade_rl.rl.universal_normalization import (
    UniversalTradeChannelStatistics,
    UniversalTradeSequenceNormalizer,
)
from trade_rl.workflows.universal_trade_rl_data_provenance import (
    UniversalTradeRLFitProvenance,
    UniversalTradeRLFitPurpose,
    build_universal_trade_rl_fit_provenance,
)
from trade_rl.workflows.universal_trade_rl_u1_contract import UniversalTradeRLU1Contract
from trade_rl.workflows.universal_trade_rl_u2_contract import (
    UniversalTradeRLU2Contract,
    build_universal_trade_rl_u2_contract,
    build_universal_trade_rl_u2_training_config,
)
from trade_rl.workflows.universal_trade_rl_u2_time_partition import (
    UniversalTradeRLU2TimePartition,
    build_universal_trade_rl_u2_time_partition,
)
from trade_rl.workflows.universal_trade_rl_universe_access import (
    UniversalTradeRLAccessPhase,
    UniversalTradeRLUniverseAccess,
)
from trade_rl.workflows.universal_trade_rl_universe_config import (
    UniversalTradeRLSymbolSource,
)
from trade_rl.workflows.universal_trade_rl_universe_manifest import (
    UniversalTradeRLUniverseManifest,
    build_universal_trade_rl_universe_manifest,
)

_STEP_NS = 15 * 60 * 1_000_000_000
_BARS_PER_DAY = 96
_START_NS = _STEP_NS * 2_000_000
_TOTAL_BARS = 620 * _BARS_PER_DAY
_POLICY_CONTRACT_DIGEST = "7" * 64
_NORMALIZER_VERSION = "universal_trade_sequence_normalizer_v1"
_NORMALIZER_SEMANTICS = "equal_symbol_source_event_moments_v1"


@dataclass(frozen=True, slots=True)
class U2PreflightFixture:
    manifest: UniversalTradeRLUniverseManifest
    time_partition: UniversalTradeRLU2TimePartition
    normalizer_provenance: UniversalTradeRLFitProvenance
    normalizer: UniversalTradeSequenceNormalizer
    u1_contract: UniversalTradeRLU1Contract
    rl_training_provenance: UniversalTradeRLFitProvenance
    u2_contract: UniversalTradeRLU2Contract


class SpyBoundedLoader:
    def __init__(self) -> None:
        self.calls: list[object] = []

    def __call__(self, request: object) -> object:
        self.calls.append(request)
        return request


def _module():
    try:
        return importlib.import_module(
            "trade_rl.workflows.universal_trade_rl_u2_preflight"
        )
    except ModuleNotFoundError:
        pytest.fail("Universal Trade RL U2 training preflight is not implemented")


def _manifest(*, btc_digest_char: str = "a") -> UniversalTradeRLUniverseManifest:
    config = UniversalTradeRLUniverseConfig(
        train_symbols=("BTCUSDT", "ETHUSDT"),
        development_symbols=("SOLUSDT",),
        admission_symbols=("XRPUSDT",),
    )
    sources = tuple(
        UniversalTradeRLSymbolSource(
            symbol=symbol,
            dataset_digest=digest_char * 64,
            first_timestamp_ns=_START_NS,
            last_timestamp_ns=_START_NS + (_TOTAL_BARS - 1) * _STEP_NS,
            row_count=_TOTAL_BARS,
        )
        for symbol, digest_char in (
            ("BTCUSDT", btc_digest_char),
            ("ETHUSDT", "b"),
            ("SOLUSDT", "c"),
            ("XRPUSDT", "d"),
        )
    )
    return build_universal_trade_rl_universe_manifest(config=config, sources=sources)


def _fit_provenance(
    *,
    manifest: UniversalTradeRLUniverseManifest,
    purpose: UniversalTradeRLFitPurpose,
    cutoff_ns: int,
) -> UniversalTradeRLFitProvenance:
    access = UniversalTradeRLUniverseAccess.for_phase(
        manifest=manifest,
        phase=UniversalTradeRLAccessPhase.TRAIN,
    )
    return build_universal_trade_rl_fit_provenance(
        manifest=manifest,
        access=access,
        purpose=purpose,
        source_symbols=("BTCUSDT", "ETHUSDT"),
        knowledge_cutoff=cutoff_ns,
    )


def _normalizer(
    *,
    manifest: UniversalTradeRLUniverseManifest,
    provenance: UniversalTradeRLFitProvenance,
    cutoff_ns: int,
) -> UniversalTradeSequenceNormalizer:
    train_entries = tuple(
        entry
        for entry in manifest.entries
        if entry.role is UniversalTradeRLSymbolRole.TRAIN
    )
    train_symbols = tuple(entry.symbol for entry in train_entries)
    source_dataset_digests = tuple(
        (entry.symbol, entry.dataset_digest) for entry in train_entries
    )
    channel = UniversalTradeChannelStatistics(
        timeframe="15m",
        feature_names=("test_feature",),
        mean=np.asarray([0.0], dtype=np.float64),
        scale=np.asarray([1.0], dtype=np.float64),
        per_symbol_sample_counts=tuple(
            (symbol, (1,)) for symbol in train_symbols
        ),
    )
    statistics_digest = content_digest(
        {
            "version": _NORMALIZER_VERSION,
            "statistics_semantics": _NORMALIZER_SEMANTICS,
            "contract_digest": _POLICY_CONTRACT_DIGEST,
            "source_dataset_digests": source_dataset_digests,
            "knowledge_cutoff_ns": cutoff_ns,
            "clip_value": 10.0,
            "channels": (channel.digest_payload(),),
        }
    )
    artifact_digest = content_digest(
        {
            "version": _NORMALIZER_VERSION,
            "statistics_digest": statistics_digest,
            "universe_manifest_digest": manifest.digest,
            "provenance_digest": provenance.digest,
            "contract_digest": _POLICY_CONTRACT_DIGEST,
        }
    )
    return UniversalTradeSequenceNormalizer(
        channels=(channel,),
        contract_digest=_POLICY_CONTRACT_DIGEST,
        train_symbols=train_symbols,
        source_dataset_digests=source_dataset_digests,
        knowledge_cutoff_ns=cutoff_ns,
        universe_manifest_digest=manifest.digest,
        provenance_digest=provenance.digest,
        statistics_digest=statistics_digest,
        digest=artifact_digest,
        clip_value=10.0,
    )


def _u1_contract(
    *,
    manifest: UniversalTradeRLUniverseManifest,
    normalizer: UniversalTradeSequenceNormalizer,
    normalizer_provenance: UniversalTradeRLFitProvenance,
) -> UniversalTradeRLU1Contract:
    return UniversalTradeRLU1Contract(
        universe_manifest_digest=manifest.digest,
        u0_identity_digest="8" * 64,
        policy_contract_digest=_POLICY_CONTRACT_DIGEST,
        normalizer_digest=normalizer.digest,
        normalizer_provenance_digest=normalizer_provenance.digest,
        normalizer_knowledge_cutoff_ns=normalizer.knowledge_cutoff_ns,
        normalizer_clip_value=10.0,
        observation_schema_digest="1" * 64,
        state_layout_digest="2" * 64,
        policy_state_fields=("current_weight",),
        runtime_config_digest="3" * 64,
        execution_policy_digest="4" * 64,
        pretrade_risk_digest="5" * 64,
        portfolio_risk_digest="6" * 64,
    )


def _fixture() -> U2PreflightFixture:
    manifest = _manifest()
    time_partition = build_universal_trade_rl_u2_time_partition(manifest=manifest)
    normalizer_provenance = _fit_provenance(
        manifest=manifest,
        purpose=UniversalTradeRLFitPurpose.FEATURE_NORMALIZATION,
        cutoff_ns=time_partition.fit_end_ns,
    )
    normalizer = _normalizer(
        manifest=manifest,
        provenance=normalizer_provenance,
        cutoff_ns=time_partition.fit_end_ns,
    )
    u1_contract = _u1_contract(
        manifest=manifest,
        normalizer=normalizer,
        normalizer_provenance=normalizer_provenance,
    )
    rl_training_provenance = _fit_provenance(
        manifest=manifest,
        purpose=UniversalTradeRLFitPurpose.RL_TRAINING,
        cutoff_ns=time_partition.fit_end_ns,
    )
    u2_contract = build_universal_trade_rl_u2_contract(
        manifest=manifest,
        u1_contract=u1_contract,
        time_partition=time_partition,
        rl_training_provenance=rl_training_provenance,
        training_config=build_universal_trade_rl_u2_training_config(),
    )
    return U2PreflightFixture(
        manifest=manifest,
        time_partition=time_partition,
        normalizer_provenance=normalizer_provenance,
        normalizer=normalizer,
        u1_contract=u1_contract,
        rl_training_provenance=rl_training_provenance,
        u2_contract=u2_contract,
    )


@pytest.fixture(scope="module")
def u2_fixture() -> U2PreflightFixture:
    return _fixture()


def _closure(module: object, fixture: U2PreflightFixture):
    return module.build_universal_trade_rl_u2_training_source_closure(
        manifest=fixture.manifest,
        u1_contract=fixture.u1_contract,
        u2_contract=fixture.u2_contract,
        time_partition=fixture.time_partition,
        normalizer=fixture.normalizer,
        normalizer_provenance=fixture.normalizer_provenance,
    )


@pytest.mark.parametrize("forbidden_symbol", ("SOLUSDT", "XRPUSDT"))
def test_u2_preflight_rejects_non_train_symbol_before_numeric_loader_call(
    u2_fixture: U2PreflightFixture,
    forbidden_symbol: str,
) -> None:
    module = _module()
    closure = _closure(module, u2_fixture)
    loader = SpyBoundedLoader()

    with pytest.raises(ValueError, match="Train|training|scope|symbol"):
        module.load_universal_trade_rl_u2_fit_sources(
            closure=closure,
            requested_symbols=(forbidden_symbol,),
            loader=loader,
        )

    assert loader.calls == []


def test_u2_preflight_passes_fit_bounded_request_before_numeric_read(
    u2_fixture: U2PreflightFixture,
) -> None:
    module = _module()
    closure = _closure(module, u2_fixture)
    loader = SpyBoundedLoader()

    loaded = module.load_universal_trade_rl_u2_fit_sources(
        closure=closure,
        requested_symbols=("BTCUSDT",),
        loader=loader,
    )

    assert len(loader.calls) == 1
    request = loader.calls[0]
    assert loaded == (request,)
    assert request.symbol == "BTCUSDT"
    assert request.dataset_digest == "a" * 64
    assert request.fit_first_timestamp_ns == closure.fit_first_timestamp_ns
    assert request.fit_last_timestamp_ns == closure.fit_last_timestamp_ns
    assert (
        request.fit_stop_timestamp_ns_exclusive
        == closure.fit_last_timestamp_ns + _STEP_NS
    )
    assert request.fit_bar_count == u2_fixture.time_partition.window("fit").bar_count
    assert request.source_last_timestamp_ns > request.fit_last_timestamp_ns
    assert request.source_row_count > request.fit_bar_count


def test_u2_preflight_returns_only_verified_train_sources_and_fit_bounds(
    u2_fixture: U2PreflightFixture,
) -> None:
    module = _module()
    closure = _closure(module, u2_fixture)

    assert closure.u2_contract_digest == u2_fixture.u2_contract.digest
    assert closure.universe_manifest_digest == u2_fixture.manifest.digest
    assert closure.u1_contract_digest == u2_fixture.u1_contract.digest
    assert closure.normalizer_digest == u2_fixture.normalizer.digest
    assert (
        closure.normalizer_provenance_digest
        == u2_fixture.normalizer_provenance.digest
    )
    assert closure.time_partition_digest == u2_fixture.time_partition.digest
    assert closure.fit_first_timestamp_ns == u2_fixture.time_partition.window(
        "fit"
    ).first_timestamp_ns
    assert closure.fit_last_timestamp_ns == u2_fixture.time_partition.fit_end_ns
    assert tuple(source.symbol for source in closure.sources) == (
        "BTCUSDT",
        "ETHUSDT",
    )
    assert tuple(source.dataset_digest for source in closure.sources) == (
        "a" * 64,
        "b" * 64,
    )
    assert all(
        source.fit_last_timestamp_ns == closure.fit_last_timestamp_ns
        for source in closure.sources
    )


def test_u2_preflight_round_trips_canonically(
    u2_fixture: U2PreflightFixture,
) -> None:
    module = _module()
    closure = _closure(module, u2_fixture)

    restored = module.U2TrainingSourceClosure.from_payload(closure.to_payload())

    assert restored == closure
    assert restored.digest == closure.digest
    assert restored.to_payload() == closure.to_payload()


def test_u2_preflight_rejects_wrong_u0_manifest(
    u2_fixture: U2PreflightFixture,
) -> None:
    module = _module()
    wrong_manifest = _manifest(btc_digest_char="9")

    with pytest.raises(ValueError, match="U0|manifest|universe|identity"):
        module.build_universal_trade_rl_u2_training_source_closure(
            manifest=wrong_manifest,
            u1_contract=u2_fixture.u1_contract,
            u2_contract=u2_fixture.u2_contract,
            time_partition=u2_fixture.time_partition,
            normalizer=u2_fixture.normalizer,
            normalizer_provenance=u2_fixture.normalizer_provenance,
        )


def test_u2_preflight_rejects_wrong_normalizer_provenance(
    u2_fixture: U2PreflightFixture,
) -> None:
    module = _module()
    wrong_provenance = _fit_provenance(
        manifest=u2_fixture.manifest,
        purpose=UniversalTradeRLFitPurpose.FORECAST_FIT,
        cutoff_ns=u2_fixture.time_partition.fit_end_ns,
    )

    with pytest.raises(ValueError, match="normalizer|provenance|FEATURE_NORMALIZATION"):
        module.build_universal_trade_rl_u2_training_source_closure(
            manifest=u2_fixture.manifest,
            u1_contract=u2_fixture.u1_contract,
            u2_contract=u2_fixture.u2_contract,
            time_partition=u2_fixture.time_partition,
            normalizer=u2_fixture.normalizer,
            normalizer_provenance=wrong_provenance,
        )


def test_u2_preflight_rejects_normalizer_cutoff_after_or_before_fit(
    u2_fixture: U2PreflightFixture,
) -> None:
    module = _module()
    wrong_cutoff = u2_fixture.time_partition.fit_end_ns - _STEP_NS
    wrong_provenance = _fit_provenance(
        manifest=u2_fixture.manifest,
        purpose=UniversalTradeRLFitPurpose.FEATURE_NORMALIZATION,
        cutoff_ns=wrong_cutoff,
    )
    wrong_normalizer = _normalizer(
        manifest=u2_fixture.manifest,
        provenance=wrong_provenance,
        cutoff_ns=wrong_cutoff,
    )

    with pytest.raises(ValueError, match="normalizer|cutoff|FIT"):
        module.build_universal_trade_rl_u2_training_source_closure(
            manifest=u2_fixture.manifest,
            u1_contract=u2_fixture.u1_contract,
            u2_contract=u2_fixture.u2_contract,
            time_partition=u2_fixture.time_partition,
            normalizer=wrong_normalizer,
            normalizer_provenance=wrong_provenance,
        )


def test_u2_preflight_rejects_wrong_u1_contract_digest(
    u2_fixture: U2PreflightFixture,
) -> None:
    module = _module()
    wrong_u1_contract = replace(
        u2_fixture.u1_contract,
        runtime_config_digest="9" * 64,
        digest="",
    )

    with pytest.raises(ValueError, match="U1|contract|digest|identity"):
        module.build_universal_trade_rl_u2_training_source_closure(
            manifest=u2_fixture.manifest,
            u1_contract=wrong_u1_contract,
            u2_contract=u2_fixture.u2_contract,
            time_partition=u2_fixture.time_partition,
            normalizer=u2_fixture.normalizer,
            normalizer_provenance=u2_fixture.normalizer_provenance,
        )


def test_u2_preflight_rejects_normalizer_source_dataset_identity_drift(
    u2_fixture: U2PreflightFixture,
) -> None:
    module = _module()
    wrong_manifest = _manifest(btc_digest_char="9")
    wrong_provenance = _fit_provenance(
        manifest=wrong_manifest,
        purpose=UniversalTradeRLFitPurpose.FEATURE_NORMALIZATION,
        cutoff_ns=u2_fixture.time_partition.fit_end_ns,
    )
    wrong_normalizer = _normalizer(
        manifest=wrong_manifest,
        provenance=wrong_provenance,
        cutoff_ns=u2_fixture.time_partition.fit_end_ns,
    )

    with pytest.raises(ValueError, match="normalizer|dataset|source|manifest|identity"):
        module.build_universal_trade_rl_u2_training_source_closure(
            manifest=u2_fixture.manifest,
            u1_contract=u2_fixture.u1_contract,
            u2_contract=u2_fixture.u2_contract,
            time_partition=u2_fixture.time_partition,
            normalizer=wrong_normalizer,
            normalizer_provenance=wrong_provenance,
        )
