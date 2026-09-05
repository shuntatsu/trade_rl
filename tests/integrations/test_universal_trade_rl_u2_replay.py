from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from tests.rl.universal_trade_test_support import (
    make_u1_feature_specs,
    make_u1_market,
    make_u1_wrapper,
)
from trade_rl.artifacts.hashing import content_digest
from trade_rl.data.market import MarketDataset
from trade_rl.domain.universal_trade_rl_universe import UniversalTradeRLUniverseConfig
from trade_rl.rl.universal_normalization import (
    UniversalTradeSequenceNormalizer,
    build_universal_trade_sequence_normalizer,
)
from trade_rl.rl.universal_trade_contract import UniversalTradePolicyContract
from trade_rl.rl.universal_trade_environment import UniversalTradeEnvironment
from trade_rl.workflows.universal_trade_rl_data_provenance import (
    UniversalTradeRLFitPurpose,
    build_universal_trade_rl_fit_provenance,
)
from trade_rl.workflows.universal_trade_rl_run_identity import (
    UniversalTradeRLRunIdentity,
    UniversalTradeRLRunStage,
)
from trade_rl.workflows.universal_trade_rl_u1_contract import (
    UniversalTradeRLU1Contract,
    build_universal_trade_rl_u1_contract,
)
from trade_rl.workflows.universal_trade_rl_u2_contract import (
    UniversalTradeRLU2Contract,
    build_universal_trade_rl_u2_contract,
    build_universal_trade_rl_u2_training_config,
)
from trade_rl.workflows.universal_trade_rl_u2_evaluation import (
    UniversalTradeRLU2DevelopmentScopeClosure,
    UniversalTradeRLU2EvaluationScope,
    build_universal_trade_rl_u2_development_scope_closure,
)
from trade_rl.workflows.universal_trade_rl_u2_replay import (
    UniversalTradeRLU2DevelopmentReplaySession,
    build_universal_trade_rl_u2_development_replay_session,
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

_BARS_PER_DAY = 96
_TOTAL_BARS = 620 * _BARS_PER_DAY


@dataclass(slots=True)
class ReplayIntegrationFixture:
    manifest: UniversalTradeRLUniverseManifest
    partition: UniversalTradeRLU2TimePartition
    u2_contract: UniversalTradeRLU2Contract
    u1_contract: UniversalTradeRLU1Contract
    policy_contract: UniversalTradePolicyContract
    normalizer: UniversalTradeSequenceNormalizer
    closure: UniversalTradeRLU2DevelopmentScopeClosure
    session: UniversalTradeRLU2DevelopmentReplaySession
    sources: dict[str, MarketDataset]


def _timestamp_ns(value: np.datetime64) -> int:
    return int(value.astype("datetime64[ns]").astype(np.int64))


def _source(
    *,
    symbol: str,
    dataset_digest: str,
    first_timestamp_ns: int,
    last_timestamp_ns: int,
    row_count: int,
) -> UniversalTradeRLSymbolSource:
    return UniversalTradeRLSymbolSource(
        symbol=symbol,
        dataset_digest=dataset_digest,
        first_timestamp_ns=first_timestamp_ns,
        last_timestamp_ns=last_timestamp_ns,
        row_count=row_count,
    )


@pytest.fixture(scope="module")
def replay_fixture() -> ReplayIntegrationFixture:
    sources = {
        "BTCUSDT": make_u1_market(symbol="BTCUSDT", n_bars=_TOTAL_BARS),
        "SOLUSDT": make_u1_market(
            symbol="SOLUSDT",
            n_bars=_TOTAL_BARS,
            price_scale=1.2,
            feature_level=0.2,
        ),
    }
    first_ns = _timestamp_ns(sources["BTCUSDT"].timestamps[0])
    last_ns = _timestamp_ns(sources["BTCUSDT"].timestamps[-1])
    config = UniversalTradeRLUniverseConfig(
        train_symbols=("BTCUSDT",),
        development_symbols=("SOLUSDT",),
        admission_symbols=("XRPUSDT",),
    )
    manifest = build_universal_trade_rl_universe_manifest(
        config=config,
        sources=(
            _source(
                symbol="BTCUSDT",
                dataset_digest=sources["BTCUSDT"].dataset_id,
                first_timestamp_ns=first_ns,
                last_timestamp_ns=last_ns,
                row_count=_TOTAL_BARS,
            ),
            _source(
                symbol="SOLUSDT",
                dataset_digest=sources["SOLUSDT"].dataset_id,
                first_timestamp_ns=first_ns,
                last_timestamp_ns=last_ns,
                row_count=_TOTAL_BARS,
            ),
            _source(
                symbol="XRPUSDT",
                dataset_digest=content_digest({"fixture": "sealed-admission"}),
                first_timestamp_ns=first_ns,
                last_timestamp_ns=last_ns,
                row_count=_TOTAL_BARS,
            ),
        ),
    )
    partition = build_universal_trade_rl_u2_time_partition(manifest=manifest)
    train_access = UniversalTradeRLUniverseAccess.for_phase(
        manifest=manifest,
        phase=UniversalTradeRLAccessPhase.TRAIN,
    )
    normalizer_provenance = build_universal_trade_rl_fit_provenance(
        manifest=manifest,
        access=train_access,
        purpose=UniversalTradeRLFitPurpose.FEATURE_NORMALIZATION,
        source_symbols=("BTCUSDT",),
        knowledge_cutoff=partition.fit_end_ns,
    )
    policy_contract = UniversalTradePolicyContract(
        feature_specs=make_u1_feature_specs()
    )
    normalizer = build_universal_trade_sequence_normalizer(
        symbol_datasets={"BTCUSDT": sources["BTCUSDT"]},
        contract=policy_contract,
        source_dataset_digests=normalizer_provenance.source_dataset_digests,
        knowledge_cutoff_ns=partition.fit_end_ns,
        universe_manifest_digest=manifest.digest,
        provenance_digest=normalizer_provenance.digest,
    )
    u0_identity = UniversalTradeRLRunIdentity(
        stage=UniversalTradeRLRunStage.UNIVERSE_MATERIALIZATION,
        universe_manifest_digest=manifest.digest,
        model_config_digest=None,
        fit_provenance_digests=(),
    )
    reference_environment = make_u1_wrapper(
        dataset=sources["BTCUSDT"],
        contract=policy_contract,
        normalizer=normalizer,
    )
    try:
        u1_contract = build_universal_trade_rl_u1_contract(
            manifest=manifest,
            u0_identity=u0_identity,
            normalizer_provenance=normalizer_provenance,
            environment=reference_environment,
        )
    finally:
        reference_environment.close()
    rl_training_provenance = build_universal_trade_rl_fit_provenance(
        manifest=manifest,
        access=train_access,
        purpose=UniversalTradeRLFitPurpose.RL_TRAINING,
        source_symbols=("BTCUSDT",),
        knowledge_cutoff=partition.fit_end_ns,
    )
    u2_contract = build_universal_trade_rl_u2_contract(
        manifest=manifest,
        u1_contract=u1_contract,
        time_partition=partition,
        rl_training_provenance=rl_training_provenance,
        training_config=build_universal_trade_rl_u2_training_config(),
    )
    closure = build_universal_trade_rl_u2_development_scope_closure(
        manifest=manifest,
        time_partition=partition,
        u2_contract=u2_contract,
    )
    locators = {
        "BTCUSDT": "fixture://BTCUSDT",
        "SOLUSDT": "fixture://SOLUSDT",
    }
    by_locator = {
        locators[symbol]: dataset for symbol, dataset in sources.items()
    }

    def source_loader(locator: object) -> MarketDataset:
        return by_locator[str(locator)]

    def environment_factory(dataset: MarketDataset) -> UniversalTradeEnvironment:
        return make_u1_wrapper(
            dataset=dataset,
            contract=policy_contract,
            normalizer=normalizer,
        )

    session = build_universal_trade_rl_u2_development_replay_session(
        manifest=manifest,
        time_partition=partition,
        u2_contract=u2_contract,
        u1_contract=u1_contract,
        policy_contract=policy_contract,
        normalizer=normalizer,
        supplied_scope_closure=closure,
        artifact_locators=locators,
        source_loader=source_loader,
        environment_factory=environment_factory,
    )
    return ReplayIntegrationFixture(
        manifest=manifest,
        partition=partition,
        u2_contract=u2_contract,
        u1_contract=u1_contract,
        policy_contract=policy_contract,
        normalizer=normalizer,
        closure=closure,
        session=session,
        sources=sources,
    )


def _scope(fixture: ReplayIntegrationFixture, *, cell: str) -> UniversalTradeRLU2EvaluationScope:
    return next(scope for scope in fixture.closure.scopes if scope.cell == cell)


def test_u2_replay_resets_actual_u1_environment_on_exact_scope_boundaries(
    replay_fixture: ReplayIntegrationFixture,
) -> None:
    scope = _scope(replay_fixture, cell="B")
    environment = replay_fixture.session._create_verified_environment(scope)
    try:
        _observation, info = replay_fixture.session._reset_scope_environment(
            environment,
            scope,
            evaluation_seed=1,
        )
        expected_runtime_end = scope.outcome_stop_bar_index_exclusive - 1
        assert info["start_index"] == scope.evaluation_start_bar_index
        assert info["end_index"] == expected_runtime_end
        assert environment.base_env.start_index == scope.evaluation_start_bar_index
        assert environment.base_env.current_index == scope.evaluation_start_bar_index
        assert environment.base_env.end_index == expected_runtime_end
    finally:
        environment.close()


def test_u2_replay_rejects_reused_mutable_u1_environment_before_stepping(
    replay_fixture: ReplayIntegrationFixture,
) -> None:
    scope = _scope(replay_fixture, cell="B")
    dataset = replay_fixture.session.datasets[scope.concrete_symbol]
    shared = make_u1_wrapper(
        dataset=dataset,
        contract=replay_fixture.policy_contract,
        normalizer=replay_fixture.normalizer,
    )
    original_factory = replay_fixture.session.environment_factory
    replay_fixture.session.environment_factory = lambda _dataset: shared
    first: UniversalTradeEnvironment | None = None
    try:
        first = replay_fixture.session._create_verified_environment(scope)
        with pytest.raises(ValueError, match="reuse|shared|fresh|mutable"):
            replay_fixture.session._create_verified_environment(scope)
    finally:
        replay_fixture.session.environment_factory = original_factory
        if first is not None:
            first.close()
        else:
            shared.close()
