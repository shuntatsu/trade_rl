from __future__ import annotations

import importlib
import math
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import cast

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
from trade_rl.rl.universal_episode_router import DeterministicBalancedInstrumentRouter
from trade_rl.rl.universal_instrument_binding import InstrumentDatasetBinding
from trade_rl.rl.universal_normalization import (
    UniversalTradeSequenceNormalizer,
    build_universal_trade_sequence_normalizer,
)
from trade_rl.rl.universal_single_instrument_env import EpisodeRoutedSingleInstrumentEnv
from trade_rl.rl.universal_trade_contract import UniversalTradePolicyContract
from trade_rl.rl.universal_trade_environment import UniversalTradeEnvironment
from trade_rl.workflows.universal_trade_rl_data_provenance import (
    UniversalTradeRLFitProvenance,
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
from trade_rl.workflows.universal_trade_rl_u2_preflight import (
    U2TrainingSource,
    U2TrainingSourceClosure,
)
from trade_rl.workflows.universal_trade_rl_u2_time_partition import U2_DECISION_STEP_NS
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

_U2_ENVIRONMENT_MODULE = "trade_rl.workflows.universal_trade_rl_u2_environment"
_RUN_SEED = 17


def _timestamp_ns(value: np.datetime64) -> int:
    return int(value.astype("datetime64[ns]").astype(np.int64))


def _u2_builder() -> Callable[..., EpisodeRoutedSingleInstrumentEnv]:
    module = importlib.import_module(_U2_ENVIRONMENT_MODULE)
    builder = getattr(module, "build_universal_trade_rl_u2_environment")
    if not callable(builder):
        raise TypeError("U2 environment builder must be callable")
    return cast(Callable[..., EpisodeRoutedSingleInstrumentEnv], builder)


@dataclass(frozen=True, slots=True)
class U2EnvironmentFixture:
    datasets: dict[str, MarketDataset]
    manifest: UniversalTradeRLUniverseManifest
    normalizer_provenance: UniversalTradeRLFitProvenance
    policy_contract: UniversalTradePolicyContract
    normalizer: UniversalTradeSequenceNormalizer
    u1_contract: UniversalTradeRLU1Contract
    closure: U2TrainingSourceClosure
    bindings: tuple[InstrumentDatasetBinding, ...]

    @property
    def train_symbols(self) -> tuple[str, ...]:
        return tuple(source.symbol for source in self.closure.sources)


@pytest.fixture(scope="module")
def u2_environment_fixture() -> U2EnvironmentFixture:
    datasets = {
        "BTCUSDT": make_u1_market(symbol="BTCUSDT", n_bars=10_000),
        "ETHUSDT": make_u1_market(
            symbol="ETHUSDT",
            n_bars=10_000,
            price_scale=1.25,
            feature_level=0.2,
        ),
    }
    first_ns = _timestamp_ns(datasets["BTCUSDT"].timestamps[0])
    last_ns = _timestamp_ns(datasets["BTCUSDT"].timestamps[-1])
    row_count = datasets["BTCUSDT"].n_bars

    config = UniversalTradeRLUniverseConfig(
        train_symbols=("BTCUSDT", "ETHUSDT"),
        development_symbols=("SOLUSDT",),
        admission_symbols=("ADAUSDT",),
    )
    sources = (
        UniversalTradeRLSymbolSource(
            symbol="ADAUSDT",
            dataset_digest=content_digest({"fixture": "ADAUSDT"}),
            first_timestamp_ns=first_ns,
            last_timestamp_ns=last_ns,
            row_count=row_count,
        ),
        UniversalTradeRLSymbolSource(
            symbol="BTCUSDT",
            dataset_digest=datasets["BTCUSDT"].dataset_id,
            first_timestamp_ns=first_ns,
            last_timestamp_ns=last_ns,
            row_count=row_count,
        ),
        UniversalTradeRLSymbolSource(
            symbol="ETHUSDT",
            dataset_digest=datasets["ETHUSDT"].dataset_id,
            first_timestamp_ns=first_ns,
            last_timestamp_ns=last_ns,
            row_count=row_count,
        ),
        UniversalTradeRLSymbolSource(
            symbol="SOLUSDT",
            dataset_digest=content_digest({"fixture": "SOLUSDT"}),
            first_timestamp_ns=first_ns,
            last_timestamp_ns=last_ns,
            row_count=row_count,
        ),
    )
    manifest = build_universal_trade_rl_universe_manifest(
        config=config,
        sources=sources,
    )
    u0_identity = UniversalTradeRLRunIdentity(
        stage=UniversalTradeRLRunStage.UNIVERSE_MATERIALIZATION,
        universe_manifest_digest=manifest.digest,
        model_config_digest=None,
        fit_provenance_digests=(),
    )
    access = UniversalTradeRLUniverseAccess.for_phase(
        manifest=manifest,
        phase=UniversalTradeRLAccessPhase.TRAIN,
    )
    normalizer_provenance = build_universal_trade_rl_fit_provenance(
        manifest=manifest,
        access=access,
        purpose=UniversalTradeRLFitPurpose.FEATURE_NORMALIZATION,
        source_symbols=config.train_symbols,
        knowledge_cutoff=last_ns,
    )
    policy_contract = UniversalTradePolicyContract(
        feature_specs=make_u1_feature_specs()
    )
    normalizer = build_universal_trade_sequence_normalizer(
        symbol_datasets=datasets,
        contract=policy_contract,
        source_dataset_digests=normalizer_provenance.source_dataset_digests,
        knowledge_cutoff_ns=last_ns,
        universe_manifest_digest=manifest.digest,
        provenance_digest=normalizer_provenance.digest,
    )
    reference_environment = make_u1_wrapper(
        dataset=datasets["BTCUSDT"],
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

    fit_stop_ns = last_ns + U2_DECISION_STEP_NS
    closure = U2TrainingSourceClosure(
        u2_contract_digest=content_digest({"fixture": "u2_contract"}),
        universe_manifest_digest=manifest.digest,
        u1_contract_digest=u1_contract.digest,
        normalizer_digest=normalizer.digest,
        normalizer_provenance_digest=normalizer_provenance.digest,
        time_partition_digest=content_digest({"fixture": "u2_time_partition"}),
        fit_first_timestamp_ns=first_ns,
        fit_last_timestamp_ns=last_ns,
        fit_stop_timestamp_ns_exclusive=fit_stop_ns,
        fit_bar_count=row_count,
        sources=tuple(
            U2TrainingSource(
                symbol=symbol,
                dataset_digest=datasets[symbol].dataset_id,
                source_first_timestamp_ns=first_ns,
                source_last_timestamp_ns=last_ns,
                source_row_count=row_count,
                fit_first_timestamp_ns=first_ns,
                fit_last_timestamp_ns=last_ns,
                fit_stop_timestamp_ns_exclusive=fit_stop_ns,
                fit_bar_count=row_count,
            )
            for symbol in config.train_symbols
        ),
    )
    bindings = tuple(
        InstrumentDatasetBinding(
            concrete_symbol=symbol,
            source_dataset_id=datasets[symbol].dataset_id,
            symbol_dataset_digest=datasets[symbol].dataset_id,
            execution_metadata_digest=content_digest(
                {"fixture": "execution", "symbol": symbol}
            ),
            instrument_descriptor_digest=content_digest(
                {"fixture": "descriptor", "symbol": symbol}
            ),
            split="train",
        )
        for symbol in config.train_symbols
    )
    return U2EnvironmentFixture(
        datasets=datasets,
        manifest=manifest,
        normalizer_provenance=normalizer_provenance,
        policy_contract=policy_contract,
        normalizer=normalizer,
        u1_contract=u1_contract,
        closure=closure,
        bindings=bindings,
    )


def _factory(
    fixture: U2EnvironmentFixture,
    *,
    max_abs_weight: float = 1.0,
    normalizer: UniversalTradeSequenceNormalizer | None = None,
    datasets: dict[str, MarketDataset] | None = None,
    calls: list[str] | None = None,
) -> Callable[[InstrumentDatasetBinding], UniversalTradeEnvironment]:
    resolved_normalizer = fixture.normalizer if normalizer is None else normalizer
    resolved_datasets = fixture.datasets if datasets is None else datasets

    def build(binding: InstrumentDatasetBinding) -> UniversalTradeEnvironment:
        if calls is not None:
            calls.append(binding.concrete_symbol)
        return make_u1_wrapper(
            dataset=resolved_datasets[binding.concrete_symbol],
            max_abs_weight=max_abs_weight,
            contract=fixture.policy_contract,
            normalizer=resolved_normalizer,
        )

    return build


def _build(
    fixture: U2EnvironmentFixture,
    *,
    bindings: tuple[InstrumentDatasetBinding, ...] | None = None,
    environment_factory: Callable[[InstrumentDatasetBinding], UniversalTradeEnvironment]
    | None = None,
    environment_index: int = 0,
) -> EpisodeRoutedSingleInstrumentEnv:
    return _u2_builder()(
        closure=fixture.closure,
        u1_contract=fixture.u1_contract,
        policy_contract=fixture.policy_contract,
        normalizer=fixture.normalizer,
        bindings=fixture.bindings if bindings is None else bindings,
        environment_factory=(
            _factory(fixture) if environment_factory is None else environment_factory
        ),
        run_seed=_RUN_SEED,
        environment_index=environment_index,
    )


def _active_child(
    environment: EpisodeRoutedSingleInstrumentEnv,
) -> UniversalTradeEnvironment:
    active = environment._active_environment
    if not isinstance(active, UniversalTradeEnvironment):
        raise AssertionError("active U2 child must be a UniversalTradeEnvironment")
    return active


def _force_episode_boundary(environment: EpisodeRoutedSingleInstrumentEnv) -> None:
    # This isolates reset/cache hygiene without paying for a full 720h rollout.
    environment._episode_complete = True
    environment._completed_episode_count += 1


def _assert_cash_runtime(environment: UniversalTradeEnvironment) -> None:
    runtime = environment.base_env.universal_trade_runtime_snapshot()
    assert runtime.policy_requested_weight == pytest.approx(0.0)
    assert runtime.pending_target_weight == pytest.approx(0.0)
    assert runtime.pending_target_active is False
    assert runtime.risk_projected_weight == pytest.approx(0.0)
    assert runtime.current_weight == pytest.approx(0.0)
    assert runtime.previous_action == pytest.approx(0.0)
    assert runtime.pending_notional_ratio == pytest.approx(0.0)
    assert runtime.pending_order_type_code == pytest.approx(0.0)
    assert runtime.pending_order_status_code == pytest.approx(0.0)
    assert runtime.pending_order_triggered is False
    assert runtime.current_gross_exposure == pytest.approx(0.0)
    assert runtime.current_net_exposure == pytest.approx(0.0)
    assert runtime.cash_weight == pytest.approx(1.0)


def test_u2_environment_reuses_exact_u1_surface_and_has_no_context_channels(
    u2_environment_fixture: U2EnvironmentFixture,
) -> None:
    calls: list[str] = []
    environment = _build(
        u2_environment_fixture,
        environment_factory=_factory(u2_environment_fixture, calls=calls),
    )
    try:
        assert sorted(calls) == sorted(u2_environment_fixture.train_symbols)
        assert environment.action_space.shape == (1,)
        assert environment._instrument_context_provider is None
        assert environment._v4_context_provider is None

        observation, _info = environment.reset(seed=_RUN_SEED)
        child = _active_child(environment)
        assert (
            child.contract.digest
            == u2_environment_fixture.u1_contract.policy_contract_digest
        )
        assert child.sequence_normalizer is u2_environment_fixture.normalizer
        assert (
            child.sequence_normalizer.digest
            == u2_environment_fixture.u1_contract.normalizer_digest
        )
        assert child.observation_space == environment.observation_space
        assert child.action_space == environment.action_space
        assert set(observation) == set(child.observation_space.spaces)
        assert "instrument_context" not in observation
        assert "local_cross_market_context" not in observation
        assert "global_market_context" not in observation
        assert "causal_beta" not in observation
    finally:
        environment.close()


def test_u2_environment_rejects_binding_source_drift_before_child_factory_call(
    u2_environment_fixture: U2EnvironmentFixture,
) -> None:
    calls: list[str] = []
    bindings = list(u2_environment_fixture.bindings)
    bindings[0] = replace(bindings[0], symbol_dataset_digest="f" * 64)

    with pytest.raises(ValueError, match="binding|source|dataset|digest"):
        _build(
            u2_environment_fixture,
            bindings=tuple(bindings),
            environment_factory=_factory(u2_environment_fixture, calls=calls),
        )

    assert calls == []


def test_u2_environment_prevalidates_noninitial_child_economic_identity(
    u2_environment_fixture: U2EnvironmentFixture,
) -> None:
    router = DeterministicBalancedInstrumentRouter(
        train_symbols=u2_environment_fixture.train_symbols,
        partition_digest=u2_environment_fixture.closure.time_partition_digest,
        run_seed=_RUN_SEED,
        environment_index=0,
    )
    initial_symbol = router.route(0).concrete_symbol
    drift_symbol = next(
        symbol
        for symbol in u2_environment_fixture.train_symbols
        if symbol != initial_symbol
    )
    calls: list[str] = []

    def factory(binding: InstrumentDatasetBinding) -> UniversalTradeEnvironment:
        calls.append(binding.concrete_symbol)
        return make_u1_wrapper(
            dataset=u2_environment_fixture.datasets[binding.concrete_symbol],
            max_abs_weight=(0.5 if binding.concrete_symbol == drift_symbol else 1.0),
            contract=u2_environment_fixture.policy_contract,
            normalizer=u2_environment_fixture.normalizer,
        )

    with pytest.raises(ValueError, match="risk|pretrade|U1|contract|economic"):
        _build(u2_environment_fixture, environment_factory=factory)

    assert initial_symbol in calls
    assert drift_symbol in calls


def test_u2_environment_rejects_wrong_normalizer_generation(
    u2_environment_fixture: U2EnvironmentFixture,
) -> None:
    wrong_normalizer = build_universal_trade_sequence_normalizer(
        symbol_datasets=u2_environment_fixture.datasets,
        contract=u2_environment_fixture.policy_contract,
        source_dataset_digests=(
            u2_environment_fixture.normalizer_provenance.source_dataset_digests
        ),
        knowledge_cutoff_ns=u2_environment_fixture.normalizer.knowledge_cutoff_ns,
        universe_manifest_digest=u2_environment_fixture.manifest.digest,
        provenance_digest=u2_environment_fixture.normalizer_provenance.digest,
        clip_value=5.0,
    )

    with pytest.raises(ValueError, match="normalizer|U1|contract"):
        _build(
            u2_environment_fixture,
            environment_factory=_factory(
                u2_environment_fixture,
                normalizer=wrong_normalizer,
            ),
        )


def test_u2_environment_rejects_child_with_post_fit_bar(
    u2_environment_fixture: U2EnvironmentFixture,
) -> None:
    expanded = {
        symbol: make_u1_market(
            symbol=symbol,
            n_bars=u2_environment_fixture.closure.fit_bar_count + 1,
            price_scale=(1.0 if symbol == "BTCUSDT" else 1.25),
            feature_level=(0.0 if symbol == "BTCUSDT" else 0.2),
        )
        for symbol in u2_environment_fixture.train_symbols
    }
    bindings = tuple(
        replace(
            binding,
            source_dataset_id=expanded[binding.concrete_symbol].dataset_id,
        )
        for binding in u2_environment_fixture.bindings
    )

    with pytest.raises(ValueError, match="FIT|fit|timestamp|bar"):
        _build(
            u2_environment_fixture,
            bindings=bindings,
            environment_factory=_factory(
                u2_environment_fixture,
                datasets=expanded,
            ),
        )


def test_u2_environment_preserves_complete_balanced_symbol_cycles(
    u2_environment_fixture: U2EnvironmentFixture,
) -> None:
    environment = _build(u2_environment_fixture)
    try:
        routes = tuple(environment._router.route(index) for index in range(12))
        width = len(u2_environment_fixture.train_symbols)
        for offset in range(0, len(routes), width):
            cycle = routes[offset : offset + width]
            assert {route.concrete_symbol for route in cycle} == set(
                u2_environment_fixture.train_symbols
            )
            assert tuple(route.routing_position for route in cycle) == tuple(
                range(width)
            )
    finally:
        environment.close()


def test_u2_environment_cached_symbol_reset_clears_policy_and_execution_state(
    u2_environment_fixture: U2EnvironmentFixture,
) -> None:
    environment = _build(u2_environment_fixture)
    try:
        environment.reset(seed=_RUN_SEED)
        first_binding = environment.active_episode_binding
        first_child = _active_child(environment)
        first_symbol = first_binding.dataset_binding.concrete_symbol

        environment.step(np.asarray([1.0], dtype=np.float32))
        contaminated = first_child.base_env.universal_trade_runtime_snapshot()
        assert contaminated.previous_action == pytest.approx(1.0)

        _force_episode_boundary(environment)
        environment.reset()
        second_symbol = (
            environment.active_episode_binding.dataset_binding.concrete_symbol
        )
        assert second_symbol != first_symbol
        _assert_cash_runtime(_active_child(environment))

        for _ in range(2):
            _force_episode_boundary(environment)
            environment.reset()
            if (
                environment.active_episode_binding.dataset_binding.concrete_symbol
                == first_symbol
            ):
                break
        else:
            raise AssertionError(
                "cached first symbol did not reappear within next cycle"
            )

        assert _active_child(environment) is first_child
        _assert_cash_runtime(first_child)
    finally:
        environment.close()


def test_u2_environment_many_deterministic_resets_cannot_cross_fit_end(
    u2_environment_fixture: U2EnvironmentFixture,
) -> None:
    checked = 0
    for environment_index in (0, 1):
        routed = _build(
            u2_environment_fixture,
            environment_index=environment_index,
        )
        try:
            child = cast(UniversalTradeEnvironment, routed._reference_environment)
            for offset in range(512):
                _observation, info = child.reset(
                    seed=environment_index * 10_000 + offset,
                )
                end_index = int(info["end_index"])
                end_ns = _timestamp_ns(child.dataset.timestamps[end_index])
                assert end_ns <= u2_environment_fixture.closure.fit_last_timestamp_ns
                checked += 1
        finally:
            routed.close()

    assert checked == 1_024


def test_u2_environment_reward_remains_realized_u1_net_log_growth(
    u2_environment_fixture: U2EnvironmentFixture,
) -> None:
    environment = _build(u2_environment_fixture)
    try:
        environment.reset(seed=_RUN_SEED)
        environment.step(np.asarray([1.0], dtype=np.float32))
        before = float(environment.hybrid.portfolio_value)
        _observation, reward, terminated, truncated, _info = environment.step(
            np.asarray([1.0], dtype=np.float32)
        )
        after = float(environment.hybrid.portfolio_value)

        assert terminated is False
        assert truncated is False
        assert reward == pytest.approx(100.0 * math.log(after / before), abs=1e-10)
    finally:
        environment.close()
