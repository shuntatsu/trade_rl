"""FIT-only routed U1 environment assembly for Universal Trade RL U2."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.data.market import MarketDataset
from trade_rl.rl.universal_episode_router import InstrumentRoute
from trade_rl.rl.universal_instrument_binding import (
    InstrumentDatasetBinding,
    validate_training_instrument_bindings,
)
from trade_rl.rl.universal_normalization import UniversalTradeSequenceNormalizer
from trade_rl.rl.universal_single_instrument_env import EpisodeRoutedSingleInstrumentEnv
from trade_rl.rl.universal_trade_contract import UniversalTradePolicyContract
from trade_rl.rl.universal_trade_environment import UniversalTradeEnvironment
from trade_rl.workflows.universal_trade_rl_u1_contract import (
    UniversalTradeRLU1Contract,
    require_universal_trade_rl_u1_environment_contract,
)
from trade_rl.workflows.universal_trade_rl_u2_preflight import (
    U2TrainingSource,
    U2TrainingSourceClosure,
)
from trade_rl.workflows.universal_trade_rl_u2_time_partition import U2_DECISION_STEP_NS

U2EnvironmentFactory = Callable[[InstrumentDatasetBinding], UniversalTradeEnvironment]

_FORBIDDEN_CONTEXT_KEYS = frozenset(
    {
        "instrument_context",
        "local_cross_market_context",
        "local_cross_market_available",
        "local_cross_market_staleness_hours",
        "global_market_context",
        "global_market_available",
        "global_market_staleness_hours",
        "causal_beta",
        "causal_beta_available",
    }
)


def _non_negative_integer(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _training_sources(
    closure: U2TrainingSourceClosure,
) -> dict[str, U2TrainingSource]:
    return {source.symbol: source for source in closure.sources}


def _require_frozen_generation(
    *,
    closure: U2TrainingSourceClosure,
    u1_contract: UniversalTradeRLU1Contract,
    policy_contract: UniversalTradePolicyContract,
    normalizer: UniversalTradeSequenceNormalizer,
) -> tuple[str, ...]:
    if not isinstance(closure, U2TrainingSourceClosure):
        raise TypeError("U2 environment source closure is invalid")
    if not isinstance(u1_contract, UniversalTradeRLU1Contract):
        raise TypeError("U2 environment U1 contract is invalid")
    if not isinstance(policy_contract, UniversalTradePolicyContract):
        raise TypeError("U2 environment policy contract is invalid")
    if not isinstance(normalizer, UniversalTradeSequenceNormalizer):
        raise TypeError("U2 environment normalizer is invalid")

    if closure.u1_contract_digest != u1_contract.digest:
        raise ValueError("U2 environment U1 contract identity mismatch")
    if closure.universe_manifest_digest != u1_contract.universe_manifest_digest:
        raise ValueError("U2 environment U1 universe identity mismatch")
    if policy_contract.digest != u1_contract.policy_contract_digest:
        raise ValueError("U2 environment policy contract identity mismatch")
    if closure.normalizer_digest != u1_contract.normalizer_digest:
        raise ValueError("U2 environment closure/U1 normalizer identity mismatch")
    if closure.normalizer_provenance_digest != u1_contract.normalizer_provenance_digest:
        raise ValueError("U2 environment closure/U1 normalizer provenance mismatch")
    if normalizer.digest != closure.normalizer_digest:
        raise ValueError("U2 environment normalizer generation mismatch")
    if normalizer.provenance_digest != closure.normalizer_provenance_digest:
        raise ValueError("U2 environment normalizer provenance mismatch")
    if normalizer.universe_manifest_digest != closure.universe_manifest_digest:
        raise ValueError("U2 environment normalizer universe identity mismatch")
    if normalizer.contract_digest != policy_contract.digest:
        raise ValueError("U2 environment normalizer policy contract mismatch")
    if normalizer.knowledge_cutoff_ns != u1_contract.normalizer_knowledge_cutoff_ns:
        raise ValueError("U2 environment normalizer knowledge cutoff mismatch")
    if normalizer.knowledge_cutoff_ns != closure.fit_last_timestamp_ns:
        raise ValueError("U2 environment normalizer cutoff must equal FIT end")
    if normalizer.clip_value != u1_contract.normalizer_clip_value:
        raise ValueError("U2 environment normalizer clip value mismatch")

    train_symbols = tuple(source.symbol for source in closure.sources)
    if normalizer.train_symbols != train_symbols:
        raise ValueError("U2 environment normalizer Train symbol scope mismatch")
    expected_sources = tuple(
        (source.symbol, source.dataset_digest) for source in closure.sources
    )
    if normalizer.source_dataset_digests != expected_sources:
        raise ValueError("U2 environment normalizer source identity mismatch")
    return train_symbols


def _require_binding_closure(
    *,
    train_symbols: tuple[str, ...],
    sources: Mapping[str, U2TrainingSource],
    bindings: Sequence[InstrumentDatasetBinding],
) -> dict[str, InstrumentDatasetBinding]:
    resolved = validate_training_instrument_bindings(train_symbols, bindings)
    for symbol in train_symbols:
        binding = resolved[symbol]
        source = sources[symbol]
        if binding.symbol_dataset_digest != source.dataset_digest:
            raise ValueError(
                f"U2 environment binding source dataset digest mismatch for {symbol}"
            )
    return resolved


def _dataset_timestamps_ns(dataset: MarketDataset) -> np.ndarray:
    timestamps = (
        np.asarray(dataset.timestamps).astype("datetime64[ns]").astype(np.int64)
    )
    if timestamps.ndim != 1 or timestamps.size != dataset.n_bars:
        raise ValueError("U2 environment dataset timestamp layout is invalid")
    return timestamps


def build_universal_trade_rl_u2_instrument_bindings(
    *,
    closure: U2TrainingSourceClosure,
    fit_datasets: Mapping[str, MarketDataset],
    u1_contract: UniversalTradeRLU1Contract,
) -> tuple[InstrumentDatasetBinding, ...]:
    """Derive canonical U2 bindings from verified FIT datasets and frozen U1 identity."""

    if not isinstance(closure, U2TrainingSourceClosure):
        raise TypeError("U2 binding derivation requires a verified source closure")
    if not isinstance(u1_contract, UniversalTradeRLU1Contract):
        raise TypeError("U2 binding derivation requires a U1 contract")
    if not isinstance(fit_datasets, Mapping):
        raise TypeError("U2 binding FIT datasets must be a mapping")
    if closure.u1_contract_digest != u1_contract.digest:
        raise ValueError("U2 binding U1 contract identity mismatch")

    expected_symbols = tuple(source.symbol for source in closure.sources)
    observed_symbols = tuple(fit_datasets)
    if len(observed_symbols) != len(expected_symbols) or set(observed_symbols) != set(
        expected_symbols
    ):
        raise ValueError("U2 binding FIT dataset closure must equal Train symbols")

    instrument_descriptor_digest = content_digest(
        {
            "schema_version": "universal_trade_rl_u2_instrument_descriptor_disabled_v1",
            "instrument_context_enabled": False,
            "v4_context_enabled": False,
        }
    )
    bindings: list[InstrumentDatasetBinding] = []
    for source in closure.sources:
        fit_dataset = fit_datasets[source.symbol]
        if not isinstance(fit_dataset, MarketDataset):
            raise TypeError("U2 binding FIT dataset must be a MarketDataset")
        if fit_dataset.symbols != (source.symbol,):
            raise ValueError("U2 binding FIT dataset symbol mismatch")
        if fit_dataset.n_bars != source.fit_bar_count:
            raise ValueError("U2 binding FIT dataset bar count mismatch")

        timestamps_ns = _dataset_timestamps_ns(fit_dataset)
        expected_timestamps_ns = source.fit_first_timestamp_ns + np.arange(
            source.fit_bar_count,
            dtype=np.int64,
        ) * np.int64(U2_DECISION_STEP_NS)
        if not np.array_equal(timestamps_ns, expected_timestamps_ns):
            raise ValueError("U2 binding FIT dataset timestamps mismatch")
        if int(timestamps_ns[-1]) != source.fit_last_timestamp_ns:
            raise ValueError("U2 binding FIT dataset crosses the FIT end")

        execution_metadata_digest = content_digest(
            {
                "schema_version": "universal_trade_rl_u2_execution_binding_v1",
                "fit_dataset_id": fit_dataset.dataset_id,
                "u1_execution_policy_digest": u1_contract.execution_policy_digest,
                "u1_pretrade_risk_digest": u1_contract.pretrade_risk_digest,
                "u1_portfolio_risk_digest": u1_contract.portfolio_risk_digest,
            }
        )
        bindings.append(
            InstrumentDatasetBinding(
                concrete_symbol=source.symbol,
                source_dataset_id=fit_dataset.dataset_id,
                symbol_dataset_digest=source.dataset_digest,
                execution_metadata_digest=execution_metadata_digest,
                instrument_descriptor_digest=instrument_descriptor_digest,
                split="train",
            )
        )
    return tuple(bindings)


def _require_fit_dataset(
    *,
    environment: UniversalTradeEnvironment,
    binding: InstrumentDatasetBinding,
    source: U2TrainingSource,
) -> None:
    dataset = environment.dataset
    if not isinstance(dataset, MarketDataset):
        raise TypeError("U2 child environment must expose a MarketDataset")
    if dataset.symbols != (binding.concrete_symbol,):
        raise ValueError("U2 child environment dataset symbol mismatch")
    if dataset.dataset_id != binding.source_dataset_id:
        raise ValueError("U2 child environment dataset identity mismatch")
    if dataset.n_bars != source.fit_bar_count:
        raise ValueError("U2 child environment must contain exactly the FIT bars")

    timestamps_ns = _dataset_timestamps_ns(dataset)
    expected = source.fit_first_timestamp_ns + np.arange(
        source.fit_bar_count,
        dtype=np.int64,
    ) * np.int64(U2_DECISION_STEP_NS)
    if not np.array_equal(timestamps_ns, expected):
        raise ValueError(
            "U2 child environment timestamps must equal the dense FIT grid"
        )
    if int(timestamps_ns[-1]) != source.fit_last_timestamp_ns:
        raise ValueError("U2 child environment crosses the FIT end")
    if source.fit_stop_timestamp_ns_exclusive != (
        int(timestamps_ns[-1]) + U2_DECISION_STEP_NS
    ):
        raise ValueError("U2 child environment FIT exclusive stop mismatch")


def _require_child_surface(
    *,
    environment: UniversalTradeEnvironment,
    binding: InstrumentDatasetBinding,
    policy_contract: UniversalTradePolicyContract,
    normalizer: UniversalTradeSequenceNormalizer,
    u1_contract: UniversalTradeRLU1Contract,
    source: U2TrainingSource,
) -> None:
    require_universal_trade_rl_u1_environment_contract(
        contract=u1_contract,
        environment=environment,
    )
    if environment.contract.digest != policy_contract.digest:
        raise ValueError("U2 child policy contract mismatch")
    child_normalizer = environment.sequence_normalizer
    if not isinstance(child_normalizer, UniversalTradeSequenceNormalizer):
        raise ValueError("U2 child requires the frozen sequence normalizer")
    if child_normalizer.digest != normalizer.digest:
        raise ValueError("U2 child normalizer generation mismatch")
    if tuple(environment.action_names) != (f"target_weight:{binding.concrete_symbol}",):
        raise ValueError("U2 child action names do not match its concrete symbol")
    if environment.action_space.shape != (1,):
        raise ValueError("U2 child action space must remain scalar target weight")

    spaces = getattr(environment.observation_space, "spaces", None)
    if isinstance(spaces, Mapping) and _FORBIDDEN_CONTEXT_KEYS.intersection(spaces):
        raise ValueError("U2 child observation contains forbidden context channels")
    _require_fit_dataset(
        environment=environment,
        binding=binding,
        source=source,
    )


def _require_equal_spaces(
    *,
    reference: UniversalTradeEnvironment,
    candidate: UniversalTradeEnvironment,
) -> None:
    if candidate.observation_space != reference.observation_space:
        raise ValueError("U2 child observation space mismatch")
    if candidate.action_space != reference.action_space:
        raise ValueError("U2 child action space mismatch")
    if float(candidate.initial_capital) != float(reference.initial_capital):
        raise ValueError("U2 child initial capital mismatch")
    if float(candidate.decision_hours) != float(reference.decision_hours):
        raise ValueError("U2 child decision cadence mismatch")


def _close_unique(
    environments: Sequence[UniversalTradeEnvironment],
    *,
    suppress_errors: bool,
) -> None:
    first_error: Exception | None = None
    closed_ids: set[int] = set()
    for environment in environments:
        object_id = id(environment)
        if object_id in closed_ids:
            continue
        closed_ids.add(object_id)
        try:
            environment.close()
        except Exception as error:
            if first_error is None:
                first_error = error
    if first_error is not None and not suppress_errors:
        raise first_error


class _PrevalidatedEnvironmentFactory:
    def __init__(self, environments: Mapping[str, UniversalTradeEnvironment]) -> None:
        self._environments = dict(environments)
        self._issued_symbols: set[str] = set()

    def __call__(self, binding: InstrumentDatasetBinding) -> UniversalTradeEnvironment:
        environment = self._environments.get(binding.concrete_symbol)
        if environment is None:
            raise KeyError(binding.concrete_symbol)
        self._issued_symbols.add(binding.concrete_symbol)
        return environment

    def close_unissued(self) -> None:
        unissued = tuple(
            environment
            for symbol, environment in self._environments.items()
            if symbol not in self._issued_symbols
        )
        _close_unique(unissued, suppress_errors=False)

    def close_all(self) -> None:
        _close_unique(tuple(self._environments.values()), suppress_errors=True)


class _OwnedU2RoutedEnvironment(EpisodeRoutedSingleInstrumentEnv):
    def __init__(
        self,
        *,
        owned_factory: _PrevalidatedEnvironmentFactory,
        train_symbols: tuple[str, ...],
        partition_digest: str,
        bindings: tuple[InstrumentDatasetBinding, ...],
        run_seed: int,
        environment_index: int,
        training_contract_digest: str,
    ) -> None:
        self._owned_factory = owned_factory
        self._u2_closed = False
        super().__init__(
            train_symbols=train_symbols,
            partition_digest=partition_digest,
            bindings=bindings,
            environment_factory=owned_factory,
            run_seed=run_seed,
            environment_index=environment_index,
            instrument_context_provider=None,
            v4_context_provider=None,
            training_contract_digest=training_contract_digest,
            max_cached_environments=None,
        )

    @property
    def run_seed(self) -> int:
        return self._run_seed

    @property
    def environment_index(self) -> int:
        return self._environment_index

    @property
    def canonical_probe_seed(self) -> int:
        return self._run_seed + self._environment_index

    def _episode_seed(
        self,
        *,
        route: InstrumentRoute,
        binding: InstrumentDatasetBinding,
    ) -> int:
        digest = content_digest(
            {
                "schema_version": "universal_trade_rl_u2_episode_seed_v1",
                "run_seed": self._run_seed,
                "partition_digest": self._router.partition_digest,
                "environment_index": self._environment_index,
                "completed_episode_count": route.completed_episode_count,
                "fit_dataset_id": binding.source_dataset_id,
            }
        )
        return int(digest[:8], 16)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        if seed is not None:
            resolved_seed = _non_negative_integer(seed, field="seed")
            if resolved_seed != self.canonical_probe_seed:
                raise ValueError(
                    "U2 reset seed must equal run_seed + environment_index"
                )
        return super().reset(seed=self._run_seed, options=options)

    def close(self) -> None:
        if self._u2_closed:
            return
        self._u2_closed = True
        first_error: Exception | None = None
        try:
            super().close()
        except Exception as error:
            first_error = error
        try:
            self._owned_factory.close_unissued()
        except Exception as error:
            if first_error is None:
                first_error = error
        if first_error is not None:
            raise first_error


def build_universal_trade_rl_u2_environment(
    *,
    closure: U2TrainingSourceClosure,
    u1_contract: UniversalTradeRLU1Contract,
    policy_contract: UniversalTradePolicyContract,
    normalizer: UniversalTradeSequenceNormalizer,
    bindings: tuple[InstrumentDatasetBinding, ...],
    environment_factory: U2EnvironmentFactory,
    run_seed: int,
    environment_index: int,
) -> EpisodeRoutedSingleInstrumentEnv:
    """Build one prevalidated balanced U2 environment from exact FIT children."""

    resolved_run_seed = _non_negative_integer(run_seed, field="U2 run_seed")
    resolved_environment_index = _non_negative_integer(
        environment_index,
        field="U2 environment_index",
    )
    if not callable(environment_factory):
        raise TypeError("U2 environment_factory must be callable")

    train_symbols = _require_frozen_generation(
        closure=closure,
        u1_contract=u1_contract,
        policy_contract=policy_contract,
        normalizer=normalizer,
    )
    sources = _training_sources(closure)
    resolved_bindings = _require_binding_closure(
        train_symbols=train_symbols,
        sources=sources,
        bindings=bindings,
    )

    built: dict[str, UniversalTradeEnvironment] = {}
    object_ids: set[int] = set()
    reference: UniversalTradeEnvironment | None = None
    try:
        for symbol in train_symbols:
            binding = resolved_bindings[symbol]
            environment = environment_factory(binding)
            if not isinstance(environment, UniversalTradeEnvironment):
                raise TypeError(
                    "U2 environment_factory must return UniversalTradeEnvironment"
                )
            if id(environment) in object_ids:
                raise ValueError(
                    "U2 environment_factory reused one child across symbols"
                )
            object_ids.add(id(environment))
            built[symbol] = environment
            _require_child_surface(
                environment=environment,
                binding=binding,
                policy_contract=policy_contract,
                normalizer=normalizer,
                u1_contract=u1_contract,
                source=sources[symbol],
            )
            if reference is None:
                reference = environment
            else:
                _require_equal_spaces(reference=reference, candidate=environment)

        ordered_bindings = tuple(resolved_bindings[symbol] for symbol in train_symbols)
        owned_factory = _PrevalidatedEnvironmentFactory(built)
        try:
            return _OwnedU2RoutedEnvironment(
                owned_factory=owned_factory,
                train_symbols=train_symbols,
                partition_digest=closure.time_partition_digest,
                bindings=ordered_bindings,
                run_seed=resolved_run_seed,
                environment_index=resolved_environment_index,
                training_contract_digest=closure.u2_contract_digest,
            )
        except Exception:
            owned_factory.close_all()
            raise
    except Exception:
        _close_unique(tuple(built.values()), suppress_errors=True)
        raise


__all__ = [
    "U2EnvironmentFactory",
    "build_universal_trade_rl_u2_environment",
    "build_universal_trade_rl_u2_instrument_bindings",
]
