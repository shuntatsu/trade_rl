"""Deterministic synthetic Development replay boundary for Universal Trade RL U2."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from weakref import WeakValueDictionary

from trade_rl.data.market import MarketDataset
from trade_rl.domain.common import require_sha256
from trade_rl.rl.universal_normalization import UniversalTradeSequenceNormalizer
from trade_rl.rl.universal_trade_contract import UniversalTradePolicyContract
from trade_rl.rl.universal_trade_environment import (
    UniversalTradeEnvironment,
    UniversalTradeMarketEnv,
)
from trade_rl.workflows.universal_trade_rl_u1_contract import (
    UniversalTradeRLU1Contract,
    require_universal_trade_rl_u1_environment_contract,
)
from trade_rl.workflows.universal_trade_rl_u2_contract import (
    U2_TRAINING_SEEDS,
    UniversalTradeRLU2Contract,
)
from trade_rl.workflows.universal_trade_rl_u2_evaluation import (
    UniversalTradeRLU2DevelopmentScopeClosure,
    UniversalTradeRLU2EvaluationScope,
    build_universal_trade_rl_u2_development_scope_closure,
)
from trade_rl.workflows.universal_trade_rl_u2_evaluation_dataset import (
    U2EvaluationSourceArtifactLoader,
    U2EvaluationSourceArtifactLocator,
    load_universal_trade_rl_u2_development_evaluation_datasets,
)
from trade_rl.workflows.universal_trade_rl_u2_time_partition import (
    UniversalTradeRLU2TimePartition,
)
from trade_rl.workflows.universal_trade_rl_universe_manifest import (
    UniversalTradeRLUniverseManifest,
)

U2ReplayEnvironmentFactory = Callable[[MarketDataset], UniversalTradeEnvironment]


class UniversalTradeRLU2ReplayVariant(str, Enum):
    """The four preregistered deterministic Development replay variants."""

    CANDIDATE = "candidate"
    CASH = "cash"
    CONSTANT_LONG = "constant_long"
    CONSTANT_SHORT = "constant_short"


@dataclass(frozen=True, slots=True)
class UniversalTradeRLU2ReplayRequest:
    """One paired candidate/baseline replay identity before numeric execution."""

    scope_digest: str
    policy_variant: UniversalTradeRLU2ReplayVariant
    evaluation_seed: int
    paired_candidate_checkpoint_digest: str

    def __post_init__(self) -> None:
        require_sha256(self.scope_digest, field="U2 replay scope digest")
        if not isinstance(self.policy_variant, UniversalTradeRLU2ReplayVariant):
            raise TypeError("U2 replay policy variant is invalid")
        if (
            isinstance(self.evaluation_seed, bool)
            or not isinstance(self.evaluation_seed, int)
            or self.evaluation_seed not in U2_TRAINING_SEEDS
        ):
            raise ValueError("U2 replay evaluation seed must be preregistered")
        require_sha256(
            self.paired_candidate_checkpoint_digest,
            field="U2 replay paired candidate checkpoint digest",
        )


@dataclass(slots=True)
class UniversalTradeRLU2DevelopmentReplaySession:
    """Own one canonical synthetic Development dataset generation for replay."""

    manifest: UniversalTradeRLUniverseManifest
    time_partition: UniversalTradeRLU2TimePartition
    u2_contract: UniversalTradeRLU2Contract
    u1_contract: UniversalTradeRLU1Contract
    policy_contract: UniversalTradePolicyContract
    normalizer: UniversalTradeSequenceNormalizer
    scope_closure: UniversalTradeRLU2DevelopmentScopeClosure
    datasets: dict[str, MarketDataset]
    environment_factory: U2ReplayEnvironmentFactory
    _issued_environments: WeakValueDictionary[int, UniversalTradeEnvironment] = field(
        default_factory=WeakValueDictionary,
        init=False,
        repr=False,
    )
    _issued_base_environments: WeakValueDictionary[int, UniversalTradeMarketEnv] = field(
        default_factory=WeakValueDictionary,
        init=False,
        repr=False,
    )

    @property
    def scope_closure_digest(self) -> str:
        return self.scope_closure.digest

    @property
    def evaluation_dataset_ids(self) -> tuple[tuple[str, str], ...]:
        return tuple((symbol, dataset.dataset_id) for symbol, dataset in self.datasets.items())

    def scope(self, scope_digest: str) -> UniversalTradeRLU2EvaluationScope:
        require_sha256(scope_digest, field="U2 replay scope digest")
        matches = tuple(
            scope for scope in self.scope_closure.scopes if scope.digest == scope_digest
        )
        if len(matches) != 1:
            raise ValueError("U2 replay scope is not present exactly once")
        return matches[0]

    def _require_canonical_scope(
        self,
        scope: UniversalTradeRLU2EvaluationScope,
    ) -> UniversalTradeRLU2EvaluationScope:
        if not isinstance(scope, UniversalTradeRLU2EvaluationScope):
            raise TypeError("U2 replay requires an evaluation scope")
        canonical = self.scope(scope.digest)
        if canonical != scope:
            raise ValueError("U2 replay scope identity drifted from the canonical closure")
        return canonical

    def _create_verified_environment(
        self,
        scope: UniversalTradeRLU2EvaluationScope,
    ) -> UniversalTradeEnvironment:
        canonical = self._require_canonical_scope(scope)
        dataset = self.datasets.get(canonical.concrete_symbol)
        if not isinstance(dataset, MarketDataset):
            raise TypeError("U2 replay scope dataset is unavailable")
        if dataset.symbols != (canonical.concrete_symbol,):
            raise ValueError("U2 replay dataset symbol mismatch")
        if dataset.dataset_id != canonical.evaluation_dataset_digest:
            raise ValueError("U2 replay dataset identity mismatch")

        environment = self.environment_factory(dataset)
        if not isinstance(environment, UniversalTradeEnvironment):
            raise TypeError("U2 replay environment factory must return UniversalTradeEnvironment")

        object_id = id(environment)
        existing = self._issued_environments.get(object_id)
        if existing is environment:
            raise ValueError("U2 replay requires a fresh mutable U1 environment per variant")
        if existing is not None:
            raise RuntimeError("U2 replay observed a live U1 environment object-id collision")

        base_environment = environment.base_env
        base_id = id(base_environment)
        existing_base = self._issued_base_environments.get(base_id)
        if existing_base is base_environment:
            raise ValueError("U2 replay requires a fresh mutable U1 base environment per variant")
        if existing_base is not None:
            raise RuntimeError("U2 replay observed a live U1 base object-id collision")

        try:
            require_universal_trade_rl_u1_environment_contract(
                contract=self.u1_contract,
                environment=environment,
            )
            if environment.contract.digest != self.policy_contract.digest:
                raise ValueError("U2 replay policy contract mismatch")
            child_normalizer = environment.sequence_normalizer
            if not isinstance(child_normalizer, UniversalTradeSequenceNormalizer):
                raise ValueError("U2 replay requires the frozen sequence normalizer")
            if child_normalizer.digest != self.normalizer.digest:
                raise ValueError("U2 replay normalizer generation mismatch")
            if child_normalizer.provenance_digest != self.normalizer.provenance_digest:
                raise ValueError("U2 replay normalizer provenance mismatch")
            if environment.dataset.dataset_id != canonical.evaluation_dataset_digest:
                raise ValueError("U2 replay environment dataset identity mismatch")
            if environment.dataset.symbols != (canonical.concrete_symbol,):
                raise ValueError("U2 replay environment dataset symbol mismatch")
        except Exception:
            environment.close()
            raise

        self._issued_environments[object_id] = environment
        self._issued_base_environments[base_id] = base_environment
        return environment

    def _reset_scope_environment(
        self,
        environment: UniversalTradeEnvironment,
        scope: UniversalTradeRLU2EvaluationScope,
        *,
        evaluation_seed: int,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        canonical = self._require_canonical_scope(scope)
        issued = self._issued_environments.get(id(environment))
        if issued is not environment:
            raise ValueError("U2 replay environment was not issued by this session")
        if (
            isinstance(evaluation_seed, bool)
            or not isinstance(evaluation_seed, int)
            or evaluation_seed not in self.u2_contract.training_seeds
        ):
            raise ValueError("U2 replay evaluation seed must be preregistered")

        observation, info = environment.reset(
            seed=evaluation_seed,
            options={"start_idx": canonical.evaluation_start_bar_index},
        )
        expected_start = canonical.evaluation_start_bar_index
        expected_end = canonical.outcome_stop_bar_index_exclusive - 1
        if info.get("start_index") != expected_start:
            raise RuntimeError("U2 replay reset start index drifted")
        if info.get("end_index") != expected_end:
            raise RuntimeError("U2 replay runtime end index drifted")
        base = environment.base_env
        if base.start_index != expected_start or base.current_index != expected_start:
            raise RuntimeError("U2 replay environment start state drifted")
        if base.end_index != expected_end:
            raise RuntimeError("U2 replay environment end state drifted")
        return observation, info


def _require_runtime_generation(
    *,
    u2_contract: UniversalTradeRLU2Contract,
    u1_contract: UniversalTradeRLU1Contract,
    policy_contract: UniversalTradePolicyContract,
    normalizer: UniversalTradeSequenceNormalizer,
    environment_factory: U2ReplayEnvironmentFactory,
) -> None:
    if not isinstance(u1_contract, UniversalTradeRLU1Contract):
        raise TypeError("U2 replay requires a U1 contract")
    if not isinstance(policy_contract, UniversalTradePolicyContract):
        raise TypeError("U2 replay requires a policy contract")
    if not isinstance(normalizer, UniversalTradeSequenceNormalizer):
        raise TypeError("U2 replay requires the frozen sequence normalizer")
    if not callable(environment_factory):
        raise TypeError("U2 replay environment factory must be callable")
    if u2_contract.u1_contract_digest != u1_contract.digest:
        raise ValueError("U2 replay U1 contract identity mismatch")
    if u1_contract.policy_contract_digest != policy_contract.digest:
        raise ValueError("U2 replay policy contract identity mismatch")
    if u2_contract.u1_normalizer_digest != normalizer.digest:
        raise ValueError("U2 replay normalizer generation mismatch")
    if normalizer.contract_digest != policy_contract.digest:
        raise ValueError("U2 replay normalizer policy contract mismatch")


def build_universal_trade_rl_u2_development_replay_session(
    *,
    manifest: UniversalTradeRLUniverseManifest,
    time_partition: UniversalTradeRLU2TimePartition,
    u2_contract: UniversalTradeRLU2Contract,
    u1_contract: UniversalTradeRLU1Contract,
    policy_contract: UniversalTradePolicyContract,
    normalizer: UniversalTradeSequenceNormalizer,
    supplied_scope_closure: UniversalTradeRLU2DevelopmentScopeClosure,
    artifact_locators: Mapping[str, U2EvaluationSourceArtifactLocator],
    source_loader: U2EvaluationSourceArtifactLoader,
    environment_factory: U2ReplayEnvironmentFactory,
) -> UniversalTradeRLU2DevelopmentReplaySession:
    """Validate canonical metadata before crossing the Development numeric boundary."""

    if not isinstance(manifest, UniversalTradeRLUniverseManifest):
        raise TypeError("U2 replay requires a U0 manifest")
    if not isinstance(time_partition, UniversalTradeRLU2TimePartition):
        raise TypeError("U2 replay requires a time partition")
    if not isinstance(u2_contract, UniversalTradeRLU2Contract):
        raise TypeError("U2 replay requires a U2 contract")
    if not isinstance(
        supplied_scope_closure,
        UniversalTradeRLU2DevelopmentScopeClosure,
    ):
        raise TypeError("U2 replay requires a Development scope closure")

    canonical_scope_closure = build_universal_trade_rl_u2_development_scope_closure(
        manifest=manifest,
        time_partition=time_partition,
        u2_contract=u2_contract,
    )
    if supplied_scope_closure != canonical_scope_closure:
        raise ValueError("U2 Development replay supplied closure is not canonical")

    _require_runtime_generation(
        u2_contract=u2_contract,
        u1_contract=u1_contract,
        policy_contract=policy_contract,
        normalizer=normalizer,
        environment_factory=environment_factory,
    )
    if not callable(source_loader):
        raise TypeError("U2 replay source loader must be callable")

    datasets = load_universal_trade_rl_u2_development_evaluation_datasets(
        manifest=manifest,
        scope_closure=canonical_scope_closure,
        artifact_locators=artifact_locators,
        loader=source_loader,
    )
    return UniversalTradeRLU2DevelopmentReplaySession(
        manifest=manifest,
        time_partition=time_partition,
        u2_contract=u2_contract,
        u1_contract=u1_contract,
        policy_contract=policy_contract,
        normalizer=normalizer,
        scope_closure=canonical_scope_closure,
        datasets=datasets,
        environment_factory=environment_factory,
    )


__all__ = [
    "U2ReplayEnvironmentFactory",
    "UniversalTradeRLU2DevelopmentReplaySession",
    "UniversalTradeRLU2ReplayRequest",
    "UniversalTradeRLU2ReplayVariant",
    "build_universal_trade_rl_u2_development_replay_session",
]
