"""Executable runtime assembly for Universal single-instrument training."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any

from trade_rl.artifacts.atomic_write import atomic_write_bytes
from trade_rl.artifacts.codec import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest
from trade_rl.artifacts.verified_file import file_digest
from trade_rl.data.artifact import (
    load_market_dataset_artifact,
    publish_market_dataset_artifact,
)
from trade_rl.data.contracts import (
    InstrumentContract,
    InstrumentExecutionRule,
    VolumeUnit,
)
from trade_rl.domain.common import require_sha256
from trade_rl.integrations.sb3_training import StableBaselines3Backend
from trade_rl.integrations.universal_pretraining import (
    UniversalPretrainingBundle,
    build_universal_pretraining_hook,
)
from trade_rl.learning.episode_oracle_teacher import EpisodeOracleBatch
from trade_rl.risk.portfolio import PortfolioRiskModel
from trade_rl.risk.pretrade import PreTradeRisk
from trade_rl.rl.actions import ACTION_SCHEMA, ActionMode, ActionSpec
from trade_rl.rl.environment import ResidualMarketEnv
from trade_rl.rl.universal_instrument_binding import InstrumentDatasetBinding
from trade_rl.rl.universal_instrument_context import CausalInstrumentContextProvider
from trade_rl.rl.universal_single_instrument_env import EpisodeRoutedSingleInstrumentEnv
from trade_rl.strategies.trend import TrendStrategy
from trade_rl.workflows.universal_teacher_runtime import (
    build_universal_oracle_batches,
    build_universal_pretraining_bundle_from_batches,
    build_universal_symbol_teacher_environment,
)
from trade_rl.workflows.universal_training import (
    universal_training_contract_digest,
)


def _aware_datetime(value: object, *, field: str) -> datetime:
    if isinstance(value, datetime):
        resolved = value
    elif isinstance(value, str):
        token = value.strip().replace("Z", "+00:00")
        if not token:
            raise ValueError(f"{field} must not be empty")
        try:
            resolved = datetime.fromisoformat(token)
        except ValueError as error:
            raise ValueError(f"{field} must be an ISO-8601 datetime") from error
    else:
        raise TypeError(f"{field} must be a datetime or ISO-8601 string")
    if resolved.tzinfo is None or resolved.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware")
    return resolved.astimezone(UTC)


def _positive_number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{field} must be numeric")
    resolved = float(value)
    if not resolved > 0.0:
        raise ValueError(f"{field} must be positive")
    return resolved


def validate_universal_training_config(run_config: Any) -> None:
    """Reject any run configuration that violates the maintained Universal surface."""

    training = getattr(run_config, "training", None)
    action = getattr(run_config, "action", None)
    environment = getattr(run_config, "environment", None)
    if training is None or environment is None or not isinstance(action, ActionSpec):
        raise TypeError(
            "Universal training requires a complete TrainingRunConfig surface"
        )
    if (
        ActionMode(action.mode) is not ActionMode.TARGET_WEIGHT
        or action.target_weight_count != 1
        or action.alpha_enabled
        or action.risk_tilt_enabled
        or action.n_factors != 0
    ):
        raise ValueError(
            "Universal training requires exactly one scalar target-weight action"
        )
    if getattr(run_config, "alpha_artifact", None) is not None:
        raise ValueError(
            "Universal target-weight training does not accept alpha artifacts"
        )
    if getattr(run_config, "factor_artifact", None) is not None:
        raise ValueError(
            "Universal target-weight training does not accept factor artifacts"
        )
    if getattr(training, "observation_encoder", None) != "hierarchical_sequence_v2":
        raise ValueError("Universal training requires hierarchical_sequence_v2")
    if not bool(getattr(environment, "structured_sequence_observation", False)):
        raise ValueError("Universal training requires structured sequence observations")
    if not bool(getattr(environment, "finite_horizon_observation", False)):
        raise ValueError("Universal training requires finite-horizon observations")


def concrete_action_spec_digest(action: ActionSpec, symbol: str) -> str:
    """Bind the generic one-action specification to one concrete child environment."""

    if not isinstance(action, ActionSpec):
        raise TypeError("action must be an ActionSpec")
    if not isinstance(symbol, str) or not symbol:
        raise ValueError("concrete symbol must be non-empty")
    if (
        ActionMode(action.mode) is not ActionMode.TARGET_WEIGHT
        or action.target_weight_count != 1
    ):
        raise ValueError(
            "Universal concrete action identity requires one target-weight action"
        )
    return content_digest(
        {
            "schema_version": ACTION_SCHEMA,
            "alpha_enabled": action.alpha_enabled,
            "mode": ActionMode(action.mode).value,
            "risk_tilt_enabled": action.risk_tilt_enabled,
            "n_factors": action.n_factors,
            "names": action.names_for_symbols((symbol,)),
            "residual_scale": action.residual_scale,
            "target_weight_count": action.target_weight_count,
            "validation_mode": action.validation_mode.value,
        }
    )


def build_universal_instrument_contracts(
    metadata_resolution: Any,
    *,
    train_symbols: Sequence[str],
) -> dict[str, InstrumentContract]:
    """Build train-only instrument contracts used by causal descriptor generation."""

    symbols = tuple(train_symbols)
    if (
        not symbols
        or len(set(symbols)) != len(symbols)
        or any(not value for value in symbols)
    ):
        raise ValueError("Universal train_symbols must be non-empty and unique")
    raw_metadata = getattr(metadata_resolution, "metadata", None)
    if not isinstance(raw_metadata, Mapping):
        raise TypeError("Universal metadata resolution must expose metadata")
    raw_histories = getattr(metadata_resolution, "execution_rule_histories", None)
    if raw_histories is not None and not isinstance(raw_histories, Mapping):
        raise TypeError("Universal execution_rule_histories must be a mapping or null")

    contracts: dict[str, InstrumentContract] = {}
    for symbol in symbols:
        raw = raw_metadata.get(symbol)
        if not isinstance(raw, Mapping):
            raise ValueError(f"Universal metadata is missing train symbol {symbol}")
        delisted_raw = raw.get("delisted_at")
        delisted_at = (
            None
            if delisted_raw is None
            else _aware_datetime(delisted_raw, field=f"{symbol}.delisted_at")
        )
        execution_rules: tuple[InstrumentExecutionRule, ...] = ()
        if raw_histories is not None:
            history = raw_histories.get(symbol)
            if not isinstance(history, (tuple, list)) or any(
                not isinstance(item, InstrumentExecutionRule) for item in history
            ):
                raise ValueError(
                    f"Universal execution rules are missing or invalid for {symbol}"
                )
            execution_rules = tuple(history)
        contracts[symbol] = InstrumentContract(
            symbol=symbol,
            listed_at=_aware_datetime(
                raw.get("listed_at"), field=f"{symbol}.listed_at"
            ),
            delisted_at=delisted_at,
            volume_unit=VolumeUnit.QUOTE_NOTIONAL,
            tick_size=_positive_number(
                raw.get("tick_size"), field=f"{symbol}.tick_size"
            ),
            lot_size=_positive_number(raw.get("lot_size"), field=f"{symbol}.lot_size"),
            minimum_notional=_positive_number(
                raw.get("minimum_notional"),
                field=f"{symbol}.minimum_notional",
            ),
            execution_rules=execution_rules,
        )
    return contracts


def _build_universal_concrete_environment(
    dataset: Any,
    *,
    run_config: Any,
    normalizers: tuple[Any, Any],
) -> ResidualMarketEnv:
    flat_normalizer, sequence_normalizer = normalizers
    return ResidualMarketEnv(
        dataset,
        trend_strategy=TrendStrategy(run_config.trend),
        alpha_enabled=False,
        factor_count=0,
        action_spec=run_config.action,
        pre_trade_risk=PreTradeRisk(run_config.risk),
        portfolio_risk=PortfolioRiskModel(run_config.portfolio_risk),
        normalizer=flat_normalizer,
        sequence_normalizer=sequence_normalizer,
        config=run_config.environment,
    )


def publish_universal_train_dataset_artifacts(
    datasets: Mapping[str, Any],
    *,
    train_symbols: Sequence[str],
    artifact_root: Path,
) -> dict[str, Path]:
    """Publish or strictly reuse immutable train-only single-symbol datasets."""

    symbols = tuple(train_symbols)
    if (
        not symbols
        or len(set(symbols)) != len(symbols)
        or any(not symbol for symbol in symbols)
    ):
        raise ValueError("Universal train_symbols must be non-empty and unique")
    if set(datasets) != set(symbols):
        raise ValueError(
            "Universal dataset artifact scope must exactly match train_symbols"
        )
    artifact_root.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for symbol in symbols:
        dataset = datasets[symbol]
        if tuple(getattr(dataset, "symbols", ())) != (symbol,):
            raise ValueError("Universal dataset artifact symbol identity mismatch")
        dataset_id = getattr(dataset, "dataset_id", None)
        if not isinstance(dataset_id, str):
            raise ValueError("Universal dataset artifact identity is unavailable")
        destination = artifact_root / symbol
        if destination.exists():
            existing = load_market_dataset_artifact(destination)
            if (
                tuple(getattr(existing, "symbols", ())) != (symbol,)
                or getattr(existing, "dataset_id", None) != dataset_id
            ):
                raise ValueError(
                    "Universal existing dataset artifact identity mismatch"
                )
        else:
            publish_market_dataset_artifact(destination, dataset)
        paths[symbol] = destination
    return paths


@dataclass(frozen=True, slots=True)
class UniversalDatasetArtifactEnvironmentFactory:
    """Load one concrete dataset lazily from immutable artifacts inside each worker."""

    dataset_artifact_paths: Mapping[str, Path]
    run_config: Any
    normalizers: Mapping[str, tuple[Any, Any]]

    def __post_init__(self) -> None:
        paths = {
            str(symbol): Path(value)
            for symbol, value in self.dataset_artifact_paths.items()
        }
        normalizers = dict(self.normalizers)
        if not paths or set(paths) != set(normalizers):
            raise ValueError(
                "Universal dataset artifact paths and normalizers must share scope"
            )
        if any(not symbol for symbol in paths):
            raise ValueError("Universal dataset artifact symbols must be non-empty")
        object.__setattr__(self, "dataset_artifact_paths", paths)
        object.__setattr__(self, "normalizers", normalizers)

    def __call__(self, binding: InstrumentDatasetBinding) -> Any:
        if not isinstance(binding, InstrumentDatasetBinding):
            raise TypeError("binding must be an InstrumentDatasetBinding")
        if binding.split != "train":
            raise ValueError("Universal training factory accepts train bindings only")
        symbol = binding.concrete_symbol
        try:
            dataset_path = self.dataset_artifact_paths[symbol]
            normalizers = self.normalizers[symbol]
        except KeyError as error:
            raise ValueError(
                "Universal dataset artifact binding is outside factory scope"
            ) from error
        dataset = load_market_dataset_artifact(dataset_path)
        if tuple(getattr(dataset, "symbols", ())) != (symbol,):
            raise ValueError("Universal dataset artifact symbol identity mismatch")
        dataset_id = getattr(dataset, "dataset_id", None)
        if (
            dataset_id != binding.source_dataset_id
            or dataset_id != binding.symbol_dataset_digest
        ):
            raise ValueError("Universal dataset artifact identity mismatch")
        return _build_universal_concrete_environment(
            dataset,
            run_config=self.run_config,
            normalizers=normalizers,
        )


@dataclass(frozen=True, slots=True)
class UniversalRoutedEnvironmentFactory:
    """Pickle-safe routed environment factory with explicit vector-worker identity."""

    train_symbols: tuple[str, ...]
    partition_digest: str
    bindings: tuple[InstrumentDatasetBinding, ...]
    concrete_environment_factory: Callable[[InstrumentDatasetBinding], Any]
    instrument_context_provider: CausalInstrumentContextProvider | None
    training_contract_digest: str
    run_seed: int
    max_cached_environments: int | None = 1

    def __post_init__(self) -> None:
        symbols = tuple(self.train_symbols)
        if not symbols or len(set(symbols)) != len(symbols):
            raise ValueError("Universal routed factory train_symbols are invalid")
        if tuple(binding.concrete_symbol for binding in self.bindings) != symbols:
            raise ValueError(
                "Universal routed factory bindings must follow train_symbols"
            )
        if any(binding.split != "train" for binding in self.bindings):
            raise ValueError(
                "Universal routed training factory accepts train bindings only"
            )
        if not callable(self.concrete_environment_factory):
            raise TypeError("concrete_environment_factory must be callable")
        if (
            isinstance(self.run_seed, bool)
            or not isinstance(self.run_seed, int)
            or self.run_seed < 0
        ):
            raise ValueError("Universal routed factory run_seed must be non-negative")

    @property
    def digest(self) -> str:
        return content_digest(
            {
                "bindings": tuple(binding.digest for binding in self.bindings),
                "partition_digest": self.partition_digest,
                "run_seed": self.run_seed,
                "max_cached_environments": self.max_cached_environments,
                "schema_version": "universal_routed_environment_factory_v1",
                "training_contract_digest": self.training_contract_digest,
                "train_symbols": self.train_symbols,
            }
        )

    def _create(self, environment_index: int) -> EpisodeRoutedSingleInstrumentEnv:
        return EpisodeRoutedSingleInstrumentEnv(
            train_symbols=self.train_symbols,
            partition_digest=self.partition_digest,
            bindings=self.bindings,
            environment_factory=self.concrete_environment_factory,
            run_seed=self.run_seed,
            environment_index=environment_index,
            instrument_context_provider=self.instrument_context_provider,
            training_contract_digest=self.training_contract_digest,
            max_cached_environments=self.max_cached_environments,
        )

    def __call__(self) -> EpisodeRoutedSingleInstrumentEnv:
        return self._create(0)

    def for_environment_index(
        self,
        index: int,
    ) -> Callable[[], EpisodeRoutedSingleInstrumentEnv]:
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise ValueError("Universal environment index must be non-negative")
        return partial(self._create, index)


def build_universal_bindings(
    *,
    datasets: Mapping[str, Any],
    contracts: Mapping[str, Any],
    catalog: Any,
    train_symbols: Sequence[str],
) -> tuple[InstrumentDatasetBinding, ...]:
    """Bind each concrete train dataset to immutable metadata and descriptor evidence."""

    symbols = tuple(train_symbols)
    if set(datasets) != set(symbols) or set(contracts) != set(symbols):
        raise ValueError("Universal bindings must close exactly over train_symbols")
    raw_metadata = getattr(catalog, "per_symbol_metadata_digests", None)
    if not isinstance(raw_metadata, (tuple, list)):
        raise TypeError("Universal catalog metadata digests are unavailable")
    metadata_by_symbol = {str(symbol): str(digest) for symbol, digest in raw_metadata}
    missing = set(symbols) - set(metadata_by_symbol)
    if missing:
        raise ValueError(
            f"Universal catalog metadata digests are missing train symbols: {sorted(missing)}"
        )

    bindings: list[InstrumentDatasetBinding] = []
    for symbol in symbols:
        dataset_id = getattr(datasets[symbol], "dataset_id", None)
        if not isinstance(dataset_id, str):
            raise ValueError(f"Universal dataset identity is unavailable for {symbol}")
        contract = contracts[symbol]
        payload_method = getattr(contract, "canonical_payload", None)
        if not callable(payload_method):
            raise TypeError(
                "Universal instrument contract must expose canonical_payload"
            )
        bindings.append(
            InstrumentDatasetBinding(
                concrete_symbol=symbol,
                source_dataset_id=dataset_id,
                symbol_dataset_digest=dataset_id,
                execution_metadata_digest=metadata_by_symbol[symbol],
                instrument_descriptor_digest=content_digest(payload_method()),
                split="train",
            )
        )
    return tuple(bindings)


@dataclass(frozen=True, slots=True)
class UniversalTrainingRuntime:
    """Immutable candidate-specific Universal runtime identity."""

    train_symbols: tuple[str, ...]
    catalog_digest: str
    partition_digest: str
    split_manifest_digest: str
    feature_schema_digest: str
    statistics_digest: str
    instrument_context_schema_digest: str
    training_contract_digest: str
    routed_environment_factory: UniversalRoutedEnvironmentFactory
    pretraining_artifact_digest: str | None = None

    def __post_init__(self) -> None:
        symbols = tuple(self.train_symbols)
        if not symbols or len(set(symbols)) != len(symbols):
            raise ValueError(
                "Universal runtime train_symbols must be non-empty and unique"
            )
        for field_name, value in (
            ("catalog_digest", self.catalog_digest),
            ("partition_digest", self.partition_digest),
            ("split_manifest_digest", self.split_manifest_digest),
            ("feature_schema_digest", self.feature_schema_digest),
            ("statistics_digest", self.statistics_digest),
            ("instrument_context_schema_digest", self.instrument_context_schema_digest),
            ("training_contract_digest", self.training_contract_digest),
        ):
            require_sha256(value, field=f"Universal runtime {field_name}")
        if self.pretraining_artifact_digest is not None:
            require_sha256(
                self.pretraining_artifact_digest,
                field="Universal runtime pretraining_artifact_digest",
            )
        if not isinstance(
            self.routed_environment_factory, UniversalRoutedEnvironmentFactory
        ):
            raise TypeError(
                "Universal runtime routed_environment_factory must be UniversalRoutedEnvironmentFactory"
            )
        if self.routed_environment_factory.train_symbols != symbols:
            raise ValueError("Universal runtime routed symbol scope mismatch")
        if self.routed_environment_factory.partition_digest != self.partition_digest:
            raise ValueError("Universal runtime routed partition identity mismatch")
        if (
            self.routed_environment_factory.training_contract_digest
            != self.training_contract_digest
        ):
            raise ValueError("Universal runtime routed training contract mismatch")
        object.__setattr__(self, "train_symbols", symbols)

    def with_pretraining_artifact(
        self, artifact_digest: str
    ) -> UniversalTrainingRuntime:
        require_sha256(
            artifact_digest,
            field="Universal runtime pretraining_artifact_digest",
        )
        return replace(self, pretraining_artifact_digest=artifact_digest)


def build_universal_training_runtime(
    *,
    train_symbols: Sequence[str],
    catalog_digest: str,
    partition_digest: str,
    split_manifest_digest: str,
    feature_schema_digest: str,
    statistics_digest: str,
    instrument_context_schema_digest: str,
    routed_environment_factory: UniversalRoutedEnvironmentFactory,
    training: Any,
) -> UniversalTrainingRuntime:
    """Rebind one routed factory to the exact architecture-specific training identity."""

    if not isinstance(routed_environment_factory, UniversalRoutedEnvironmentFactory):
        raise TypeError(
            "routed_environment_factory must be a UniversalRoutedEnvironmentFactory"
        )
    digest_payload = getattr(training, "digest_payload", None)
    if not callable(digest_payload):
        raise TypeError("Universal training config must expose digest_payload")
    training_config_digest = content_digest(digest_payload())
    training_contract_digest = universal_training_contract_digest(
        partition_digest=partition_digest,
        feature_schema_digest=feature_schema_digest,
        statistics_digest=statistics_digest,
        instrument_context_schema_digest=instrument_context_schema_digest,
        training_config_digest=training_config_digest,
    )
    bound_factory = replace(
        routed_environment_factory,
        training_contract_digest=training_contract_digest,
    )
    return UniversalTrainingRuntime(
        train_symbols=tuple(train_symbols),
        catalog_digest=catalog_digest,
        partition_digest=partition_digest,
        split_manifest_digest=split_manifest_digest,
        feature_schema_digest=feature_schema_digest,
        statistics_digest=statistics_digest,
        instrument_context_schema_digest=instrument_context_schema_digest,
        training_contract_digest=training_contract_digest,
        routed_environment_factory=bound_factory,
    )


def assemble_universal_sb3_training_backend(
    *,
    routed_environment_factory: UniversalRoutedEnvironmentFactory,
    training: Any,
    fold_train_range: tuple[int, int],
    normalizer_digest: str,
    feature_schema_digest: str,
    oracle_batches: Mapping[str, EpisodeOracleBatch] | None = None,
    verbose: int = 0,
) -> tuple[StableBaselines3Backend, UniversalPretrainingBundle]:
    """Assemble the maintained U4 Oracle -> BC/critic -> SB3 training path."""

    if not isinstance(routed_environment_factory, UniversalRoutedEnvironmentFactory):
        raise TypeError(
            "routed_environment_factory must be a UniversalRoutedEnvironmentFactory"
        )
    behavior_cloning_epochs = getattr(training, "behavior_cloning_epochs", None)
    if (
        isinstance(behavior_cloning_epochs, bool)
        or not isinstance(behavior_cloning_epochs, int)
        or behavior_cloning_epochs <= 0
    ):
        raise ValueError("Universal U4 requires behavior cloning to be enabled")
    if getattr(training, "behavior_cloning_teacher", None) != "oracle":
        raise ValueError("Universal U4 requires the Oracle behavior-cloning teacher")
    behavior_cloning_seed = getattr(training, "behavior_cloning_seed", None)
    if (
        isinstance(behavior_cloning_seed, bool)
        or not isinstance(behavior_cloning_seed, int)
        or not 0 <= behavior_cloning_seed <= 0xFFFFFFFF
    ):
        raise ValueError(
            "Universal U4 requires an explicit uint32 behavior_cloning_seed"
        )
    n_envs = getattr(training, "n_envs", None)
    if isinstance(n_envs, bool) or not isinstance(n_envs, int) or n_envs <= 0:
        raise ValueError("Universal U4 requires a positive n_envs")
    gamma = getattr(training, "gamma", None)
    if isinstance(gamma, bool) or not isinstance(gamma, int | float):
        raise ValueError("Universal U4 gamma must be numeric")
    gamma_value = float(gamma)
    if not 0.0 < gamma_value <= 1.0:
        raise ValueError("Universal U4 gamma must be in (0, 1]")
    validation_fraction = getattr(
        training, "behavior_cloning_validation_fraction", None
    )
    if (
        isinstance(validation_fraction, bool)
        or not isinstance(validation_fraction, int | float)
        or not 0.0 <= float(validation_fraction) < 0.5
    ):
        raise ValueError(
            "Universal U4 behavior_cloning_validation_fraction must be in [0, 0.5)"
        )
    if isinstance(verbose, bool) or not isinstance(verbose, int) or verbose < 0:
        raise ValueError("Universal U4 verbose must be a non-negative integer")
    provider = routed_environment_factory.instrument_context_provider
    if not callable(provider):
        raise ValueError("Universal U4 requires an instrument context provider")

    if oracle_batches is None:
        batches = build_universal_oracle_batches(
            train_symbols=routed_environment_factory.train_symbols,
            bindings=routed_environment_factory.bindings,
            concrete_environment_factory=(
                routed_environment_factory.concrete_environment_factory
            ),
            fold_train_range=fold_train_range,
            behavior_cloning_seed=behavior_cloning_seed,
            n_envs=n_envs,
        )
    else:
        batches = dict(oracle_batches)
        if set(batches) != set(routed_environment_factory.train_symbols):
            raise ValueError(
                "Universal U4 oracle_batches must exactly match train_symbols"
            )
        if any(not isinstance(batch, EpisodeOracleBatch) for batch in batches.values()):
            raise TypeError(
                "Universal U4 oracle_batches must contain EpisodeOracleBatch"
            )
    bundle = build_universal_pretraining_bundle_from_batches(
        train_symbols=routed_environment_factory.train_symbols,
        bindings=routed_environment_factory.bindings,
        batches=batches,
        concrete_environment_factory=(
            routed_environment_factory.concrete_environment_factory
        ),
        instrument_context_provider=provider,
        partition_digest=routed_environment_factory.partition_digest,
        training_contract_digest=routed_environment_factory.training_contract_digest,
        run_seed=routed_environment_factory.run_seed,
        gamma=gamma_value,
        validation_fraction=float(validation_fraction),
        normalizer_digest=normalizer_digest,
        feature_schema_digest=feature_schema_digest,
    )
    binding_by_symbol = {
        binding.concrete_symbol: binding
        for binding in routed_environment_factory.bindings
    }
    symbol_environment_factories = {
        symbol: partial(
            build_universal_symbol_teacher_environment,
            symbol=symbol,
            binding=binding_by_symbol[symbol],
            concrete_environment_factory=(
                routed_environment_factory.concrete_environment_factory
            ),
            instrument_context_provider=provider,
            partition_digest=routed_environment_factory.partition_digest,
            training_contract_digest=(
                routed_environment_factory.training_contract_digest
            ),
            run_seed=routed_environment_factory.run_seed + index,
        )
        for index, symbol in enumerate(routed_environment_factory.train_symbols)
    }
    backend = StableBaselines3Backend(
        routed_environment_factory,
        verbose=verbose,
        universal_pretraining_hook=build_universal_pretraining_hook(
            bundle,
            symbol_environment_factories=symbol_environment_factories,
        ),
    )
    return backend, bundle


def train_universal_seeds(
    *,
    runtime: Any,
    training: Any,
    backend: Any,
    output_root: Path,
    architecture_name: str,
) -> dict[str, object]:
    """Train every configured member and publish one non-research-success manifest."""

    seeds = tuple(getattr(training, "seeds", ()))
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Universal training seeds must be non-empty and unique")
    if any(
        isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
        for seed in seeds
    ):
        raise ValueError("Universal training seeds must be non-negative integers")
    digest_payload = getattr(training, "digest_payload", None)
    if not callable(digest_payload):
        raise TypeError("Universal training config must expose digest_payload")
    training_config_digest = content_digest(digest_payload())
    output_root.mkdir(parents=True, exist_ok=True)

    members: list[dict[str, object]] = []
    for seed in seeds:
        policy_path = output_root / f"seed-{seed}" / "policy.zip"
        result = backend.train(seed=seed, config=training, output_path=policy_path)
        checkpoint_path = Path(getattr(result, "checkpoint_path", policy_path))
        if checkpoint_path != policy_path or not policy_path.is_file():
            raise RuntimeError(
                "Universal backend did not publish the requested policy path"
            )
        environment_digest = getattr(result, "environment_digest", None)
        if not isinstance(environment_digest, str):
            raise ValueError("Universal backend environment digest is unavailable")
        architecture_digest = getattr(result, "architecture_digest", None)
        actual_timesteps = getattr(result, "actual_timesteps", None)
        if isinstance(actual_timesteps, bool) or not isinstance(actual_timesteps, int):
            raise ValueError("Universal backend actual_timesteps is unavailable")
        members.append(
            {
                "seed": seed,
                "policy_file": policy_path.relative_to(output_root).as_posix(),
                "policy_digest": file_digest(policy_path, field="Universal policy"),
                "environment_digest": environment_digest,
                "architecture_digest": architecture_digest,
                "actual_timesteps": actual_timesteps,
            }
        )

    payload: dict[str, object] = {
        "schema_version": "universal_training_run_v1",
        "architecture_name": architecture_name,
        "train_symbols": list(getattr(runtime, "train_symbols")),
        "catalog_digest": getattr(runtime, "catalog_digest"),
        "partition_digest": getattr(runtime, "partition_digest"),
        "split_manifest_digest": getattr(runtime, "split_manifest_digest"),
        "feature_schema_digest": getattr(runtime, "feature_schema_digest"),
        "statistics_digest": getattr(runtime, "statistics_digest"),
        "instrument_context_schema_digest": getattr(
            runtime, "instrument_context_schema_digest"
        ),
        "training_contract_digest": getattr(runtime, "training_contract_digest"),
        "pretraining_artifact_digest": getattr(
            runtime, "pretraining_artifact_digest", None
        ),
        "training_config_digest": training_config_digest,
        "members": members,
        "research_success": False,
        "research_success_reason": "sealed zero-shot evidence not evaluated by training runner",
    }
    run_digest = content_digest(payload)
    manifest = {**payload, "run_digest": run_digest}
    atomic_write_bytes(
        output_root / "universal-training.json",
        canonical_json_bytes(manifest),
    )
    return manifest


__all__ = [
    "build_universal_training_runtime",
    "UniversalTrainingRuntime",
    "assemble_universal_sb3_training_backend",
    "UniversalDatasetArtifactEnvironmentFactory",
    "UniversalRoutedEnvironmentFactory",
    "build_universal_bindings",
    "train_universal_seeds",
    "build_universal_instrument_contracts",
    "concrete_action_spec_digest",
    "publish_universal_train_dataset_artifacts",
    "validate_universal_training_config",
]
