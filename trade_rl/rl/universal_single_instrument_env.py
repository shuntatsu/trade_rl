"""Episode-routed Gymnasium facade for universal single-instrument training."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Final, cast

import gymnasium as gym
import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.data.universal_features import UNIVERSAL_INSTRUMENT_DESCRIPTOR_NAMES
from trade_rl.domain.common import require_sha256
from trade_rl.rl.actions import ActionMode, ActionSpec, ActionValidationMode
from trade_rl.rl.universal_episode_router import (
    DeterministicBalancedInstrumentRouter,
    InstrumentRoute,
)
from trade_rl.rl.universal_instrument_binding import (
    GENERIC_INSTRUMENT_SYMBOLS,
    GENERIC_TARGET_WEIGHT_ACTION_NAMES,
    InstrumentDatasetBinding,
    InstrumentEpisodeBinding,
    validate_training_instrument_bindings,
)
from trade_rl.rl.universal_v4_context import V4ContextProvider

INSTRUMENT_EPISODE_INFO_KEY: Final = "instrument_episode_binding"
INSTRUMENT_EPISODE_DIGEST_INFO_KEY: Final = "instrument_episode_binding_digest"
UNIVERSAL_OBSERVATION_SCHEMA: Final = "universal_single_instrument_observation_v1"
_V4_OBSERVATION_KEYS: Final = (
    "local_cross_market_context",
    "local_cross_market_available",
    "local_cross_market_staleness_hours",
    "global_market_context",
    "global_market_available",
    "global_market_staleness_hours",
    "causal_beta",
    "causal_beta_available",
)

ConcreteSingleInstrumentEnv = gym.Env[Any, np.ndarray]
InstrumentEnvironmentFactory = Callable[
    [InstrumentDatasetBinding],
    ConcreteSingleInstrumentEnv,
]
InstrumentContextProvider = Callable[
    [object, InstrumentDatasetBinding],
    np.ndarray,
]


def _require_non_negative_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _episode_boundary(
    info: Mapping[str, Any],
    *,
    field: str,
) -> int:
    if field not in info:
        raise ValueError(f"concrete reset info is missing {field}")
    return _require_non_negative_int(info[field], field=field)


def _optional_digest(value: object) -> str | None:
    if value is None:
        return None
    digest = getattr(value, "statistics_digest", None)
    if digest is None:
        digest = getattr(value, "digest", None)
    if digest is None:
        return None
    if not isinstance(digest, str):
        raise TypeError("training component digest must be a string")
    require_sha256(digest, field="training component digest")
    return digest


class EpisodeRoutedSingleInstrumentEnv(gym.Env[Any, np.ndarray]):
    """Expose one generic action while routing complete episodes by symbol."""

    metadata = {"render_modes": []}
    is_universal_single_instrument: Final = True

    def __init__(
        self,
        *,
        train_symbols: Sequence[str],
        partition_digest: str,
        bindings: Sequence[InstrumentDatasetBinding],
        environment_factory: InstrumentEnvironmentFactory,
        run_seed: int,
        environment_index: int,
        instrument_context_provider: InstrumentContextProvider | None = None,
        v4_context_provider: V4ContextProvider | None = None,
        training_contract_digest: str | None = None,
        max_cached_environments: int | None = None,
    ) -> None:
        super().__init__()
        if not callable(environment_factory):
            raise TypeError("environment_factory must be callable")
        if instrument_context_provider is not None and not callable(
            instrument_context_provider
        ):
            raise TypeError("instrument_context_provider must be callable")
        if v4_context_provider is not None and not isinstance(
            v4_context_provider, V4ContextProvider
        ):
            raise TypeError("v4_context_provider must be a V4ContextProvider")
        if training_contract_digest is not None:
            training_contract_digest = require_sha256(
                training_contract_digest,
                field="training_contract_digest",
            )
        if max_cached_environments is not None and (
            isinstance(max_cached_environments, bool)
            or not isinstance(max_cached_environments, int)
            or max_cached_environments <= 0
        ):
            raise ValueError(
                "max_cached_environments must be null or a positive integer"
            )
        self._bindings = validate_training_instrument_bindings(
            train_symbols,
            bindings,
        )
        self._router = DeterministicBalancedInstrumentRouter(
            train_symbols=tuple(self._bindings),
            partition_digest=partition_digest,
            run_seed=run_seed,
            environment_index=environment_index,
        )
        self._environment_factory = environment_factory
        self._instrument_context_provider = instrument_context_provider
        self._v4_context_provider = v4_context_provider
        if v4_context_provider is not None:
            missing_v4_symbols = set(self._bindings) - set(
                v4_context_provider.contexts
            )
            if missing_v4_symbols:
                raise ValueError(
                    "V4 context provider does not cover routed symbols: "
                    f"{sorted(missing_v4_symbols)}"
                )
        self._training_contract_digest = training_contract_digest
        self._training_identity_enabled = training_contract_digest is not None
        self._max_cached_environments = max_cached_environments
        self._run_seed = self._router.run_seed
        self._environment_index = self._router.environment_index
        self._environments: dict[str, ConcreteSingleInstrumentEnv] = {}
        self._environment_object_ids: set[int] = set()
        self._active_environment: ConcreteSingleInstrumentEnv | None = None
        self._active_episode_binding: InstrumentEpisodeBinding | None = None
        self._episode_complete = True
        self._completed_episode_count = 0

        initial_route = self._router.route(0)
        initial_environment = self._load_environment(initial_route)
        self._reference_environment = initial_environment
        self._concrete_observation_space = initial_environment.observation_space
        self.observation_space = self._policy_observation_space(
            initial_environment.observation_space
        )
        self.action_space = cast(
            gym.spaces.Space[np.ndarray],
            initial_environment.action_space,
        )
        self.action_spec = ActionSpec(
            mode=ActionMode.TARGET_WEIGHT,
            risk_tilt_enabled=False,
            target_weight_count=1,
            validation_mode=ActionValidationMode.FAIL_CLOSED,
        )
        self.metadata = dict(getattr(initial_environment, "metadata", self.metadata))
        self._sequence_layout_metadata = self._resolve_sequence_layout_metadata(
            initial_environment
        )

        raw_initial_capital = getattr(initial_environment, "initial_capital", None)
        raw_decision_hours = getattr(initial_environment, "decision_hours", None)
        self._initial_capital = (
            None if raw_initial_capital is None else float(raw_initial_capital)
        )
        self._decision_hours = (
            None if raw_decision_hours is None else float(raw_decision_hours)
        )
        concrete_observation_digest = getattr(
            initial_environment,
            "observation_contract_digest",
            None,
        )
        source_environment_digest = getattr(
            initial_environment,
            "environment_digest",
            None,
        )
        if self._training_identity_enabled:
            if (
                self._initial_capital is None
                or not np.isfinite(self._initial_capital)
                or self._initial_capital <= 0.0
            ):
                raise ValueError(
                    "concrete environment initial_capital must be positive"
                )
            if (
                self._decision_hours is None
                or not np.isfinite(self._decision_hours)
                or self._decision_hours <= 0.0
            ):
                raise ValueError("concrete environment decision_hours must be positive")
            if not isinstance(concrete_observation_digest, str):
                raise TypeError(
                    "concrete environment must expose observation_contract_digest"
                )
            require_sha256(
                concrete_observation_digest,
                field="concrete observation_contract_digest",
            )
            if not isinstance(source_environment_digest, str):
                raise TypeError("concrete environment must expose environment_digest")
            require_sha256(
                source_environment_digest,
                field="concrete environment_digest",
            )
        elif source_environment_digest is not None:
            if not isinstance(source_environment_digest, str):
                raise TypeError("concrete environment_digest must be a string")
            require_sha256(
                source_environment_digest,
                field="concrete environment_digest",
            )

        context_schema_digest = getattr(
            instrument_context_provider,
            "schema_digest",
            None,
        )
        if context_schema_digest is not None:
            require_sha256(
                context_schema_digest,
                field="instrument context schema digest",
            )
        v4_context_schema_digest = (
            None
            if v4_context_provider is None
            else v4_context_provider.schema_digest
        )
        if v4_context_schema_digest is not None:
            require_sha256(
                v4_context_schema_digest,
                field="V4 context schema digest",
            )
        self._observation_contract_digest = content_digest(
            {
                "concrete_observation_contract_digest": (
                    concrete_observation_digest
                    if training_contract_digest is None
                    else None
                ),
                "instrument_context_schema_digest": context_schema_digest,
                "schema_version": UNIVERSAL_OBSERVATION_SCHEMA,
                "training_contract_digest": training_contract_digest,
                "v4_context_schema_digest": v4_context_schema_digest,
            }
        )
        self._environment_digest = content_digest(
            {
                "router_digest": self._router.digest,
                "schema_version": "universal_routed_environment_v1",
                "source_environment_digest": (
                    source_environment_digest
                    if training_contract_digest is None
                    else None
                ),
                "training_contract_digest": training_contract_digest,
            }
        )

    def _policy_observation_space(
        self,
        observation_space: gym.spaces.Space[Any],
    ) -> gym.spaces.Space[Any]:
        instrument_provider = self._instrument_context_provider
        v4_provider = self._v4_context_provider
        if instrument_provider is None and v4_provider is None:
            return observation_space
        if not isinstance(observation_space, gym.spaces.Dict):
            raise TypeError("Universal context requires a Dict concrete observation space")
        spaces = dict(observation_space.spaces)
        if instrument_provider is not None:
            if "instrument_context" in spaces:
                raise ValueError("concrete observation already contains instrument_context")
            spaces["instrument_context"] = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(1, len(UNIVERSAL_INSTRUMENT_DESCRIPTOR_NAMES)),
                dtype=np.float32,
            )
        if v4_provider is not None:
            collisions = tuple(key for key in _V4_OBSERVATION_KEYS if key in spaces)
            if collisions:
                raise ValueError(
                    f"concrete observation already contains V4 context keys: {collisions}"
                )
            local_shape = (1, v4_provider.local_width)
            global_shape = (1, v4_provider.global_width)
            spaces.update(
                {
                    "local_cross_market_context": gym.spaces.Box(
                        low=-np.inf, high=np.inf, shape=local_shape, dtype=np.float32
                    ),
                    "local_cross_market_available": gym.spaces.Box(
                        low=0.0, high=1.0, shape=local_shape, dtype=np.float32
                    ),
                    "local_cross_market_staleness_hours": gym.spaces.Box(
                        low=0.0, high=np.inf, shape=local_shape, dtype=np.float32
                    ),
                    "global_market_context": gym.spaces.Box(
                        low=-np.inf, high=np.inf, shape=global_shape, dtype=np.float32
                    ),
                    "global_market_available": gym.spaces.Box(
                        low=0.0, high=1.0, shape=global_shape, dtype=np.float32
                    ),
                    "global_market_staleness_hours": gym.spaces.Box(
                        low=0.0, high=np.inf, shape=global_shape, dtype=np.float32
                    ),
                    "causal_beta": gym.spaces.Box(
                        low=-3.0, high=3.0, shape=(1, 1), dtype=np.float32
                    ),
                    "causal_beta_available": gym.spaces.Box(
                        low=0.0, high=1.0, shape=(1, 1), dtype=np.float32
                    ),
                }
            )
        return gym.spaces.Dict(spaces)

    def _resolve_sequence_layout_metadata(
        self,
        environment: ConcreteSingleInstrumentEnv,
    ) -> dict[str, Any] | None:
        raw = getattr(environment, "sequence_layout_metadata", None)
        if raw is None:
            if self._v4_context_provider is None:
                return None
            resolved: dict[str, Any] = {}
        else:
            if not isinstance(raw, dict):
                raise TypeError("concrete sequence_layout_metadata must be a dict")
            resolved = dict(raw)
        if self._instrument_context_provider is not None:
            resolved["instrument_context_width"] = len(
                UNIVERSAL_INSTRUMENT_DESCRIPTOR_NAMES
            )
        if self._v4_context_provider is not None:
            resolved["v4_local_context_width"] = self._v4_context_provider.local_width
            resolved["v4_global_context_width"] = self._v4_context_provider.global_width
            resolved["v4_context_schema_digest"] = (
                self._v4_context_provider.schema_digest
            )
        return resolved

    def _training_surface(
        self,
        environment: ConcreteSingleInstrumentEnv,
    ) -> dict[str, object]:
        return {
            "action_space": environment.action_space,
            "decision_hours": float(getattr(environment, "decision_hours")),
            "initial_capital": float(getattr(environment, "initial_capital")),
            "normalizer_digest": _optional_digest(
                getattr(environment, "normalizer", None)
            ),
            "observation_contract_digest": (
                None
                if self._training_identity_enabled
                else getattr(environment, "observation_contract_digest", None)
            ),
            "observation_schema": getattr(environment, "observation_schema", None),
            "sequence_layout_metadata": self._resolve_sequence_layout_metadata(
                environment
            ),
            "sequence_normalizer_digest": _optional_digest(
                getattr(environment, "sequence_normalizer", None)
            ),
        }

    @property
    def policy_symbols(self) -> tuple[str, ...]:
        return GENERIC_INSTRUMENT_SYMBOLS

    @property
    def symbols(self) -> tuple[str, ...]:
        return GENERIC_INSTRUMENT_SYMBOLS

    @property
    def action_names(self) -> tuple[str, ...]:
        return GENERIC_TARGET_WEIGHT_ACTION_NAMES

    @property
    def action_spec_digest(self) -> str:
        return content_digest(
            {
                "action_names": GENERIC_TARGET_WEIGHT_ACTION_NAMES,
                "mode": ActionMode.TARGET_WEIGHT.value,
                "schema_version": "universal_scalar_target_weight_action_v1",
                "size": 1,
            }
        )

    @property
    def observation_schema(self) -> str:
        return UNIVERSAL_OBSERVATION_SCHEMA

    @property
    def observation_contract_digest(self) -> str:
        return self._observation_contract_digest

    @property
    def environment_digest(self) -> str:
        return self._environment_digest

    @property
    def initial_capital(self) -> float:
        if self._initial_capital is None:
            raise AttributeError("Universal training identity is not enabled")
        return self._initial_capital

    @property
    def decision_hours(self) -> float:
        if self._decision_hours is None:
            raise AttributeError("Universal training identity is not enabled")
        return self._decision_hours

    @property
    def sequence_layout_metadata(self) -> dict[str, Any] | None:
        return (
            None
            if self._sequence_layout_metadata is None
            else dict(self._sequence_layout_metadata)
        )

    @property
    def pre_trade_risk(self) -> object | None:
        return getattr(self._reference_environment, "pre_trade_risk", None)

    @property
    def normalizer(self) -> object | None:
        return getattr(self._reference_environment, "normalizer", None)

    @property
    def sequence_normalizer(self) -> object | None:
        return getattr(self._reference_environment, "sequence_normalizer", None)

    @property
    def normalizer_digest(self) -> str | None:
        flat = _optional_digest(
            getattr(self._reference_environment, "normalizer", None)
        )
        sequence = _optional_digest(
            getattr(self._reference_environment, "sequence_normalizer", None)
        )
        if flat is None:
            return sequence
        if sequence is None:
            return flat
        return content_digest(
            {
                "flat": flat,
                "schema_version": "universal_policy_normalizer_bundle_v1",
                "sequence": sequence,
            }
        )

    @property
    def alpha_artifact_digest(self) -> str | None:
        return getattr(self._reference_environment, "alpha_artifact_digest", None)

    @property
    def factor_artifact_digest(self) -> str | None:
        return getattr(self._reference_environment, "factor_artifact_digest", None)

    @property
    def router_digest(self) -> str:
        return self._router.digest

    @property
    def completed_episode_count(self) -> int:
        return self._completed_episode_count

    @property
    def current_index(self) -> int:
        environment = self._active_environment
        if environment is None:
            raise RuntimeError("environment must be reset before current_index access")
        value = getattr(environment, "current_index", None)
        return _require_non_negative_int(value, field="current_index")

    @property
    def dataset(self) -> Any:
        """Expose the active concrete dataset for evaluation and telemetry."""

        environment = self._active_environment
        if environment is None:
            raise RuntimeError("environment must be reset before dataset access")
        dataset = getattr(environment, "dataset", None)
        if dataset is None:
            raise AttributeError("active environment does not expose a dataset")
        return dataset

    @property
    def minimum_start_index(self) -> int:
        """Expose the concrete lower bound used by exact rollout evaluators."""

        value = getattr(self._reference_environment, "minimum_start_index", None)
        return _require_non_negative_int(value, field="minimum_start_index")

    @property
    def hybrid(self) -> Any:
        """Expose the active policy portfolio for economic audits."""

        environment = self._active_environment
        if environment is None:
            raise RuntimeError("environment must be reset before hybrid access")
        value = getattr(environment, "hybrid", None)
        if value is None:
            raise AttributeError("active environment does not expose hybrid")
        return value

    @property
    def shadow(self) -> Any:
        """Expose the active baseline portfolio for economic audits."""

        environment = self._active_environment
        if environment is None:
            raise RuntimeError("environment must be reset before shadow access")
        value = getattr(environment, "shadow", None)
        if value is None:
            raise AttributeError("active environment does not expose shadow")
        return value

    @property
    def active_episode_binding(self) -> InstrumentEpisodeBinding:
        binding = self._active_episode_binding
        if binding is None:
            raise RuntimeError("environment must be reset before binding access")
        return binding

    def _validate_concrete_environment(
        self,
        environment: ConcreteSingleInstrumentEnv,
        binding: InstrumentDatasetBinding,
    ) -> None:
        dataset = getattr(environment, "dataset", None)
        if dataset is None:
            raise TypeError("concrete environment must expose its dataset")
        raw_symbols = getattr(dataset, "symbols", None)
        if (
            not isinstance(raw_symbols, (tuple, list))
            or not raw_symbols
            or any(not isinstance(symbol, str) or not symbol for symbol in raw_symbols)
        ):
            raise TypeError("concrete environment dataset symbols are invalid")
        symbols = tuple(raw_symbols)
        if symbols != (binding.concrete_symbol,):
            raise ValueError(
                "concrete environment dataset symbol does not match binding"
            )
        if getattr(dataset, "dataset_id", None) != binding.source_dataset_id:
            raise ValueError(
                "concrete environment dataset identity does not match binding"
            )

        action_spec = getattr(environment, "action_spec", None)
        if not isinstance(action_spec, ActionSpec):
            raise TypeError("concrete environment must expose an ActionSpec")
        if (
            action_spec.mode is not ActionMode.TARGET_WEIGHT
            or action_spec.target_weight_count != 1
            or action_spec.size != 1
        ):
            raise ValueError("concrete environment must use one target-weight action")
        concrete_action_names = tuple(getattr(environment, "action_names", ()))
        expected_action_names = (f"target_weight:{binding.concrete_symbol}",)
        if concrete_action_names != expected_action_names:
            raise ValueError("concrete environment action names do not match binding")
        if not isinstance(environment.action_space, gym.spaces.Box):
            raise TypeError("concrete environment action space must be a Box")
        if environment.action_space.shape != (1,):
            raise ValueError("concrete environment action space must have shape (1,)")
        if not isinstance(environment.observation_space, gym.spaces.Space):
            raise TypeError("concrete environment observation space is invalid")

    def _require_space_compatibility(
        self,
        environment: ConcreteSingleInstrumentEnv,
    ) -> None:
        if environment.observation_space != self._concrete_observation_space:
            raise ValueError("concrete environment observation space mismatch")
        if environment.action_space != self.action_space:
            raise ValueError("concrete environment action space mismatch")
        if self._training_identity_enabled and (
            self._training_surface(environment)
            != self._training_surface(self._reference_environment)
        ):
            raise ValueError("concrete environment training contract mismatch")

    @staticmethod
    def _close_rejected_environment(
        environment: ConcreteSingleInstrumentEnv,
    ) -> None:
        try:
            environment.close()
        except Exception:
            pass

    def _evict_cached_environment_for(self, symbol: str) -> bool:
        limit = self._max_cached_environments
        if (
            limit is None
            or symbol in self._environments
            or len(self._environments) < limit
        ):
            return False
        if not self._episode_complete:
            raise RuntimeError(
                "cannot evict a child environment during an active episode"
            )
        victim_symbol = next(iter(self._environments))
        victim = self._environments.pop(victim_symbol)
        self._environment_object_ids.discard(id(victim))
        if self._active_environment is victim:
            self._active_environment = None
            self._active_episode_binding = None
        reference_evicted = (
            hasattr(self, "_reference_environment")
            and self._reference_environment is victim
        )
        victim.close()
        return reference_evicted

    def _load_environment(
        self,
        route: InstrumentRoute,
    ) -> ConcreteSingleInstrumentEnv:
        symbol = route.concrete_symbol
        cached = self._environments.get(symbol)
        if cached is not None:
            return cached

        reference_evicted = self._evict_cached_environment_for(symbol)
        binding = self._bindings[symbol]
        environment = self._environment_factory(binding)
        if not isinstance(environment, gym.Env):
            raise TypeError("environment_factory must return a Gymnasium environment")
        object_id = id(environment)
        if object_id in self._environment_object_ids:
            raise ValueError(
                "environment_factory reused one environment across symbols"
            )
        try:
            self._validate_concrete_environment(environment, binding)
            if self._environments and hasattr(self, "_reference_environment"):
                self._require_space_compatibility(environment)
        except Exception:
            self._close_rejected_environment(environment)
            raise
        self._environments[symbol] = environment
        self._environment_object_ids.add(object_id)
        if reference_evicted:
            self._reference_environment = environment
        return environment

    def _episode_seed(
        self,
        *,
        route: InstrumentRoute,
        binding: InstrumentDatasetBinding,
    ) -> int:
        digest = content_digest(
            {
                "completed_episode_count": route.completed_episode_count,
                "dataset_binding_digest": binding.digest,
                "environment_index": self._environment_index,
                "partition_digest": self._router.partition_digest,
                "run_seed": self._run_seed,
                "schema_version": "instrument_episode_seed_v1",
            }
        )
        return int(digest[:8], 16)

    @staticmethod
    def _instrument_info(
        info: Mapping[str, Any],
        binding: InstrumentEpisodeBinding,
    ) -> dict[str, Any]:
        resolved = dict(info)
        resolved[INSTRUMENT_EPISODE_INFO_KEY] = binding.to_json_dict()
        resolved[INSTRUMENT_EPISODE_DIGEST_INFO_KEY] = binding.digest
        return resolved

    def _policy_observation(
        self,
        observation: Any,
        *,
        environment: ConcreteSingleInstrumentEnv,
        binding: InstrumentDatasetBinding,
    ) -> Any:
        instrument_provider = self._instrument_context_provider
        v4_provider = self._v4_context_provider
        if instrument_provider is None and v4_provider is None:
            return observation
        if not isinstance(observation, Mapping):
            raise TypeError("Universal context requires mapping observations")
        resolved = dict(observation)
        if instrument_provider is not None:
            context = np.asarray(
                instrument_provider(environment, binding), dtype=np.float32
            )
            expected = (1, len(UNIVERSAL_INSTRUMENT_DESCRIPTOR_NAMES))
            if context.shape != expected or not np.isfinite(context).all():
                raise ValueError("instrument_context must be a finite (1, 9) matrix")
            resolved["instrument_context"] = context.copy()
        if v4_provider is not None:
            decision_index = _require_non_negative_int(
                getattr(environment, "current_index", None),
                field="V4 decision_index",
            )
            v4 = v4_provider.resolve(
                symbol=binding.concrete_symbol,
                decision_index=decision_index,
            )
            resolved.update(
                {
                    "local_cross_market_context": v4.local_values.copy(),
                    "local_cross_market_available": v4.local_available.copy(),
                    "local_cross_market_staleness_hours": (
                        v4.local_staleness_hours.copy()
                    ),
                    "global_market_context": v4.global_values.copy(),
                    "global_market_available": v4.global_available.copy(),
                    "global_market_staleness_hours": (
                        v4.global_staleness_hours.copy()
                    ),
                    "causal_beta": v4.beta.copy(),
                    "causal_beta_available": v4.beta_available.copy(),
                }
            )
        return resolved

    @property
    def canonical_probe_seed(self) -> int:
        """Return the only reset seed accepted by this immutable environment."""

        return self._run_seed

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        if not self._episode_complete:
            raise RuntimeError("cannot reset while an active episode is routed")
        if seed is not None:
            resolved_seed = _require_non_negative_int(seed, field="seed")
            if resolved_seed != self._run_seed:
                raise ValueError("reset seed must equal the immutable run_seed")
        super().reset(seed=self._run_seed)

        route = self._router.route(self._completed_episode_count)
        environment = self._load_environment(route)
        dataset_binding = self._bindings[route.concrete_symbol]
        episode_seed = self._episode_seed(route=route, binding=dataset_binding)
        observation, raw_info = environment.reset(
            seed=episode_seed,
            options=options,
        )
        if not isinstance(raw_info, Mapping):
            raise TypeError("concrete environment reset info must be a mapping")
        episode_start = _episode_boundary(raw_info, field="start_index")
        episode_stop = _episode_boundary(raw_info, field="end_index")
        episode_binding = InstrumentEpisodeBinding(
            dataset_binding=dataset_binding,
            episode_start=episode_start,
            episode_stop=episode_stop,
            episode_seed=episode_seed,
            environment_index=self._environment_index,
            completed_episode_count=route.completed_episode_count,
            routing_cycle=route.routing_cycle,
            routing_position=route.routing_position,
        )
        self._active_environment = environment
        self._active_episode_binding = episode_binding
        self._episode_complete = False
        return self._policy_observation(
            observation,
            environment=environment,
            binding=dataset_binding,
        ), self._instrument_info(raw_info, episode_binding)

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        environment = self._active_environment
        binding = self._active_episode_binding
        if environment is None or binding is None:
            raise RuntimeError("environment must be reset before step")
        if self._episode_complete:
            raise RuntimeError("step called after the routed episode completed")

        observation, reward, terminated, truncated, raw_info = environment.step(action)
        if not isinstance(terminated, bool):
            raise TypeError("terminated must be a boolean")
        if not isinstance(truncated, bool):
            raise TypeError("truncated must be a boolean")
        if not isinstance(raw_info, Mapping):
            raise TypeError("concrete environment step info must be a mapping")
        info = self._instrument_info(raw_info, binding)
        policy_observation = self._policy_observation(
            observation,
            environment=environment,
            binding=binding.dataset_binding,
        )
        if terminated or truncated:
            self._episode_complete = True
            self._completed_episode_count += 1
        return policy_observation, float(reward), terminated, truncated, info

    def close(self) -> None:
        first_error: Exception | None = None
        for environment in tuple(self._environments.values()):
            try:
                environment.close()
            except Exception as error:
                if first_error is None:
                    first_error = error
        if first_error is not None:
            raise first_error


__all__ = [
    "INSTRUMENT_EPISODE_DIGEST_INFO_KEY",
    "INSTRUMENT_EPISODE_INFO_KEY",
    "UNIVERSAL_OBSERVATION_SCHEMA",
    "EpisodeRoutedSingleInstrumentEnv",
    "InstrumentContextProvider",
    "InstrumentEnvironmentFactory",
]
