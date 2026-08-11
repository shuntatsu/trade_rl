"""Episode-routed Gymnasium facade for universal single-instrument training."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Final, cast

import gymnasium as gym
import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.rl.actions import ActionMode, ActionSpec
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

INSTRUMENT_EPISODE_INFO_KEY: Final = "instrument_episode_binding"
INSTRUMENT_EPISODE_DIGEST_INFO_KEY: Final = "instrument_episode_binding_digest"

ConcreteSingleInstrumentEnv = gym.Env[Any, np.ndarray]
InstrumentEnvironmentFactory = Callable[
    [InstrumentDatasetBinding],
    ConcreteSingleInstrumentEnv,
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


class EpisodeRoutedSingleInstrumentEnv(gym.Env[Any, np.ndarray]):
    """Expose one generic action while routing complete episodes by symbol."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        train_symbols: Sequence[str],
        partition_digest: str,
        bindings: Sequence[InstrumentDatasetBinding],
        environment_factory: InstrumentEnvironmentFactory,
        run_seed: int,
        environment_index: int,
    ) -> None:
        super().__init__()
        if not callable(environment_factory):
            raise TypeError("environment_factory must be callable")
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
        self.observation_space = initial_environment.observation_space
        self.action_space = cast(
            gym.spaces.Space[np.ndarray],
            initial_environment.action_space,
        )
        self.metadata = dict(getattr(initial_environment, "metadata", self.metadata))

    @property
    def policy_symbols(self) -> tuple[str, ...]:
        return GENERIC_INSTRUMENT_SYMBOLS

    @property
    def action_names(self) -> tuple[str, ...]:
        return GENERIC_TARGET_WEIGHT_ACTION_NAMES

    @property
    def router_digest(self) -> str:
        return self._router.digest

    @property
    def completed_episode_count(self) -> int:
        return self._completed_episode_count

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
        if environment.observation_space != self.observation_space:
            raise ValueError("concrete environment observation space mismatch")
        if environment.action_space != self.action_space:
            raise ValueError("concrete environment action space mismatch")

    @staticmethod
    def _close_rejected_environment(
        environment: ConcreteSingleInstrumentEnv,
    ) -> None:
        try:
            environment.close()
        except Exception:
            pass

    def _load_environment(
        self,
        route: InstrumentRoute,
    ) -> ConcreteSingleInstrumentEnv:
        symbol = route.concrete_symbol
        cached = self._environments.get(symbol)
        if cached is not None:
            return cached

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
            if self._environments:
                self._require_space_compatibility(environment)
        except Exception:
            self._close_rejected_environment(environment)
            raise
        self._environments[symbol] = environment
        self._environment_object_ids.add(object_id)
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

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        if not self._episode_complete:
            raise RuntimeError("cannot reset while an active episode is routed")
        if seed is not None:
            resolved_seed = _require_non_negative_int(
                seed,
                field="seed",
            )
            if resolved_seed != self._run_seed:
                raise ValueError("reset seed must equal the immutable run_seed")
        super().reset(seed=self._run_seed)

        route = self._router.route(self._completed_episode_count)
        environment = self._load_environment(route)
        dataset_binding = self._bindings[route.concrete_symbol]
        episode_seed = self._episode_seed(
            route=route,
            binding=dataset_binding,
        )
        observation, raw_info = environment.reset(
            seed=episode_seed,
            options=options,
        )
        if not isinstance(raw_info, Mapping):
            raise TypeError("concrete environment reset info must be a mapping")
        episode_start = _episode_boundary(
            raw_info,
            field="start_index",
        )
        episode_stop = _episode_boundary(
            raw_info,
            field="end_index",
        )
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
        return observation, self._instrument_info(
            raw_info,
            episode_binding,
        )

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
        if terminated or truncated:
            self._episode_complete = True
            self._completed_episode_count += 1
        return observation, float(reward), terminated, truncated, info

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
    "EpisodeRoutedSingleInstrumentEnv",
    "InstrumentEnvironmentFactory",
]
