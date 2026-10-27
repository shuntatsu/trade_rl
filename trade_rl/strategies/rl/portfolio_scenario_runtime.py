"""Runtime bindings for opt-in v2 portfolio execution scenarios."""

from __future__ import annotations

import hashlib
import importlib
from copy import deepcopy
from pathlib import Path
from typing import cast

import gymnasium as gym
import numpy as np

from trade_rl._validation import require_sha256
from trade_rl.artifacts.canonical import canonical_json_bytes, to_json_value
from trade_rl.artifacts.hashing import content_digest
from trade_rl.artifacts.verified_file import open_regular_binary
from trade_rl.simulation.objectives.binding import BoundObjectiveClock
from trade_rl.strategies.rl.portfolio_env import SharedPortfolioGymEnv
from trade_rl.strategies.rl.portfolio_inference_bundle import (
    _space_descriptor,
    _strict_object,
    capture_portfolio_source_runtime,
)
from trade_rl.strategies.rl.portfolio_runtime_scenario import (
    PortfolioRuntimeExecutionScenarioV2,
)
from trade_rl.strategies.rl.portfolio_transform import (
    PortfolioObservationTransform,
    PortfolioScenarioObservationRuntimeBindingV2,
    _dataset_timestamp_ns,
    _datetime_ns,
    _execution_payload,
    _risk_payload,
)

_SCENARIO_RUNTIME_SOURCE_DOMAIN = b"trade_rl.portfolio_scenario_inference_source_v2\0"
_SCENARIO_RUNTIME_SOURCE_SCHEMA = "portfolio_scenario_inference_source_runtime_v2"
_SCENARIO_RUNTIME_SOURCE_MODULES = (
    "trade_rl.strategies.rl.portfolio_runtime_scenario",
    "trade_rl.strategies.rl.portfolio_scenario_runtime",
    "trade_rl.strategies.rl.portfolio_scenario_inference",
)


def _strict_string_tuple(value: object, *, field: str) -> tuple[str, ...]:
    if type(value) is not list or any(
        type(item) is not str for item in cast(list[object], value)
    ):
        raise ValueError(f"{field} must be a list of strings")
    return tuple(cast(list[str], value))


def capture_portfolio_scenario_inference_source_digest() -> str:
    """Digest selected v1 source/runtime identity plus the explicit v2 owners."""

    selected = capture_portfolio_source_runtime()
    selected_modules = selected["source_modules"]
    if type(selected_modules) is not list:
        raise ValueError("selected source runtime module roster is malformed")
    portable_selected = {
        "scope": selected["scope"],
        "runtime": selected["runtime"],
        "source_modules": [
            {
                "module": item["module"],
                "bytes": item["bytes"],
                "sha256": item["sha256"],
            }
            for item in cast(list[dict[str, object]], selected_modules)
        ],
    }
    scenario_modules: list[dict[str, object]] = []
    for name in _SCENARIO_RUNTIME_SOURCE_MODULES:
        module = importlib.import_module(name)
        origin = getattr(module, "__file__", None)
        if type(origin) is not str:
            raise ValueError(f"scenario source owner {name} has no file origin")
        source_path = Path(origin)
        if source_path.suffix != ".py" or source_path.is_symlink():
            raise ValueError(f"scenario source owner {name} is not Python source")
        resolved = source_path.resolve(strict=True)
        digest = hashlib.sha256()
        size = 0
        with open_regular_binary(
            resolved, field=f"scenario source owner {name}"
        ) as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
                size += len(chunk)
        scenario_modules.append(
            {"module": name, "bytes": size, "sha256": digest.hexdigest()}
        )
    payload = {
        "schema": _SCENARIO_RUNTIME_SOURCE_SCHEMA,
        "selected_source_runtime": portable_selected,
        "scenario_source_modules": scenario_modules,
        "scope": (
            "selected direct v1 source/runtime plus explicit v2 scenario owners; "
            "not a complete transitive source or in-memory code closure"
        ),
    }
    return hashlib.sha256(
        _SCENARIO_RUNTIME_SOURCE_DOMAIN + canonical_json_bytes(payload)
    ).hexdigest()


def create_portfolio_scenario_runtime_binding_v2(
    transform: PortfolioObservationTransform,
    training_bound: BoundObjectiveClock,
    runtime_env: SharedPortfolioGymEnv,
    *,
    training_context: dict[str, object],
    training_context_sha256: str,
    checkpoint_sha256: str,
    scenario: PortfolioRuntimeExecutionScenarioV2,
    expected_scenario_sha256: str,
    scenario_inference_source_sha256: str,
    require_unreset: bool,
) -> PortfolioScenarioObservationRuntimeBindingV2:
    """Validate training/base/runtime contracts and bind two distinct clocks."""

    require_sha256(training_context_sha256, field="training_context_sha256")
    require_sha256(checkpoint_sha256, field="checkpoint_sha256")
    require_sha256(expected_scenario_sha256, field="expected_scenario_sha256")
    require_sha256(
        scenario_inference_source_sha256,
        field="scenario_inference_source_sha256",
    )
    if type(transform) is not PortfolioObservationTransform:
        raise TypeError("transform must be exactly PortfolioObservationTransform")
    if type(training_bound) is not BoundObjectiveClock:
        raise TypeError("training_bound must be exactly BoundObjectiveClock")
    if type(runtime_env) is not SharedPortfolioGymEnv:
        raise TypeError("runtime_env must be exactly SharedPortfolioGymEnv")
    if type(scenario) is not PortfolioRuntimeExecutionScenarioV2:
        raise TypeError("scenario must be exactly PortfolioRuntimeExecutionScenarioV2")
    if type(training_context) is not dict:
        raise TypeError("training_context must be a strict object")
    scenario.to_bytes()  # Re-run the immutable declaration's strict validation.
    if scenario.digest != expected_scenario_sha256:
        raise ValueError("scenario digest differs from its external expected pin")
    if (
        scenario.training_context_sha256 != training_context_sha256
        or scenario.transform_digest != transform.digest
        or scenario.checkpoint_sha256 != checkpoint_sha256
    ):
        raise ValueError("scenario training, transform, or checkpoint anchor differs")
    if require_unreset and runtime_env._session is not None:
        raise ValueError("runtime_env must not have been reset before bundle loading")

    context = _strict_object(training_context["context"], field="training context")
    financial = _strict_object(
        training_context["financial_context"], field="financial_context"
    )
    declared_base_cost = to_json_value(scenario.base_execution_cost)
    declared_runtime_cost = to_json_value(scenario.runtime_execution_cost)
    if (
        canonical_json_bytes(declared_base_cost)
        != canonical_json_bytes(financial["execution_cost"])
        or content_digest(_execution_payload(scenario.base_execution_cost))
        != transform.execution_config_digest
    ):
        raise ValueError("scenario base execution cost differs from fitted training")
    if type(runtime_env.execution_cost) is not type(
        scenario.runtime_execution_cost
    ) or canonical_json_bytes(
        to_json_value(runtime_env.execution_cost)
    ) != canonical_json_bytes(declared_runtime_cost):
        raise ValueError("runtime environment execution cost differs from scenario")

    base_agent_stop = (
        transform.stop_index - scenario.base_execution_cost.order_latency_bars - 1
    )
    if (
        training_bound.digest != transform.bound_digest
        or training_bound.objective.digest != transform.objective_digest
        or training_bound.clock.digest != transform.clock_digest
        or transform.source_dataset_id != context["dataset_id"]
        or transform.start_index != context["start_index"]
        or transform.stop_index != context["stop_index"]
        or transform.agent_stop_index != base_agent_stop
    ):
        raise ValueError("original training clock differs from the fitted transform")

    dataset = runtime_env.dataset
    profile_digest = (
        None
        if runtime_env.market_order_profile is None
        else runtime_env.market_order_profile.digest
    )
    if (
        dataset.dataset_id != transform.source_dataset_id
        or tuple(dataset.symbols) != transform.symbols
        or tuple(dataset.feature_names) != transform.feature_names
        or tuple(dataset.global_feature_names) != transform.global_feature_names
        or tuple(runtime_env.target_abs_weights)
        != tuple(cast(list[object], financial["target_abs_weights"]))
        or runtime_env.initial_capital != transform.initial_capital
        or canonical_json_bytes(to_json_value(runtime_env.risk_config))
        != canonical_json_bytes(financial["risk_config"])
        or content_digest(_risk_payload(runtime_env.risk_config))
        != transform.risk_config_digest
        or profile_digest != transform.market_order_profile_digest
        or canonical_json_bytes(
            None
            if runtime_env.market_order_profile is None
            else runtime_env.market_order_profile.canonical_payload()
        )
        != canonical_json_bytes(financial["market_order_profile"])
    ):
        raise ValueError("runtime Dataset or financial context differs from training")

    if transform.stop_index >= dataset.n_bars:
        raise ValueError("training stop is outside the runtime Dataset")
    if _datetime_ns(training_bound.objective.evaluation_start) != _dataset_timestamp_ns(
        runtime_env, transform.start_index
    ) or _datetime_ns(
        training_bound.objective.evaluation_stop_exclusive
    ) != _dataset_timestamp_ns(runtime_env, transform.stop_index):
        raise ValueError("training bound timestamps differ from the same Dataset")

    training_objective = training_bound.objective.payload()
    runtime_objective = runtime_env.bound.objective.payload()
    for key in ("evaluation_start", "evaluation_stop_exclusive"):
        training_objective.pop(key)
        runtime_objective.pop(key)
    if canonical_json_bytes(training_objective) != canonical_json_bytes(
        runtime_objective
    ):
        raise ValueError("runtime objective semantics differ from training")
    training_clock = training_bound.clock.payload()
    runtime_clock = runtime_env.bound.clock.payload()
    training_clock.pop("economic_horizon_seconds")
    runtime_clock.pop("economic_horizon_seconds")
    if canonical_json_bytes(training_clock) != canonical_json_bytes(runtime_clock):
        raise ValueError("runtime clock semantics differ from training")

    if not (
        0 <= runtime_env.start_index < runtime_env.stop_index < dataset.n_bars
        and runtime_env.start_index >= transform.stop_index
        and runtime_env.bound.clock.decision_interval_seconds
        == transform.native_interval_seconds
        and _datetime_ns(runtime_env.bound.objective.evaluation_start)
        == _dataset_timestamp_ns(runtime_env, runtime_env.start_index)
        and _datetime_ns(runtime_env.bound.objective.evaluation_stop_exclusive)
        == _dataset_timestamp_ns(runtime_env, runtime_env.stop_index)
    ):
        raise ValueError("runtime window is not a valid forward Dataset interval")
    runtime_agent_stop = (
        runtime_env.stop_index - runtime_env.execution_cost.order_latency_bars - 1
    )
    if not runtime_env.start_index < runtime_agent_stop <= runtime_env.stop_index:
        raise ValueError("runtime window has no decision before terminal reservation")
    transform.local_normalizer.validate_training_scope(
        dataset,
        tuple(range(dataset.n_symbols)),
        transform.start_index,
        transform.agent_stop_index,
    )

    layout = _strict_object(training_context["policy_layout"], field="policy_layout")
    expected_observation_keys = _strict_string_tuple(
        layout.get("observation_keys"), field="policy_layout.observation_keys"
    )
    if (
        _space_descriptor(transform.observation_space) != layout["observation_space"]
        or _space_descriptor(runtime_env.action_space) != layout["action_space"]
        or tuple(runtime_env.observation_keys) != expected_observation_keys
    ):
        raise ValueError("runtime policy spaces differ from training")
    return PortfolioScenarioObservationRuntimeBindingV2(
        training_context_sha256=training_context_sha256,
        transform_digest=transform.digest,
        checkpoint_sha256=checkpoint_sha256,
        scenario_sha256=expected_scenario_sha256,
        scenario_inference_source_sha256=scenario_inference_source_sha256,
        base_execution_policy_digest=scenario.base_execution_cost.execution_policy_digest,
        training_execution_config_digest=transform.execution_config_digest,
        runtime_execution_policy_digest=runtime_env.execution_cost.execution_policy_digest,
        training_bound_digest=training_bound.digest,
        runtime_bound_digest=runtime_env.bound.digest,
        dataset_id=dataset.dataset_id,
        training_start_index=transform.start_index,
        training_stop_index=transform.stop_index,
        training_agent_stop_index=transform.agent_stop_index,
        runtime_start_index=runtime_env.start_index,
        runtime_stop_index=runtime_env.stop_index,
        runtime_agent_stop_index=runtime_agent_stop,
        runtime_start_ns=_datetime_ns(runtime_env.bound.objective.evaluation_start),
        runtime_horizon_seconds=runtime_env.bound.clock.economic_horizon_seconds,
        native_interval_seconds=runtime_env.bound.clock.decision_interval_seconds,
    )


def validate_portfolio_scenario_runtime_binding_v2(
    binding: PortfolioScenarioObservationRuntimeBindingV2,
    transform: PortfolioObservationTransform,
    training_bound: BoundObjectiveClock,
    runtime_env: SharedPortfolioGymEnv,
    *,
    training_context: dict[str, object],
    training_context_sha256: str,
    checkpoint_sha256: str,
    scenario: PortfolioRuntimeExecutionScenarioV2,
    expected_scenario_sha256: str,
    scenario_inference_source_sha256: str,
) -> None:
    if type(binding) is not PortfolioScenarioObservationRuntimeBindingV2:
        raise TypeError(
            "binding must be exactly PortfolioScenarioObservationRuntimeBindingV2"
        )
    expected = create_portfolio_scenario_runtime_binding_v2(
        transform,
        training_bound,
        runtime_env,
        training_context=training_context,
        training_context_sha256=training_context_sha256,
        checkpoint_sha256=checkpoint_sha256,
        scenario=scenario,
        expected_scenario_sha256=expected_scenario_sha256,
        scenario_inference_source_sha256=scenario_inference_source_sha256,
        require_unreset=False,
    )
    if binding != expected:
        raise ValueError("scenario runtime binding differs from current environment")


class PortfolioScenarioGymObservationWrapperV2(gym.ObservationWrapper):
    """Apply frozen training statistics under a checked v2 runtime binding."""

    def __init__(
        self,
        env: SharedPortfolioGymEnv,
        transform: PortfolioObservationTransform,
        training_bound: BoundObjectiveClock,
        runtime_binding: PortfolioScenarioObservationRuntimeBindingV2,
        *,
        training_context: dict[str, object],
        training_context_sha256: str,
        checkpoint_sha256: str,
        scenario: PortfolioRuntimeExecutionScenarioV2,
        expected_scenario_sha256: str,
        scenario_inference_source_sha256: str,
    ) -> None:
        if type(env) is not SharedPortfolioGymEnv:
            raise TypeError("env must be exactly SharedPortfolioGymEnv")
        if type(transform) is not PortfolioObservationTransform:
            raise TypeError("transform must be exactly PortfolioObservationTransform")
        if type(training_bound) is not BoundObjectiveClock:
            raise TypeError("training_bound must be exactly BoundObjectiveClock")
        if type(runtime_binding) is not PortfolioScenarioObservationRuntimeBindingV2:
            raise TypeError(
                "runtime_binding must be exactly "
                "PortfolioScenarioObservationRuntimeBindingV2"
            )
        self.transform = transform
        self.training_bound = training_bound
        self.runtime_binding = runtime_binding
        self._training_context = deepcopy(training_context)
        self._training_context_sha256 = training_context_sha256
        self._checkpoint_sha256 = checkpoint_sha256
        self._scenario = scenario
        self._expected_scenario_sha256 = expected_scenario_sha256
        self._scenario_inference_source_sha256 = scenario_inference_source_sha256
        self._poisoned = False
        self._last_failure: str | None = None
        super().__init__(env)
        self.validate_environment(require_unreset=True)
        if (
            capture_portfolio_scenario_inference_source_digest()
            != scenario_inference_source_sha256
        ):
            raise ValueError("current scenario inference source differs from its pin")
        self.observation_space = transform.observation_space

    def validate_environment(self, *, require_unreset: bool = False) -> None:
        if (
            require_unreset
            and cast(SharedPortfolioGymEnv, self.env)._session is not None
        ):
            raise ValueError(
                "runtime_env must not have been reset before bundle loading"
            )
        validate_portfolio_scenario_runtime_binding_v2(
            self.runtime_binding,
            self.transform,
            self.training_bound,
            cast(SharedPortfolioGymEnv, self.env),
            training_context=self._training_context,
            training_context_sha256=self._training_context_sha256,
            checkpoint_sha256=self._checkpoint_sha256,
            scenario=self._scenario,
            expected_scenario_sha256=self._expected_scenario_sha256,
            scenario_inference_source_sha256=self._scenario_inference_source_sha256,
        )

    @property
    def last_failure(self) -> str | None:
        return self._last_failure

    def observation(self, observation: object) -> dict[str, np.ndarray]:
        if self._poisoned:
            raise RuntimeError(
                "scenario transform wrapper is unusable; reset is required"
            )
        self.validate_environment()
        return self.transform.transform_scenario_observation_v2(
            observation, runtime_binding=self.runtime_binding
        )

    def reset(
        self, *, seed: int | None = None, options: dict[str, object] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, object]]:
        try:
            self.validate_environment()
            raw, info = cast(SharedPortfolioGymEnv, self.env).reset(
                seed=seed, options=options
            )
            self._poisoned = False
            transformed = self.observation(raw)
        except BaseException as error:
            self._poisoned = True
            self._last_failure = f"{type(error).__name__}: {error}"
            raise
        self._poisoned = False
        self._last_failure = None
        return transformed, info

    def step(
        self, action: object
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, object]]:
        if self._poisoned:
            raise RuntimeError(
                "scenario transform wrapper is unusable; reset is required"
            )
        try:
            self.validate_environment()
            raw, reward, terminated, truncated, info = cast(
                SharedPortfolioGymEnv, self.env
            ).step(action)
            transformed = self.observation(raw)
        except BaseException as error:
            self._poisoned = True
            self._last_failure = f"{type(error).__name__}: {error}"
            raise
        return transformed, reward, terminated, truncated, info


__all__ = [
    "PortfolioScenarioGymObservationWrapperV2",
    "capture_portfolio_scenario_inference_source_digest",
    "create_portfolio_scenario_runtime_binding_v2",
    "validate_portfolio_scenario_runtime_binding_v2",
]
