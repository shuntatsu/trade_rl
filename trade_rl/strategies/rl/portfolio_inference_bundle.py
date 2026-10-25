"""Strict opt-in inference bundles for the shared-portfolio PPO path."""

from __future__ import annotations

import hashlib
import importlib
import json
import platform
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, cast

import gymnasium as gym
import numpy as np

from trade_rl._validation import require_sha256
from trade_rl.artifacts.atomic_write import atomic_write_bytes
from trade_rl.artifacts.canonical import canonical_json_bytes, to_json_value
from trade_rl.artifacts.hashing import content_digest
from trade_rl.artifacts.verified_file import (
    file_digest_and_size,
    open_regular_binary,
    verified_private_copy,
)
from trade_rl.simulation.objectives.binding import BoundObjectiveClock
from trade_rl.simulation.objectives.clock import FinancialClockContract
from trade_rl.simulation.objectives.contract import CapitalContract, ObjectiveContract
from trade_rl.strategies.portfolio import (
    PortfolioDecisionSnapshot,
    PortfolioTargetProposal,
)
from trade_rl.strategies.portfolio_policy_codec import (
    encode_portfolio_decision_snapshot,
    portfolio_action_to_proposal,
)
from trade_rl.strategies.rl.portfolio_env import SharedPortfolioGymEnv
from trade_rl.strategies.rl.portfolio_transform import (
    PortfolioObservationRuntimeBinding,
    PortfolioObservationTransform,
)

_SOURCE_MODULES = (
    "trade_rl._validation",
    "trade_rl.artifacts.atomic_write",
    "trade_rl.artifacts.canonical",
    "trade_rl.artifacts.hashing",
    "trade_rl.artifacts.verified_file",
    "trade_rl.data.market",
    "trade_rl.data.market_order_rules",
    "trade_rl.risk.pretrade",
    "trade_rl.simulation.accounting",
    "trade_rl.simulation.bar_path",
    "trade_rl.simulation.diagnostics.funding",
    "trade_rl.simulation.execution",
    "trade_rl.simulation.liquidity",
    "trade_rl.simulation.native_transition",
    "trade_rl.simulation.objectives.binding",
    "trade_rl.simulation.objectives.clock",
    "trade_rl.simulation.objectives.contract",
    "trade_rl.simulation.orders.model",
    "trade_rl.simulation.quantities",
    "trade_rl.simulation.stateful.execution",
    "trade_rl.simulation.stateful.bar_lifecycle",
    "trade_rl.simulation.stateful.order_transitions",
    "trade_rl.simulation.stateful.portfolio_session",
    "trade_rl.simulation.stateful.runtime",
    "trade_rl.simulation.stateful.symbol_fills",
    "trade_rl.simulation.targets.execution",
    "trade_rl.strategies.interface",
    "trade_rl.strategies.portfolio",
    "trade_rl.strategies.portfolio_observation",
    "trade_rl.strategies.portfolio_policy_codec",
    "trade_rl.strategies.position_intent",
    "trade_rl.strategies.rl.portfolio_env",
    "trade_rl.strategies.rl.portfolio_episode_journal",
    "trade_rl.strategies.rl.portfolio_policy_sample_journal",
    "trade_rl.strategies.rl.portfolio_transform",
    "trade_rl.strategies.rl.portfolio_training",
    "trade_rl.strategies.rl.portfolio_inference_bundle",
    "trade_rl.strategies.rl.ppo_normalization",
)
_SOURCE_SCOPE = "selected direct execution, context and policy owners; not a complete transitive closure"
_TRAINING_SCHEMA = "shared_portfolio_ppo_training_v1"
_TRAINING_CONTEXT_SCHEMA = "shared_portfolio_training_context_v1"
_BUNDLE_SCHEMA = "shared_portfolio_inference_bundle_v1"
_BUNDLE_FILES = {
    "training_context.json",
    "training_manifest.json",
    "policy.zip",
    "manifest.json",
}


@dataclass(frozen=True, slots=True)
class PortfolioInferenceBundleReceipt:
    """Identity of one immutable opt-in portfolio inference bundle."""

    root: Path
    bundle_sha256: str
    training_manifest_sha256: str
    training_completion_sha256: str
    training_context_sha256: str
    checkpoint_sha256: str


@dataclass(frozen=True, slots=True)
class PortfolioInferencePolicy:
    """Deterministic prediction facade bound to a validated forward environment."""

    _policy: Any
    _transform: PortfolioObservationTransform
    _action_space: gym.Space[Any]
    runtime_binding: PortfolioObservationRuntimeBinding
    _target_abs_weights: tuple[float, ...]

    def predict(self, raw_observation: object) -> np.ndarray:
        """Transform a current native observation and predict one joint action."""

        transformed = self._transform.transform_observation(
            raw_observation, runtime_binding=self.runtime_binding
        )
        prediction = self._policy.predict(transformed, deterministic=True)
        action = prediction[0] if isinstance(prediction, tuple) else prediction
        resolved = np.asarray(action)
        if (
            resolved.ndim != 1
            or resolved.dtype.kind not in "iu"
            or not self._action_space.contains(resolved)
        ):
            raise ValueError("PPO returned an action outside its bound action space")
        return resolved.copy()

    def decide_portfolio(
        self, snapshot: PortfolioDecisionSnapshot
    ) -> PortfolioTargetProposal:
        """Validate one source snapshot and map its policy action to native targets.

        The snapshot contains no Dataset identity. Callers must obtain it from
        the verified Dataset and runtime context bound when this policy loaded.
        """

        if not isinstance(snapshot, PortfolioDecisionSnapshot):
            raise TypeError("snapshot must be a PortfolioDecisionSnapshot")
        binding = self.runtime_binding
        if (
            snapshot.symbols != self._transform.symbols
            or snapshot.feature_names != self._transform.feature_names
            or snapshot.global_feature_names != self._transform.global_feature_names
        ):
            raise ValueError("snapshot roster differs from the bound policy")
        if (
            snapshot.agent_stop_index != binding.runtime_agent_stop_index
            or not (
                binding.runtime_start_index
                <= snapshot.index
                < binding.runtime_agent_stop_index
            )
            or snapshot.remaining_decisions
            != binding.runtime_agent_stop_index - snapshot.index
        ):
            raise ValueError("snapshot decision window differs from the bound policy")
        expected_timestamp_ns = (
            binding.runtime_start_ns
            + (snapshot.index - binding.runtime_start_index)
            * binding.native_interval_seconds
            * 1_000_000_000
        )
        if snapshot.timestamp_ns != expected_timestamp_ns:
            raise ValueError("snapshot timestamp differs from the bound policy clock")
        raw_observation = encode_portfolio_decision_snapshot(snapshot)
        action = self.predict(raw_observation)
        return portfolio_action_to_proposal(action, snapshot, self._target_abs_weights)


def capture_portfolio_source_runtime() -> dict[str, object]:
    """Record selected imported source owners and the actual PPO runtime versions."""

    modules: list[dict[str, object]] = []
    for name in _SOURCE_MODULES:
        module = importlib.import_module(name)
        origin = getattr(module, "__file__", None)
        if type(origin) is not str:
            raise ValueError(f"source owner {name} has no file origin")
        source_path = Path(origin)
        if source_path.suffix != ".py" or source_path.is_symlink():
            raise ValueError(f"source owner {name} is not a regular Python source file")
        resolved = source_path.resolve(strict=True)
        digest = __import__("hashlib").sha256()
        size = 0
        with open_regular_binary(resolved, field=f"source owner {name}") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
                size += len(chunk)
        modules.append(
            {
                "module": name,
                "origin": resolved.as_posix(),
                "bytes": size,
                "sha256": digest.hexdigest(),
            }
        )

    numpy = importlib.import_module("numpy")
    gymnasium = importlib.import_module("gymnasium")
    torch = importlib.import_module("torch")
    stable_baselines3 = importlib.import_module("stable_baselines3")
    return {
        "scope": _SOURCE_SCOPE,
        "runtime": {
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "platform_system": platform.system(),
            "platform_release": platform.release(),
            "platform_machine": platform.machine(),
            "packages": {
                "numpy": str(numpy.__version__),
                "gymnasium": str(gymnasium.__version__),
                "stable_baselines3": str(stable_baselines3.__version__),
                "torch": str(torch.__version__),
            },
        },
        "source_modules": modules,
    }


def _space_number(value: object) -> int | float | str:
    if isinstance(value, bool):
        return value
    if not isinstance(value, (int, float)):
        raise TypeError("space bounds must be numeric")
    number = float(value)
    if np.isnan(number):
        raise ValueError("space bounds must not contain NaN")
    if np.isposinf(number):
        return "+Infinity"
    if np.isneginf(number):
        return "-Infinity"
    return number


def _space_array(value: object) -> object:
    array = np.asarray(value)
    if array.ndim == 0:
        return _space_number(array.item())
    return [_space_array(item) for item in array.tolist()]


def _space_descriptor(space: gym.Space[Any]) -> dict[str, object]:
    if isinstance(space, gym.spaces.Dict):
        keys = tuple(space.spaces)
        return {
            "kind": f"{type(space).__module__}.{type(space).__qualname__}",
            "field_order": list(keys),
            "fields": {key: _space_descriptor(space.spaces[key]) for key in keys},
        }
    if isinstance(space, gym.spaces.MultiDiscrete):
        return {
            "kind": f"{type(space).__module__}.{type(space).__qualname__}",
            "shape": [int(value) for value in space.shape],
            "nvec": [int(value) for value in np.asarray(space.nvec).tolist()],
            "start": [int(value) for value in np.asarray(space.start).tolist()],
            "dtype": np.dtype(space.dtype).name,
        }
    shape = getattr(space, "shape", None)
    dtype = getattr(space, "dtype", None)
    if shape is None or dtype is None:
        raise TypeError("policy spaces must expose shape and dtype")
    descriptor: dict[str, object] = {
        "kind": f"{type(space).__module__}.{type(space).__qualname__}",
        "shape": [int(value) for value in shape],
        "dtype": np.dtype(dtype).name,
    }
    if isinstance(space, gym.spaces.Box):
        descriptor["low"] = _space_array(space.low)
        descriptor["high"] = _space_array(space.high)
    elif isinstance(space, gym.spaces.MultiBinary):
        descriptor["n"] = [int(value) for value in np.asarray(space.n).reshape(-1)]
    return descriptor


def _strict_object(value: object, *, field: str) -> dict[str, object]:
    if type(value) is not dict or any(type(key) is not str for key in value):
        raise ValueError(f"{field} must be a JSON object")
    return cast(dict[str, object], value)


def _exact_keys(value: dict[str, object], keys: set[str], *, field: str) -> None:
    if set(value) != keys:
        raise ValueError(f"{field} has missing or extra fields")


def _strict_int(value: object, *, field: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}")
    return value


def _string_array(value: object, *, field: str) -> list[str]:
    if type(value) is not list or any(type(item) is not str for item in value):
        raise ValueError(f"{field} must be a string array")
    return cast(list[str], value)


def _strict_number(value: object, *, field: str, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a non-boolean number")
    resolved = float(value)
    if not np.isfinite(resolved) or (positive and resolved <= 0.0):
        raise ValueError(
            f"{field} must be finite and positive"
            if positive
            else f"{field} must be finite"
        )
    return resolved


def _read_canonical_json(
    path: Path, *, expected_digest: str, field: str
) -> tuple[dict[str, object], bytes]:
    require_sha256(expected_digest, field=f"{field}.expected_digest")
    with open_regular_binary(path, field=field) as handle:
        payload = handle.read(16 * 1024 * 1024 + 1)
    if len(payload) > 16 * 1024 * 1024:
        raise ValueError(f"{field} exceeds the metadata size limit")
    if hashlib.sha256(payload).hexdigest() != expected_digest:
        raise ValueError(f"{field} digest mismatch")

    return _decode_canonical_json(payload, field=field), payload


def _decode_canonical_json(payload: bytes, *, field: str) -> dict[str, object]:
    if len(payload) > 16 * 1024 * 1024:
        raise ValueError(f"{field} exceeds the metadata size limit")

    def reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f"{field} contains a duplicate JSON key")
            result[key] = item
        return result

    try:
        decoded = json.loads(
            payload.decode("utf-8"), object_pairs_hook=reject_duplicate_keys
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{field} is not valid UTF-8 JSON") from error
    result = _strict_object(decoded, field=field)
    if canonical_json_bytes(result) != payload:
        raise ValueError(f"{field} is not canonical JSON")
    return result


def _read_unpinned_json(path: Path, *, field: str) -> dict[str, object]:
    with open_regular_binary(path, field=field) as handle:
        payload = handle.read(16 * 1024 * 1024 + 1)
    return _decode_canonical_json(payload, field=field)


def _validated_root(path: str | Path, *, field: str, must_exist: bool) -> Path:
    root = Path(path).expanduser().absolute()
    if root.is_symlink():
        raise ValueError(f"{field} must not be a symlink")
    if must_exist:
        if not root.is_dir():
            raise FileNotFoundError(f"{field} must be an existing directory")
        return root.resolve(strict=True)
    if root.exists():
        raise FileExistsError(f"{field} already exists: {root}")
    if not root.parent.is_dir() or root.parent.is_symlink():
        raise FileNotFoundError(f"{field} parent must be a regular existing directory")
    return root


def _safe_member(root: Path, relative: object, *, field: str) -> Path:
    if type(relative) is not str or not relative or "\\" in relative:
        raise ValueError(f"{field} must be a canonical relative path")
    parts = relative.split("/")
    if any(part in {"", ".", ".."} for part in parts) or relative.startswith("/"):
        raise ValueError(f"{field} must not escape its artifact root")
    current = root
    for part in parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"{field} must not traverse symlinks")
    if not current.is_file():
        raise FileNotFoundError(f"{field} is not a regular file")
    if not current.resolve(strict=True).is_relative_to(root.resolve(strict=True)):
        raise ValueError(f"{field} escapes its artifact root")
    return current


def _read_json_member(
    root: Path, relative: str, *, expected_digest: str, field: str
) -> dict[str, object]:
    path = _safe_member(root, relative, field=field)
    result, _ = _read_canonical_json(path, expected_digest=expected_digest, field=field)
    return result


def _objective_from_payload(value: object) -> ObjectiveContract:
    objective = _strict_object(value, field="objective")
    _exact_keys(
        objective,
        {
            "schema",
            "objective_id",
            "tax_treatment",
            "infrastructure_cost_treatment",
            "external_cash_flow_convention",
            "capital",
            "evaluation_start",
            "evaluation_stop_exclusive",
            "terminal_valuation",
            "economics_digest",
            "risk_digest",
            "deployment_recipe_digest",
            "maximum_drawdown",
        },
        field="objective",
    )
    capital = _strict_object(objective["capital"], field="objective.capital")
    _exact_keys(
        capital, {"mode", "currency", "initial_equities"}, field="objective.capital"
    )
    equities = capital["initial_equities"]
    if type(equities) is not list:
        raise ValueError("objective capital initial_equities must be an array")
    account_mode = capital["mode"]
    currency = capital["currency"]
    terminal_valuation = objective["terminal_valuation"]
    if type(account_mode) is not str or account_mode not in {
        "independent_symbol",
        "shared_portfolio",
    }:
        raise ValueError("objective account mode is unsupported")
    if type(currency) is not str or not currency:
        raise ValueError("objective currency must be non-empty")
    if type(terminal_valuation) is not str or terminal_valuation not in {
        "settled",
        "marked_continuation",
    }:
        raise ValueError("objective terminal valuation is unsupported")
    economics_digest = require_sha256(
        cast(str, objective["economics_digest"]), field="objective economics_digest"
    )
    risk_digest = require_sha256(
        cast(str, objective["risk_digest"]), field="objective risk_digest"
    )
    deployment_digest = require_sha256(
        cast(str, objective["deployment_recipe_digest"]),
        field="objective deployment_recipe_digest",
    )
    maximum_drawdown = _strict_number(
        objective["maximum_drawdown"], field="objective maximum_drawdown"
    )
    try:
        result = ObjectiveContract(
            capital=CapitalContract(
                account_mode=cast(
                    Literal["independent_symbol", "shared_portfolio"], account_mode
                ),
                currency=currency,
                initial_equities=tuple(equities),
            ),
            evaluation_start=datetime.fromisoformat(
                cast(str, objective["evaluation_start"])
            ),
            evaluation_stop_exclusive=datetime.fromisoformat(
                cast(str, objective["evaluation_stop_exclusive"])
            ),
            terminal_valuation=cast(
                Literal["settled", "marked_continuation"], terminal_valuation
            ),
            economics_digest=economics_digest,
            risk_digest=risk_digest,
            deployment_recipe_digest=deployment_digest,
            maximum_drawdown=maximum_drawdown,
        )
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("objective typed payload is invalid") from error
    if canonical_json_bytes(result.payload()) != canonical_json_bytes(objective):
        raise ValueError("objective typed payload is not canonical")
    return result


def _clock_from_payload(value: object) -> FinancialClockContract:
    clock = _strict_object(value, field="financial_clock")
    _exact_keys(
        clock,
        {
            "schema",
            "decision_interval_seconds",
            "execution_interval_seconds",
            "reward_interval_seconds",
            "economic_horizon_seconds",
            "rollout_steps",
            "gamma",
            "gae_lambda",
            "reward_schema",
        },
        field="financial_clock",
    )
    reward_schema = clock["reward_schema"]
    if type(reward_schema) is not str or reward_schema not in {
        "net_log_return_v1",
        "equity_delta_v1",
    }:
        raise ValueError("financial clock reward schema is unsupported")
    try:
        result = FinancialClockContract(
            decision_interval_seconds=_strict_int(
                clock["decision_interval_seconds"],
                field="decision_interval_seconds",
                minimum=1,
            ),
            execution_interval_seconds=_strict_int(
                clock["execution_interval_seconds"],
                field="execution_interval_seconds",
                minimum=1,
            ),
            reward_interval_seconds=_strict_int(
                clock["reward_interval_seconds"],
                field="reward_interval_seconds",
                minimum=1,
            ),
            economic_horizon_seconds=_strict_int(
                clock["economic_horizon_seconds"],
                field="economic_horizon_seconds",
                minimum=1,
            ),
            rollout_steps=_strict_int(
                clock["rollout_steps"], field="rollout_steps", minimum=1
            ),
            gamma=_strict_number(clock["gamma"], field="gamma"),
            gae_lambda=_strict_number(clock["gae_lambda"], field="gae_lambda"),
            reward_schema=cast(
                Literal["net_log_return_v1", "equity_delta_v1"], reward_schema
            ),
        )
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("financial clock typed payload is invalid") from error
    if canonical_json_bytes(result.payload()) != canonical_json_bytes(clock):
        raise ValueError("financial clock typed payload is not canonical")
    return result


def _validated_training_context(
    value: object, manifest: dict[str, object]
) -> tuple[PortfolioObservationTransform, BoundObjectiveClock, dict[str, object]]:
    context = _strict_object(value, field="training_context")
    _exact_keys(
        context,
        {
            "schema",
            "training_context_digest",
            "context",
            "transform",
            "transform_digest",
            "objective",
            "objective_digest",
            "financial_clock",
            "financial_clock_digest",
            "bound_digest",
            "financial_context",
            "policy_layout",
            "training_config",
            "observed_source_runtime",
            "caller_declared_research_context",
            "research_attestation",
        },
        field="training_context",
    )
    if context["schema"] != _TRAINING_CONTEXT_SCHEMA:
        raise ValueError("training context schema differs")
    declared_context = _strict_object(
        context["context"], field="training_context.context"
    )
    if content_digest(declared_context) != context["training_context_digest"]:
        raise ValueError("training context digest does not match its declared context")
    if (
        manifest.get("context") != declared_context
        or manifest.get("context_digest") != context["training_context_digest"]
    ):
        raise ValueError("training context differs from the completed attempt manifest")

    expected_context_keys = {
        "dataset_id",
        "symbols",
        "feature_names",
        "global_feature_names",
        "start_index",
        "stop_index",
        "initial_capital",
        "target_abs_weights",
        "bound_digest",
        "transform_digest",
        "execution_cost",
        "risk_config",
        "market_order_profile_digest",
    }
    _exact_keys(
        declared_context, expected_context_keys, field="training context declaration"
    )
    symbols = _string_array(declared_context["symbols"], field="symbols")
    features = _string_array(declared_context["feature_names"], field="feature_names")
    globals_ = _string_array(
        declared_context["global_feature_names"], field="global_feature_names"
    )
    weights = declared_context["target_abs_weights"]
    if (
        not symbols
        or len(set(symbols)) != len(symbols)
        or any(not item for item in symbols)
    ):
        raise ValueError("training context symbols must be ordered and unique")
    if len(set(features)) != len(features) or any(not item for item in features):
        raise ValueError("training context features must be ordered and unique")
    if len(set(globals_)) != len(globals_) or any(not item for item in globals_):
        raise ValueError("training context global features must be ordered and unique")
    if type(weights) is not list or len(weights) != len(symbols):
        raise ValueError("training context target weights must align with symbols")
    typed_weights = cast(list[object], weights)
    if any(
        _strict_number(item, field="target_abs_weights") < 0.0 for item in typed_weights
    ):
        raise ValueError("training context target weights must be non-negative")
    start = _strict_int(declared_context["start_index"], field="start_index")
    stop = _strict_int(declared_context["stop_index"], field="stop_index", minimum=1)
    if start >= stop:
        raise ValueError("training context index window is empty")
    initial_capital = _strict_number(
        declared_context["initial_capital"], field="initial_capital", positive=True
    )
    for name in ("dataset_id", "bound_digest", "transform_digest"):
        require_sha256(
            cast(str, declared_context[name]), field=f"training context {name}"
        )
    profile_digest = declared_context["market_order_profile_digest"]
    if profile_digest is not None:
        require_sha256(cast(str, profile_digest), field="market_order_profile_digest")
    if (
        type(declared_context["execution_cost"]) is not dict
        or type(declared_context["risk_config"]) is not dict
    ):
        raise ValueError("training context execution and risk values must be objects")

    objective = _objective_from_payload(context["objective"])
    clock = _clock_from_payload(context["financial_clock"])
    bound = BoundObjectiveClock(objective, clock)
    if (
        objective.digest != context["objective_digest"]
        or clock.digest != context["financial_clock_digest"]
        or bound.digest != context["bound_digest"]
        or bound.digest != declared_context["bound_digest"]
        or objective.capital.account_mode != "shared_portfolio"
        or len(objective.capital.initial_equities) != 1
        or objective.capital.initial_equities[0] != initial_capital
        or clock.terminal_profit_aligned is False
        or clock.economic_horizon_seconds
        != (stop - start) * clock.decision_interval_seconds
    ):
        raise ValueError(
            "typed objective/clock binding differs from the training declaration"
        )

    transform = PortfolioObservationTransform.from_payload(
        context["transform"], expected_digest=cast(str, context["transform_digest"])
    )
    if (
        transform.digest != context["transform_digest"]
        or transform.bound_digest != bound.digest
        or transform.objective_digest != objective.digest
        or transform.clock_digest != clock.digest
        or transform.source_dataset_id != declared_context["dataset_id"]
        or transform.symbols != tuple(symbols)
        or transform.feature_names != tuple(features)
        or transform.global_feature_names != tuple(globals_)
        or transform.start_index != start
        or transform.stop_index != stop
        or transform.initial_capital != initial_capital
        or transform.risk_config_digest
        != content_digest(declared_context["risk_config"])
        or transform.execution_config_digest
        != content_digest(declared_context["execution_cost"])
        or transform.market_order_profile_digest != profile_digest
    ):
        raise ValueError("transform differs from the typed training declaration")

    financial = _strict_object(context["financial_context"], field="financial_context")
    _exact_keys(
        financial,
        {
            "initial_capital",
            "target_abs_weights",
            "execution_cost",
            "risk_config",
            "market_order_profile",
            "market_order_profile_digest",
        },
        field="financial_context",
    )
    if (
        financial["initial_capital"] != initial_capital
        or financial["target_abs_weights"] != weights
        or financial["execution_cost"] != declared_context["execution_cost"]
        or financial["risk_config"] != declared_context["risk_config"]
        or financial["market_order_profile_digest"] != profile_digest
    ):
        raise ValueError("financial context differs from the training declaration")
    profile = financial["market_order_profile"]
    if profile is None:
        if profile_digest is not None:
            raise ValueError("market profile digest has no profile payload")
    else:
        profile_payload = _strict_object(profile, field="market_order_profile")
        if content_digest(profile_payload) != profile_digest:
            raise ValueError("market profile payload digest differs")

    layout = _strict_object(context["policy_layout"], field="policy_layout")
    _exact_keys(
        layout,
        {
            "observation_schema",
            "observation_keys",
            "observation_space",
            "symbols",
            "feature_names",
            "global_feature_names",
            "action_space",
            "action_codes",
            "target_abs_weights",
        },
        field="policy_layout",
    )
    expected_action_space = _space_descriptor(
        gym.spaces.MultiDiscrete(np.full(len(symbols), 4, dtype=np.int64))
    )
    expected_layout = {
        "observation_schema": "shared_portfolio_dict_transform_v1",
        "observation_keys": list(transform.observation_keys),
        "observation_space": _space_descriptor(transform.observation_space),
        "symbols": symbols,
        "feature_names": features,
        "global_feature_names": globals_,
        "action_space": expected_action_space,
        "action_codes": [0, 1, 2, 3],
        "target_abs_weights": weights,
    }
    if canonical_json_bytes(layout) != canonical_json_bytes(expected_layout):
        raise ValueError("policy layout differs from the typed training context")

    training_config = _strict_object(
        context["training_config"], field="training_config"
    )
    _exact_keys(
        training_config,
        {
            "effective_config",
            "requested_timesteps",
            "execution_seed",
            "reset_seed",
            "ppo_seed",
        },
        field="training_config",
    )
    effective = _strict_object(
        training_config["effective_config"], field="effective_config"
    )
    _exact_keys(
        effective,
        {
            "total_timesteps",
            "n_steps",
            "batch_size",
            "n_epochs",
            "reset_seed",
            "ppo_seed",
            "learning_rate",
            "gae_lambda",
            "clip_range",
            "ent_coef",
            "vf_coef",
            "max_grad_norm",
            "target_kl",
            "gamma",
            "normalize_advantage",
            "normalize_reward",
            "device",
            "n_envs",
            "policy",
            "policy_net_arch",
            "policy_activation",
            "policy_features_extractor",
            "policy_ortho_init",
            "use_sde",
        },
        field="training effective_config",
    )
    if (
        effective.get("policy") != "MultiInputPolicy"
        or effective.get("device") != "cpu"
        or type(effective.get("gamma")) is not float
        or effective.get("gamma") != 1.0
        or effective.get("normalize_reward") is not False
        or effective.get("normalize_advantage") is not True
        or type(effective.get("n_envs")) is not int
        or effective.get("n_envs") != 1
        or effective.get("n_steps") != clock.rollout_steps
        or effective.get("gae_lambda") != clock.gae_lambda
        or effective.get("policy_net_arch") != {"pi": [64, 64], "vf": [64, 64]}
        or effective.get("policy_activation") != "torch.nn.Tanh"
        or effective.get("policy_features_extractor")
        != "stable_baselines3.common.torch_layers.CombinedExtractor"
        or effective.get("policy_ortho_init") is not True
        or effective.get("use_sde") is not False
        or canonical_json_bytes(effective)
        != canonical_json_bytes(manifest.get("effective_config"))
        or training_config["requested_timesteps"] != manifest.get("requested_timesteps")
    ):
        raise ValueError("effective training configuration is unsupported")
    if (
        training_config["requested_timesteps"] != effective.get("total_timesteps")
        or training_config["execution_seed"] != manifest.get("execution_seed")
        or training_config["reset_seed"] != manifest.get("reset_seed")
        or training_config["ppo_seed"] != manifest.get("ppo_seed")
    ):
        raise ValueError("training configuration differs from the attempt manifest")
    execution_seed = _strict_int(
        training_config["execution_seed"], field="execution_seed"
    )
    reset_seed = _strict_int(training_config["reset_seed"], field="reset_seed")
    ppo_seed = _strict_int(training_config["ppo_seed"], field="ppo_seed")
    if len({execution_seed, reset_seed, ppo_seed}) != 3:
        raise ValueError("execution, reset, and PPO seeds must be distinct")

    source_runtime = _validate_source_runtime(context["observed_source_runtime"])
    if context["caller_declared_research_context"] is not None:
        raise ValueError("caller-declared research labels are not inference provenance")
    if (
        context["research_attestation"]
        != "software training context; not economic evidence"
    ):
        raise ValueError("training context research attestation differs")
    return transform, bound, source_runtime


def _validate_source_runtime(value: object) -> dict[str, object]:
    source = _strict_object(value, field="observed_source_runtime")
    _exact_keys(
        source, {"scope", "runtime", "source_modules"}, field="observed_source_runtime"
    )
    runtime = _strict_object(source["runtime"], field="source runtime")
    _exact_keys(
        runtime,
        {
            "python_version",
            "python_implementation",
            "platform_system",
            "platform_release",
            "platform_machine",
            "packages",
        },
        field="source runtime",
    )
    packages = _strict_object(runtime["packages"], field="source runtime packages")
    if set(packages) != {"numpy", "gymnasium", "stable_baselines3", "torch"}:
        raise ValueError("source runtime package roster differs")
    for field, item in runtime.items():
        if field == "packages":
            if any(
                type(version) is not str or not version for version in packages.values()
            ):
                raise ValueError(
                    "source runtime package versions must be non-empty strings"
                )
        elif type(item) is not str or not item:
            raise ValueError(f"source runtime {field} must be a non-empty string")
    modules = source["source_modules"]
    if type(modules) is not list or len(modules) != len(_SOURCE_MODULES):
        raise ValueError("source runtime module roster differs")
    names: list[str] = []
    for item in modules:
        entry = _strict_object(item, field="source runtime module")
        _exact_keys(
            entry,
            {"module", "origin", "bytes", "sha256"},
            field="source runtime module",
        )
        if type(entry["module"]) is not str or type(entry["origin"]) is not str:
            raise ValueError("source runtime module identity is malformed")
        _strict_int(entry["bytes"], field="source module bytes", minimum=1)
        require_sha256(cast(str, entry["sha256"]), field="source module sha256")
        names.append(cast(str, entry["module"]))
    if tuple(names) != _SOURCE_MODULES or source["scope"] != _SOURCE_SCOPE:
        raise ValueError("source runtime ordered owner scope differs")
    return source


def _source_runtime_matches(recorded: object, observed: dict[str, object]) -> bool:
    try:
        source = _validate_source_runtime(recorded)
    except (TypeError, ValueError):
        return False
    if source["runtime"] != observed["runtime"]:
        return False
    recorded_modules = cast(list[dict[str, object]], source["source_modules"])
    observed_modules = cast(list[dict[str, object]], observed["source_modules"])
    return [
        (item["module"], item["bytes"], item["sha256"]) for item in recorded_modules
    ] == [(item["module"], item["bytes"], item["sha256"]) for item in observed_modules]


def _write_copy_verified(
    source: Path,
    destination: Path,
    *,
    expected_digest: str,
    expected_size: int,
    field: str,
) -> None:
    require_sha256(expected_digest, field=f"{field}.expected_digest")
    digest = hashlib.sha256()
    size = 0
    with open_regular_binary(source, field=field) as src, destination.open("xb") as dst:
        while True:
            chunk = src.read(1024 * 1024)
            if not chunk:
                break
            dst.write(chunk)
            digest.update(chunk)
            size += len(chunk)
        dst.flush()
        import os

        os.fsync(dst.fileno())
    if size != expected_size or digest.hexdigest() != expected_digest:
        raise ValueError(f"{field} changed while publishing the inference bundle")


def _load_policy_from_verified_path(path: Path) -> Any:
    try:
        sb3 = importlib.import_module("stable_baselines3")
        ppo = getattr(sb3, "PPO")
    except (ImportError, AttributeError) as error:
        raise RuntimeError(
            "stable-baselines3 and torch are required; install the train-sb3 extra"
        ) from error
    return ppo.load(str(path), device="cpu", print_system_info=False)


def _validate_loaded_policy_architecture(
    model: object, *, effective_config: dict[str, object]
) -> None:
    """Check the deserialized PPO against the fixed, declared policy architecture."""

    try:
        sb3_policies = importlib.import_module("stable_baselines3.common.policies")
        torch = importlib.import_module("torch")
        torch_layers = importlib.import_module("stable_baselines3.common.torch_layers")
        policy_type = getattr(sb3_policies, "MultiInputActorCriticPolicy")
        extractor_type = getattr(torch_layers, "CombinedExtractor")
        nn = getattr(torch, "nn")
    except (ImportError, AttributeError) as error:
        raise RuntimeError(
            "stable-baselines3 and torch are required; install the train-sb3 extra"
        ) from error

    actual_policy = getattr(model, "policy", None)
    if type(actual_policy) is not policy_type:
        raise ValueError("loaded PPO policy architecture differs from training")
    if getattr(actual_policy, "activation_fn", None) is not getattr(nn, "Tanh"):
        raise ValueError("loaded PPO policy activation differs from training")
    if type(getattr(actual_policy, "features_extractor", None)) is not extractor_type:
        raise ValueError("loaded PPO policy features extractor differs from training")

    declared_arch = _strict_object(
        effective_config.get("policy_net_arch"), field="policy_net_arch"
    )
    mlp_extractor = getattr(actual_policy, "mlp_extractor", None)
    if mlp_extractor is None:
        raise ValueError("loaded PPO policy has no MLP extractor")
    for branch, network_name in (("pi", "policy_net"), ("vf", "value_net")):
        widths = declared_arch.get(branch)
        if type(widths) is not list or any(
            type(width) is not int or width < 1 for width in widths
        ):
            raise ValueError("declared PPO policy architecture is malformed")
        network = getattr(mlp_extractor, network_name, None)
        if type(network) is not getattr(nn, "Sequential"):
            raise ValueError("loaded PPO policy architecture differs from training")
        layers = tuple(cast(Any, network).children())
        if len(layers) != 2 * len(widths):
            raise ValueError("loaded PPO policy architecture differs from training")
        for index, width in enumerate(widths):
            linear = layers[2 * index]
            activation = layers[2 * index + 1]
            if (
                type(linear) is not getattr(nn, "Linear")
                or type(getattr(linear, "out_features", None)) is not int
                or linear.out_features != width
                or type(activation) is not getattr(nn, "Tanh")
            ):
                raise ValueError("loaded PPO policy architecture differs from training")


def _validate_training_manifest(value: dict[str, object]) -> dict[str, object]:
    _exact_keys(
        value,
        {
            "schema",
            "status",
            "effective_config",
            "requested_timesteps",
            "realized_timesteps_contract",
            "execution_seed",
            "reset_seed",
            "ppo_seed",
            "context",
            "context_digest",
            "policy_version_semantics",
            "episode_journal_path",
            "join_journal_path",
            "rollouts_directory",
            "checkpoints_directory",
        },
        field="training manifest",
    )
    if (
        value["schema"] != _TRAINING_SCHEMA
        or value["status"] != "started"
        or value["realized_timesteps_contract"]
        != "exact total_timesteps; no rollout rounding"
        or value["policy_version_semantics"]
        != "sha256 of the exact saved PPO checkpoint zip bytes before this rollout; not an inference-provenance attestation"
        or value["episode_journal_path"] != "native"
        or value["join_journal_path"] != "joins"
        or value["rollouts_directory"] != "rollouts"
        or value["checkpoints_directory"] != "checkpoints"
    ):
        raise ValueError("training manifest schema or fixed semantics differ")
    effective = _strict_object(value["effective_config"], field="effective_config")
    expected_keys = {
        "total_timesteps",
        "n_steps",
        "batch_size",
        "n_epochs",
        "reset_seed",
        "ppo_seed",
        "learning_rate",
        "gae_lambda",
        "clip_range",
        "ent_coef",
        "vf_coef",
        "max_grad_norm",
        "target_kl",
        "gamma",
        "normalize_advantage",
        "normalize_reward",
        "device",
        "n_envs",
        "policy",
        "policy_net_arch",
        "policy_activation",
        "policy_features_extractor",
        "policy_ortho_init",
        "use_sde",
    }
    _exact_keys(effective, expected_keys, field="effective_config")
    total = _strict_int(
        effective["total_timesteps"], field="total_timesteps", minimum=1
    )
    n_steps = _strict_int(effective["n_steps"], field="n_steps", minimum=2)
    batch_size = _strict_int(effective["batch_size"], field="batch_size", minimum=2)
    _strict_int(effective["n_epochs"], field="n_epochs", minimum=1)
    if (
        total < n_steps
        or total % n_steps
        or batch_size > n_steps
        or n_steps % batch_size
    ):
        raise ValueError("effective PPO rollout dimensions are inconsistent")
    requested = _strict_int(
        value["requested_timesteps"], field="requested_timesteps", minimum=1
    )
    execution_seed = _strict_int(value["execution_seed"], field="execution_seed")
    reset_seed = _strict_int(value["reset_seed"], field="reset_seed")
    ppo_seed = _strict_int(value["ppo_seed"], field="ppo_seed")
    if (
        requested != total
        or effective["policy"] != "MultiInputPolicy"
        or effective["device"] != "cpu"
        or type(effective["n_envs"]) is not int
        or effective["n_envs"] != 1
        or type(effective["gamma"]) is not float
        or effective["gamma"] != 1.0
        or effective["normalize_advantage"] is not True
        or effective["normalize_reward"] is not False
        or effective["policy_net_arch"] != {"pi": [64, 64], "vf": [64, 64]}
        or effective["policy_activation"] != "torch.nn.Tanh"
        or effective["policy_features_extractor"]
        != "stable_baselines3.common.torch_layers.CombinedExtractor"
        or effective["policy_ortho_init"] is not True
        or effective["use_sde"] is not False
    ):
        raise ValueError("effective PPO policy configuration is unsupported")
    for field in (
        "learning_rate",
        "gae_lambda",
        "clip_range",
        "ent_coef",
        "vf_coef",
        "max_grad_norm",
    ):
        _strict_number(effective[field], field=f"effective_config.{field}")
    if (
        _strict_number(effective["learning_rate"], field="learning_rate") <= 0.0
        or not 0.0 <= _strict_number(effective["gae_lambda"], field="gae_lambda") <= 1.0
        or _strict_number(effective["clip_range"], field="clip_range") <= 0.0
        or _strict_number(effective["ent_coef"], field="ent_coef") < 0.0
        or _strict_number(effective["vf_coef"], field="vf_coef") < 0.0
        or _strict_number(effective["max_grad_norm"], field="max_grad_norm") <= 0.0
    ):
        raise ValueError("effective PPO numeric parameters are out of range")
    if effective["target_kl"] is not None:
        _strict_number(
            effective["target_kl"], field="effective_config.target_kl", positive=True
        )
    if reset_seed != _strict_int(
        effective["reset_seed"], field="effective_config.reset_seed"
    ):
        raise ValueError("manifest reset seed differs from effective configuration")
    if ppo_seed != _strict_int(
        effective["ppo_seed"], field="effective_config.ppo_seed"
    ):
        raise ValueError("manifest PPO seed differs from effective configuration")
    if len({execution_seed, reset_seed, ppo_seed}) != 3:
        raise ValueError("execution, reset, and PPO seeds must be distinct")
    require_sha256(
        cast(str, value["context_digest"]), field="training manifest context_digest"
    )
    return effective


def _validate_completion(
    completion: dict[str, object],
    *,
    manifest: dict[str, object],
    expected_checkpoint_sha256: str,
    attempt_root: Path,
) -> tuple[list[dict[str, object]], str]:
    _exact_keys(
        completion,
        {
            "schema",
            "status",
            "requested_timesteps",
            "realized_timesteps",
            "rollout_count",
            "joined_transition_count",
            "optimizer_step_calls",
            "sb3_epoch_count",
            "rollouts",
            "final_checkpoint_path",
            "final_checkpoint_sha256",
        },
        field="training completion",
    )
    if completion["schema"] != _TRAINING_SCHEMA or completion["status"] != "completed":
        raise ValueError("training attempt has no completed status")
    effective = _validate_training_manifest(manifest)
    total = _strict_int(
        effective["total_timesteps"], field="total_timesteps", minimum=1
    )
    n_steps = _strict_int(effective["n_steps"], field="n_steps", minimum=2)
    rollouts = completion["rollouts"]
    if type(rollouts) is not list:
        raise ValueError("completion rollouts must be an array")
    expected_rollouts = total // n_steps
    if (
        _strict_int(completion["requested_timesteps"], field="requested_timesteps")
        != total
        or _strict_int(completion["realized_timesteps"], field="realized_timesteps")
        != total
        or _strict_int(completion["rollout_count"], field="rollout_count")
        != expected_rollouts
        or len(rollouts) != expected_rollouts
        or _strict_int(
            completion["joined_transition_count"], field="joined_transition_count"
        )
        != total
    ):
        raise ValueError("training completion does not cover the exact rollout budget")
    _strict_int(completion["optimizer_step_calls"], field="optimizer_step_calls")
    _strict_int(completion["sb3_epoch_count"], field="sb3_epoch_count")
    require_sha256(expected_checkpoint_sha256, field="expected_checkpoint_sha256")
    if completion["final_checkpoint_sha256"] != expected_checkpoint_sha256:
        raise ValueError("final checkpoint digest differs from the trusted receipt")
    checkpoint_path = completion["final_checkpoint_path"]
    if type(checkpoint_path) is not str or not checkpoint_path:
        raise ValueError("final checkpoint path is missing")
    previous_path = "checkpoints/initial.zip"
    initial_path = _safe_member(attempt_root, previous_path, field="initial checkpoint")
    previous_digest, _ = file_digest_and_size(initial_path, field="initial checkpoint")
    optimizer_steps = 0
    sb3_epochs = 0
    for index, item in enumerate(rollouts, start=1):
        record = _strict_object(item, field=f"rollout {index}")
        _exact_keys(
            record,
            {
                "schema",
                "rollout_index",
                "realized_num_timesteps",
                "sample_count",
                "joined_transition_count",
                "optimizer_step_calls",
                "sb3_epoch_count",
                "checkpoint_path",
                "checkpoint_sha256",
            },
            field=f"rollout {index}",
        )
        if (
            record["schema"] != _TRAINING_SCHEMA
            or _strict_int(record["rollout_index"], field="rollout_index") != index
            or _strict_int(
                record["realized_num_timesteps"], field="realized_num_timesteps"
            )
            != index * n_steps
            or _strict_int(record["sample_count"], field="sample_count") != n_steps
            or _strict_int(
                record["joined_transition_count"], field="joined_transition_count"
            )
            != index * n_steps
        ):
            raise ValueError(f"rollout {index} completion is inconsistent")
        prior_optimizer_steps = optimizer_steps
        prior_sb3_epochs = sb3_epochs
        optimizer_steps += _strict_int(
            record["optimizer_step_calls"], field="rollout optimizer_step_calls"
        )
        sb3_epochs += _strict_int(
            record["sb3_epoch_count"], field="rollout sb3_epoch_count"
        )
        expected_path = f"checkpoints/after-rollout-{index:06d}.zip"
        if record["checkpoint_path"] != expected_path:
            raise ValueError(f"rollout {index} checkpoint path is not canonical")
        digest = require_sha256(
            cast(str, record["checkpoint_sha256"]),
            field="rollout checkpoint_sha256",
        )
        started_path = f"rollouts/rollout-{index:06d}/rollout_started.json"
        started = _read_unpinned_json(
            _safe_member(attempt_root, started_path, field=started_path),
            field=started_path,
        )
        _exact_keys(
            started,
            {
                "schema",
                "rollout_index",
                "starting_num_timesteps",
                "starting_optimizer_step_calls",
                "starting_epoch_count",
                "checkpoint_path",
                "policy_version",
                "sample_directory",
            },
            field=f"rollout {index} start",
        )
        if (
            started["schema"] != _TRAINING_SCHEMA
            or _strict_int(started["rollout_index"], field="started.rollout_index")
            != index
            or _strict_int(
                started["starting_num_timesteps"], field="starting_num_timesteps"
            )
            != (index - 1) * n_steps
            or started["checkpoint_path"] != previous_path
            or started["policy_version"] != previous_digest
            or started["sample_directory"]
            != f"rollouts/rollout-{index:06d}/policy-samples"
            or _strict_int(
                started["starting_optimizer_step_calls"],
                field="starting_optimizer_step_calls",
            )
            != prior_optimizer_steps
            or _strict_int(
                started["starting_epoch_count"], field="starting_epoch_count"
            )
            != prior_sb3_epochs
        ):
            raise ValueError(f"rollout {index} start does not follow its predecessor")
        completed_path = f"rollouts/rollout-{index:06d}/rollout_completed.json"
        completed = _read_unpinned_json(
            _safe_member(attempt_root, completed_path, field=completed_path),
            field=completed_path,
        )
        if completed != record:
            raise ValueError(
                f"rollout {index} completion file differs from the final receipt"
            )
        actual_checkpoint, _ = file_digest_and_size(
            _safe_member(
                attempt_root, expected_path, field=f"rollout {index} checkpoint"
            ),
            field=f"rollout {index} checkpoint",
        )
        if actual_checkpoint != digest:
            raise ValueError(
                f"rollout {index} checkpoint bytes differ from its receipt"
            )
        previous_path = expected_path
        previous_digest = digest
    if (
        checkpoint_path != previous_path
        or expected_checkpoint_sha256 != previous_digest
    ):
        raise ValueError(
            "final checkpoint is not the last completed rollout checkpoint"
        )
    if optimizer_steps != _strict_int(
        completion["optimizer_step_calls"], field="optimizer_step_calls"
    ) or sb3_epochs != _strict_int(
        completion["sb3_epoch_count"], field="sb3_epoch_count"
    ):
        raise ValueError(
            "completion optimizer/epoch counts differ from rollout receipts"
        )
    return cast(list[dict[str, object]], rollouts), cast(str, checkpoint_path)


def publish_portfolio_inference_bundle(
    attempt_root: str | Path,
    destination: str | Path,
    *,
    expected_manifest_sha256: str,
    expected_completion_sha256: str,
    expected_training_context_sha256: str,
    expected_checkpoint_sha256: str,
) -> PortfolioInferenceBundleReceipt:
    """Publish one verified completed PPO checkpoint as a separate versioned bundle.

    The expected digests must come from the separately retained training receipt.
    Output roots are single-use; incomplete exports remain visible and are not retried.
    """

    for name, value in (
        ("expected_manifest_sha256", expected_manifest_sha256),
        ("expected_completion_sha256", expected_completion_sha256),
        ("expected_training_context_sha256", expected_training_context_sha256),
        ("expected_checkpoint_sha256", expected_checkpoint_sha256),
    ):
        require_sha256(value, field=name)
    source_root = _validated_root(attempt_root, field="attempt_root", must_exist=True)
    manifest_path = _safe_member(
        source_root, "manifest.json", field="training manifest"
    )
    manifest, manifest_bytes = _read_canonical_json(
        manifest_path,
        expected_digest=expected_manifest_sha256,
        field="training manifest",
    )
    _validate_training_manifest(manifest)
    completion_path = _safe_member(
        source_root, "completion.json", field="training completion"
    )
    completion, _ = _read_canonical_json(
        completion_path,
        expected_digest=expected_completion_sha256,
        field="training completion",
    )
    _, checkpoint_relative = _validate_completion(
        completion,
        manifest=manifest,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
        attempt_root=source_root,
    )
    context_path = _safe_member(
        source_root, "training_context.json", field="training context"
    )
    training_context, context_bytes = _read_canonical_json(
        context_path,
        expected_digest=expected_training_context_sha256,
        field="training context",
    )
    _, _, _ = _validated_training_context(training_context, manifest)
    checkpoint_path = _safe_member(
        source_root, checkpoint_relative, field="final checkpoint"
    )
    checkpoint_sha256, checkpoint_size = file_digest_and_size(
        checkpoint_path, field="final checkpoint"
    )
    if checkpoint_sha256 != expected_checkpoint_sha256:
        raise ValueError("final checkpoint bytes differ from the trusted receipt")
    _safe_member(source_root, "checkpoints/initial.zip", field="initial checkpoint")

    output_root = _validated_root(destination, field="destination", must_exist=False)
    output_root.mkdir(parents=False, exist_ok=False)
    manifest_written = False
    try:
        context_output = output_root / "training_context.json"
        with context_output.open("xb") as handle:
            handle.write(context_bytes)
            handle.flush()
            import os

            os.fsync(handle.fileno())
        training_manifest_output = output_root / "training_manifest.json"
        with training_manifest_output.open("xb") as handle:
            handle.write(manifest_bytes)
            handle.flush()
            import os

            os.fsync(handle.fileno())
        policy_output = output_root / "policy.zip"
        _write_copy_verified(
            checkpoint_path,
            policy_output,
            expected_digest=expected_checkpoint_sha256,
            expected_size=checkpoint_size,
            field="final checkpoint",
        )
        bundle_manifest: dict[str, object] = {
            "schema": _BUNDLE_SCHEMA,
            "training_attempt": {
                "manifest_sha256": expected_manifest_sha256,
                "completion_sha256": expected_completion_sha256,
                "training_context_sha256": expected_training_context_sha256,
                "training_context_digest": training_context["training_context_digest"],
                "final_checkpoint_sha256": expected_checkpoint_sha256,
            },
            "files": {
                "training_context.json": {
                    "sha256": expected_training_context_sha256,
                    "size_bytes": len(context_bytes),
                },
                "training_manifest.json": {
                    "sha256": expected_manifest_sha256,
                    "size_bytes": len(manifest_bytes),
                },
                "policy.zip": {
                    "sha256": expected_checkpoint_sha256,
                    "size_bytes": checkpoint_size,
                },
            },
        }
        manifest_payload = canonical_json_bytes(bundle_manifest)
        atomic_write_bytes(output_root / "manifest.json", manifest_payload)
        manifest_written = True
        return PortfolioInferenceBundleReceipt(
            root=output_root,
            bundle_sha256=content_digest(bundle_manifest),
            training_manifest_sha256=expected_manifest_sha256,
            training_completion_sha256=expected_completion_sha256,
            training_context_sha256=expected_training_context_sha256,
            checkpoint_sha256=expected_checkpoint_sha256,
        )
    except BaseException as error:
        if not manifest_written:
            failure = {
                "schema": "shared_portfolio_inference_bundle_failure_v1",
                "status": "failed",
                "error_type": f"{type(error).__module__}.{type(error).__qualname__}",
                "error_message": str(error),
                "expected_checkpoint_sha256": expected_checkpoint_sha256,
            }
            try:
                atomic_write_bytes(
                    output_root / "failure.json", canonical_json_bytes(failure)
                )
            except BaseException as record_error:
                error.add_note(
                    "Bundle failure receipt also failed: "
                    f"{type(record_error).__name__}: {record_error}"
                )
        raise


def _validate_runtime_environment(
    runtime_env: SharedPortfolioGymEnv,
    *,
    transform: PortfolioObservationTransform,
    training_bound: BoundObjectiveClock,
    training_context: dict[str, object],
) -> PortfolioObservationRuntimeBinding:
    if not isinstance(runtime_env, SharedPortfolioGymEnv):
        raise TypeError("runtime_env must be a SharedPortfolioGymEnv")
    if runtime_env._session is not None:
        raise ValueError("runtime_env must not have been reset before bundle loading")
    financial = _strict_object(
        training_context["financial_context"], field="financial_context"
    )
    profile_payload = financial["market_order_profile"]
    current_profile = (
        None
        if runtime_env.market_order_profile is None
        else runtime_env.market_order_profile.canonical_payload()
    )
    if (
        tuple(runtime_env.target_abs_weights)
        != tuple(cast(list[object], financial["target_abs_weights"]))
        or canonical_json_bytes(to_json_value(runtime_env.execution_cost))
        != canonical_json_bytes(financial["execution_cost"])
        or canonical_json_bytes(to_json_value(runtime_env.risk_config))
        != canonical_json_bytes(financial["risk_config"])
        or canonical_json_bytes(current_profile)
        != canonical_json_bytes(profile_payload)
    ):
        raise ValueError(
            "runtime financial or target-weight context differs from training"
        )
    binding = PortfolioObservationRuntimeBinding.create(
        transform, training_bound, runtime_env
    )
    layout = _strict_object(training_context["policy_layout"], field="policy_layout")
    if (
        _space_descriptor(transform.observation_space) != layout["observation_space"]
        or _space_descriptor(runtime_env.action_space) != layout["action_space"]
    ):
        raise ValueError("runtime policy spaces differ from training")
    return binding


def load_portfolio_inference_bundle(
    root: str | Path,
    *,
    expected_bundle_sha256: str,
    expected_manifest_sha256: str,
    expected_completion_sha256: str,
    expected_training_context_sha256: str,
    expected_checkpoint_sha256: str,
    runtime_env: SharedPortfolioGymEnv,
) -> PortfolioInferencePolicy:
    """Load a separately pinned bundle after validating its forward environment.

    The checkpoint digest must be independently retained from a trusted training
    completion receipt; a digest copied only from an untrusted bundle is not trust.
    """

    for name, value in (
        ("expected_bundle_sha256", expected_bundle_sha256),
        ("expected_manifest_sha256", expected_manifest_sha256),
        ("expected_completion_sha256", expected_completion_sha256),
        ("expected_training_context_sha256", expected_training_context_sha256),
        ("expected_checkpoint_sha256", expected_checkpoint_sha256),
    ):
        require_sha256(value, field=name)
    try:
        observed_runtime = capture_portfolio_source_runtime()
    except (ImportError, AttributeError) as error:
        raise RuntimeError(
            "stable-baselines3 and torch are required; install the train-sb3 extra"
        ) from error

    bundle_root = _validated_root(root, field="bundle root", must_exist=True)
    children = {entry.name for entry in bundle_root.iterdir()}
    if children != _BUNDLE_FILES:
        raise ValueError("inference bundle has missing or extra files")
    manifest, _ = _read_canonical_json(
        _safe_member(bundle_root, "manifest.json", field="bundle manifest"),
        expected_digest=expected_bundle_sha256,
        field="bundle manifest",
    )
    _exact_keys(
        manifest, {"schema", "training_attempt", "files"}, field="bundle manifest"
    )
    if manifest["schema"] != _BUNDLE_SCHEMA:
        raise ValueError("inference bundle schema differs")
    attempt = _strict_object(
        manifest["training_attempt"], field="bundle training_attempt"
    )
    _exact_keys(
        attempt,
        {
            "manifest_sha256",
            "completion_sha256",
            "training_context_sha256",
            "training_context_digest",
            "final_checkpoint_sha256",
        },
        field="bundle training_attempt",
    )
    external_pins = {
        "manifest_sha256": expected_manifest_sha256,
        "completion_sha256": expected_completion_sha256,
        "training_context_sha256": expected_training_context_sha256,
        "final_checkpoint_sha256": expected_checkpoint_sha256,
    }
    if any(attempt[key] != digest for key, digest in external_pins.items()):
        raise ValueError(
            "bundle attempt identities differ from separately held receipts"
        )
    files = _strict_object(manifest["files"], field="bundle files")
    if set(files) != {"training_context.json", "training_manifest.json", "policy.zip"}:
        raise ValueError("inference bundle file roster differs")
    file_records: dict[str, dict[str, object]] = {}
    for name, expected_digest in (
        ("training_context.json", expected_training_context_sha256),
        ("training_manifest.json", expected_manifest_sha256),
        ("policy.zip", expected_checkpoint_sha256),
    ):
        record = _strict_object(files[name], field=f"bundle files.{name}")
        _exact_keys(record, {"sha256", "size_bytes"}, field=f"bundle files.{name}")
        if record["sha256"] != expected_digest:
            raise ValueError(f"bundle {name} digest differs from trusted receipt")
        _strict_int(record["size_bytes"], field=f"bundle {name} size", minimum=1)
        file_records[name] = record
    context_path = _safe_member(
        bundle_root, "training_context.json", field="training context"
    )
    training_context, context_bytes = _read_canonical_json(
        context_path,
        expected_digest=expected_training_context_sha256,
        field="training context",
    )
    if len(context_bytes) != file_records["training_context.json"]["size_bytes"]:
        raise ValueError("training context size differs from bundle manifest")
    training_manifest_path = _safe_member(
        bundle_root, "training_manifest.json", field="training manifest"
    )
    training_manifest, training_manifest_bytes = _read_canonical_json(
        training_manifest_path,
        expected_digest=expected_manifest_sha256,
        field="training manifest",
    )
    if (
        len(training_manifest_bytes)
        != file_records["training_manifest.json"]["size_bytes"]
    ):
        raise ValueError("training manifest size differs from bundle manifest")
    _validate_training_manifest(training_manifest)
    if attempt["training_context_digest"] != training_context.get(
        "training_context_digest"
    ):
        raise ValueError("bundle context digest differs")
    transform, training_bound, source_runtime = _validated_training_context(
        training_context, training_manifest
    )
    if not _source_runtime_matches(source_runtime, observed_runtime):
        raise ValueError("observed source/runtime differs from training")
    runtime_binding = _validate_runtime_environment(
        runtime_env,
        transform=transform,
        training_bound=training_bound,
        training_context=training_context,
    )
    policy_path = _safe_member(bundle_root, "policy.zip", field="policy checkpoint")
    checkpoint_size = file_records["policy.zip"]["size_bytes"]
    with verified_private_copy(
        policy_path,
        expected_digest=expected_checkpoint_sha256,
        expected_size_bytes=cast(int, checkpoint_size),
        field="policy checkpoint",
        filename="policy.zip",
    ) as private_checkpoint:
        policy = _load_policy_from_verified_path(private_checkpoint)
    policy_observation_space = getattr(policy, "observation_space", None)
    policy_action_space = getattr(policy, "action_space", None)
    if (
        policy_observation_space is None
        or policy_action_space is None
        or _space_descriptor(policy_observation_space)
        != _space_descriptor(transform.observation_space)
        or _space_descriptor(policy_action_space)
        != _space_descriptor(runtime_env.action_space)
    ):
        raise ValueError("loaded PPO policy spaces differ from the validated bundle")
    effective_config = _strict_object(
        training_manifest["effective_config"], field="effective_config"
    )
    _validate_loaded_policy_architecture(policy, effective_config=effective_config)
    return PortfolioInferencePolicy(
        _policy=policy,
        _transform=transform,
        _action_space=runtime_env.action_space,
        runtime_binding=runtime_binding,
        _target_abs_weights=tuple(runtime_env.target_abs_weights),
    )
