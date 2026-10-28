"""Extract per-coordinate probabilities from one transformed PPO observation.

This opt-in diagnostic accepts an already-loaded Stable-Baselines3 PPO model.
It does not load checkpoints, sample or update a policy, or attest symbol order,
dataset identity, training provenance, or economic results.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from trade_rl.strategies.portfolio_policy_codec import (
    ACTION_CODES,
    ACTION_FLAT,
    ACTION_HOLD,
    ACTION_LONG,
    ACTION_SHORT,
)

_SCHEMA = "issue810_portfolio_action_probabilities_v1"
_SB3_VERSION = "2.3.2"
_TORCH_VERSION = "2.4.1"
# The pinned package metadata is exactly 2.4.1. Runtime builds may append a
# valid PEP 440 local label, such as +cpu or +cu121.
_TORCH_RUNTIME_VERSION_PATTERN = re.compile(
    rf"{re.escape(_TORCH_VERSION)}(?:\+[A-Za-z0-9]+(?:[._-][A-Za-z0-9]+)*)?",
    re.IGNORECASE,
)
_ACTIONS = (
    ("HOLD", ACTION_HOLD),
    ("SHORT", ACTION_SHORT),
    ("FLAT", ACTION_FLAT),
    ("LONG", ACTION_LONG),
)
_NORMALIZATION_TOLERANCE = 1e-6


@dataclass(frozen=True)
class _Runtime:
    torch: Any
    multi_discrete_type: type[Any]
    ppo_type: type[Any]
    multi_categorical_type: type[Any]
    stable_baselines3_version: str
    torch_version: str
    torch_runtime_version: str


def _load_pinned_runtime() -> _Runtime:
    try:
        stable_baselines3_version = importlib.metadata.version("stable-baselines3")
        torch_version = importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError:
        raise RuntimeError(
            "install the locked train-sb3 extra before extracting probabilities"
        ) from None

    if stable_baselines3_version != _SB3_VERSION:
        raise RuntimeError(
            f"stable-baselines3 {_SB3_VERSION} is required; "
            f"found {stable_baselines3_version!r}"
        )
    if torch_version != _TORCH_VERSION:
        raise RuntimeError(
            f"torch {_TORCH_VERSION} is required; found {torch_version!r}"
        )

    try:
        gymnasium: Any = importlib.import_module("gymnasium")
        torch: Any = importlib.import_module("torch")
        stable_baselines3: Any = importlib.import_module("stable_baselines3")
        distributions: Any = importlib.import_module(
            "stable_baselines3.common.distributions"
        )
    except ImportError:
        raise RuntimeError(
            "the locked train-sb3 runtime could not be imported"
        ) from None

    torch_runtime_version = str(torch.__version__)
    if _TORCH_RUNTIME_VERSION_PATTERN.fullmatch(torch_runtime_version) is None:
        raise RuntimeError(
            "torch runtime build does not match the locked 2.4.1 package: "
            f"found {torch_runtime_version!r}"
        )
    return _Runtime(
        torch=torch,
        multi_discrete_type=gymnasium.spaces.MultiDiscrete,
        ppo_type=stable_baselines3.PPO,
        multi_categorical_type=distributions.MultiCategoricalDistribution,
        stable_baselines3_version=stable_baselines3_version,
        torch_version=torch_version,
        torch_runtime_version=torch_runtime_version,
    )


def _ordered_symbols(
    symbol_names: Sequence[str], *, coordinate_count: int
) -> list[str]:
    if isinstance(symbol_names, (str, bytes)) or not isinstance(symbol_names, Sequence):
        raise TypeError("ordered_symbols must be a sequence of caller-declared names")
    names = list(symbol_names)
    if len(names) != coordinate_count:
        raise ValueError("ordered symbol count must match action coordinates")
    if any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError("ordered symbol names must be non-empty strings")
    if len(set(names)) != len(names):
        raise ValueError("ordered symbol names must be unique")
    return names


def _probability_row(
    component: object,
    *,
    torch: Any,
    coordinate_index: int,
) -> list[float]:
    probabilities = getattr(component, "probs", None)
    if not isinstance(probabilities, torch.Tensor):
        raise TypeError(f"coordinate {coordinate_index} probabilities must be a tensor")
    if not probabilities.is_floating_point():
        raise TypeError(
            f"coordinate {coordinate_index} probabilities must be floating point"
        )
    if tuple(probabilities.shape) != (1, 4):
        raise ValueError(
            f"coordinate {coordinate_index} probability shape must be (1, 4)"
        )

    values = probabilities.detach().to(device="cpu", dtype=torch.float64).numpy()[0]
    if not np.isfinite(values).all():
        raise ValueError(f"coordinate {coordinate_index} probabilities must be finite")
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError(
            f"coordinate {coordinate_index} probabilities must be within [0, 1]"
        )
    total = math.fsum(float(value) for value in values)
    if not math.isclose(
        total,
        1.0,
        rel_tol=_NORMALIZATION_TOLERANCE,
        abs_tol=_NORMALIZATION_TOLERANCE,
    ):
        raise ValueError(f"coordinate {coordinate_index} probabilities must sum to one")
    return [float(value) for value in values]


def extract_portfolio_action_probabilities(
    model: Any,
    observation: Mapping[str, object],
    *,
    ordered_symbols: Sequence[str],
) -> dict[str, object]:
    """Return the four action probabilities for each declared MultiDiscrete axis.

    ``observation`` must be one already-transformed Dict observation. The symbol
    names are caller declarations aligned to the model's action-space order; this
    function does not compare them with Dataset, transform, or checkpoint metadata.
    """

    runtime = _load_pinned_runtime()
    if tuple(code for _, code in _ACTIONS) != ACTION_CODES:
        raise RuntimeError("portfolio action labels differ from the action codec")
    if not isinstance(model, runtime.ppo_type):
        raise TypeError("model must be an already-loaded Stable-Baselines3 PPO")
    if not isinstance(observation, Mapping) or not observation:
        raise TypeError("observation must be a non-empty transformed Dict observation")

    policy = getattr(model, "policy", None)
    if policy is None:
        raise TypeError("PPO model does not expose a policy")
    action_space = getattr(policy, "action_space", None)
    if not isinstance(action_space, runtime.multi_discrete_type):
        raise TypeError("PPO policy action space must be MultiDiscrete")
    nvec = np.asarray(action_space.nvec)
    if nvec.ndim != 1 or nvec.size == 0:
        raise ValueError(
            "MultiDiscrete action space must have one-dimensional coordinates"
        )
    if not np.issubdtype(nvec.dtype, np.integer):
        raise TypeError("MultiDiscrete action dimensions must be integers")
    if np.any(nvec != 4):
        raise ValueError("every action coordinate must have exactly four actions")
    start = np.asarray(action_space.start)
    if start.shape != nvec.shape or np.any(start != 0):
        raise ValueError("MultiDiscrete action coordinates must be zero-based")
    symbols = _ordered_symbols(ordered_symbols, coordinate_count=int(nvec.size))

    modules = tuple(policy.modules())
    if not modules or not callable(getattr(policy, "set_training_mode", None)):
        raise TypeError("PPO policy does not expose the pinned module-mode API")
    module_modes = tuple((module, bool(module.training)) for module in modules)
    try:
        policy.set_training_mode(False)
        observation_tensor, was_vectorized = policy.obs_to_tensor(dict(observation))
        if was_vectorized:
            raise ValueError("observation must be unbatched")
        with runtime.torch.no_grad():
            action_distribution = policy.get_distribution(observation_tensor)
        if not isinstance(action_distribution, runtime.multi_categorical_type):
            raise TypeError("expected a MultiCategoricalDistribution")
        components = getattr(action_distribution, "distribution", None)
        if not isinstance(components, list):
            raise TypeError(
                "pinned MultiCategoricalDistribution must expose its component list"
            )
        if len(components) != int(nvec.size):
            raise ValueError(
                "distribution component count must match action coordinates"
            )

        coordinates: list[dict[str, object]] = []
        action_names = [name for name, _ in _ACTIONS]
        for index, (symbol, component) in enumerate(zip(symbols, components)):
            row = _probability_row(
                component, torch=runtime.torch, coordinate_index=index
            )
            coordinates.append(
                {
                    "coordinate_index": index,
                    "symbol": symbol,
                    "probabilities": dict(zip(action_names, row, strict=True)),
                }
            )
    finally:
        # Restore flags directly: recursive train() calls can overwrite a
        # shared child again when it is reachable through a later parent.
        for module, training in module_modes:
            module.training = training

    return {
        "schema": _SCHEMA,
        "stable_baselines3_version": runtime.stable_baselines3_version,
        "torch_version": runtime.torch_version,
        "torch_runtime_version": runtime.torch_runtime_version,
        "action_order": [name for name, _ in _ACTIONS],
        "ordered_symbols": symbols,
        "symbol_order_source": "caller_declared_not_verified",
        "scope": "single_observation_probabilities_only_no_dataset_or_economic_attestation",
        "probability_api": "SB3 2.3.2 MultiCategoricalDistribution.distribution[*].probs",
        "coordinates": coordinates,
    }
