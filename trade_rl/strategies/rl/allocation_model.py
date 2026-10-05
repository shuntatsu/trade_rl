"""Explicit four-action allocation PPO inference with a frozen receipt."""

from __future__ import annotations

import json
from numbers import Integral
from typing import Any

import gymnasium as gym
import numpy as np

from trade_rl._validation import require_sha256
from trade_rl.artifacts import canonical_json_bytes
from trade_rl.strategies.rl.allocation_manifest import validate_allocation_manifest


def validate_allocation_model(model: Any, manifest: dict[str, Any]) -> None:
    observation, action = model.observation_space, model.action_space
    count = len(manifest["recipe"]["feature_names"]) + len(
        manifest["recipe"]["observation"]["fields"]
    )
    if (
        not isinstance(observation, gym.spaces.Box)
        or observation.shape != (count,)
        or observation.dtype != np.dtype(np.float32)
        or not np.isneginf(observation.low).all()
        or not np.isposinf(observation.high).all()
        or not isinstance(action, gym.spaces.Discrete)
        or action.n != 4
        or action.start != 0
    ):
        raise ValueError(
            "policy spaces differ from the allocation observation/action contract"
        )
    training = manifest["training"]
    ppo = training["ppo"]
    for key in (
        "gamma",
        "gae_lambda",
        "n_steps",
        "batch_size",
        "n_epochs",
        "learning_rate",
    ):
        if getattr(model, key, None) != ppo[key]:
            raise ValueError(f"actual PPO {key} differs from its receipt")
    if (
        str(model.device) != ppo["device"]
        or model.policy.net_arch != ppo["net_arch"]
        or model.num_timesteps != training["actual_timesteps"]
        or model.seed != training["seed"]
        or getattr(model, "n_envs", None) != 1
    ):
        raise ValueError("actual PPO runtime differs from its training receipt")


class AllocationPPOPolicy:
    """Deterministic allocation inference; no research or trading authorization."""

    def __init__(self, model: Any, manifest: dict[str, Any]) -> None:
        normalized = validate_allocation_manifest(manifest)
        validate_allocation_model(model, normalized)
        self.model = model
        self._manifest = canonical_json_bytes(normalized)

    @property
    def manifest(self) -> dict[str, Any]:
        return json.loads(self._manifest)

    def action(self, observation: np.ndarray, *, runtime_recipe_digest: str) -> int:
        require_sha256(runtime_recipe_digest, field="runtime_recipe_digest")
        manifest = self.manifest
        if runtime_recipe_digest != manifest["recipe_digest"]:
            raise ValueError("runtime recipe differs from the frozen allocation policy")
        validate_allocation_model(self.model, manifest)
        with np.errstate(over="raise", invalid="raise"):
            try:
                original = np.asarray(observation)
                if original.dtype.kind not in "iuf":
                    raise ValueError("allocation observation must contain real numbers")
                values = original.astype(np.float32)
            except (TypeError, ValueError, FloatingPointError) as error:
                raise ValueError(
                    "allocation observation must fit finite float32 values"
                ) from error
        if (
            values.shape != self.model.observation_space.shape
            or not np.isfinite(values).all()
        ):
            raise ValueError(
                "allocation observation differs from its finite vector contract"
            )
        raw, _ = self.model.predict(values, deterministic=True)
        codes = np.asarray(raw).reshape(-1)
        if (
            codes.size != 1
            or isinstance(codes[0], (bool, np.bool_))
            or not isinstance(codes[0], Integral)
            or not 0 <= int(codes[0]) <= 3
        ):
            raise ValueError(
                "allocation action must be one integer within {0, 1, 2, 3}"
            )
        return int(codes[0])
