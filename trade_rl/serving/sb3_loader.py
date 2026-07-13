"""Stable-Baselines3 serving loader for validated ensemble bundles."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path, PurePosixPath
from typing import Any, Final

import numpy as np

from trade_rl.serving.bundle import ServingBundle

SB3_POLICY_LOADER_NAME: Final = "policy-loader.json"
SB3_POLICY_LOADER_SCHEMA: Final = "sb3_policy_loader_v1"
_SUPPORTED_ALGORITHMS: Final = frozenset({"ppo", "sac", "td3", "tqc"})


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    return value


def _safe_relative_path(value: object, *, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(
        part in {"", ".", ".."} for part in path.parts
    ):
        raise ValueError(f"{field} contains an unsafe member path")
    return path.as_posix()


class _SB3EnsemblePolicy:
    def __init__(
        self,
        models: tuple[Any, ...],
        *,
        observation_size: int,
        action_size: int,
    ) -> None:
        if not models:
            raise ValueError("SB3 ensemble must contain at least one member")
        self.models = models
        self.observation_size = observation_size
        self.action_size = action_size

    def predict(self, observation: np.ndarray) -> np.ndarray:
        vector = np.asarray(observation, dtype=np.float32).reshape(-1)
        if vector.shape != (self.observation_size,) or not np.isfinite(vector).all():
            raise ValueError("observation does not match the SB3 ensemble contract")
        actions: list[np.ndarray] = []
        for member_index, model in enumerate(self.models):
            try:
                raw, _ = model.predict(vector, deterministic=True)
            except Exception as error:
                raise ValueError(
                    f"SB3 ensemble member {member_index} prediction failed"
                ) from error
            action = np.asarray(raw, dtype=np.float32).reshape(-1)
            if action.shape != (self.action_size,):
                raise ValueError(
                    f"SB3 ensemble member {member_index} action shape mismatch"
                )
            if not np.isfinite(action).all():
                raise ValueError(
                    f"SB3 ensemble member {member_index} action must be finite"
                )
            if np.any(action < -1.0) or np.any(action > 1.0):
                raise ValueError(
                    f"SB3 ensemble member {member_index} action violates bounds"
                )
            actions.append(action)
        averaged = np.mean(np.stack(actions, axis=0), axis=0, dtype=np.float64)
        if not np.isfinite(averaged).all():
            raise ValueError("SB3 ensemble mean action must be finite")
        return averaged.astype(np.float32)


class StableBaselines3PolicyLoader:
    """Load every declared SB3 member and fail closed on partial ensembles."""

    def __init__(
        self,
        *,
        model_loader: Callable[[str, Path], Any] | None = None,
    ) -> None:
        self.model_loader = model_loader or self._load_model

    @staticmethod
    def _load_model(algorithm: str, path: Path) -> Any:
        if algorithm == "ppo":
            from stable_baselines3 import PPO

            return PPO.load(str(path), device="cpu")
        if algorithm == "sac":
            from stable_baselines3 import SAC

            return SAC.load(str(path), device="cpu")
        if algorithm == "td3":
            from stable_baselines3 import TD3

            return TD3.load(str(path), device="cpu")
        if algorithm == "tqc":
            from sb3_contrib import TQC

            return TQC.load(str(path), device="cpu")
        raise ValueError("unsupported Stable-Baselines3 algorithm")

    def load(self, bundle: ServingBundle) -> _SB3EnsemblePolicy:
        manifest_path = bundle.root / SB3_POLICY_LOADER_NAME
        declared_files = {item.path for item in bundle.manifest.files}
        if SB3_POLICY_LOADER_NAME not in declared_files:
            raise ValueError("serving bundle does not declare policy-loader.json")
        if not manifest_path.is_file():
            raise ValueError("serving bundle policy loader manifest is missing")
        payload = _mapping(
            json.loads(manifest_path.read_text(encoding="utf-8")),
            field="SB3 policy loader manifest",
        )
        if payload.get("schema_version") != SB3_POLICY_LOADER_SCHEMA:
            raise ValueError("unsupported SB3 policy loader schema")
        algorithm_value = payload.get("algorithm")
        if not isinstance(algorithm_value, str):
            raise ValueError("SB3 policy loader algorithm must be a string")
        algorithm = algorithm_value.lower()
        if algorithm not in _SUPPORTED_ALGORITHMS:
            raise ValueError("SB3 policy loader algorithm is unsupported")
        raw_members = payload.get("members")
        if not isinstance(raw_members, list) or not raw_members:
            raise ValueError("SB3 policy loader members must be a non-empty list")
        members = tuple(
            _safe_relative_path(item, field=f"members[{index}]")
            for index, item in enumerate(raw_members)
        )
        if len(set(members)) != len(members):
            raise ValueError("SB3 policy loader members must be unique")
        models: list[Any] = []
        for relative in members:
            if relative not in declared_files:
                raise ValueError(
                    f"SB3 policy member is not declared by the serving bundle: {relative}"
                )
            if not relative.endswith("/policy.zip") and relative != "policy.zip":
                raise ValueError("SB3 policy member path must end with policy.zip")
            path = bundle.root / relative
            if not path.is_file():
                raise ValueError(f"SB3 policy member is missing: {relative}")
            model = self.model_loader(algorithm, path)
            if model is None or not callable(getattr(model, "predict", None)):
                raise ValueError(f"SB3 policy member could not be loaded: {relative}")
            models.append(model)
        return _SB3EnsemblePolicy(
            tuple(models),
            observation_size=bundle.manifest.observation_size,
            action_size=bundle.manifest.action_size,
        )
