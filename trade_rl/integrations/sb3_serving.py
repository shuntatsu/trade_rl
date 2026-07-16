"""Stable-Baselines3 serving integration for flat and structured bundles."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping
from pathlib import Path, PurePosixPath
from typing import Any, Final

import numpy as np
from gymnasium import spaces

from trade_rl.data.market import MarketDataset
from trade_rl.rl.normalization import ObservationNormalizer
from trade_rl.rl.observations import ObservationBuilder
from trade_rl.rl.sequence_observations import (
    SequenceObservationBuilder,
    SequenceWindowSpec,
    build_structured_policy_observation,
)
from trade_rl.serving.bundle import ServingBundle
from trade_rl.serving.sequence_normalizer import (
    SEQUENCE_NORMALIZER_ARTIFACT_NAME,
    load_sequence_feature_normalizer,
)

SB3_POLICY_LOADER_NAME: Final = "policy-loader.json"
SB3_POLICY_LOADER_SCHEMA: Final = "sb3_policy_loader_v1"
SB3_STRUCTURED_POLICY_LOADER_SCHEMA: Final = "sb3_policy_loader_v2"
_SUPPORTED_ALGORITHMS: Final = frozenset({"ppo", "sac", "td3", "tqc"})


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    return value


def _safe_relative_path(value: object, *, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"{field} contains an unsafe member path")
    return path.as_posix()


def _validated_action(
    raw: object, *, action_size: int, member_index: int
) -> np.ndarray:
    action = np.asarray(raw, dtype=np.float32).reshape(-1)
    if action.shape != (action_size,):
        raise ValueError(f"SB3 ensemble member {member_index} action shape mismatch")
    if not np.isfinite(action).all():
        raise ValueError(f"SB3 ensemble member {member_index} action must be finite")
    if np.any(action < -1.0) or np.any(action > 1.0):
        raise ValueError(f"SB3 ensemble member {member_index} action violates bounds")
    return action


def _mean_action(actions: list[np.ndarray]) -> np.ndarray:
    averaged = np.mean(np.stack(actions, axis=0), axis=0, dtype=np.float64)
    if not np.isfinite(averaged).all():
        raise ValueError("SB3 ensemble mean action must be finite")
    return np.asarray(averaged, dtype=np.float32)


class _SB3EnsemblePolicy:
    def __init__(
        self, models: tuple[Any, ...], *, observation_size: int, action_size: int
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
            actions.append(
                _validated_action(
                    raw, action_size=self.action_size, member_index=member_index
                )
            )
        return _mean_action(actions)


class _SB3StructuredSequenceEnsemblePolicy:
    """Reconstruct the exact training Dict observation from a rolling dataset."""

    def __init__(
        self,
        models: tuple[Any, ...],
        *,
        action_size: int,
        dataset_reference: Mapping[str, object],
        builder: SequenceObservationBuilder,
        normalizer: ObservationNormalizer,
        sequence_normalizer: Any,
        n_factors: int,
        finite_horizon: bool,
    ) -> None:
        if not models:
            raise ValueError("SB3 structured ensemble must contain at least one member")
        first_space = getattr(models[0], "observation_space", None)
        if not isinstance(first_space, spaces.Dict):
            raise ValueError(
                "structured SB3 member must expose a Dict observation space"
            )
        self.models = models
        self.action_size = action_size
        self.dataset_reference = dict(dataset_reference)
        self.builder = builder
        self.normalizer = normalizer
        self.sequence_normalizer = sequence_normalizer
        self.n_factors = n_factors
        self.finite_horizon = finite_horizon
        self.observation_space = first_space
        expected = {
            key: (space.shape, np.dtype(space.dtype))
            for key, space in first_space.spaces.items()
        }
        for model in models[1:]:
            candidate = getattr(model, "observation_space", None)
            if not isinstance(candidate, spaces.Dict):
                raise ValueError("all structured SB3 members must expose Dict spaces")
            observed = {
                key: (space.shape, np.dtype(space.dtype))
                for key, space in candidate.spaces.items()
            }
            if observed != expected:
                raise ValueError("structured SB3 member observation spaces disagree")

    @staticmethod
    def _reference_names(value: object, *, field: str) -> tuple[str, ...]:
        if not isinstance(value, list) or any(
            not isinstance(item, str) for item in value
        ):
            raise ValueError(f"dataset reference {field} must be a list of strings")
        return tuple(value)

    def _validate_dataset(self, dataset: MarketDataset) -> None:
        expected_symbols = self._reference_names(
            self.dataset_reference.get("symbols"), field="symbols"
        )
        expected_features = self._reference_names(
            self.dataset_reference.get("feature_names"), field="feature_names"
        )
        expected_globals = self._reference_names(
            self.dataset_reference.get("global_feature_names"),
            field="global_feature_names",
        )
        expected_bar_hours = self.dataset_reference.get("bar_hours")
        expected_feature_config_digest = self.dataset_reference.get(
            "feature_config_digest"
        )
        if (
            not isinstance(expected_feature_config_digest, str)
            or len(expected_feature_config_digest) != 64
            or any(
                character not in "0123456789abcdef"
                for character in expected_feature_config_digest
            )
        ):
            raise ValueError(
                "dataset reference feature_config_digest is missing or invalid"
            )
        if dataset.symbols != expected_symbols:
            raise ValueError("serving rolling dataset symbols do not match training")
        if dataset.feature_names != expected_features:
            raise ValueError(
                "serving rolling dataset feature order does not match training"
            )
        if dataset.global_feature_names != expected_globals:
            raise ValueError(
                "serving rolling dataset global feature order does not match training"
            )
        if dataset.feature_config_digest != expected_feature_config_digest:
            raise ValueError(
                "serving rolling dataset feature recipe does not match training"
            )
        if (
            isinstance(expected_bar_hours, bool)
            or not isinstance(expected_bar_hours, (int, float))
            or not math.isclose(
                dataset.bar_hours,
                float(expected_bar_hours),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise ValueError(
                "serving rolling dataset bar cadence does not match training"
            )
        if (
            self.builder.layout_digest(dataset)
            != self.sequence_normalizer.sequence_schema_digest
        ):
            raise ValueError(
                "serving rolling dataset sequence layout does not match training"
            )

    def _validate_observation(
        self, observation: Mapping[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        expected_keys = set(self.observation_space.spaces)
        if set(observation) != expected_keys:
            raise ValueError("structured observation keys do not match policy contract")
        result: dict[str, np.ndarray] = {}
        for key, space in self.observation_space.spaces.items():
            array = np.asarray(observation[key], dtype=space.dtype)
            if array.shape != space.shape or not np.isfinite(array).all():
                raise ValueError(f"structured observation component {key} is invalid")
            result[key] = array
        return result

    def smoke_observation(self) -> dict[str, np.ndarray]:
        result: dict[str, np.ndarray] = {}
        for key, space in self.observation_space.spaces.items():
            if space.shape is None or space.dtype is None:
                raise ValueError(
                    "structured observation space must declare shape and dtype"
                )
            result[key] = np.zeros(space.shape, dtype=space.dtype)
        return result

    def predict(self, observation: Mapping[str, np.ndarray]) -> np.ndarray:
        structured = self._validate_observation(observation)
        actions: list[np.ndarray] = []
        for member_index, model in enumerate(self.models):
            try:
                raw, _ = model.predict(structured, deterministic=True)
            except Exception as error:
                raise ValueError(
                    f"SB3 ensemble member {member_index} prediction failed"
                ) from error
            actions.append(
                _validated_action(
                    raw, action_size=self.action_size, member_index=member_index
                )
            )
        return _mean_action(actions)

    def predict_from_dataset(
        self,
        dataset: MarketDataset,
        *,
        index: int,
        current_flat: np.ndarray,
    ) -> np.ndarray:
        self._validate_dataset(dataset)
        layout = ObservationBuilder(
            action_size=self.action_size,
            n_factors=self.n_factors,
            finite_horizon=self.finite_horizon,
        ).layout(dataset)
        raw = np.asarray(current_flat, dtype=np.float32).reshape(-1)
        if raw.shape != (layout.size,) or not np.isfinite(raw).all():
            raise ValueError("current flat observation does not match serving layout")
        current = self.normalizer.transform(raw)
        sequence = self.builder.build(dataset, index=index)
        structured = build_structured_policy_observation(
            sequence=sequence,
            current_flat=current,
            layout=layout,
            n_features=dataset.n_features,
            sequence_normalizer=self.sequence_normalizer,
        )
        structured["decision_index"] = np.asarray([index], dtype=np.int64)
        return self.predict(structured)


class StableBaselines3PolicyLoader:
    """Load every declared SB3 member and fail closed on partial ensembles."""

    def __init__(
        self, *, model_loader: Callable[[str, Path], Any] | None = None
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

    def load(self, bundle: ServingBundle) -> Any:
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
        schema = payload.get("schema_version")
        if schema not in {
            SB3_POLICY_LOADER_SCHEMA,
            SB3_STRUCTURED_POLICY_LOADER_SCHEMA,
        }:
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
        if schema == SB3_POLICY_LOADER_SCHEMA:
            return _SB3EnsemblePolicy(
                tuple(models),
                observation_size=bundle.manifest.observation_size,
                action_size=bundle.manifest.action_size,
            )
        if payload.get("observation_mode") != "structured_sequence":
            raise ValueError("SB3 v2 loader requires structured_sequence mode")
        required_paths = {
            "dataset_reference": "dataset-reference.json",
            "environment": "environment.json",
            "normalizer": "normalizer.json",
            "sequence_normalizer": SEQUENCE_NORMALIZER_ARTIFACT_NAME,
        }
        resolved_paths: dict[str, str] = {}
        for field, expected in required_paths.items():
            relative = _safe_relative_path(payload.get(field), field=field)
            if relative != expected or relative not in declared_files:
                raise ValueError(f"structured SB3 loader {field} is not declared")
            resolved_paths[field] = relative
        if not isinstance(bundle.normalizer, ObservationNormalizer):
            raise ValueError(
                "structured SB3 loader requires the flat normalizer sidecar"
            )
        dataset_reference = _mapping(
            json.loads(
                (bundle.root / resolved_paths["dataset_reference"]).read_text(
                    encoding="utf-8"
                )
            ),
            field="dataset reference",
        )
        environment = _mapping(
            json.loads(
                (bundle.root / resolved_paths["environment"]).read_text(
                    encoding="utf-8"
                )
            ),
            field="training environment",
        )
        action = _mapping(environment.get("action"), field="environment.action")
        env_config = _mapping(
            environment.get("environment"), field="environment.environment"
        )
        if env_config.get("structured_sequence_observation") is not True:
            raise ValueError("structured loader environment does not enable sequences")
        raw_windows = env_config.get("sequence_windows")
        if not isinstance(raw_windows, list) or not raw_windows:
            raise ValueError("structured loader sequence windows are missing")
        windows: list[SequenceWindowSpec] = []
        for index, raw in enumerate(raw_windows):
            if not isinstance(raw, list) or len(raw) != 2:
                raise ValueError(f"sequence_windows[{index}] is invalid")
            windows.append(SequenceWindowSpec(str(raw[0]), int(raw[1])))
        n_factors = action.get("n_factors", 0)
        if (
            isinstance(n_factors, bool)
            or not isinstance(n_factors, int)
            or n_factors < 0
        ):
            raise ValueError("structured loader factor count is invalid")
        finite_horizon = env_config.get("finite_horizon_observation", False)
        if not isinstance(finite_horizon, bool):
            raise ValueError("structured loader finite horizon flag is invalid")
        sequence_normalizer = load_sequence_feature_normalizer(bundle.root)
        return _SB3StructuredSequenceEnsemblePolicy(
            tuple(models),
            action_size=bundle.manifest.action_size,
            dataset_reference=dataset_reference,
            builder=SequenceObservationBuilder(windows=tuple(windows)),
            normalizer=bundle.normalizer,
            sequence_normalizer=sequence_normalizer,
            n_factors=n_factors,
            finite_horizon=finite_horizon,
        )


__all__ = [
    "SB3_POLICY_LOADER_NAME",
    "SB3_POLICY_LOADER_SCHEMA",
    "SB3_STRUCTURED_POLICY_LOADER_SCHEMA",
    "StableBaselines3PolicyLoader",
]
