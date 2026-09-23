"""Shared deterministic three-action adapter for maintained RL strategies."""

from __future__ import annotations

from numbers import Integral
from typing import Protocol

import numpy as np

from trade_rl.strategies.interface import StrategyObservation
from trade_rl.strategies.position_intent import PositionIntent
from trade_rl.strategies.rl.ppo_normalization import PPOFeatureNormalizer

PPO_OBSERVATION_SCHEMA = "ppo_observation_v2"
PPO_GLOBAL_FEATURE_NAMES: tuple[str, ...] = ()


def ppo_observation_contract_payload() -> dict[str, object]:
    """Return the frozen shared RL observation contract used by PPO and A2C."""

    return {
        "schema_version": PPO_OBSERVATION_SCHEMA,
        "global_feature_names": list(PPO_GLOBAL_FEATURE_NAMES),
        "includes_local_feature_staleness": True,
        "layout": [
            "local_values",
            "local_available",
            "local_staleness",
            "current_intent",
            "current_weight",
        ],
    }


class _PredictPolicy(Protocol):
    def predict(
        self,
        observation: np.ndarray,
        *,
        deterministic: bool = True,
    ) -> tuple[object, object]: ...


def _validated_indices(feature_indices: tuple[int, ...]) -> tuple[int, ...]:
    indices = tuple(feature_indices)
    if not indices:
        raise ValueError("feature_indices must be non-empty and unique")
    if any(
        isinstance(index, bool) or not isinstance(index, int) or index < 0
        for index in indices
    ):
        raise ValueError("feature_indices must contain non-negative integers")
    if len(set(indices)) != len(indices):
        raise ValueError("feature_indices must be non-empty and unique")
    return indices


def _encode_observation(
    observation: StrategyObservation,
    feature_indices: tuple[int, ...],
    feature_normalizer: PPOFeatureNormalizer | None = None,
) -> np.ndarray:
    indices = _validated_indices(feature_indices)
    if max(indices) >= observation.features.size:
        raise ValueError("feature index is outside observation features")

    selected = np.asarray(observation.features[list(indices)], dtype=np.float64)
    available = np.asarray(
        observation.feature_available[list(indices)],
        dtype=np.bool_,
    )
    if observation.feature_staleness is None:
        raise ValueError("PPO Observation v2 requires feature staleness")
    staleness = np.asarray(
        observation.feature_staleness[list(indices)],
        dtype=np.float64,
    )
    return _encode_observation_fields(
        selected,
        available,
        staleness,
        observation.current_intent,
        observation.current_weight,
        feature_normalizer,
    )


def _encode_observation_fields(
    selected: np.ndarray,
    available: np.ndarray,
    staleness: np.ndarray,
    current_intent: PositionIntent,
    current_weight: float,
    feature_normalizer: PPOFeatureNormalizer | None = None,
) -> np.ndarray:
    """Encode validated local fields without constructing a public observation."""

    selected = np.asarray(selected, dtype=np.float64)
    available = np.asarray(available, dtype=np.bool_)
    staleness = np.asarray(staleness, dtype=np.float64)
    finite = np.isfinite(selected)
    usable = available & finite
    values = np.where(usable, selected, 0.0)
    if feature_normalizer is not None:
        values = feature_normalizer.transform(selected, usable)

    state = np.asarray([float(current_intent), current_weight], dtype=np.float64)
    encoded = np.concatenate(
        (
            values,
            usable.astype(np.float64),
            staleness,
            state,
        ),
    ).astype(np.float32)
    encoded.setflags(write=False)
    return encoded


def _intent_from_action(action: object, *, family: str = "PPO") -> PositionIntent:
    values = np.asarray(action).reshape(-1)
    if values.size != 1:
        raise ValueError(f"{family} action must contain exactly one value")
    raw = values[0]
    if isinstance(raw, (bool, np.bool_)) or not isinstance(raw, Integral):
        raise ValueError(f"{family} action must be an integer in {{0, 1, 2}}")
    value = int(raw)
    if value not in {0, 1, 2}:
        raise ValueError(f"{family} action must be an integer in {{0, 1, 2}}")
    return PositionIntent(value - 1)


class _ThreeActionIntentStrategy:
    """Map a deterministic discrete 0/1/2 policy into target intent."""

    _family_name = "policy"

    def __init__(
        self,
        policy: _PredictPolicy,
        *,
        feature_indices: tuple[int, ...],
        feature_names: tuple[str, ...] | None = None,
        feature_normalizer: PPOFeatureNormalizer | None = None,
    ) -> None:
        self.policy = policy
        indices = _validated_indices(feature_indices)
        if feature_names is None:
            selected_names = (
                None
                if feature_normalizer is None
                else tuple(feature_normalizer.feature_names)
            )
        else:
            selected_names = tuple(feature_names)
            if (
                len(selected_names) != len(indices)
                or any(not isinstance(name, str) or not name for name in selected_names)
                or len(set(selected_names)) != len(selected_names)
            ):
                raise ValueError(
                    "feature_names must match feature_indices with unique non-empty strings"
                )
        if feature_normalizer is not None:
            feature_normalizer.validate_features(indices)
            if selected_names != feature_normalizer.feature_names:
                raise ValueError(
                    "strategy feature names differ from fitted normalization"
                )
        self._feature_indices = indices
        self._feature_names = selected_names
        self._feature_normalizer = feature_normalizer

    @property
    def feature_indices(self) -> tuple[int, ...]:
        return self._feature_indices

    @property
    def feature_names(self) -> tuple[str, ...] | None:
        return self._feature_names

    @property
    def feature_normalizer(self) -> PPOFeatureNormalizer | None:
        return self._feature_normalizer

    def decide(self, observation: StrategyObservation) -> PositionIntent:
        encoded = _encode_observation(
            observation,
            self.feature_indices,
            self.feature_normalizer,
        )
        action, _ = self.policy.predict(encoded, deterministic=True)
        return _intent_from_action(action, family=self._family_name)


__all__ = [
    "PPO_GLOBAL_FEATURE_NAMES",
    "PPO_OBSERVATION_SCHEMA",
    "_ThreeActionIntentStrategy",
    "ppo_observation_contract_payload",
]
