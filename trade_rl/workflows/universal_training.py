"""Train-only assembly primitives for universal single-instrument research."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.common import require_sha256
from trade_rl.learning.episode_oracle_teacher import EpisodeOracleBatch
from trade_rl.learning.episode_teacher_artifact import EpisodeSupervisedPolicyDataset
from trade_rl.rl.normalization import ObservationNormalizer
from trade_rl.rl.observations import OBSERVATION_SCHEMA, ObservationBuilder, observation_layout
from trade_rl.rl.sequence_normalization import SequenceFeatureNormalizer
from trade_rl.rl.sequence_observations import (
    DEFAULT_SEQUENCE_WINDOWS,
    SequenceObservationBuilder,
    SequenceWindowSpec,
)
from trade_rl.rl.universal_normalization import SymbolBalancedStandardNormalizer

_MAINTAINED_TIMEFRAMES = ("15m", "1h", "4h", "1d")


def _readonly_bool_vector(value: object, *, field_name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.bool_).reshape(-1).copy(order="C")
    if result.size == 0:
        raise ValueError(f"{field_name} must be non-empty")
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class UniversalBoundObservationNormalizer(ObservationNormalizer):
    """Bind shared market statistics to one dataset-specific flat observation."""

    market_feature_count: int = 0
    constant_mask: np.ndarray = field(
        default_factory=lambda: np.asarray([], dtype=np.bool_)
    )
    statistics_digest: str = ""

    def __post_init__(self) -> None:
        if (
            isinstance(self.market_feature_count, bool)
            or not isinstance(self.market_feature_count, int)
            or self.market_feature_count <= 0
        ):
            raise ValueError("Universal market_feature_count must be positive")
        constant_mask = _readonly_bool_vector(
            self.constant_mask,
            field_name="Universal flat constant_mask",
        )
        if constant_mask.shape != (self.market_feature_count,):
            raise ValueError("Universal flat constant_mask does not match market features")
        statistics_digest = require_sha256(
            self.statistics_digest,
            field="Universal flat statistics_digest",
        )
        object.__setattr__(self, "constant_mask", constant_mask)
        object.__setattr__(self, "statistics_digest", statistics_digest)
        ObservationNormalizer.__post_init__(self)
        if self.market_feature_count > self.size:
            raise ValueError("Universal market features exceed observation size")

    def digest_payload(self) -> dict[str, object]:
        return {
            **ObservationNormalizer.digest_payload(self),
            "constant_mask": tuple(bool(value) for value in self.constant_mask),
            "market_feature_count": self.market_feature_count,
            "statistics_digest": self.statistics_digest,
            "universal_binding_schema": "universal_flat_normalizer_binding_v1",
        }

    def transform(self, observation: np.ndarray) -> np.ndarray:
        vector = np.asarray(observation, dtype=np.float64).reshape(-1)
        if vector.shape != self.mean.shape or not np.isfinite(vector).all():
            raise ValueError("observation does not match the Universal flat normalizer")
        n_features = self.market_feature_count
        if self.size < 2 * n_features:
            raise ValueError("Universal flat observation is missing availability channels")
        available = vector[n_features : 2 * n_features] > 0.5
        market = np.clip(
            (vector[:n_features] - self.mean[:n_features]) / self.scale[:n_features],
            -self.clip,
            self.clip,
        )
        market = np.where(self.constant_mask | ~available, 0.0, market)
        result = vector.copy(order="C")
        result[:n_features] = market
        return result.astype(np.float32)

    def transform_batch(self, observations: np.ndarray) -> np.ndarray:
        matrix = np.asarray(observations, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1:] != self.mean.shape:
            raise ValueError("observation batch does not match Universal flat normalizer")
        if not np.isfinite(matrix).all():
            raise ValueError("Universal flat observation batch must be finite")
        n_features = self.market_feature_count
        available = matrix[:, n_features : 2 * n_features] > 0.5
        market = np.clip(
            (matrix[:, :n_features] - self.mean[None, :n_features])
            / self.scale[None, :n_features],
            -self.clip,
            self.clip,
        )
        market = np.where(self.constant_mask[None, :] | ~available, 0.0, market)
        result = matrix.copy(order="C")
        result[:, :n_features] = market
        return result.astype(np.float32)


@dataclass(frozen=True, slots=True)
class UniversalBoundSequenceNormalizer(SequenceFeatureNormalizer):
    """Bind the same shared market statistics to one dataset sequence layout."""

    constant_mask_by_timeframe: Mapping[str, np.ndarray] = field(default_factory=dict)
    statistics_digest: str = ""

    def __post_init__(self) -> None:
        if tuple(self.constant_mask_by_timeframe) != _MAINTAINED_TIMEFRAMES:
            raise ValueError("Universal sequence constant masks must match timeframes")
        resolved: dict[str, np.ndarray] = {}
        for timeframe in _MAINTAINED_TIMEFRAMES:
            mask = _readonly_bool_vector(
                self.constant_mask_by_timeframe[timeframe],
                field_name=f"{timeframe}.constant_mask",
            )
            if mask.shape != (len(self.feature_names[timeframe]),):
                raise ValueError("Universal sequence constant mask width mismatch")
            resolved[timeframe] = mask
        statistics_digest = require_sha256(
            self.statistics_digest,
            field="Universal sequence statistics_digest",
        )
        object.__setattr__(
            self,
            "constant_mask_by_timeframe",
            MappingProxyType(resolved),
        )
        object.__setattr__(self, "statistics_digest", statistics_digest)
        SequenceFeatureNormalizer.__post_init__(self)

    def digest_payload(self) -> dict[str, object]:
        return {
            **SequenceFeatureNormalizer.digest_payload(self),
            "constant_mask": {
                key: tuple(bool(value) for value in self.constant_mask_by_timeframe[key])
                for key in _MAINTAINED_TIMEFRAMES
            },
            "statistics_digest": self.statistics_digest,
            "universal_binding_schema": "universal_sequence_normalizer_binding_v1",
        }

    def transform(
        self,
        timeframe: str,
        values: np.ndarray,
        available: np.ndarray,
        *,
        feature_names: tuple[str, ...],
    ) -> np.ndarray:
        if timeframe not in self.feature_names:
            raise ValueError("Universal sequence timeframe is unknown")
        if tuple(feature_names) != self.feature_names[timeframe]:
            raise ValueError("Universal sequence feature order does not match normalizer")
        array = np.asarray(values, dtype=np.float64)
        mask = np.asarray(available, dtype=np.bool_)
        if array.shape != mask.shape or array.shape[-1] != len(feature_names):
            raise ValueError("Universal sequence values and availability shapes differ")
        if not np.isfinite(array).all():
            raise ValueError("Universal sequence values must be finite")
        result = np.clip(
            (array - self.center[timeframe]) / self.scale[timeframe],
            -self.clip,
            self.clip,
        )
        constants = self.constant_mask_by_timeframe[timeframe]
        result = np.where(mask & ~constants, result, 0.0)
        return np.asarray(result, dtype=np.float32)


def _feature_groups(
    feature_names: Sequence[str],
) -> tuple[
    dict[str, tuple[str, ...]],
    dict[str, np.ndarray],
]:
    names = tuple(str(name) for name in feature_names)
    grouped_names: dict[str, tuple[str, ...]] = {}
    grouped_indices: dict[str, np.ndarray] = {}
    observed: list[int] = []
    for timeframe in _MAINTAINED_TIMEFRAMES:
        prefix = f"{timeframe}__"
        indices = tuple(index for index, name in enumerate(names) if name.startswith(prefix))
        if not indices:
            raise ValueError(f"Universal feature contract is missing {timeframe} channels")
        grouped_names[timeframe] = tuple(names[index] for index in indices)
        grouped_indices[timeframe] = np.asarray(indices, dtype=np.int64)
        observed.extend(indices)
    if tuple(sorted(observed)) != tuple(range(len(names))):
        raise ValueError("Universal feature names must belong to exactly one maintained timeframe")
    return grouped_names, grouped_indices


def bind_universal_normalizers(
    dataset: Any,
    *,
    shared: SymbolBalancedStandardNormalizer,
    action_spec_digest: str,
    action_size: int = 1,
    n_factors: int = 0,
    finite_horizon: bool = True,
    sequence_windows: tuple[SequenceWindowSpec, ...] = DEFAULT_SEQUENCE_WINDOWS,
    candidate_config_digest: str | None = None,
) -> tuple[UniversalBoundObservationNormalizer, UniversalBoundSequenceNormalizer]:
    """Bind one train-only shared normalizer to a concrete single-symbol dataset."""

    require_sha256(action_spec_digest, field="Universal action_spec_digest")
    raw_symbols = tuple(getattr(dataset, "symbols", ()))
    if len(raw_symbols) != 1:
        raise ValueError("Universal normalizer binding requires one concrete symbol")
    feature_names = tuple(getattr(dataset, "feature_names", ()))
    if len(feature_names) != len(shared.mean):
        raise ValueError("Universal shared statistics width does not match dataset features")
    grouped_names, grouped_indices = _feature_groups(feature_names)

    layout = observation_layout(
        dataset,
        action_size=action_size,
        n_factors=n_factors,
        finite_horizon=finite_horizon,
    )
    mean = np.zeros(layout.size, dtype=np.float64)
    scale = np.ones(layout.size, dtype=np.float64)
    mean[: len(feature_names)] = shared.mean
    scale[: len(feature_names)] = shared.std
    flat_builder = ObservationBuilder(
        action_size=action_size,
        n_factors=n_factors,
        finite_horizon=finite_horizon,
    )
    train_start, train_end = shared.fold_train_range
    flat = UniversalBoundObservationNormalizer(
        mean=mean,
        scale=scale,
        train_start=train_start,
        train_end=train_end,
        clip=shared.clip,
        passthrough_indices=tuple(range(len(feature_names), layout.size)),
        dataset_id=str(dataset.dataset_id),
        source_dataset_id=str(dataset.dataset_id),
        absolute_train_start=train_start,
        absolute_train_end=train_end,
        observation_schema=OBSERVATION_SCHEMA,
        observation_schema_digest=flat_builder.schema_digest(dataset),
        action_spec_digest=action_spec_digest,
        candidate_config_digest=candidate_config_digest,
        market_feature_count=len(feature_names),
        constant_mask=np.asarray(shared.constant_mask, dtype=np.bool_),
        statistics_digest=shared.statistics_digest,
    )

    centers: dict[str, np.ndarray] = {}
    scales: dict[str, np.ndarray] = {}
    counts: dict[str, np.ndarray] = {}
    constants: dict[str, np.ndarray] = {}
    sample_counts = np.asarray(shared.sample_count_per_feature, dtype=np.int64)
    shared_constant = np.asarray(shared.constant_mask, dtype=np.bool_)
    for timeframe in _MAINTAINED_TIMEFRAMES:
        indices = grouped_indices[timeframe]
        centers[timeframe] = np.asarray(shared.mean[indices], dtype=np.float64)
        scales[timeframe] = np.asarray(shared.std[indices], dtype=np.float64)
        counts[timeframe] = np.asarray(sample_counts[indices], dtype=np.int64)
        constants[timeframe] = np.asarray(shared_constant[indices], dtype=np.bool_)
    sequence_builder = SequenceObservationBuilder(windows=tuple(sequence_windows))
    sequence = UniversalBoundSequenceNormalizer(
        feature_names=grouped_names,
        center=centers,
        scale=scales,
        train_start=train_start,
        train_end=train_end,
        dataset_id=str(dataset.dataset_id),
        source_dataset_id=str(dataset.dataset_id),
        sequence_schema_digest=sequence_builder.layout_digest(dataset),
        sample_count=counts,
        minimum_samples_per_channel=1,
        clip=shared.clip,
        constant_mask_by_timeframe=constants,
        statistics_digest=shared.statistics_digest,
    )
    return flat, sequence


def validate_universal_dataset_scope(
    datasets: Mapping[str, Any],
    *,
    train_symbols: Sequence[str],
) -> dict[str, Any]:
    """Reject any dataset that is outside the immutable train-symbol partition."""

    symbols = tuple(train_symbols)
    if not symbols or len(set(symbols)) != len(symbols) or any(not value for value in symbols):
        raise ValueError("Universal train_symbols must be non-empty and unique")
    if set(datasets) != set(symbols):
        raise ValueError("Universal datasets must exactly match train_symbols")
    return {symbol: datasets[symbol] for symbol in symbols}


def _discounted_return_to_go(rewards: Sequence[float], *, gamma: float) -> np.ndarray:
    if not math.isfinite(gamma) or not 0.0 < gamma <= 1.0:
        raise ValueError("Universal teacher gamma must be in (0, 1]")
    result = np.empty(len(rewards), dtype=np.float64)
    running = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        running = float(rewards[index]) + gamma * running
        result[index] = running
    return result


@dataclass(frozen=True, slots=True)
class UniversalCollectedTeacher:
    dataset: EpisodeSupervisedPolicyDataset
    critic_targets: np.ndarray

    def __post_init__(self) -> None:
        values = np.asarray(self.critic_targets, dtype=np.float32).reshape(-1).copy(order="C")
        if values.shape != (self.dataset.sample_count,) or not np.isfinite(values).all():
            raise ValueError("Universal critic targets must align with teacher samples")
        values.setflags(write=False)
        object.__setattr__(self, "critic_targets", values)


def collect_universal_episode_teacher(
    environment: Any,
    batch: EpisodeOracleBatch,
    *,
    teacher_config_digest: str,
    gamma: float,
) -> UniversalCollectedTeacher:
    """Collect full multi-dataset-safe Dict observations and finite-horizon targets."""

    require_sha256(teacher_config_digest, field="Universal teacher_config_digest")
    expected_keys: tuple[str, ...] | None = None
    expected_shapes: dict[str, tuple[int, ...]] = {}
    observations: dict[str, list[np.ndarray]] = {}
    actions: list[np.ndarray] = []
    decision_indices: list[int] = []
    episode_ids: list[int] = []
    return_targets: list[np.ndarray] = []

    for contract, raw_targets in zip(batch.contracts, batch.targets, strict=True):
        targets = np.asarray(raw_targets, dtype=np.float32)
        expected_steps = contract.stop - contract.start - 1
        if targets.ndim != 2 or len(targets) != expected_steps:
            raise ValueError("Universal Oracle targets do not match episode contract")
        observation, info = environment.reset(
            options={
                "start_idx": contract.start,
                "episode_bars": expected_steps,
                "initial_state_mode": contract.initial_state_mode,
            }
        )
        if not isinstance(observation, Mapping):
            raise TypeError("Universal teacher requires full Dict observations")
        if int(info.get("start_index", -1)) != contract.start:
            raise ValueError("Universal teacher reset start mismatch")
        if "current_weights" in observation:
            current = np.asarray(observation["current_weights"], dtype=np.float64)
            if not np.allclose(current, contract.initial_weights, atol=1e-6, rtol=0.0):
                raise ValueError("Universal teacher reset weights mismatch contract")

        rewards: list[float] = []
        for offset, target in enumerate(targets):
            expected_index = contract.start + offset
            current_index = getattr(environment, "current_index", expected_index)
            if int(current_index) != expected_index:
                raise ValueError("Universal teacher environment left its episode contract")
            keys = tuple(sorted(observation))
            if expected_keys is None:
                expected_keys = keys
                observations = {key: [] for key in keys}
                expected_shapes = {
                    key: tuple(np.asarray(observation[key]).shape) for key in keys
                }
            if keys != expected_keys:
                raise ValueError("Universal teacher observation keys changed")
            for key in keys:
                value = np.asarray(observation[key])
                if tuple(value.shape) != expected_shapes[key]:
                    raise ValueError("Universal teacher observation shape changed")
                observations[key].append(value.copy(order="C"))
            actions.append(np.asarray(target, dtype=np.float32).copy(order="C"))
            decision_indices.append(expected_index)
            episode_ids.append(contract.episode_index)
            observation, reward, terminated, truncated, _ = environment.step(target)
            resolved_reward = float(reward)
            if not math.isfinite(resolved_reward):
                raise ValueError("Universal teacher reward must be finite")
            rewards.append(resolved_reward)
            final_step = offset == expected_steps - 1
            if bool(terminated or truncated) != final_step:
                raise ValueError("Universal teacher environment ended outside contract")
        return_targets.append(_discounted_return_to_go(rewards, gamma=gamma))

    if expected_keys is None or not actions:
        raise ValueError("Universal teacher requires at least one episode sample")
    environment_digest = getattr(environment, "environment_digest", None)
    action_spec_digest = getattr(environment, "action_spec_digest", None)
    if not isinstance(environment_digest, str):
        raise ValueError("Universal teacher environment digest is unavailable")
    if not isinstance(action_spec_digest, str):
        raise ValueError("Universal teacher action specification digest is unavailable")
    dataset = EpisodeSupervisedPolicyDataset(
        observations={key: np.stack(values, axis=0) for key, values in observations.items()},
        actions=np.stack(actions, axis=0),
        dataset_id=batch.dataset_id,
        train_start=min(contract.start for contract in batch.contracts),
        train_stop=max(contract.stop for contract in batch.contracts),
        environment_digest=environment_digest,
        action_spec_digest=action_spec_digest,
        teacher_config_digest=teacher_config_digest,
        decision_indices=np.asarray(decision_indices, dtype=np.int64),
        episode_ids=np.asarray(episode_ids, dtype=np.int64),
        solver_provenance=batch.solver_provenance,
    )
    return UniversalCollectedTeacher(
        dataset=dataset,
        critic_targets=np.concatenate(return_targets).astype(np.float32, copy=False),
    )


def universal_training_contract_digest(
    *,
    partition_digest: str,
    feature_schema_digest: str,
    statistics_digest: str,
    instrument_context_schema_digest: str,
    training_config_digest: str,
) -> str:
    """Bind the dataset-independent Universal policy/training surface."""

    for field_name, value in (
        ("partition_digest", partition_digest),
        ("feature_schema_digest", feature_schema_digest),
        ("statistics_digest", statistics_digest),
        ("instrument_context_schema_digest", instrument_context_schema_digest),
        ("training_config_digest", training_config_digest),
    ):
        require_sha256(value, field=field_name)
    return content_digest(
        {
            "feature_schema_digest": feature_schema_digest,
            "instrument_context_schema_digest": instrument_context_schema_digest,
            "partition_digest": partition_digest,
            "schema_version": "universal_training_contract_v1",
            "statistics_digest": statistics_digest,
            "training_config_digest": training_config_digest,
        }
    )


__all__ = [
    "UniversalBoundObservationNormalizer",
    "UniversalBoundSequenceNormalizer",
    "UniversalCollectedTeacher",
    "bind_universal_normalizers",
    "collect_universal_episode_teacher",
    "universal_training_contract_digest",
    "validate_universal_dataset_scope",
]
