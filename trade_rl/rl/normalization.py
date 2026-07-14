"""Leakage-safe, content-addressed observation normalization."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.common import require_sha256

NORMALIZER_SCHEMA = "observation_normalizer_v2"


def _readonly_vector(value: np.ndarray, *, field_name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64).reshape(-1).copy()
    if vector.size == 0 or not np.isfinite(vector).all():
        raise ValueError(f"{field_name} must be a non-empty finite vector")
    vector.setflags(write=False)
    return vector


def _validate_digest(value: str | None, *, field: str) -> None:
    if value is not None:
        require_sha256(value, field=field)


@dataclass(frozen=True, slots=True)
class ObservationNormalizer:
    """Statistics fitted on one explicit training range and frozen thereafter."""

    mean: np.ndarray
    scale: np.ndarray
    train_start: int
    train_end: int
    clip: float = 10.0
    epsilon: float = 1e-8
    passthrough_indices: tuple[int, ...] = ()
    # ``dataset_id`` is the exact dataset/view used to fit the observation matrix.
    dataset_id: str | None = None
    # ``source_dataset_id`` is the full immutable source dataset from which a
    # fold view was derived. This lets evaluation environments on the full
    # dataset verify the normalizer without accepting unrelated statistics.
    source_dataset_id: str | None = None
    source_dataset_artifact_digest: str | None = None
    absolute_train_start: int | None = None
    absolute_train_end: int | None = None
    observation_schema: str = "baseline_residual_observation_v3"
    observation_schema_digest: str | None = None
    action_spec_digest: str | None = None
    alpha_artifact_digest: str | None = None
    factor_artifact_digest: str | None = None
    candidate_config_digest: str | None = None
    schema_version: str = NORMALIZER_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        mean = _readonly_vector(self.mean, field_name="mean")
        scale = _readonly_vector(self.scale, field_name="scale")
        if mean.shape != scale.shape:
            raise ValueError("normalizer mean and scale must have identical shapes")
        if np.any(scale <= 0.0):
            raise ValueError("normalizer scale must be strictly positive")
        if (
            isinstance(self.train_start, bool)
            or isinstance(self.train_end, bool)
            or not isinstance(self.train_start, int)
            or not isinstance(self.train_end, int)
            or self.train_start < 0
            or self.train_end <= self.train_start
        ):
            raise ValueError(
                "normalizer training range must be a non-empty index range"
            )
        if not math.isfinite(self.clip) or self.clip <= 0.0:
            raise ValueError("normalizer clip must be finite and positive")
        if not math.isfinite(self.epsilon) or self.epsilon <= 0.0:
            raise ValueError("normalizer epsilon must be finite and positive")
        passthrough = tuple(self.passthrough_indices)
        if any(
            isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index < mean.size
            for index in passthrough
        ):
            raise ValueError("passthrough_indices are outside the observation vector")
        if len(set(passthrough)) != len(passthrough):
            raise ValueError("passthrough_indices must be unique")
        passthrough = tuple(sorted(passthrough))
        for field_name, value in (
            ("normalizer.dataset_id", self.dataset_id),
            ("normalizer.source_dataset_id", self.source_dataset_id),
            (
                "normalizer.source_dataset_artifact_digest",
                self.source_dataset_artifact_digest,
            ),
            ("normalizer.observation_schema_digest", self.observation_schema_digest),
            ("normalizer.action_spec_digest", self.action_spec_digest),
            ("normalizer.alpha_artifact_digest", self.alpha_artifact_digest),
            ("normalizer.factor_artifact_digest", self.factor_artifact_digest),
            ("normalizer.candidate_config_digest", self.candidate_config_digest),
        ):
            _validate_digest(value, field=field_name)
        absolute_start = self.absolute_train_start
        absolute_end = self.absolute_train_end
        if (absolute_start is None) != (absolute_end is None):
            raise ValueError("absolute normalizer range must be specified as a pair")
        if absolute_start is not None and absolute_end is not None:
            if (
                isinstance(absolute_start, bool)
                or isinstance(absolute_end, bool)
                or not isinstance(absolute_start, int)
                or not isinstance(absolute_end, int)
                or absolute_start < 0
                or absolute_end <= absolute_start
            ):
                raise ValueError("absolute normalizer training range is invalid")
        if not self.observation_schema:
            raise ValueError("observation_schema must be non-empty")
        if self.schema_version not in {NORMALIZER_SCHEMA, "observation_normalizer_v1"}:
            raise ValueError("unsupported normalizer schema")
        resolved_schema = NORMALIZER_SCHEMA
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "passthrough_indices", passthrough)
        object.__setattr__(self, "schema_version", resolved_schema)
        expected = content_digest(self.digest_payload())
        if self.digest and self.digest != expected:
            raise ValueError("normalizer digest does not match its content")
        object.__setattr__(self, "digest", expected)

    @property
    def size(self) -> int:
        return int(self.mean.size)

    def digest_payload(self) -> dict[str, object]:
        return {
            "absolute_train_end": self.absolute_train_end,
            "absolute_train_start": self.absolute_train_start,
            "action_spec_digest": self.action_spec_digest,
            "alpha_artifact_digest": self.alpha_artifact_digest,
            "candidate_config_digest": self.candidate_config_digest,
            "clip": self.clip,
            "dataset_id": self.dataset_id,
            "epsilon": self.epsilon,
            "factor_artifact_digest": self.factor_artifact_digest,
            "mean": tuple(float(value) for value in self.mean),
            "observation_schema": self.observation_schema,
            "observation_schema_digest": self.observation_schema_digest,
            "passthrough_indices": self.passthrough_indices,
            "scale": tuple(float(value) for value in self.scale),
            "schema_version": self.schema_version,
            "source_dataset_artifact_digest": self.source_dataset_artifact_digest,
            "source_dataset_id": self.source_dataset_id,
            "train_end": self.train_end,
            "train_start": self.train_start,
        }

    @classmethod
    def fit(
        cls,
        observations: np.ndarray,
        *,
        train_start: int,
        train_end: int,
        clip: float = 10.0,
        epsilon: float = 1e-8,
        passthrough_indices: tuple[int, ...] = (),
        dataset_id: str | None = None,
        source_dataset_id: str | None = None,
        source_dataset_artifact_digest: str | None = None,
        absolute_train_start: int | None = None,
        absolute_train_end: int | None = None,
        observation_schema: str = "baseline_residual_observation_v3",
        observation_schema_digest: str | None = None,
        action_spec_digest: str | None = None,
        alpha_artifact_digest: str | None = None,
        factor_artifact_digest: str | None = None,
        candidate_config_digest: str | None = None,
    ) -> ObservationNormalizer:
        matrix = np.asarray(observations, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
            raise ValueError("observations must be a non-empty two-dimensional matrix")
        if not np.isfinite(matrix).all():
            raise ValueError("observations must contain only finite values")
        if (
            isinstance(train_start, bool)
            or isinstance(train_end, bool)
            or not isinstance(train_start, int)
            or not isinstance(train_end, int)
            or not 0 <= train_start < train_end <= matrix.shape[0]
        ):
            raise ValueError("training range is outside the observation matrix")
        fitted = matrix[train_start:train_end]
        mean = fitted.mean(axis=0)
        std = fitted.std(axis=0, ddof=0)
        scale = np.where(std > epsilon, std, 1.0)
        passthrough = tuple(passthrough_indices)
        if any(
            isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index < matrix.shape[1]
            for index in passthrough
        ):
            raise ValueError("passthrough_indices are outside the observation matrix")
        if passthrough:
            selected = np.asarray(passthrough, dtype=np.int64)
            mean[selected] = 0.0
            scale[selected] = 1.0
        return cls(
            mean=mean,
            scale=scale,
            train_start=train_start,
            train_end=train_end,
            clip=clip,
            epsilon=epsilon,
            passthrough_indices=passthrough,
            dataset_id=dataset_id,
            source_dataset_id=source_dataset_id,
            source_dataset_artifact_digest=source_dataset_artifact_digest,
            absolute_train_start=absolute_train_start,
            absolute_train_end=absolute_train_end,
            observation_schema=observation_schema,
            observation_schema_digest=observation_schema_digest,
            action_spec_digest=action_spec_digest,
            alpha_artifact_digest=alpha_artifact_digest,
            factor_artifact_digest=factor_artifact_digest,
            candidate_config_digest=candidate_config_digest,
        )

    def transform(self, observation: np.ndarray) -> np.ndarray:
        vector = np.asarray(observation, dtype=np.float64).reshape(-1)
        if vector.shape != self.mean.shape or not np.isfinite(vector).all():
            raise ValueError("observation does not match the fitted normalizer")
        normalized = np.clip((vector - self.mean) / self.scale, -self.clip, self.clip)
        if self.passthrough_indices:
            selected = np.asarray(self.passthrough_indices, dtype=np.int64)
            normalized[selected] = vector[selected]
        return normalized.astype(np.float32)

    def transform_batch(self, observations: np.ndarray) -> np.ndarray:
        matrix = np.asarray(observations, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1:] != self.mean.shape:
            raise ValueError("observation batch does not match the fitted normalizer")
        if not np.isfinite(matrix).all():
            raise ValueError("observation batch must contain only finite values")
        normalized = np.clip(
            (matrix - self.mean[None, :]) / self.scale[None, :],
            -self.clip,
            self.clip,
        )
        if self.passthrough_indices:
            selected = np.asarray(self.passthrough_indices, dtype=np.int64)
            normalized[:, selected] = matrix[:, selected]
        return normalized.astype(np.float32)
