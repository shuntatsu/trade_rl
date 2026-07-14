"""Residual policy ensemble identity records."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime

from trade_rl.domain.common import (
    domain_content_digest,
    require_aware_datetime,
    require_non_empty,
    require_sha256,
)


@dataclass(frozen=True, slots=True)
class PolicyMember:
    """One immutable policy checkpoint in an ensemble."""

    seed: int
    checkpoint_digest: str

    def __post_init__(self) -> None:
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
        ):
            raise ValueError("seed must be non-negative")
        require_sha256(self.checkpoint_digest, field="checkpoint_digest")

    def digest_payload(self) -> dict[str, object]:
        return {"checkpoint_digest": self.checkpoint_digest, "seed": self.seed}


@dataclass(frozen=True, slots=True)
class PolicyEnsembleManifest:
    """Manifest for a complete baseline-residual policy ensemble."""

    digest: str
    dataset_id: str
    action_schema: str
    observation_schema: str
    training_config_digest: str
    environment_digest: str
    initial_capital: float
    requested_timesteps: int
    actual_timesteps: int
    resolved_device: str
    expected_members: int
    members: tuple[PolicyMember, ...]
    created_at: datetime
    action_size: int = 2
    action_names: tuple[str, ...] = ()
    action_spec_digest: str | None = None
    observation_size: int | None = None
    alpha_artifact_digest: str | None = None
    factor_artifact_digest: str | None = None
    normalizer_digest: str | None = None
    schema_version: str = "policy_ensemble_v4"

    def __post_init__(self) -> None:
        require_sha256(self.digest, field="digest")
        require_sha256(self.dataset_id, field="dataset_id")
        require_non_empty(self.action_schema, field="action_schema")
        require_non_empty(self.observation_schema, field="observation_schema")
        require_sha256(
            self.training_config_digest,
            field="training_config_digest",
        )
        require_sha256(self.environment_digest, field="environment_digest")
        if not math.isfinite(self.initial_capital) or self.initial_capital <= 0.0:
            raise ValueError("initial_capital must be finite and positive")
        require_non_empty(self.resolved_device, field="resolved_device")
        require_aware_datetime(self.created_at, field="created_at")
        require_non_empty(self.schema_version, field="schema_version")
        if self.requested_timesteps <= 0:
            raise ValueError("requested_timesteps must be positive")
        if self.actual_timesteps < self.requested_timesteps:
            raise ValueError("actual_timesteps cannot be below requested_timesteps")
        if self.expected_members <= 0:
            raise ValueError("expected_members must be positive")
        if len(self.members) != self.expected_members:
            raise ValueError(
                "policy ensemble member count does not match expected_members"
            )
        if (
            isinstance(self.action_size, bool)
            or not isinstance(self.action_size, int)
            or self.action_size <= 0
        ):
            raise ValueError("action_size must be a positive integer")
        if len(self.action_names) != self.action_size:
            raise ValueError("action_names must match action_size")
        if len(set(self.action_names)) != len(self.action_names) or any(
            not name for name in self.action_names
        ):
            raise ValueError("action_names must be unique and non-empty")
        if self.action_spec_digest is None:
            raise ValueError("action_spec_digest is required")
        require_sha256(self.action_spec_digest, field="action_spec_digest")
        if self.observation_size is not None and (
            isinstance(self.observation_size, bool)
            or not isinstance(self.observation_size, int)
            or self.observation_size <= 0
        ):
            raise ValueError("observation_size must be a positive integer")
        for field_name, value in (
            ("alpha_artifact_digest", self.alpha_artifact_digest),
            ("factor_artifact_digest", self.factor_artifact_digest),
            ("normalizer_digest", self.normalizer_digest),
        ):
            if value is not None:
                require_sha256(value, field=field_name)
        seeds = tuple(member.seed for member in self.members)
        if len(set(seeds)) != len(seeds):
            raise ValueError("policy ensemble seeds must be unique")
        digests = tuple(member.checkpoint_digest for member in self.members)
        if len(set(digests)) != len(digests):
            raise ValueError("policy ensemble checkpoint digests must be unique")
        if self.digest != domain_content_digest(self.digest_payload()):
            raise ValueError("policy ensemble digest does not match content")

    def digest_payload(self) -> dict[str, object]:
        return {
            "action_names": self.action_names,
            "action_schema": self.action_schema,
            "action_size": self.action_size,
            "action_spec_digest": self.action_spec_digest,
            "actual_timesteps": self.actual_timesteps,
            "alpha_artifact_digest": self.alpha_artifact_digest,
            "created_at": self.created_at,
            "dataset_id": self.dataset_id,
            "environment_digest": self.environment_digest,
            "factor_artifact_digest": self.factor_artifact_digest,
            "initial_capital": self.initial_capital,
            "members": tuple(member.digest_payload() for member in self.members),
            "normalizer_digest": self.normalizer_digest,
            "observation_schema": self.observation_schema,
            "observation_size": self.observation_size,
            "requested_timesteps": self.requested_timesteps,
            "resolved_device": self.resolved_device,
            "schema_version": self.schema_version,
            "training_config_digest": self.training_config_digest,
        }
