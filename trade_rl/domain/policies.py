"""Residual policy ensemble identity records."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime

from trade_rl.domain.common import (
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
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        require_sha256(self.checkpoint_digest, field="checkpoint_digest")


@dataclass(frozen=True, slots=True)
class PolicyEnsembleManifest:
    """Manifest for a complete baseline-residual PPO ensemble."""

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
    schema_version: str = "policy_ensemble_v3"

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
        seeds = tuple(member.seed for member in self.members)
        if len(set(seeds)) != len(seeds):
            raise ValueError("policy ensemble seeds must be unique")
        digests = tuple(member.checkpoint_digest for member in self.members)
        if len(set(digests)) != len(digests):
            raise ValueError("policy ensemble checkpoint digests must be unique")
