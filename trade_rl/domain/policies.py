"""Residual policy ensemble identity records."""

from __future__ import annotations

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
    expected_members: int
    members: tuple[PolicyMember, ...]
    created_at: datetime
    schema_version: str = "policy_ensemble_v1"

    def __post_init__(self) -> None:
        require_sha256(self.digest, field="digest")
        require_sha256(self.dataset_id, field="dataset_id")
        require_non_empty(self.action_schema, field="action_schema")
        require_aware_datetime(self.created_at, field="created_at")
        require_non_empty(self.schema_version, field="schema_version")
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
