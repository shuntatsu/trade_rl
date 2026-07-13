"""Residual-policy ensemble training orchestration and backend isolation."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol

import gymnasium as gym

from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.common import require_aware_datetime, require_sha256
from trade_rl.domain.datasets import DatasetManifest
from trade_rl.domain.policies import PolicyEnsembleManifest, PolicyMember
from trade_rl.rl.actions import ACTION_SCHEMA

DEFAULT_RESIDUAL_GAMMA = 0.99
MIN_RESIDUAL_GAMMA = 0.95


class PolicyTrainingBackend(Protocol):
    """Backend boundary that writes exactly one policy checkpoint."""

    def train(
        self,
        *,
        seed: int,
        timesteps: int,
        gamma: float,
        output_path: Path,
    ) -> Path: ...


@dataclass(frozen=True, slots=True)
class ResidualTrainingConfig:
    timesteps: int
    seeds: tuple[int, ...]
    gamma: float = DEFAULT_RESIDUAL_GAMMA
    allow_low_gamma: bool = False

    def __post_init__(self) -> None:
        if self.timesteps <= 0:
            raise ValueError("timesteps must be positive")
        if not math.isfinite(self.gamma) or not 0.0 < self.gamma <= 1.0:
            raise ValueError("gamma must be within (0, 1]")
        if not isinstance(self.allow_low_gamma, bool):
            raise ValueError("allow_low_gamma must be a boolean")
        if self.gamma < MIN_RESIDUAL_GAMMA and not self.allow_low_gamma:
            raise ValueError(
                f"residual gamma below {MIN_RESIDUAL_GAMMA} requires "
                "allow_low_gamma=True for an explicit research ablation"
            )
        if not self.seeds:
            raise ValueError("seeds must not be empty")
        if any(seed < 0 for seed in self.seeds):
            raise ValueError("seeds must be non-negative")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be unique")


def _file_digest(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"training backend did not create checkpoint: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def train_residual_ensemble(
    *,
    dataset: DatasetManifest,
    environment_dataset_id: str,
    config: ResidualTrainingConfig,
    backend: PolicyTrainingBackend,
    output_dir: Path,
    created_at: datetime,
) -> PolicyEnsembleManifest:
    """Train one checkpoint per seed and bind all outputs to dataset identity."""

    require_sha256(environment_dataset_id, field="environment_dataset_id")
    require_aware_datetime(created_at, field="created_at")
    if dataset.dataset_id != environment_dataset_id:
        raise ValueError("dataset identity mismatch between manifest and environment")
    output_dir.mkdir(parents=True, exist_ok=True)

    members: list[PolicyMember] = []
    for member_index, seed in enumerate(config.seeds):
        checkpoint = output_dir / f"member-{member_index:03d}" / "policy.zip"
        resolved = backend.train(
            seed=seed,
            timesteps=config.timesteps,
            gamma=config.gamma,
            output_path=checkpoint,
        )
        resolved_path = Path(resolved)
        if resolved_path.resolve() != checkpoint.resolve():
            raise ValueError("training backend returned an unexpected checkpoint path")
        members.append(
            PolicyMember(
                seed=seed,
                checkpoint_digest=_file_digest(resolved_path),
            )
        )

    digest_payload = {
        "action_schema": ACTION_SCHEMA,
        "created_at": created_at,
        "dataset_id": dataset.dataset_id,
        "members": tuple(
            {
                "checkpoint_digest": member.checkpoint_digest,
                "seed": member.seed,
            }
            for member in members
        ),
        "schema_version": "policy_ensemble_v1",
        "training": {
            "allow_low_gamma": config.allow_low_gamma,
            "gamma": config.gamma,
            "timesteps": config.timesteps,
        },
    }
    return PolicyEnsembleManifest(
        digest=content_digest(digest_payload),
        dataset_id=dataset.dataset_id,
        action_schema=ACTION_SCHEMA,
        expected_members=len(config.seeds),
        members=tuple(members),
        created_at=created_at,
    )


class StableBaselines3PPOBackend:
    """Stable-Baselines3 adapter kept outside domain and workflow code."""

    def __init__(
        self,
        environment_factory: Callable[[], gym.Env],
        *,
        policy: str = "MlpPolicy",
        verbose: int = 0,
    ) -> None:
        self.environment_factory = environment_factory
        self.policy = policy
        self.verbose = verbose

    def train(
        self,
        *,
        seed: int,
        timesteps: int,
        gamma: float,
        output_path: Path,
    ) -> Path:
        from stable_baselines3 import PPO

        environment = self.environment_factory()
        try:
            model = PPO(
                self.policy,
                environment,
                gamma=gamma,
                seed=seed,
                verbose=self.verbose,
            )
            model.learn(total_timesteps=timesteps)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            save_target = output_path.with_suffix("")
            model.save(str(save_target))
            created = save_target.with_suffix(".zip")
            if created != output_path:
                created.replace(output_path)
            return output_path
        finally:
            environment.close()
