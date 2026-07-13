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
from trade_rl.domain.common import (
    require_aware_datetime,
    require_non_empty,
    require_sha256,
)
from trade_rl.domain.datasets import DatasetManifest
from trade_rl.domain.policies import PolicyEnsembleManifest, PolicyMember
from trade_rl.rl.actions import ACTION_SCHEMA
from trade_rl.rl.observations import OBSERVATION_SCHEMA


def gamma_from_half_life(*, decision_hours: float, half_life_hours: float) -> float:
    """Convert a real-time discount half-life to a per-decision gamma."""

    if not math.isfinite(decision_hours) or decision_hours <= 0.0:
        raise ValueError("decision_hours must be finite and positive")
    if not math.isfinite(half_life_hours) or half_life_hours <= 0.0:
        raise ValueError("half_life_hours must be finite and positive")
    gamma = math.exp(math.log(0.5) * decision_hours / half_life_hours)
    if not 0.0 < gamma <= 1.0:
        raise ValueError("resolved gamma must be within (0, 1]")
    return gamma


@dataclass(frozen=True, slots=True)
class ResidualTrainingConfig:
    timesteps: int
    gamma: float
    seeds: tuple[int, ...]
    learning_rate: float = 3e-4
    n_steps: int = 2_048
    batch_size: int = 64
    n_epochs: int = 10
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    normalize_advantage: bool = True
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    policy: str = "MlpPolicy"
    device: str = "auto"

    def __post_init__(self) -> None:
        for integer_field_name, integer_value in (
            ("timesteps", self.timesteps),
            ("n_steps", self.n_steps),
            ("batch_size", self.batch_size),
            ("n_epochs", self.n_epochs),
        ):
            if (
                isinstance(integer_value, bool)
                or not isinstance(integer_value, int)
                or integer_value <= 0
            ):
                raise ValueError(
                    f"{integer_field_name} must be a positive integer"
                )
        if self.n_steps % self.batch_size != 0:
            raise ValueError("batch_size must divide n_steps for one environment")
        if not math.isfinite(self.gamma) or not 0.0 < self.gamma <= 1.0:
            raise ValueError("gamma must be within (0, 1]")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be finite and positive")
        if not math.isfinite(self.gae_lambda) or not 0.0 < self.gae_lambda <= 1.0:
            raise ValueError("gae_lambda must be within (0, 1]")
        if not math.isfinite(self.clip_range) or self.clip_range <= 0.0:
            raise ValueError("clip_range must be finite and positive")
        for coefficient_field_name, coefficient_value in (
            ("ent_coef", self.ent_coef),
            ("vf_coef", self.vf_coef),
        ):
            if not math.isfinite(coefficient_value) or coefficient_value < 0.0:
                raise ValueError(
                    f"{coefficient_field_name} must be finite and non-negative"
                )
        if not math.isfinite(self.max_grad_norm) or self.max_grad_norm <= 0.0:
            raise ValueError("max_grad_norm must be finite and positive")
        if not isinstance(self.normalize_advantage, bool):
            raise ValueError("normalize_advantage must be a boolean")
        require_non_empty(self.policy, field="policy")
        require_non_empty(self.device, field="device")
        if not self.seeds:
            raise ValueError("seeds must not be empty")
        if any(seed < 0 for seed in self.seeds):
            raise ValueError("seeds must be non-negative")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be unique")

    @property
    def rounded_timesteps(self) -> int:
        """Single-environment PPO work rounded to complete rollout batches."""

        return math.ceil(self.timesteps / self.n_steps) * self.n_steps

    def digest_payload(self) -> dict[str, object]:
        return {
            "batch_size": self.batch_size,
            "clip_range": self.clip_range,
            "device": self.device,
            "ent_coef": self.ent_coef,
            "gae_lambda": self.gae_lambda,
            "gamma": self.gamma,
            "learning_rate": self.learning_rate,
            "max_grad_norm": self.max_grad_norm,
            "n_epochs": self.n_epochs,
            "n_steps": self.n_steps,
            "normalize_advantage": self.normalize_advantage,
            "policy": self.policy,
            "seeds": self.seeds,
            "timesteps": self.timesteps,
            "vf_coef": self.vf_coef,
        }


@dataclass(frozen=True, slots=True)
class PolicyTrainingResult:
    """One backend run with observed work and device identity."""

    checkpoint_path: Path
    actual_timesteps: int
    resolved_device: str

    def __post_init__(self) -> None:
        if self.actual_timesteps <= 0:
            raise ValueError("actual_timesteps must be positive")
        require_non_empty(self.resolved_device, field="resolved_device")


class PolicyTrainingBackend(Protocol):
    """Backend boundary that writes exactly one policy checkpoint."""

    def train(
        self,
        *,
        seed: int,
        config: ResidualTrainingConfig,
        output_path: Path,
    ) -> PolicyTrainingResult: ...


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
    """Train one checkpoint per seed and bind all outputs to full run identity."""

    require_sha256(environment_dataset_id, field="environment_dataset_id")
    require_aware_datetime(created_at, field="created_at")
    if dataset.dataset_id != environment_dataset_id:
        raise ValueError("dataset identity mismatch between manifest and environment")
    output_dir.mkdir(parents=True, exist_ok=True)

    members: list[PolicyMember] = []
    actual_timesteps: set[int] = set()
    resolved_devices: set[str] = set()
    for member_index, seed in enumerate(config.seeds):
        checkpoint = output_dir / f"member-{member_index:03d}" / "policy.zip"
        result = backend.train(
            seed=seed,
            config=config,
            output_path=checkpoint,
        )
        resolved_path = Path(result.checkpoint_path)
        if resolved_path.resolve() != checkpoint.resolve():
            raise ValueError("training backend returned an unexpected checkpoint path")
        actual_timesteps.add(result.actual_timesteps)
        resolved_devices.add(result.resolved_device)
        members.append(
            PolicyMember(
                seed=seed,
                checkpoint_digest=_file_digest(resolved_path),
            )
        )

    if len(actual_timesteps) != 1:
        raise ValueError("ensemble members reported inconsistent actual_timesteps")
    if len(resolved_devices) != 1:
        raise ValueError("ensemble members reported inconsistent resolved devices")
    observed_timesteps = actual_timesteps.pop()
    resolved_device = resolved_devices.pop()
    training_config_digest = content_digest(config.digest_payload())
    digest_payload = {
        "action_schema": ACTION_SCHEMA,
        "actual_timesteps": observed_timesteps,
        "created_at": created_at,
        "dataset_id": dataset.dataset_id,
        "members": tuple(
            {
                "checkpoint_digest": member.checkpoint_digest,
                "seed": member.seed,
            }
            for member in members
        ),
        "observation_schema": OBSERVATION_SCHEMA,
        "requested_timesteps": config.timesteps,
        "resolved_device": resolved_device,
        "schema_version": "policy_ensemble_v2",
        "training_config_digest": training_config_digest,
    }
    return PolicyEnsembleManifest(
        digest=content_digest(digest_payload),
        dataset_id=dataset.dataset_id,
        action_schema=ACTION_SCHEMA,
        observation_schema=OBSERVATION_SCHEMA,
        training_config_digest=training_config_digest,
        requested_timesteps=config.timesteps,
        actual_timesteps=observed_timesteps,
        resolved_device=resolved_device,
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
        verbose: int = 0,
    ) -> None:
        self.environment_factory = environment_factory
        self.verbose = verbose

    def train(
        self,
        *,
        seed: int,
        config: ResidualTrainingConfig,
        output_path: Path,
    ) -> PolicyTrainingResult:
        from stable_baselines3 import PPO

        environment = self.environment_factory()
        try:
            model = PPO(
                config.policy,
                environment,
                learning_rate=config.learning_rate,
                n_steps=config.n_steps,
                batch_size=config.batch_size,
                n_epochs=config.n_epochs,
                gamma=config.gamma,
                gae_lambda=config.gae_lambda,
                clip_range=config.clip_range,
                normalize_advantage=config.normalize_advantage,
                ent_coef=config.ent_coef,
                vf_coef=config.vf_coef,
                max_grad_norm=config.max_grad_norm,
                seed=seed,
                device=config.device,
                verbose=self.verbose,
            )
            model.learn(total_timesteps=config.timesteps)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            save_target = output_path.with_suffix("")
            model.save(str(save_target))
            created = save_target.with_suffix(".zip")
            if created != output_path:
                created.replace(output_path)
            return PolicyTrainingResult(
                checkpoint_path=output_path,
                actual_timesteps=int(model.num_timesteps),
                resolved_device=str(model.device),
            )
        finally:
            environment.close()
