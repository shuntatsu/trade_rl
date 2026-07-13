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
    decision_hours: float | None = None
    discount_half_life_hours: float | None = None
    log_std_init: float = -0.5
    target_kl: float | None = 0.02
    use_sde: bool = False
    sde_sample_freq: int = -1
    policy_net_arch: tuple[int, ...] = (128, 128)
    asset_set_encoder: bool = True
    asset_embedding_dim: int = 64
    global_embedding_dim: int = 64
    algorithm: str = "ppo"
    buffer_size: int = 100_000
    learning_starts: int = 10_000
    train_freq: int = 1
    gradient_steps: int = 1

    def __post_init__(self) -> None:
        for integer_field_name, integer_value in (
            ("timesteps", self.timesteps),
            ("n_steps", self.n_steps),
            ("batch_size", self.batch_size),
            ("n_epochs", self.n_epochs),
            ("buffer_size", self.buffer_size),
            ("train_freq", self.train_freq),
            ("gradient_steps", self.gradient_steps),
        ):
            if (
                isinstance(integer_value, bool)
                or not isinstance(integer_value, int)
                or integer_value <= 0
            ):
                raise ValueError(f"{integer_field_name} must be a positive integer")
        if self.algorithm.lower() == "ppo" and self.n_steps % self.batch_size != 0:
            raise ValueError("batch_size must divide n_steps for one PPO environment")
        algorithm = self.algorithm.lower()
        if algorithm not in {"ppo", "sac", "td3", "tqc"}:
            raise ValueError("algorithm must be one of ppo, sac, td3, or tqc")
        object.__setattr__(self, "algorithm", algorithm)
        if (
            isinstance(self.learning_starts, bool)
            or not isinstance(self.learning_starts, int)
            or self.learning_starts < 0
        ):
            raise ValueError("learning_starts must be a non-negative integer")
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
        if self.decision_hours is not None and (
            not math.isfinite(self.decision_hours) or self.decision_hours <= 0.0
        ):
            raise ValueError("decision_hours must be finite and positive")
        if self.discount_half_life_hours is not None and (
            not math.isfinite(self.discount_half_life_hours)
            or self.discount_half_life_hours <= 0.0
        ):
            raise ValueError("discount_half_life_hours must be finite and positive")
        if self.discount_half_life_hours is not None and self.decision_hours is None:
            raise ValueError("discount_half_life_hours requires decision_hours")
        if (
            self.decision_hours is not None
            and self.discount_half_life_hours is not None
        ):
            expected_gamma = gamma_from_half_life(
                decision_hours=self.decision_hours,
                half_life_hours=self.discount_half_life_hours,
            )
            if not math.isclose(self.gamma, expected_gamma, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(
                    "gamma does not match the configured real-time half-life"
                )
        if not math.isfinite(self.log_std_init):
            raise ValueError("log_std_init must be finite")
        if self.target_kl is not None and (
            not math.isfinite(self.target_kl) or self.target_kl <= 0.0
        ):
            raise ValueError("target_kl must be finite and positive")
        if not isinstance(self.use_sde, bool):
            raise ValueError("use_sde must be a boolean")
        if (
            isinstance(self.sde_sample_freq, bool)
            or not isinstance(self.sde_sample_freq, int)
            or self.sde_sample_freq == 0
            or self.sde_sample_freq < -1
        ):
            raise ValueError("sde_sample_freq must be -1 or a positive integer")
        if not self.policy_net_arch or any(
            isinstance(width, bool) or not isinstance(width, int) or width <= 0
            for width in self.policy_net_arch
        ):
            raise ValueError("policy_net_arch must contain positive integers")
        if not isinstance(self.asset_set_encoder, bool):
            raise ValueError("asset_set_encoder must be a boolean")
        for field_name, value in (
            ("asset_embedding_dim", self.asset_embedding_dim),
            ("global_embedding_dim", self.global_embedding_dim),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer")

    @property
    def rounded_timesteps(self) -> int:
        if self.algorithm == "ppo":
            return math.ceil(self.timesteps / self.n_steps) * self.n_steps
        return self.timesteps

    def digest_payload(self) -> dict[str, object]:
        return {
            "algorithm": self.algorithm,
            "asset_embedding_dim": self.asset_embedding_dim,
            "asset_set_encoder": self.asset_set_encoder,
            "batch_size": self.batch_size,
            "buffer_size": self.buffer_size,
            "global_embedding_dim": self.global_embedding_dim,
            "clip_range": self.clip_range,
            "decision_hours": self.decision_hours,
            "device": self.device,
            "discount_half_life_hours": self.discount_half_life_hours,
            "ent_coef": self.ent_coef,
            "gae_lambda": self.gae_lambda,
            "gamma": self.gamma,
            "gradient_steps": self.gradient_steps,
            "learning_rate": self.learning_rate,
            "learning_starts": self.learning_starts,
            "log_std_init": self.log_std_init,
            "max_grad_norm": self.max_grad_norm,
            "n_epochs": self.n_epochs,
            "n_steps": self.n_steps,
            "normalize_advantage": self.normalize_advantage,
            "policy": self.policy,
            "policy_net_arch": self.policy_net_arch,
            "sde_sample_freq": self.sde_sample_freq,
            "seeds": self.seeds,
            "target_kl": self.target_kl,
            "timesteps": self.timesteps,
            "train_freq": self.train_freq,
            "use_sde": self.use_sde,
            "vf_coef": self.vf_coef,
        }


@dataclass(frozen=True, slots=True)
class PolicyTrainingResult:
    """One backend run with complete environment and model-shape identity."""

    checkpoint_path: Path
    actual_timesteps: int
    resolved_device: str
    environment_digest: str
    initial_capital: float
    action_size: int = 2
    action_names: tuple[str, ...] = ()
    action_spec_digest: str | None = None
    observation_size: int | None = None
    alpha_artifact_digest: str | None = None
    factor_artifact_digest: str | None = None
    normalizer_digest: str | None = None

    def __post_init__(self) -> None:
        if self.actual_timesteps <= 0:
            raise ValueError("actual_timesteps must be positive")
        require_non_empty(self.resolved_device, field="resolved_device")
        require_sha256(self.environment_digest, field="environment_digest")
        if not math.isfinite(self.initial_capital) or self.initial_capital <= 0.0:
            raise ValueError("initial_capital must be finite and positive")
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


class PolicyTrainingBackend(Protocol):
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


def _environment_identity(environment: gym.Env) -> dict[str, object]:
    unwrapped = environment.unwrapped
    environment_digest = getattr(unwrapped, "environment_digest", None)
    initial_capital = getattr(unwrapped, "initial_capital", None)
    if not isinstance(environment_digest, str):
        raise ValueError("training environment must expose environment_digest")
    require_sha256(environment_digest, field="environment_digest")
    if (
        isinstance(initial_capital, bool)
        or not isinstance(initial_capital, int | float)
        or not math.isfinite(float(initial_capital))
        or float(initial_capital) <= 0.0
    ):
        raise ValueError("training environment must expose positive initial_capital")
    action_space = getattr(environment, "action_space", None)
    observation_space = getattr(environment, "observation_space", None)
    action_shape = getattr(action_space, "shape", None)
    observation_shape = getattr(observation_space, "shape", None)
    if not action_shape or len(action_shape) != 1 or action_shape[0] <= 0:
        raise ValueError("training environment must expose a flat action space")
    if (
        not observation_shape
        or len(observation_shape) != 1
        or observation_shape[0] <= 0
    ):
        raise ValueError("training environment must expose a flat observation space")
    return {
        "environment_digest": environment_digest,
        "initial_capital": float(initial_capital),
        "action_size": int(action_shape[0]),
        "action_names": tuple(getattr(unwrapped, "action_names", ())),
        "action_spec_digest": getattr(unwrapped, "action_spec_digest", None),
        "observation_size": int(observation_shape[0]),
        "decision_hours": getattr(unwrapped, "decision_hours", None),
        "alpha_artifact_digest": getattr(unwrapped, "alpha_artifact_digest", None),
        "factor_artifact_digest": getattr(unwrapped, "factor_artifact_digest", None),
        "normalizer_digest": (
            None
            if getattr(unwrapped, "normalizer", None) is None
            else getattr(unwrapped.normalizer, "digest", None)
        ),
    }


def _validate_training_environment(
    identity: dict[str, object],
    config: ResidualTrainingConfig,
) -> None:
    action_size = int(identity["action_size"])
    action_names = identity["action_names"]
    action_spec_digest = identity["action_spec_digest"]
    if not isinstance(action_names, tuple) or len(action_names) != action_size:
        raise ValueError("training environment must expose exact action_names")
    if not isinstance(action_spec_digest, str):
        raise ValueError("training environment must expose action_spec_digest")
    require_sha256(action_spec_digest, field="action_spec_digest")
    environment_decision_hours = identity["decision_hours"]
    if config.decision_hours is not None:
        if not isinstance(environment_decision_hours, int | float):
            raise ValueError("training environment must expose decision_hours")
        if not math.isclose(
            float(environment_decision_hours),
            config.decision_hours,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("training decision_hours do not match the environment")


def train_residual_ensemble(
    *,
    dataset: DatasetManifest,
    environment_dataset_id: str,
    config: ResidualTrainingConfig,
    backend: PolicyTrainingBackend,
    output_dir: Path,
    created_at: datetime,
) -> PolicyEnsembleManifest:
    require_sha256(environment_dataset_id, field="environment_dataset_id")
    require_aware_datetime(created_at, field="created_at")
    if dataset.dataset_id != environment_dataset_id:
        raise ValueError("dataset identity mismatch between manifest and environment")
    output_dir.mkdir(parents=True, exist_ok=True)

    members: list[PolicyMember] = []
    results: list[PolicyTrainingResult] = []
    for member_index, seed in enumerate(config.seeds):
        checkpoint = output_dir / f"member-{member_index:03d}" / "policy.zip"
        result = backend.train(seed=seed, config=config, output_path=checkpoint)
        resolved_path = Path(result.checkpoint_path)
        if resolved_path.resolve() != checkpoint.resolve():
            raise ValueError("training backend returned an unexpected checkpoint path")
        results.append(result)
        members.append(
            PolicyMember(seed=seed, checkpoint_digest=_file_digest(resolved_path))
        )

    consistency_fields = (
        "actual_timesteps",
        "resolved_device",
        "environment_digest",
        "initial_capital",
        "action_size",
        "action_names",
        "action_spec_digest",
        "observation_size",
        "alpha_artifact_digest",
        "factor_artifact_digest",
        "normalizer_digest",
    )
    values: dict[str, object] = {}
    for field_name in consistency_fields:
        observed = {getattr(result, field_name) for result in results}
        if len(observed) != 1:
            raise ValueError(f"ensemble members reported inconsistent {field_name}")
        values[field_name] = observed.pop()

    training_config_digest = content_digest(config.digest_payload())
    digest_payload = {
        "action_schema": ACTION_SCHEMA,
        "action_names": values["action_names"],
        "action_size": values["action_size"],
        "action_spec_digest": values["action_spec_digest"],
        "actual_timesteps": values["actual_timesteps"],
        "alpha_artifact_digest": values["alpha_artifact_digest"],
        "created_at": created_at,
        "dataset_id": dataset.dataset_id,
        "environment_digest": values["environment_digest"],
        "factor_artifact_digest": values["factor_artifact_digest"],
        "initial_capital": values["initial_capital"],
        "members": tuple(
            {"checkpoint_digest": member.checkpoint_digest, "seed": member.seed}
            for member in members
        ),
        "normalizer_digest": values["normalizer_digest"],
        "observation_schema": OBSERVATION_SCHEMA,
        "observation_size": values["observation_size"],
        "requested_timesteps": config.timesteps,
        "resolved_device": values["resolved_device"],
        "schema_version": "policy_ensemble_v4",
        "training_config_digest": training_config_digest,
    }
    return PolicyEnsembleManifest(
        digest=content_digest(digest_payload),
        dataset_id=dataset.dataset_id,
        action_schema=ACTION_SCHEMA,
        observation_schema=OBSERVATION_SCHEMA,
        training_config_digest=training_config_digest,
        environment_digest=str(values["environment_digest"]),
        initial_capital=float(values["initial_capital"]),
        requested_timesteps=config.timesteps,
        actual_timesteps=int(values["actual_timesteps"]),
        resolved_device=str(values["resolved_device"]),
        expected_members=len(config.seeds),
        members=tuple(members),
        created_at=created_at,
        action_size=int(values["action_size"]),
        action_names=tuple(values["action_names"]),  # type: ignore[arg-type]
        action_spec_digest=str(values["action_spec_digest"]),
        observation_size=(
            None
            if values["observation_size"] is None
            else int(values["observation_size"])
        ),
        alpha_artifact_digest=values["alpha_artifact_digest"],  # type: ignore[arg-type]
        factor_artifact_digest=values["factor_artifact_digest"],  # type: ignore[arg-type]
        normalizer_digest=values["normalizer_digest"],  # type: ignore[arg-type]
    )


class StableBaselines3Backend:
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
        from stable_baselines3 import PPO, SAC, TD3

        environment = self.environment_factory()
        try:
            identity = _environment_identity(environment)
            _validate_training_environment(identity, config)
            policy_kwargs: dict[str, object] = {
                "net_arch": list(config.policy_net_arch),
            }
            if config.algorithm == "ppo":
                policy_kwargs["log_std_init"] = config.log_std_init
            if config.asset_set_encoder:
                from trade_rl.rl.policies import AssetSetFeatureExtractor

                unwrapped = environment.unwrapped
                layout = getattr(unwrapped, "layout", None)
                active_column = getattr(unwrapped, "asset_active_column", None)
                if layout is None or not isinstance(active_column, int):
                    raise ValueError(
                        "asset-set training requires environment layout metadata"
                    )
                policy_kwargs.update(
                    {
                        "features_extractor_class": AssetSetFeatureExtractor,
                        "features_extractor_kwargs": {
                            "n_symbols": layout.n_symbols,
                            "per_symbol_width": layout.per_symbol_width,
                            "global_width": layout.global_width,
                            "active_column": active_column,
                            "asset_embedding_dim": config.asset_embedding_dim,
                            "global_embedding_dim": config.global_embedding_dim,
                        },
                    }
                )
            common: dict[str, object] = {
                "learning_rate": config.learning_rate,
                "gamma": config.gamma,
                "policy_kwargs": policy_kwargs,
                "seed": seed,
                "device": config.device,
                "verbose": self.verbose,
            }
            if config.algorithm == "ppo":
                model = PPO(
                    config.policy,
                    environment,
                    n_steps=config.n_steps,
                    batch_size=config.batch_size,
                    n_epochs=config.n_epochs,
                    gae_lambda=config.gae_lambda,
                    clip_range=config.clip_range,
                    normalize_advantage=config.normalize_advantage,
                    ent_coef=config.ent_coef,
                    vf_coef=config.vf_coef,
                    max_grad_norm=config.max_grad_norm,
                    target_kl=config.target_kl,
                    use_sde=config.use_sde,
                    sde_sample_freq=config.sde_sample_freq,
                    **common,
                )
            else:
                off_policy: dict[str, object] = {
                    "buffer_size": config.buffer_size,
                    "learning_starts": config.learning_starts,
                    "batch_size": config.batch_size,
                    "train_freq": config.train_freq,
                    "gradient_steps": config.gradient_steps,
                    **common,
                }
                if config.algorithm == "sac":
                    model = SAC(
                        config.policy,
                        environment,
                        use_sde=config.use_sde,
                        sde_sample_freq=config.sde_sample_freq,
                        **off_policy,
                    )
                elif config.algorithm == "td3":
                    model = TD3(config.policy, environment, **off_policy)
                else:
                    try:
                        from sb3_contrib import TQC
                    except ImportError as error:
                        raise RuntimeError(
                            "TQC training requires the optional sb3-contrib package"
                        ) from error
                    model = TQC(
                        config.policy,
                        environment,
                        use_sde=config.use_sde,
                        sde_sample_freq=config.sde_sample_freq,
                        **off_policy,
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
                environment_digest=str(identity["environment_digest"]),
                initial_capital=float(identity["initial_capital"]),
                action_size=int(identity["action_size"]),
                action_names=tuple(identity["action_names"]),  # type: ignore[arg-type]
                action_spec_digest=str(identity["action_spec_digest"]),
                observation_size=int(identity["observation_size"]),
                alpha_artifact_digest=identity["alpha_artifact_digest"],  # type: ignore[arg-type]
                factor_artifact_digest=identity["factor_artifact_digest"],  # type: ignore[arg-type]
                normalizer_digest=identity["normalizer_digest"],  # type: ignore[arg-type]
            )
        finally:
            environment.close()


class StableBaselines3PPOBackend(StableBaselines3Backend):
    """Compatibility backend that accepts only PPO configurations."""

    def train(
        self,
        *,
        seed: int,
        config: ResidualTrainingConfig,
        output_path: Path,
    ) -> PolicyTrainingResult:
        if config.algorithm != "ppo":
            raise ValueError("StableBaselines3PPOBackend requires algorithm='ppo'")
        return super().train(seed=seed, config=config, output_path=output_path)
