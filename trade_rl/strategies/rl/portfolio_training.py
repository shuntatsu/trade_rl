"""Opt-in, single-environment joint PPO training with retained native lineage."""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from numbers import Real
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

import gymnasium as gym
import numpy as np

from trade_rl.artifacts.atomic_write import atomic_write_bytes
from trade_rl.artifacts.canonical import canonical_json_bytes, to_json_value
from trade_rl.strategies.rl.portfolio_env import SharedPortfolioGymEnv
from trade_rl.strategies.rl.portfolio_episode_journal import (
    PortfolioEpisodeJournalWrapper,
)
from trade_rl.strategies.rl.portfolio_policy_sample_journal import (
    PortfolioPolicySampleJournal,
)
from trade_rl.strategies.rl.portfolio_transform import (
    PortfolioGymObservationWrapper,
    PortfolioObservationTransform,
)

PORTFOLIO_PPO_TRAINING_SCHEMA = "shared_portfolio_ppo_training_v1"
_POLICY_VERSION_SEMANTICS = (
    "sha256 of the exact saved PPO checkpoint zip bytes before this rollout; "
    "not an inference-provenance attestation"
)
_POLICY_NET_ARCH = {"pi": [64, 64], "vf": [64, 64]}
_SEED_MAX = (1 << 32) - 1


def _positive_integer(value: object, *, field: str, minimum: int = 1) -> int:
    if type(value) is not int:
        raise TypeError(f"{field} must be a built-in integer")
    if value < minimum:
        raise ValueError(f"{field} must be at least {minimum}")
    return value


def _finite_number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field} must be a real non-boolean number")
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError(f"{field} must be finite") from error
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


@dataclass(frozen=True, slots=True)
class SharedPortfolioPPOConfig:
    """Finite explicit controls for one fresh, single-environment PPO fit."""

    total_timesteps: int
    n_steps: int
    batch_size: int
    n_epochs: int
    reset_seed: int
    ppo_seed: int
    learning_rate: float = 3e-4
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float | None = None

    def __post_init__(self) -> None:
        total = _positive_integer(self.total_timesteps, field="total_timesteps")
        n_steps = _positive_integer(self.n_steps, field="n_steps", minimum=2)
        batch_size = _positive_integer(self.batch_size, field="batch_size", minimum=2)
        _positive_integer(self.n_epochs, field="n_epochs")
        if total < n_steps or total % n_steps:
            raise ValueError("total_timesteps must be a positive multiple of n_steps")
        if batch_size > n_steps or n_steps % batch_size:
            raise ValueError("batch_size must divide n_steps")
        for name in ("reset_seed", "ppo_seed"):
            value = getattr(self, name)
            if type(value) is not int or not 0 <= value <= _SEED_MAX:
                raise ValueError(f"{name} must be an integer in [0, 2**32 - 1]")
        if self.reset_seed == self.ppo_seed:
            raise ValueError("reset_seed and ppo_seed must be distinct")

        for name in (
            "learning_rate",
            "gae_lambda",
            "clip_range",
            "ent_coef",
            "vf_coef",
            "max_grad_norm",
        ):
            object.__setattr__(
                self, name, _finite_number(getattr(self, name), field=name)
            )
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError("gae_lambda must be within [0, 1]")
        if self.clip_range <= 0.0:
            raise ValueError("clip_range must be positive")
        if self.ent_coef < 0.0 or self.vf_coef < 0.0:
            raise ValueError("ent_coef and vf_coef must be non-negative")
        if self.max_grad_norm <= 0.0:
            raise ValueError("max_grad_norm must be positive")
        if self.target_kl is not None:
            target_kl = _finite_number(self.target_kl, field="target_kl")
            if target_kl <= 0.0:
                raise ValueError("target_kl must be positive or None")
            object.__setattr__(self, "target_kl", target_kl)


@dataclass(frozen=True, slots=True)
class SharedPortfolioPPOTrainingResult:
    """Software training receipt; it is not a deployable inference bundle."""

    output_root: Path
    requested_timesteps: int
    realized_timesteps: int
    rollout_count: int
    optimizer_step_calls: int
    epoch_count: int
    final_checkpoint_sha256: str


class _ActorNativeJoinWrapper(gym.Wrapper):
    """Join one persisted forward sample to its already-persisted native step."""

    def __init__(
        self,
        env: PortfolioEpisodeJournalWrapper,
        *,
        output_root: Path,
        root: Path,
        context_digest: str,
    ) -> None:
        super().__init__(env)
        if not isinstance(env, PortfolioEpisodeJournalWrapper):
            raise TypeError("env must be the native episode journal wrapper")
        self._episode_journal = env
        self._output_root = output_root
        self.root = root
        self._events_root = self.root / "events"
        if self.root.exists() or self.root.is_symlink():
            raise FileExistsError(f"join root already exists: {self.root}")
        self.root.mkdir(parents=False, exist_ok=False)
        self._events_root.mkdir()
        self._event_sequence = 0
        self._joined_count = 0
        self._last_native_sequence = -1
        self._rollout_index: int | None = None
        self._sample_journal: PortfolioPolicySampleJournal | None = None
        self._last_sample_id = 0
        self._poisoned = False
        manifest = {
            "schema": "joint_portfolio_actor_native_join_v1",
            "context_digest": context_digest,
            "context_digest_semantics": "derived from this declared environment and transform context",
            "recorded_content": [
                "policy_sample_reference_and_hash",
                "native_episode_event_reference_and_hash",
                "exact_submitted_action_code_match",
                "native_termination_flags",
            ],
        }
        self._write_unique(self.root / "manifest.json", manifest)

    @property
    def joined_count(self) -> int:
        return self._joined_count

    @property
    def is_poisoned(self) -> bool:
        return self._poisoned

    def _write_unique(self, path: Path, value: object) -> str:
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"journal path already exists: {path}")
        payload = canonical_json_bytes(value)
        atomic_write_bytes(path, payload)
        return hashlib.sha256(payload).hexdigest()

    def begin_rollout(
        self, rollout_index: int, journal: PortfolioPolicySampleJournal
    ) -> None:
        if self._poisoned:
            raise RuntimeError("actor/native join wrapper is poisoned")
        if self._sample_journal is not None:
            raise RuntimeError("a rollout is already active")
        if type(rollout_index) is not int or rollout_index <= 0:
            raise ValueError("rollout_index must be a positive integer")
        if journal.sample_count != 0:
            raise ValueError("a rollout policy journal must start empty")
        self._rollout_index = rollout_index
        self._sample_journal = journal
        self._last_sample_id = 0

    def end_rollout(self, expected_samples: int) -> None:
        journal = self._sample_journal
        if journal is None:
            raise RuntimeError("no rollout is active")
        if (
            journal.sample_count != expected_samples
            or self._last_sample_id != expected_samples
        ):
            self._poisoned = True
            raise RuntimeError("rollout sample and native-join counts do not match")
        self._sample_journal = None
        self._rollout_index = None

    def abort_rollout(self) -> None:
        self._poisoned = True
        self._sample_journal = None
        self._rollout_index = None

    @staticmethod
    def _submitted_codes(action: object) -> tuple[int, ...]:
        if isinstance(action, (tuple, list)) and any(
            isinstance(value, (bool, np.bool_)) for value in action
        ):
            raise TypeError("submitted action codes may not contain booleans")
        values = np.asarray(action)
        if values.ndim != 1 or values.dtype.kind not in "iu":
            raise TypeError("submitted action must be a one-dimensional integer vector")
        return tuple(int(value) for value in values.tolist())

    def reset(
        self, *, seed: int | None = None, options: dict[str, object] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, object]]:
        if self._poisoned:
            raise RuntimeError("actor/native join wrapper is poisoned")
        return self.env.reset(seed=seed, options=options)

    def step(
        self, action: object
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, object]]:
        if self._poisoned:
            raise RuntimeError("actor/native join wrapper is poisoned")
        journal = self._sample_journal
        rollout_index = self._rollout_index
        try:
            if journal is None or rollout_index is None:
                raise RuntimeError("a policy rollout must be active before a step")
            sample = journal.latest_sample_reference
            if (
                sample is None
                or journal.sample_count != self._last_sample_id + 1
                or sample.sample_id != self._last_sample_id + 1
                or sample.forward_batch_size != 1
            ):
                raise RuntimeError(
                    "step requires exactly one fresh batch-one policy sample"
                )
            action_codes = self._submitted_codes(action)
            if action_codes != sample.raw_action_codes:
                raise RuntimeError(
                    "submitted action differs from the persisted policy sample"
                )

            # A later persistence failure consumes this attempt; native state is not rolled back.
            observation, reward, terminated, truncated, info = self.env.step(action)
            if not isinstance(info, Mapping):
                raise TypeError("native episode journal returned non-mapping info")
            source_action = info.get("raw_action")
            if not isinstance(source_action, (tuple, list)):
                raise TypeError("native action must be an integer sequence")
            if tuple(cast(Sequence[object], source_action)) != action_codes:
                raise RuntimeError(
                    "native action differs from the persisted policy sample"
                )
            native = self._episode_journal.latest_step_reference
            if native is None or native.sequence <= self._last_native_sequence:
                raise RuntimeError("native step event reference is missing or stale")

            sample_root = journal.root.relative_to(self._output_root).as_posix()
            actor_path = f"{sample_root}/{sample.relative_path}"
            native_path = f"native/{native.relative_path}"
            sequence = self._event_sequence
            event = {
                "schema": "joint_portfolio_actor_native_join_v1",
                "sequence": sequence,
                "rollout_index": rollout_index,
                "actor_sample": {
                    "sample_id": sample.sample_id,
                    "path": actor_path,
                    "sha256": sample.sha256,
                },
                "native_event": {
                    "sequence": native.sequence,
                    "path": native_path,
                    "sha256": native.sha256,
                },
                "raw_action_codes": list(action_codes),
                "terminated": terminated,
                "truncated": truncated,
            }
            self._write_unique(
                self._events_root / f"transition-{sequence:012d}.json", event
            )
            self._event_sequence += 1
            self._joined_count += 1
            self._last_sample_id = sample.sample_id
            self._last_native_sequence = native.sequence
            return (
                cast(dict[str, np.ndarray], observation),
                cast(float, reward),
                cast(bool, terminated),
                cast(bool, truncated),
                cast(dict[str, object], info),
            )
        except BaseException:
            self._poisoned = True
            raise


class _OptimizerStepCounter:
    def __init__(self) -> None:
        self.count = 0

    def hook(
        self, optimizer: object, args: tuple[object, ...], kwargs: dict[str, object]
    ) -> None:
        self.count += 1


class _TrainingCoordinator:
    def __init__(
        self,
        *,
        output_root: Path,
        env: SharedPortfolioGymEnv,
        transform: PortfolioObservationTransform,
        config: SharedPortfolioPPOConfig,
        context_digest: str,
        join_wrapper: _ActorNativeJoinWrapper,
        optimizer_counter: _OptimizerStepCounter,
    ) -> None:
        self.output_root = output_root
        self.env = env
        self.transform = transform
        self.config = config
        self.context_digest = context_digest
        self.join_wrapper = join_wrapper
        self.optimizer_counter = optimizer_counter
        self.checkpoint_sha256: str | None = None
        self.checkpoint_path: str | None = None
        self.rollout_index = 0
        self.current_journal: PortfolioPolicySampleJournal | None = None
        self.current_rollout_root: Path | None = None
        self.rollout_start_steps = 0
        self.rollout_start_optimizer_calls = 0
        self.rollout_start_epochs = 0
        self.rollout_closed = False
        self.completed_rollouts: list[dict[str, object]] = []

    @staticmethod
    def _checkpoint_digest(path: Path) -> str:
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError("checkpoint save did not create a regular file")
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def save_initial_checkpoint(self, model: Any) -> None:
        path = self.output_root / "checkpoints" / "initial.zip"
        model.save(str(path))
        self.checkpoint_sha256 = self._checkpoint_digest(path)
        self.checkpoint_path = path.relative_to(self.output_root).as_posix()

    def start_rollout(self, model: Any) -> None:
        if self.current_journal is not None or self.rollout_closed:
            raise RuntimeError("rollout state is not ready to start")
        if self.checkpoint_sha256 is None or self.checkpoint_path is None:
            raise RuntimeError("a saved pre-update checkpoint is required")
        self.rollout_index += 1
        rollout_root = (
            self.output_root / "rollouts" / f"rollout-{self.rollout_index:06d}"
        )
        rollout_root.mkdir(parents=False, exist_ok=False)
        journal = PortfolioPolicySampleJournal(
            model,
            root=rollout_root / "policy-samples",
            symbols=tuple(self.env.dataset.symbols),
            context_digest=self.context_digest,
            transform_digest=self.transform.digest,
            policy_version=self.checkpoint_sha256,
            policy_version_semantics=_POLICY_VERSION_SEMANTICS,
        )
        try:
            self.join_wrapper.begin_rollout(self.rollout_index, journal)
            started = {
                "schema": PORTFOLIO_PPO_TRAINING_SCHEMA,
                "rollout_index": self.rollout_index,
                "starting_num_timesteps": int(model.num_timesteps),
                "starting_optimizer_step_calls": self.optimizer_counter.count,
                "starting_epoch_count": int(model._n_updates),
                "checkpoint_path": self.checkpoint_path,
                "policy_version": self.checkpoint_sha256,
                "sample_directory": (rollout_root / "policy-samples")
                .relative_to(self.output_root)
                .as_posix(),
            }
            _write_unique(rollout_root / "rollout_started.json", started)
        except BaseException as original_error:
            self.join_wrapper.abort_rollout()
            try:
                journal.close()
            except BaseException as close_error:
                original_error.add_note(
                    "Policy journal close also failed: "
                    f"{type(close_error).__name__}: {close_error}"
                )
            raise
        self.current_journal = journal
        self.current_rollout_root = rollout_root
        self.rollout_start_steps = int(model.num_timesteps)
        self.rollout_start_optimizer_calls = self.optimizer_counter.count
        self.rollout_start_epochs = int(model._n_updates)
        self.rollout_closed = False

    def end_rollout(self) -> None:
        journal = self.current_journal
        if journal is None or self.current_rollout_root is None:
            raise RuntimeError("rollout journal was not opened")
        try:
            self.join_wrapper.end_rollout(self.config.n_steps)
            if journal.sample_count != self.config.n_steps:
                raise RuntimeError("rollout did not persist exactly n_steps samples")
        finally:
            journal.close()
            self.current_journal = None
        self.rollout_closed = True

    def close_incomplete_rollout(self) -> None:
        journal = self.current_journal
        self.current_journal = None
        self.join_wrapper.abort_rollout()
        if journal is not None:
            journal.close()

    def after_train(self, model: Any) -> None:
        if not self.rollout_closed or self.current_rollout_root is None:
            raise RuntimeError("PPO train block completed without a closed rollout")
        expected_steps = self.rollout_index * self.config.n_steps
        if int(model.num_timesteps) != expected_steps:
            raise RuntimeError(
                "PPO transition count differs from the exact rollout budget"
            )
        checkpoint_path = (
            self.output_root
            / "checkpoints"
            / (f"after-rollout-{self.rollout_index:06d}.zip")
        )
        model.save(str(checkpoint_path))
        checkpoint_digest = self._checkpoint_digest(checkpoint_path)
        optimizer_delta = (
            self.optimizer_counter.count - self.rollout_start_optimizer_calls
        )
        epoch_delta = int(model._n_updates) - self.rollout_start_epochs
        completed = {
            "schema": PORTFOLIO_PPO_TRAINING_SCHEMA,
            "rollout_index": self.rollout_index,
            "realized_num_timesteps": int(model.num_timesteps),
            "sample_count": self.config.n_steps,
            "joined_transition_count": self.join_wrapper.joined_count,
            "optimizer_step_calls": optimizer_delta,
            "sb3_epoch_count": epoch_delta,
            "checkpoint_path": checkpoint_path.relative_to(self.output_root).as_posix(),
            "checkpoint_sha256": checkpoint_digest,
        }
        _write_unique(self.current_rollout_root / "rollout_completed.json", completed)
        self.completed_rollouts.append(completed)
        self.checkpoint_path = checkpoint_path.relative_to(self.output_root).as_posix()
        self.checkpoint_sha256 = checkpoint_digest
        self.rollout_closed = False
        self.current_rollout_root = None

    def validate_training_end(self, model: Any) -> None:
        expected_rollouts = self.config.total_timesteps // self.config.n_steps
        if (
            self.current_journal is not None
            or self.current_rollout_root is not None
            or len(self.completed_rollouts) != expected_rollouts
            or int(model.num_timesteps) != self.config.total_timesteps
            or self.join_wrapper.joined_count != self.config.total_timesteps
        ):
            raise RuntimeError(
                "training ended without the complete exact rollout budget"
            )


def _write_unique(path: Path, value: object) -> str:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"training record already exists: {path}")
    payload = canonical_json_bytes(value)
    atomic_write_bytes(path, payload)
    return hashlib.sha256(payload).hexdigest()


def _context_payload(
    env: SharedPortfolioGymEnv, transform: PortfolioObservationTransform
) -> dict[str, object]:
    profile = env.market_order_profile
    return {
        "dataset_id": env.dataset.dataset_id,
        "symbols": list(env.dataset.symbols),
        "feature_names": list(env.dataset.feature_names),
        "global_feature_names": list(env.dataset.global_feature_names),
        "start_index": env.start_index,
        "stop_index": env.stop_index,
        "initial_capital": env.initial_capital,
        "target_abs_weights": list(env.target_abs_weights),
        "bound_digest": env.bound.digest,
        "transform_digest": transform.digest,
        "execution_cost": to_json_value(env.execution_cost),
        "risk_config": to_json_value(env.risk_config),
        "market_order_profile_digest": None if profile is None else profile.digest,
    }


def _validate_inputs(
    env: SharedPortfolioGymEnv,
    transform: PortfolioObservationTransform,
    config: SharedPortfolioPPOConfig,
) -> tuple[str, dict[str, object]]:
    if not isinstance(env, SharedPortfolioGymEnv):
        raise TypeError("env must be a SharedPortfolioGymEnv")
    if not isinstance(transform, PortfolioObservationTransform):
        raise TypeError("transform must be a PortfolioObservationTransform")
    if not isinstance(config, SharedPortfolioPPOConfig):
        raise TypeError("config must be a SharedPortfolioPPOConfig")
    if env._session is not None:
        raise ValueError("environment must not have been reset before training")
    transform.validate_environment(env)
    execution_seed = env.execution_cost.random_seed
    if config.reset_seed == execution_seed or config.ppo_seed == execution_seed:
        raise ValueError("execution, reset, and PPO seeds must be distinct")
    context = _context_payload(env, transform)
    digest = hashlib.sha256(canonical_json_bytes(context)).hexdigest()
    return digest, context


def fit_shared_portfolio_ppo(
    env: SharedPortfolioGymEnv,
    *,
    transform: PortfolioObservationTransform,
    output_root: str | Path,
    config: SharedPortfolioPPOConfig,
) -> SharedPortfolioPPOTrainingResult:
    """Fit one fresh CPU PPO policy and retain each sample/native transition join.

    This opt-in software trainer does not change the legacy per-symbol PPO path,
    create a deployment bundle, or establish economic performance.
    """

    context_digest, context = _validate_inputs(env, transform, config)

    # Import the optional training runtime only when this opt-in API is called.
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.logger import configure
    from stable_baselines3.common.vec_env import DummyVecEnv

    optimizer_type = torch.optim.Optimizer
    if not callable(getattr(optimizer_type, "register_step_post_hook", None)):
        raise RuntimeError("the installed Torch optimizer lacks post-step hooks")

    root = Path(output_root).expanduser().absolute()
    if root.exists() or root.is_symlink():
        raise FileExistsError(f"training attempt root already exists: {root}")
    if not root.parent.is_dir():
        raise FileNotFoundError(f"training attempt parent must exist: {root.parent}")
    root.mkdir(parents=False, exist_ok=False)

    manifest = {
        "schema": PORTFOLIO_PPO_TRAINING_SCHEMA,
        "status": "started",
        "effective_config": {
            **asdict(config),
            "gamma": 1.0,
            "normalize_advantage": True,
            "normalize_reward": False,
            "device": "cpu",
            "n_envs": 1,
            "policy": "MultiInputPolicy",
            "policy_net_arch": _POLICY_NET_ARCH,
            "policy_activation": "torch.nn.Tanh",
            "policy_ortho_init": True,
            "use_sde": False,
        },
        "requested_timesteps": config.total_timesteps,
        "realized_timesteps_contract": "exact total_timesteps; no rollout rounding",
        "execution_seed": env.execution_cost.random_seed,
        "reset_seed": config.reset_seed,
        "ppo_seed": config.ppo_seed,
        "context": context,
        "context_digest": context_digest,
        "policy_version_semantics": _POLICY_VERSION_SEMANTICS,
        "episode_journal_path": "native",
        "join_journal_path": "joins",
        "rollouts_directory": "rollouts",
        "checkpoints_directory": "checkpoints",
    }

    vector_env: Any | None = None
    episode_journal: PortfolioEpisodeJournalWrapper | None = None
    join_wrapper: _ActorNativeJoinWrapper | None = None
    optimizer_handle: Any | None = None
    coordinator: _TrainingCoordinator | None = None
    model: Any | None = None
    primary_error: BaseException | None = None
    try:
        (root / "checkpoints").mkdir()
        (root / "rollouts").mkdir()
        _write_unique(root / "manifest.json", manifest)
        transformed_env = PortfolioGymObservationWrapper(env, transform)
        episode_journal = PortfolioEpisodeJournalWrapper(
            transformed_env,
            root=root / "native",
            declared_context_digest=context_digest,
        )
        join_wrapper = _ActorNativeJoinWrapper(
            episode_journal,
            output_root=root,
            root=root / "joins",
            context_digest=context_digest,
        )
        active_join_wrapper = join_wrapper
        vector_env = DummyVecEnv([lambda: active_join_wrapper])
        if vector_env.num_envs != 1:
            raise RuntimeError("stage 1 requires exactly one vector environment")

        class _JournaledPPO(PPO):
            _training_coordinator: _TrainingCoordinator | None = None

            def _excluded_save_params(self) -> list[str]:
                return [*super()._excluded_save_params(), "_training_coordinator"]

            def train(self) -> None:
                super().train()
                active = self._training_coordinator
                if active is None:
                    raise RuntimeError("training coordinator is unavailable")
                active.after_train(self)

        model = _JournaledPPO(
            "MultiInputPolicy",
            vector_env,
            learning_rate=config.learning_rate,
            n_steps=config.n_steps,
            batch_size=config.batch_size,
            n_epochs=config.n_epochs,
            gamma=1.0,
            gae_lambda=config.gae_lambda,
            clip_range=config.clip_range,
            clip_range_vf=None,
            normalize_advantage=True,
            ent_coef=config.ent_coef,
            vf_coef=config.vf_coef,
            max_grad_norm=config.max_grad_norm,
            use_sde=False,
            policy_kwargs={
                "net_arch": {"pi": [64, 64], "vf": [64, 64]},
                "activation_fn": torch.nn.Tanh,
                "ortho_init": True,
            },
            seed=config.ppo_seed,
            device="cpu",
            verbose=0,
            tensorboard_log=None,
            target_kl=config.target_kl,
        )
        if model.device.type != "cpu":
            raise RuntimeError("stage 1 PPO must run on CPU")
        if not callable(
            getattr(model.policy.optimizer, "register_step_post_hook", None)
        ):
            raise RuntimeError("the PPO optimizer lacks post-step hooks")
        logger = configure(str(root / "sb3-logs"), format_strings=["csv"])
        model.set_logger(logger)

        counter = _OptimizerStepCounter()
        optimizer_handle = model.policy.optimizer.register_step_post_hook(counter.hook)
        coordinator = _TrainingCoordinator(
            output_root=root,
            env=env,
            transform=transform,
            config=config,
            context_digest=context_digest,
            join_wrapper=join_wrapper,
            optimizer_counter=counter,
        )
        model._training_coordinator = coordinator
        active_coordinator = coordinator
        coordinator.save_initial_checkpoint(model)
        vector_env.seed(config.reset_seed)

        class _RolloutJournalCallback(BaseCallback):
            def _on_rollout_start(self) -> None:
                active_coordinator.start_rollout(self.model)

            def _on_rollout_end(self) -> None:
                active_coordinator.end_rollout()

            def _on_step(self) -> bool:
                return True

            def _on_training_end(self) -> None:
                active_coordinator.validate_training_end(self.model)

        model.learn(
            total_timesteps=config.total_timesteps,
            callback=_RolloutJournalCallback(verbose=0),
            reset_num_timesteps=True,
            progress_bar=False,
        )
        if int(model.num_timesteps) != config.total_timesteps:
            raise RuntimeError(
                "SB3 realized a different transition count than requested"
            )
        if coordinator.checkpoint_sha256 is None:
            raise RuntimeError("final post-training checkpoint is missing")
        vector_env.close()
        vector_env = None
        optimizer_handle.remove()
        optimizer_handle = None

        completion = {
            "schema": PORTFOLIO_PPO_TRAINING_SCHEMA,
            "status": "completed",
            "requested_timesteps": config.total_timesteps,
            "realized_timesteps": int(model.num_timesteps),
            "rollout_count": len(coordinator.completed_rollouts),
            "joined_transition_count": join_wrapper.joined_count,
            "optimizer_step_calls": counter.count,
            "sb3_epoch_count": int(model._n_updates),
            "rollouts": coordinator.completed_rollouts,
            "final_checkpoint_path": coordinator.checkpoint_path,
            "final_checkpoint_sha256": coordinator.checkpoint_sha256,
        }
        _write_unique(root / "completion.json", completion)
        return SharedPortfolioPPOTrainingResult(
            output_root=root,
            requested_timesteps=config.total_timesteps,
            realized_timesteps=int(model.num_timesteps),
            rollout_count=len(coordinator.completed_rollouts),
            optimizer_step_calls=counter.count,
            epoch_count=int(model._n_updates),
            final_checkpoint_sha256=coordinator.checkpoint_sha256,
        )
    except BaseException as error:
        primary_error = error
        if vector_env is not None:
            try:
                vector_env.close()
            except BaseException as close_error:
                error.add_note(
                    "Environment close also failed: "
                    f"{type(close_error).__name__}: {close_error}"
                )
        elif episode_journal is not None:
            try:
                episode_journal.close()
            except BaseException as close_error:
                error.add_note(
                    "Episode journal close also failed: "
                    f"{type(close_error).__name__}: {close_error}"
                )
        failure: dict[str, object] = {
            "schema": PORTFOLIO_PPO_TRAINING_SCHEMA,
            "status": "failed",
            "error_type": f"{type(error).__module__}.{type(error).__qualname__}",
            "error_message": str(error),
            "realized_timesteps": (None if model is None else int(model.num_timesteps)),
            "joined_transition_count": (
                None if join_wrapper is None else join_wrapper.joined_count
            ),
            "completed_rollouts": (
                [] if coordinator is None else coordinator.completed_rollouts
            ),
            "join_poisoned": None if join_wrapper is None else join_wrapper.is_poisoned,
        }
        if coordinator is not None and coordinator.current_journal is not None:
            latest_sample = coordinator.current_journal.latest_sample_reference
            failure["latest_actor_sample"] = (
                None
                if latest_sample is None
                else {
                    "sample_id": latest_sample.sample_id,
                    "relative_path": latest_sample.relative_path,
                    "sha256": latest_sample.sha256,
                }
            )
        if episode_journal is not None:
            latest_native = episode_journal.latest_step_reference
            failure["latest_native_event"] = (
                None
                if latest_native is None
                else {
                    "sequence": latest_native.sequence,
                    "relative_path": latest_native.relative_path,
                    "sha256": latest_native.sha256,
                }
            )
        if coordinator is not None:
            try:
                coordinator.close_incomplete_rollout()
            except BaseException as journal_close_error:
                error.add_note(
                    "Incomplete policy journal close also failed: "
                    f"{type(journal_close_error).__name__}: {journal_close_error}"
                )
        try:
            _write_unique(root / "failure.json", failure)
        except BaseException as write_error:
            error.add_note(
                "Failure receipt could not be persisted: "
                f"{type(write_error).__name__}: {write_error}"
            )
        raise
    finally:
        if optimizer_handle is not None:
            try:
                optimizer_handle.remove()
            except BaseException as cleanup_error:
                if primary_error is not None:
                    primary_error.add_note(
                        "Optimizer hook removal also failed: "
                        f"{type(cleanup_error).__name__}: {cleanup_error}"
                    )
                else:
                    raise


__all__ = [
    "PORTFOLIO_PPO_TRAINING_SCHEMA",
    "SharedPortfolioPPOConfig",
    "SharedPortfolioPPOTrainingResult",
    "fit_shared_portfolio_ppo",
]
