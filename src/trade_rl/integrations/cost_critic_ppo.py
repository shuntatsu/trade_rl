"""Opt-in PPO collector with an actor-isolated Cost Critic sidecar."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from itertools import chain
from typing import Any, ClassVar, cast

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import BaseModel
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.vec_env import VecEnv

from trade_rl.artifacts.hashing import content_digest
from trade_rl.integrations.cost_rollout_buffer import CostRolloutStorage
from trade_rl.rl.cost_critics import FamilySeparatedCostCritic
from trade_rl.rl.cost_diagnostics import (
    CostHeadDiagnostics,
    FamilyGradientDiagnostics,
    build_cost_head_diagnostics,
    build_family_gradient_diagnostics,
    gradient_l2_norm,
)
from trade_rl.rl.cost_learning import (
    CostFamily,
    CostLearningSchema,
    canonical_cost_learning_schema,
)


class CostCriticPPO(PPO):
    """Train independent Cost Critics without altering PPO's actor objective."""

    algorithm_identifier: ClassVar[str] = "cost_critic_ppo"

    def __init__(
        self,
        *args: Any,
        cost_schema: CostLearningSchema | None = None,
        cost_learning_rate: float = 3e-4,
        cost_n_epochs: int = 1,
        cost_batch_size: int | None = None,
        cost_continuous_hidden_dims: tuple[int, ...] = (128, 64),
        cost_event_hidden_dims: tuple[int, ...] = (128, 64),
        cost_max_grad_norm: float = 0.5,
        _init_setup_model: bool = True,
        **kwargs: Any,
    ) -> None:
        if cost_schema is None:
            if _init_setup_model:
                raise TypeError("cost_schema must be a CostLearningSchema")
            resolved_cost_schema = canonical_cost_learning_schema()
        elif isinstance(cost_schema, CostLearningSchema):
            resolved_cost_schema = cost_schema
        else:
            raise TypeError("cost_schema must be a CostLearningSchema")
        if not np.isfinite(cost_learning_rate) or cost_learning_rate <= 0.0:
            raise ValueError("cost_learning_rate must be finite and positive")
        if isinstance(cost_n_epochs, bool) or cost_n_epochs <= 0:
            raise ValueError("cost_n_epochs must be a positive integer")
        if cost_batch_size is not None and (
            isinstance(cost_batch_size, bool) or cost_batch_size <= 0
        ):
            raise ValueError("cost_batch_size must be null or a positive integer")
        if not np.isfinite(cost_max_grad_norm) or cost_max_grad_norm <= 0.0:
            raise ValueError("cost_max_grad_norm must be finite and positive")
        self.cost_schema = resolved_cost_schema
        self.cost_learning_rate = float(cost_learning_rate)
        self.cost_n_epochs = int(cost_n_epochs)
        self.cost_batch_size = cost_batch_size
        self.cost_continuous_hidden_dims = tuple(cost_continuous_hidden_dims)
        self.cost_event_hidden_dims = tuple(cost_event_hidden_dims)
        self.cost_max_grad_norm = float(cost_max_grad_norm)
        self.cost_update_count = 0
        self.last_cost_training_metrics: dict[str, float] = {}
        self.last_cost_head_diagnostics: dict[str, CostHeadDiagnostics] = {}
        self.last_cost_family_gradient_diagnostics: FamilyGradientDiagnostics | None = (
            None
        )
        self._cost_support_totals = {name: 0.0 for name in self.cost_schema.event_names}
        self._cost_rng = np.random.default_rng(kwargs.get("seed"))
        super().__init__(  # type: ignore[misc]
            *args,
            _init_setup_model=False,
            **kwargs,
        )
        if _init_setup_model:
            self._setup_model()

    @staticmethod
    def _torch_rng_state() -> tuple[torch.Tensor, list[torch.Tensor] | None]:
        cpu_state = torch.random.get_rng_state()
        cuda_states = (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        )
        return cpu_state, cuda_states

    @staticmethod
    def _restore_torch_rng_state(
        state: tuple[torch.Tensor, list[torch.Tensor] | None],
    ) -> None:
        cpu_state, cuda_states = state
        torch.random.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)

    def _setup_model(self) -> None:
        self.cost_continuous_hidden_dims = tuple(self.cost_continuous_hidden_dims)
        self.cost_event_hidden_dims = tuple(self.cost_event_hidden_dims)
        super()._setup_model()
        features_extractor = getattr(self.policy, "features_extractor", None)
        input_dim = getattr(features_extractor, "features_dim", None)
        if not isinstance(input_dim, int) or input_dim <= 0:
            raise RuntimeError(
                "policy feature dimension is unavailable for Cost Critic"
            )
        rng_state = self._torch_rng_state()
        try:
            self.cost_critic = FamilySeparatedCostCritic(
                input_dim=input_dim,
                schema=self.cost_schema,
                continuous_hidden_dims=self.cost_continuous_hidden_dims,
                event_hidden_dims=self.cost_event_hidden_dims,
            ).to(self.device)
        finally:
            self._restore_torch_rng_state(rng_state)
        self.cost_critic_optimizer = torch.optim.Adam(
            self.cost_critic.parameters(),
            lr=self.cost_learning_rate,
        )
        self.cost_rollout_storage = CostRolloutStorage(
            buffer_size=self.n_steps,
            n_envs=self.n_envs,
            schema=self.cost_schema,
        )
        if self.cost_batch_size is None:
            self.cost_batch_size = self.batch_size
        transition_count = self.n_steps * self.n_envs
        if self.cost_batch_size > transition_count:
            raise ValueError("cost_batch_size exceeds rollout transition count")

    @staticmethod
    def _select_cost_features(features: object) -> torch.Tensor:
        """Preserve the maintained Cost Critic feature-selection semantics."""

        selected = features[1] if isinstance(features, tuple) else features
        if not isinstance(selected, torch.Tensor):
            raise RuntimeError("policy feature extraction did not return a tensor")
        return selected

    def _cost_features(self, observations: Any) -> torch.Tensor:
        return self._select_cost_features(
            self.policy.extract_features(observations)
        ).detach()

    @staticmethod
    def _resolved_device(device: torch.device | str) -> torch.device:
        """Resolve implicit CUDA devices to the process-local device index."""

        resolved = torch.device(device)
        resolved_index = getattr(resolved, "index", None)
        if resolved.type == "cuda" and resolved_index is None:
            return torch.device("cuda", torch.cuda.current_device())
        return resolved

    def _run_policy_with_cost_features(
        self,
        operation: Callable[[], Any],
    ) -> tuple[Any, torch.Tensor]:
        """Run one policy operation and capture its exact Cost Critic features."""

        if not callable(operation):
            raise TypeError("policy operation must be callable")
        policy = self.policy
        original = policy.extract_features
        namespace = getattr(policy, "__dict__", None)
        if isinstance(namespace, dict) and "extract_features" in namespace:
            had_local = True
            local_value = namespace["extract_features"]
        else:
            had_local = False
            local_value = None
        captured: list[torch.Tensor] = []

        def capture(*args: Any, **kwargs: Any) -> Any:
            features = original(*args, **kwargs)
            captured.append(self._select_cost_features(features))
            return features

        policy.extract_features = capture  # type: ignore[method-assign]
        try:
            result = operation()
        finally:
            if had_local:
                policy.extract_features = local_value  # type: ignore[method-assign]
            else:
                delattr(policy, "extract_features")
        if len(captured) != 1:
            raise RuntimeError(
                "policy operation must extract features exactly once for Cost Critic reuse"
            )
        return result, captured[0].detach()

    def _predict_cost_values(self, observations: Any) -> torch.Tensor:
        features = self._cost_features(observations)
        return self.cost_critic(features).values

    def _predict_values_with_cost_features(
        self,
        observations: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Match SB3 2.3.2 predict_values while reusing its value features."""

        features = BaseModel.extract_features(
            self.policy,
            observations,
            self.policy.vf_features_extractor,
        )
        if not isinstance(features, torch.Tensor):
            raise RuntimeError(
                "policy value feature extraction did not return a tensor"
            )
        latent_value = self.policy.mlp_extractor.forward_critic(features)
        values = self.policy.value_net(latent_value)
        return values, features.detach()

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """Collect the normal PPO rollout and aligned independent cost transitions."""

        assert self._last_obs is not None, "No previous observation was provided"
        self.policy.set_training_mode(False)
        self.cost_critic.train(False)
        n_steps = 0
        rollout_buffer.reset()
        self.cost_rollout_storage.reset()
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)
        callback.on_rollout_start()
        while n_steps < n_rollout_steps:
            if (
                self.use_sde
                and self.sde_sample_freq > 0
                and n_steps % self.sde_sample_freq == 0
            ):
                self.policy.reset_noise(env.num_envs)
            with torch.no_grad():
                obs_tensor = obs_as_tensor(cast(Any, self._last_obs), self.device)
                policy_output, cost_features = self._run_policy_with_cost_features(
                    lambda: self.policy(obs_tensor)
                )
                actions, values, log_probs = policy_output
                cost_values = self.cost_critic(cost_features).values
            actions_array = actions.cpu().numpy()
            clipped_actions = actions_array
            if isinstance(self.action_space, spaces.Box):
                if self.policy.squash_output:
                    clipped_actions = self.policy.unscale_action(clipped_actions)
                else:
                    clipped_actions = np.clip(
                        actions_array,
                        self.action_space.low,
                        self.action_space.high,
                    )
            new_obs, rewards, dones, infos = env.step(clipped_actions)
            self.num_timesteps += env.num_envs
            callback.update_locals(locals())
            if not callback.on_step():
                return False
            self._update_info_buffer(infos, dones)
            n_steps += 1
            if isinstance(self.action_space, spaces.Discrete):
                actions_array = actions_array.reshape(-1, 1)

            truncated = np.asarray(
                [
                    bool(done and info.get("TimeLimit.truncated", False))
                    for done, info in zip(dones, infos, strict=True)
                ],
                dtype=np.bool_,
            )
            terminated = np.asarray(dones, dtype=np.bool_) & ~truncated
            terminal_cost_values = np.zeros(
                (env.num_envs, len(self.cost_schema.names)),
                dtype=np.float32,
            )
            for index, done in enumerate(dones):
                terminal_observation = infos[index].get("terminal_observation")
                if done and terminal_observation is not None and truncated[index]:
                    terminal_obs = self.policy.obs_to_tensor(terminal_observation)[0]
                    with torch.no_grad():
                        terminal_values, terminal_features = (
                            self._predict_values_with_cost_features(terminal_obs)
                        )
                        terminal_value = terminal_values[0]
                        terminal_cost = self.cost_critic(terminal_features).values[0]
                    rewards[index] += self.gamma * float(terminal_value.item())
                    terminal_cost_values[index] = (
                        terminal_cost.detach().cpu().numpy().astype(np.float32)
                    )

            self.cost_rollout_storage.add_from_infos(
                infos=infos,
                cost_values=cost_values.detach().cpu().numpy(),
                terminated=terminated,
                truncated=truncated,
                terminal_cost_values=terminal_cost_values,
            )
            rollout_buffer.add(
                cast(Any, self._last_obs),
                actions_array,
                rewards,
                cast(Any, self._last_episode_starts),
                values,
                log_probs,
            )
            self._last_obs = cast(Any, new_obs)
            self._last_episode_starts = dones

        with torch.no_grad():
            final_observations = obs_as_tensor(cast(Any, new_obs), self.device)
            final_values, final_features = self._predict_values_with_cost_features(
                final_observations
            )
            final_cost_values = self.cost_critic(final_features).values
        rollout_buffer.compute_returns_and_advantage(
            last_values=final_values,
            dones=dones,
        )
        self.cost_rollout_storage.finalize(
            last_cost_values=final_cost_values.detach().cpu().numpy()
        )
        callback.update_locals(locals())
        callback.on_rollout_end()
        return True

    def _rollout_observations(self, indices: np.ndarray) -> Any:
        getter = getattr(self.rollout_buffer, "_get_samples", None)
        if not callable(getter):
            raise RuntimeError("rollout buffer cannot provide indexed observations")
        samples = getter(indices)
        return samples.observations

    def _build_cost_feature_cache(self) -> torch.Tensor:
        """Materialize one immutable post-PPO feature tensor for this rollout."""

        transition_count = self.n_steps * self.n_envs
        indices = np.arange(transition_count, dtype=np.int64)
        observations = self._rollout_observations(indices)
        policy_training = self.policy.training
        self.policy.set_training_mode(False)
        try:
            with torch.no_grad():
                cache = self._cost_features(observations)
        finally:
            self.policy.set_training_mode(policy_training)
        if cache.ndim != 2 or cache.shape[0] != transition_count:
            raise RuntimeError("Cost Critic feature cache has an invalid rollout shape")
        if cache.device != self._resolved_device(self.device):
            raise RuntimeError("Cost Critic feature cache is on the wrong device")
        if not bool(torch.isfinite(cache).all()):
            raise RuntimeError("Cost Critic feature cache contains non-finite values")
        return cache.detach()

    def _cached_cost_features(
        self,
        cache: torch.Tensor,
        indices: np.ndarray,
    ) -> torch.Tensor:
        """Select canonical rollout rows from the device-local feature cache."""

        if not isinstance(cache, torch.Tensor) or cache.ndim != 2:
            raise TypeError("Cost Critic feature cache must be a rank-two tensor")
        raw_indices = np.asarray(indices)
        if raw_indices.ndim != 1 or not np.issubdtype(raw_indices.dtype, np.integer):
            raise ValueError(
                "Cost Critic cache indices must be one-dimensional integers"
            )
        normalized = np.asarray(raw_indices, dtype=np.int64)
        if normalized.size == 0:
            raise ValueError("Cost Critic cache indices must not be empty")
        if np.any(normalized < 0) or np.any(normalized >= cache.shape[0]):
            raise ValueError("Cost Critic cache index is outside the rollout")
        tensor_indices = torch.as_tensor(
            normalized,
            dtype=torch.long,
            device=cache.device,
        )
        return cache.index_select(0, tensor_indices)

    def _cost_head_parameters(
        self,
        name: str,
    ) -> Iterable[torch.nn.Parameter]:
        modules: list[torch.nn.Module] = [self.cost_critic.value_heads[name]]
        if name in self.cost_critic.event_logit_heads:
            modules.append(self.cost_critic.event_logit_heads[name])
        return chain.from_iterable(module.parameters() for module in modules)

    def _family_head_parameters(
        self,
        family: CostFamily,
    ) -> Iterable[torch.nn.Parameter]:
        names = tuple(
            spec.name for spec in self.cost_schema.specs if spec.family is family
        )
        modules: list[torch.nn.Module] = [
            self.cost_critic.value_heads[name] for name in names
        ]
        if family is CostFamily.EVENT:
            modules.extend(self.cost_critic.event_logit_heads.values())
        return chain.from_iterable(module.parameters() for module in modules)

    def _build_cost_training_diagnostics(
        self,
        feature_cache: torch.Tensor,
    ) -> tuple[
        dict[str, CostHeadDiagnostics],
        FamilyGradientDiagnostics,
        dict[str, float],
    ]:
        transition_count = self.n_steps * self.n_envs
        indices = np.arange(transition_count, dtype=np.int64)
        critic_training = self.cost_critic.training
        self.cost_critic.train(False)
        try:
            with torch.no_grad():
                output = self.cost_critic(feature_cache)
        finally:
            self.cost_critic.train(critic_training)
        batch = self.cost_rollout_storage.sample(indices)
        value_predictions = output.values.detach().cpu().numpy()
        auxiliary_probabilities: dict[str, np.ndarray] = {}
        if output.auxiliary_event_logits is not None:
            for index, name in enumerate(output.auxiliary_event_names):
                auxiliary_probabilities[name] = (
                    torch.sigmoid(output.auxiliary_event_logits[:, index])
                    .detach()
                    .cpu()
                    .numpy()
                )

        family = build_family_gradient_diagnostics(
            continuous_adapter_parameters=self.cost_critic.continuous_adapter.parameters(),
            continuous_head_parameters=self._family_head_parameters(
                CostFamily.CONTINUOUS
            ),
            event_adapter_parameters=self.cost_critic.event_adapter.parameters(),
            event_head_parameters=self._family_head_parameters(CostFamily.EVENT),
        )
        reports: dict[str, CostHeadDiagnostics] = {}
        metrics: dict[str, float] = {
            "gradient/continuous": family.continuous_gradient_norm,
            "gradient/event": family.event_gradient_norm,
            "gradient/continuous_adapter": family.continuous_adapter_gradient_norm,
            "gradient/continuous_heads": family.continuous_head_gradient_norm,
            "gradient/event_adapter": family.event_adapter_gradient_norm,
            "gradient/event_heads": family.event_head_gradient_norm,
        }
        if family.dense_to_rare_gradient_ratio is not None:
            metrics["gradient/dense_to_rare_ratio"] = (
                family.dense_to_rare_gradient_ratio
            )

        for index, spec in enumerate(self.cost_schema.specs):
            event_probabilities = auxiliary_probabilities.get(spec.name)
            event_labels = (
                None
                if event_probabilities is None
                else (batch.costs[:, index] > 0.0).astype(np.float64)
            )
            adapter = (
                self.cost_critic.continuous_adapter
                if spec.family is CostFamily.CONTINUOUS
                else self.cost_critic.event_adapter
            )
            report = build_cost_head_diagnostics(
                name=spec.name,
                predictions=value_predictions[:, index],
                targets=batch.cost_returns[:, index],
                adapter_gradient_norm=gradient_l2_norm(adapter.parameters()),
                head_gradient_norm=gradient_l2_norm(
                    self._cost_head_parameters(spec.name)
                ),
                event_probabilities=event_probabilities,
                event_labels=event_labels,
            )
            reports[spec.name] = report
            prefix = f"diagnostic/{spec.name}"
            metrics[f"{prefix}/target_mean"] = report.target_mean
            metrics[f"{prefix}/target_std"] = report.target_std
            metrics[f"{prefix}/nonzero_rate"] = report.nonzero_rate
            metrics[f"{prefix}/positive_sample_count"] = float(
                report.positive_sample_count
            )
            metrics[f"{prefix}/value_loss"] = report.value_loss
            metrics[f"{prefix}/explained_variance"] = report.explained_variance
            metrics[f"{prefix}/adapter_gradient_norm"] = report.adapter_gradient_norm
            metrics[f"{prefix}/head_gradient_norm"] = report.head_gradient_norm
            if report.brier_score is not None:
                metrics[f"{prefix}/brier_score"] = report.brier_score
            if report.zero_only_brier_score is not None:
                metrics[f"{prefix}/zero_only_brier_score"] = (
                    report.zero_only_brier_score
                )
            if report.has_positive_support is not None:
                metrics[f"{prefix}/has_positive_support"] = float(
                    report.has_positive_support
                )
            if report.beats_zero_only_baseline is not None:
                metrics[f"{prefix}/beats_zero_only_baseline"] = float(
                    report.beats_zero_only_baseline
                )
            if report.eligible_for_promotion is not None:
                metrics[f"{prefix}/eligible_for_promotion"] = float(
                    report.eligible_for_promotion
                )
            for bin_index, calibration_bin in enumerate(report.calibration_bins):
                bin_prefix = f"{prefix}/calibration/{bin_index}"
                metrics[f"{bin_prefix}/count"] = float(calibration_bin.count)
                if calibration_bin.mean_probability is not None:
                    metrics[f"{bin_prefix}/mean_probability"] = (
                        calibration_bin.mean_probability
                    )
                if calibration_bin.event_rate is not None:
                    metrics[f"{bin_prefix}/event_rate"] = calibration_bin.event_rate
        return reports, family, metrics

    def _train_cost_critic(self) -> None:
        if not self.cost_rollout_storage.finalized:
            raise RuntimeError("cost rollout is not finalized")
        transition_count = self.n_steps * self.n_envs
        batch_size = int(self.cost_batch_size or transition_count)
        losses: list[float] = []
        head_losses: dict[str, list[float]] = {
            name: [] for name in self.cost_schema.names
        }
        rng_state = self._torch_rng_state()
        policy_training = self.policy.training
        critic_training = self.cost_critic.training
        self.policy.set_training_mode(False)
        self.cost_critic.train(True)
        completed = False
        try:
            feature_cache = self._build_cost_feature_cache()
            for _ in range(self.cost_n_epochs):
                permutation = self._cost_rng.permutation(transition_count)
                for start in range(0, transition_count, batch_size):
                    indices = permutation[start : start + batch_size]
                    features = self._cached_cost_features(feature_cache, indices)
                    output = self.cost_critic(features)
                    batch = self.cost_rollout_storage.sample(indices)
                    returns = torch.as_tensor(
                        batch.cost_returns,
                        dtype=output.values.dtype,
                        device=self.device,
                    )
                    immediate_costs = torch.as_tensor(
                        batch.costs,
                        dtype=output.values.dtype,
                        device=self.device,
                    )
                    total_loss = torch.zeros((), device=self.device)
                    for index, spec in enumerate(self.cost_schema.specs):
                        value_loss = torch.nn.functional.mse_loss(
                            output.values[:, index],
                            returns[:, index],
                        )
                        total_loss = (
                            total_loss + spec.value_loss_coefficient * value_loss
                        )
                        head_losses[spec.name].append(float(value_loss.detach().cpu()))
                    if output.auxiliary_event_logits is not None:
                        for logit_index, name in enumerate(
                            output.auxiliary_event_names
                        ):
                            schema_index = self.cost_schema.names.index(name)
                            target = (immediate_costs[:, schema_index] > 0.0).to(
                                dtype=output.values.dtype
                            )
                            coefficient = self.cost_schema[
                                name
                            ].auxiliary_event_loss_coefficient
                            classification_loss = (
                                torch.nn.functional.binary_cross_entropy_with_logits(
                                    output.auxiliary_event_logits[:, logit_index],
                                    target,
                                )
                            )
                            total_loss = total_loss + coefficient * classification_loss
                    self.cost_critic_optimizer.zero_grad()
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.cost_critic.parameters(),
                        self.cost_max_grad_norm,
                    )
                    self.cost_critic_optimizer.step()
                    self.cost_update_count += 1
                    losses.append(float(total_loss.detach().cpu()))
            completed = True
        finally:
            if not completed:
                self.cost_critic.train(critic_training)
            self.policy.set_training_mode(policy_training)
            self._restore_torch_rng_state(rng_state)

        metrics: dict[str, float] = {
            "loss/total": float(np.mean(losses)),
        }
        for name, values in head_losses.items():
            metrics[f"loss/{name}"] = float(np.mean(values))
        for name in self.cost_schema.event_names:
            index = self.cost_schema.names.index(name)
            support = float(
                np.count_nonzero(self.cost_rollout_storage.costs[:, :, index])
            )
            self._cost_support_totals[name] += support
            metrics[f"support/{name}"] = self._cost_support_totals[name]
        reports, family, diagnostic_metrics = self._build_cost_training_diagnostics(
            feature_cache
        )
        self.last_cost_head_diagnostics = reports
        self.last_cost_family_gradient_diagnostics = family
        metrics.update(diagnostic_metrics)
        self.last_cost_training_metrics = metrics
        for name, value in metrics.items():
            self.logger.record(f"cost/{name}", value)

    def checkpoint_identity_payload(self) -> dict[str, object]:
        """Return deterministic Cost Critic and rollout identity for checkpoints."""

        rollout_schema = {
            "schema_version": "cost_rollout_storage_v1",
            "buffer_size": self.cost_rollout_storage.buffer_size,
            "n_envs": self.cost_rollout_storage.n_envs,
            "cost_names": list(self.cost_schema.names),
            "cost_schema_digest": self.cost_schema.digest,
        }
        return {
            "algorithm": self.algorithm_identifier,
            "architecture_digest": self.cost_critic.architecture_digest,
            "cost_names": list(self.cost_schema.names),
            "cost_schema_digest": self.cost_schema.digest,
            "rollout_schema_digest": content_digest(rollout_schema),
        }

    def train(self) -> None:
        """Run the unchanged PPO update, then the isolated Cost Critic update."""

        super().train()
        self._train_cost_critic()

    def _get_torch_save_params(self) -> tuple[list[str], list[str]]:
        state_dicts, variables = super()._get_torch_save_params()
        return (
            [*state_dicts, "cost_critic", "cost_critic_optimizer"],
            variables,
        )


__all__ = ["CostCriticPPO"]
