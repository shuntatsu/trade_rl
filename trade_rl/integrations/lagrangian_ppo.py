"""PPO with rollout-frozen Lagrange multipliers and isolated Cost Critics."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from typing import Any, ClassVar, cast

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.buffers import DictRolloutBuffer, RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import explained_variance
from stable_baselines3.common.vec_env import VecEnv
from torch.nn import functional as F

from trade_rl.integrations.cost_critic_ppo import CostCriticPPO
from trade_rl.rl.environment_constraints import CONSTRAINT_COST_NAMES
from trade_rl.rl.lagrangian import (
    CompletedEpisodeBatch,
    CompletedEpisodeCostAccumulator,
    ConstraintEstimate,
    DualUpdateReport,
    LagrangianDualController,
    LagrangianSchema,
    canonical_lagrangian_schema,
)
from trade_rl.rl.lagrangian_advantages import (
    combine_lagrangian_advantages,
    normalize_cost_advantages,
)
from trade_rl.rl.lagrangian_diagnostics import (
    ConstraintCorrelationDiagnostics,
    build_constraint_correlation_diagnostics,
    build_dual_stability_diagnostics,
)
from trade_rl.rl.lagrangian_evidence import (
    LagrangianRolloutEvidence,
    build_lagrangian_rollout_evidence,
)
from trade_rl.rl.lagrangian_probe import CanonicalActionProbeEvidence


def _load_placeholder_schema() -> LagrangianSchema:
    """Return a valid temporary schema used only by the SB3 load constructor."""

    count = len(CONSTRAINT_COST_NAMES)
    return canonical_lagrangian_schema(
        names=CONSTRAINT_COST_NAMES,
        budgets=(0.0,) * count,
        dual_learning_rates=(1.0,) * count,
        ema_betas=(0.0,) * count,
        initial_multipliers=(0.0,) * count,
        max_multipliers=(1.0,) * count,
        warmup_rollouts=(0,) * count,
        update_interval_rollouts=(1,) * count,
        minimum_completed_episodes=(1,) * count,
    )


class LagrangianPPO(CostCriticPPO):
    """Apply independent cost advantages without reward shaping."""

    algorithm_identifier: ClassVar[str] = "lagrangian_ppo"
    actor_composition_mode: ClassVar[str] = "raw_lagrangian_then_sb3_normalize_v1"
    completion_semantics: ClassVar[str] = "economic_time_limit_censored_shadow_v1"

    def __init__(
        self,
        *args: Any,
        lagrangian_schema: LagrangianSchema | None = None,
        canonical_action_probe_evidence: (CanonicalActionProbeEvidence | None) = None,
        _init_setup_model: bool = True,
        **kwargs: Any,
    ) -> None:
        if lagrangian_schema is None:
            if _init_setup_model:
                raise TypeError("lagrangian_schema must be a LagrangianSchema")
            resolved_schema = _load_placeholder_schema()
        elif isinstance(lagrangian_schema, LagrangianSchema):
            resolved_schema = lagrangian_schema
        else:
            raise TypeError("lagrangian_schema must be a LagrangianSchema")

        if canonical_action_probe_evidence is not None and not isinstance(
            canonical_action_probe_evidence, CanonicalActionProbeEvidence
        ):
            raise TypeError("canonical_action_probe_evidence has an invalid type")
        self.canonical_action_probe_evidence = canonical_action_probe_evidence
        self.lagrangian_schema = resolved_schema
        self.lagrangian_controller = LagrangianDualController(resolved_schema)
        self.completed_episode_cost_accumulator: (
            CompletedEpisodeCostAccumulator | None
        ) = None
        self.frozen_lagrange_multipliers = self.lagrangian_controller.begin_rollout()
        self.last_completed_episode_batch: CompletedEpisodeBatch | None = None
        self.last_constraint_estimates: dict[str, ConstraintEstimate | None] = {
            name: None for name in resolved_schema.names
        }
        self.last_dual_update_reports: dict[str, DualUpdateReport] = {}
        self.last_constraint_correlation_diagnostics: (
            ConstraintCorrelationDiagnostics | None
        ) = None
        self.dual_report_history: list[dict[str, DualUpdateReport]] = []
        self.last_lagrangian_rollout_evidence: LagrangianRolloutEvidence | None = None
        super().__init__(
            *args,
            _init_setup_model=_init_setup_model,
            **kwargs,
        )

    def _setup_model(self) -> None:
        super()._setup_model()
        self.cost_rollout_storage.require_episode_metadata = True
        if self.lagrangian_schema.names != self.cost_schema.names:
            raise ValueError(
                "Lagrangian constraint order must match the Cost Critic schema"
            )

        controller = getattr(self, "lagrangian_controller", None)
        if not isinstance(controller, LagrangianDualController):
            controller = LagrangianDualController(self.lagrangian_schema)
            self.lagrangian_controller = controller
        elif controller.schema.digest != self.lagrangian_schema.digest:
            raise ValueError("Lagrangian controller schema identity mismatch")

        accumulator = getattr(self, "completed_episode_cost_accumulator", None)
        if accumulator is None:
            accumulator = CompletedEpisodeCostAccumulator(
                n_envs=self.n_envs,
                schema=self.lagrangian_schema,
            )
            self.completed_episode_cost_accumulator = accumulator
        elif not isinstance(accumulator, CompletedEpisodeCostAccumulator):
            raise TypeError("completed_episode_cost_accumulator has an invalid type")
        elif (
            accumulator.n_envs != self.n_envs
            or accumulator.schema.digest != self.lagrangian_schema.digest
        ):
            raise ValueError("completed episode accumulator identity mismatch")

        snapshot = getattr(self, "frozen_lagrange_multipliers", None)
        expected_shape = (len(self.lagrangian_schema.names),)
        if not isinstance(snapshot, np.ndarray) or snapshot.shape != expected_shape:
            self.frozen_lagrange_multipliers = controller.begin_rollout()
        else:
            normalized_snapshot = np.asarray(snapshot, dtype=np.float64).copy()
            if not np.all(np.isfinite(normalized_snapshot)) or np.any(
                normalized_snapshot < 0.0
            ):
                raise ValueError(
                    "frozen Lagrange multipliers must be finite and non-negative"
                )
            for index, spec in enumerate(self.lagrangian_schema.specs):
                if normalized_snapshot[index] > spec.max_multiplier + 1e-12:
                    raise ValueError(
                        f"frozen Lagrange multiplier exceeds cap for {spec.name}"
                    )
            normalized_snapshot.flags.writeable = False
            self.frozen_lagrange_multipliers = normalized_snapshot

        completed_batch = getattr(self, "last_completed_episode_batch", None)
        if completed_batch is not None and not isinstance(
            completed_batch, CompletedEpisodeBatch
        ):
            self.last_completed_episode_batch = None
        estimates = getattr(self, "last_constraint_estimates", None)
        if (
            not isinstance(estimates, dict)
            or tuple(estimates) != self.lagrangian_schema.names
        ):
            self.last_constraint_estimates = {
                name: None for name in self.lagrangian_schema.names
            }
        reports = getattr(self, "last_dual_update_reports", None)
        if not isinstance(reports, dict):
            self.last_dual_update_reports = {}
        probe_evidence = getattr(self, "canonical_action_probe_evidence", None)
        if probe_evidence is not None and not isinstance(
            probe_evidence, CanonicalActionProbeEvidence
        ):
            raise TypeError("canonical_action_probe_evidence has an invalid type")
        correlation = getattr(self, "last_constraint_correlation_diagnostics", None)
        if correlation is not None and not isinstance(
            correlation, ConstraintCorrelationDiagnostics
        ):
            raise TypeError(
                "last_constraint_correlation_diagnostics has an invalid type"
            )
        history = getattr(self, "dual_report_history", None)
        if history is None:
            self.dual_report_history = []
        elif not isinstance(history, list):
            raise TypeError("dual report history has an invalid type")
        else:
            for report_set in history:
                if not isinstance(report_set, Mapping) or tuple(report_set) != (
                    self.lagrangian_schema.names
                ):
                    raise ValueError("dual report history identity mismatch")
                for name in self.lagrangian_schema.names:
                    report = report_set[name]
                    if not isinstance(report, DualUpdateReport) or report.name != name:
                        raise ValueError("dual report history identity mismatch")
        rollout_evidence = getattr(self, "last_lagrangian_rollout_evidence", None)
        if rollout_evidence is not None and not isinstance(
            rollout_evidence, LagrangianRolloutEvidence
        ):
            raise TypeError("last_lagrangian_rollout_evidence has an invalid type")

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """Freeze one multiplier snapshot before collecting the rollout."""

        self.frozen_lagrange_multipliers = self.lagrangian_controller.begin_rollout()
        return super().collect_rollouts(
            env,
            callback,
            rollout_buffer,
            n_rollout_steps,
        )

    def _prepare_rollout_buffer_for_indexed_sampling(self) -> None:
        """Mirror SB3 2.3.2 buffer preparation while retaining batch indices."""

        buffer = cast(Any, self.rollout_buffer)
        if not bool(buffer.full):
            raise RuntimeError("PPO rollout buffer must be full before training")
        if bool(buffer.generator_ready):
            return

        tensor_names: tuple[str, ...]
        if isinstance(buffer, DictRolloutBuffer):
            for key, observation in tuple(buffer.observations.items()):
                buffer.observations[key] = buffer.swap_and_flatten(observation)
            tensor_names = ("actions", "values", "log_probs", "advantages", "returns")
        else:
            tensor_names = (
                "observations",
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
            )
        for tensor_name in tensor_names:
            buffer.__dict__[tensor_name] = buffer.swap_and_flatten(
                buffer.__dict__[tensor_name]
            )
        buffer.generator_ready = True

    def _indexed_rollout_batches(self) -> Iterator[tuple[np.ndarray, Any]]:
        """Yield SB3 rollout samples together with their canonical flat indices."""

        buffer = cast(Any, self.rollout_buffer)
        transition_count = int(buffer.buffer_size * buffer.n_envs)
        indices = np.random.permutation(transition_count)
        self._prepare_rollout_buffer_for_indexed_sampling()
        batch_size = int(self.batch_size or transition_count)
        for start in range(0, transition_count, batch_size):
            batch_indices = np.asarray(
                indices[start : start + batch_size],
                dtype=np.int64,
            )
            yield batch_indices, buffer._get_samples(batch_indices)

    def _actor_advantages(
        self,
        *,
        reward_advantages: torch.Tensor,
        cost_advantages: np.ndarray,
    ) -> torch.Tensor:
        """Compose raw Lagrangian advantages, then apply pinned PPO normalization."""

        if np.all(self.frozen_lagrange_multipliers == 0.0):
            advantages = reward_advantages
            if self.normalize_advantage and len(advantages) > 1:
                advantages = (advantages - advantages.mean()) / (
                    advantages.std() + 1e-8
                )
            return advantages

        combined = combine_lagrangian_advantages(
            reward_advantages=reward_advantages.detach().cpu().numpy(),
            cost_advantages=cost_advantages,
            multipliers=self.frozen_lagrange_multipliers,
        )
        advantages = torch.as_tensor(
            combined,
            dtype=reward_advantages.dtype,
            device=reward_advantages.device,
        )
        if self.normalize_advantage and len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages

    def _train_actor_with_lagrangian_advantages(self) -> None:
        """Run the pinned SB3 PPO loop with only the actor advantage replaced."""

        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range_schedule = cast(Callable[[float], float], self.clip_range)
        clip_range = clip_range_schedule(self._current_progress_remaining)
        if self.clip_range_vf is None:
            clip_range_vf = None
        else:
            clip_range_vf_schedule = cast(
                Callable[[float], float],
                self.clip_range_vf,
            )
            clip_range_vf = clip_range_vf_schedule(self._current_progress_remaining)

        entropy_losses: list[float] = []
        policy_gradient_losses: list[float] = []
        value_losses: list[float] = []
        clip_fractions: list[float] = []
        approx_kl_divs: list[float] = []
        continue_training = True
        last_loss = torch.zeros((), device=self.device)

        for epoch in range(self.n_epochs):
            epoch_kl_divs: list[float] = []
            for batch_indices, rollout_data in self._indexed_rollout_batches():
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = actions.long().flatten()
                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations,
                    actions,
                )
                values = values.flatten()
                cost_batch = self.cost_rollout_storage.sample(batch_indices)
                advantages = self._actor_advantages(
                    reward_advantages=rollout_data.advantages,
                    cost_advantages=cost_batch.cost_advantages,
                )

                ratio = torch.exp(log_prob - rollout_data.old_log_prob)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(
                    ratio,
                    1 - clip_range,
                    1 + clip_range,
                )
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()
                policy_gradient_losses.append(float(policy_loss.detach().cpu()))
                clip_fraction = torch.mean(
                    (torch.abs(ratio - 1) > clip_range).float()
                ).item()
                clip_fractions.append(float(clip_fraction))

                if clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + torch.clamp(
                        values - rollout_data.old_values,
                        -clip_range_vf,
                        clip_range_vf,
                    )
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(float(value_loss.detach().cpu()))

                entropy_loss = (
                    -torch.mean(-log_prob) if entropy is None else -torch.mean(entropy)
                )
                entropy_losses.append(float(entropy_loss.detach().cpu()))
                loss = (
                    policy_loss
                    + self.ent_coef * entropy_loss
                    + self.vf_coef * value_loss
                )
                last_loss = loss

                with torch.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = float(
                        torch.mean((torch.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    )
                    epoch_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(
                            "Early stopping at step "
                            f"{epoch} due to reaching max kl: {approx_kl_div:.2f}"
                        )
                    break

                self.policy.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(),
                    self.max_grad_norm,
                )
                self.policy.optimizer.step()

            self._n_updates += 1
            approx_kl_divs = epoch_kl_divs
            if not continue_training:
                break

        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(),
            self.rollout_buffer.returns.flatten(),
        )
        self.logger.record("train/entropy_loss", float(np.mean(entropy_losses)))
        self.logger.record(
            "train/policy_gradient_loss",
            float(np.mean(policy_gradient_losses)),
        )
        self.logger.record("train/value_loss", float(np.mean(value_losses)))
        self.logger.record("train/approx_kl", float(np.mean(approx_kl_divs)))
        self.logger.record("train/clip_fraction", float(np.mean(clip_fractions)))
        self.logger.record("train/loss", float(last_loss.detach().cpu()))
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record(
                "train/std",
                torch.exp(self.policy.log_std).mean().item(),
            )
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

    def _update_dual_controller(self) -> None:
        accumulator = self.completed_episode_cost_accumulator
        if not isinstance(accumulator, CompletedEpisodeCostAccumulator):
            raise RuntimeError("completed episode accumulator is unavailable")
        batch = accumulator.ingest_rollout(
            costs=self.cost_rollout_storage.costs,
            elapsed_hours=self.cost_rollout_storage.elapsed_hours,
            completion_kinds=self.cost_rollout_storage.completion_kinds,
        )
        reports = self.lagrangian_controller.update_after_rollout(
            batch.estimates,
            censored_episode_count=batch.censored_episode_count,
        )
        self.last_completed_episode_batch = batch
        self.last_constraint_estimates = batch.estimates
        self.last_dual_update_reports = reports
        self.logger.record(
            "lagrangian/completed_episode_count",
            batch.completed_episode_count,
        )
        self.logger.record(
            "lagrangian/censored_episode_count",
            batch.censored_episode_count,
        )

        for name, report in reports.items():
            prefix = f"lagrangian/{name}"
            self.logger.record(f"{prefix}/budget", report.budget)
            self.logger.record(
                f"{prefix}/multiplier_before",
                report.multiplier_before,
            )
            self.logger.record(
                f"{prefix}/multiplier_after",
                report.multiplier_after,
            )
            self.logger.record(f"{prefix}/updated", float(report.updated))
            self.logger.record(f"{prefix}/saturated", float(report.saturated))
            self.logger.record(f"{prefix}/at_lower_bound", float(report.at_lower_bound))
            self.logger.record(f"{prefix}/at_upper_cap", float(report.at_upper_cap))
            self.logger.record(
                f"{prefix}/pending_numerator_before",
                report.pending_numerator_before,
            )
            self.logger.record(
                f"{prefix}/pending_denominator_before",
                report.pending_denominator_before,
            )
            self.logger.record(
                f"{prefix}/consumed_denominator",
                report.consumed_denominator,
            )
            self.logger.record(
                f"{prefix}/censored_episode_count",
                report.censored_episode_count,
            )
            self.logger.record(f"{prefix}/rollout_count", report.rollout_count)
            self.logger.record(f"{prefix}/update_count", report.update_count)
            if report.raw_estimate is not None:
                self.logger.record(f"{prefix}/raw_estimate", report.raw_estimate)
            if report.ema_estimate is not None:
                self.logger.record(f"{prefix}/ema_estimate", report.ema_estimate)
            if report.denominator is not None:
                self.logger.record(f"{prefix}/denominator", report.denominator)
            if report.constraint_residual is not None:
                self.logger.record(
                    f"{prefix}/constraint_residual",
                    report.constraint_residual,
                )

    def _flatten_rollout_diagnostics(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return canonical transition-order copies for observational diagnostics."""

        cost_count = len(self.lagrangian_schema.names)
        raw_cost_tensor = np.asarray(
            self.cost_rollout_storage.costs,
            dtype=np.float64,
        )
        raw_cost_advantage_tensor = np.asarray(
            self.cost_rollout_storage.advantages,
            dtype=np.float64,
        )
        expected_shape = (
            self.n_steps,
            self.n_envs,
            cost_count,
        )
        if raw_cost_tensor.shape != expected_shape:
            raise ValueError("raw cost rollout shape mismatch")
        if raw_cost_advantage_tensor.shape != expected_shape:
            raise ValueError("raw cost advantage rollout shape mismatch")
        raw_costs = raw_cost_tensor.swapaxes(0, 1).reshape(-1, cost_count).copy()
        raw_cost_advantages = (
            raw_cost_advantage_tensor.swapaxes(0, 1).reshape(-1, cost_count).copy()
        )

        reward_tensor = np.asarray(
            self.rollout_buffer.advantages,
            dtype=np.float64,
        )
        if reward_tensor.ndim == 2:
            reward_advantages = reward_tensor.swapaxes(0, 1).reshape(-1).copy()
        elif reward_tensor.ndim == 1:
            reward_advantages = reward_tensor.reshape(-1).copy()
        else:
            raise ValueError("reward advantage rollout shape mismatch")
        if reward_advantages.shape[0] != raw_cost_advantages.shape[0]:
            raise ValueError("reward and cost diagnostic transition counts differ")
        return reward_advantages, raw_costs, raw_cost_advantages

    def _record_lagrangian_rollout_evidence(self) -> None:
        """Record raw actor penalties and dual evidence without mutating training data."""

        reward_advantages, raw_costs, raw_cost_advantages = (
            self._flatten_rollout_diagnostics()
        )
        diagnostics = build_constraint_correlation_diagnostics(
            cost_names=self.lagrangian_schema.names,
            raw_costs=raw_costs,
            raw_cost_advantages=raw_cost_advantages,
            normalized_cost_advantages=normalize_cost_advantages(raw_cost_advantages),
            multipliers=self.frozen_lagrange_multipliers,
            reward_advantages=reward_advantages,
        )
        self.last_constraint_correlation_diagnostics = diagnostics

        reports = {
            name: self.last_dual_update_reports[name]
            for name in self.lagrangian_schema.names
        }
        self.dual_report_history.append(reports)
        stability = build_dual_stability_diagnostics(
            cost_names=self.lagrangian_schema.names,
            report_history=tuple(self.dual_report_history),
        )
        batch = self.last_completed_episode_batch
        if not isinstance(batch, CompletedEpisodeBatch):
            raise RuntimeError("completed episode batch is unavailable")
        probe_evidence = self.canonical_action_probe_evidence
        if probe_evidence is None:
            self.last_lagrangian_rollout_evidence = None
        else:
            self.last_lagrangian_rollout_evidence = build_lagrangian_rollout_evidence(
                actor_composition_mode=self.actor_composition_mode,
                schema=self.lagrangian_schema,
                correlation_diagnostics=diagnostics,
                stability_diagnostics=stability,
                dual_reports=reports,
                probe_evidence=probe_evidence,
                completed_episode_count=batch.completed_episode_count,
                censored_episode_count=batch.censored_episode_count,
            )
        self.logger.record(
            "lagrangian/penalty_to_reward_l2_ratio",
            diagnostics.penalty_to_reward_l2_ratio,
        )

    def checkpoint_identity_payload(self) -> dict[str, object]:
        """Bind the complete constrained-optimization contract to checkpoints."""

        accumulator = self.completed_episode_cost_accumulator
        if not isinstance(accumulator, CompletedEpisodeCostAccumulator):
            raise RuntimeError("completed episode accumulator is unavailable")
        probe_evidence = self.canonical_action_probe_evidence
        payload = super().checkpoint_identity_payload()
        payload.update(
            {
                "algorithm": self.algorithm_identifier,
                "actor_composition_mode": self.actor_composition_mode,
                "completion_semantics": self.completion_semantics,
                "lagrangian_schema": self.lagrangian_schema.digest_payload(),
                "lagrangian_schema_digest": self.lagrangian_schema.digest,
                "lagrangian_cost_names": list(self.lagrangian_schema.names),
                "accumulator_state_version": accumulator.state_version,
                "controller_state_version": self.lagrangian_controller.state_version,
                "canonical_action_probe": (
                    None if probe_evidence is None else probe_evidence.digest_payload()
                ),
                "canonical_action_probe_digest": (
                    None if probe_evidence is None else probe_evidence.digest
                ),
            }
        )
        return payload

    def train(self) -> None:
        """Train actor/value, then Cost Critics, then update dual state once."""

        if not self.cost_rollout_storage.finalized:
            raise RuntimeError("cost rollout is not finalized")
        self._train_actor_with_lagrangian_advantages()
        self._train_cost_critic()
        self._update_dual_controller()
        self._record_lagrangian_rollout_evidence()


__all__ = ["LagrangianPPO"]
