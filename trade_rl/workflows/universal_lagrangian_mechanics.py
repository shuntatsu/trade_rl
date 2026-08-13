"""Short-episode Universal Lagrangian mechanics diagnostics."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, replace

from trade_rl.rl.lagrangian import DualUpdateReport
from trade_rl.rl.training_run_config import TrainingRunConfig


def build_lagrangian_mechanics_config(
    base: TrainingRunConfig,
    *,
    episode_hours: float = 8.0,
    timesteps: int = 1_024,
) -> TrainingRunConfig:
    """Shorten only diagnostic horizons while preserving the economic objective."""

    training = base.training
    if training.algorithm != "lagrangian_ppo":
        raise ValueError("mechanics smoke requires lagrangian_ppo")
    if episode_hours <= 0.0 or episode_hours % base.environment.decision_hours != 0.0:
        raise ValueError("episode_hours must be a positive whole decision count")
    if timesteps <= 0 or timesteps % training.n_steps != 0:
        raise ValueError("timesteps must be a positive multiple of n_steps")
    episode_decisions = int(round(episode_hours / base.environment.decision_hours))
    rollout_count = timesteps // training.n_steps
    completed_episode_capacity = timesteps // episode_decisions
    required_rollouts = max(training.lagrangian_warmup_rollouts) + 1
    required_episodes = max(training.lagrangian_minimum_completed_episodes)
    if rollout_count < required_rollouts:
        raise ValueError("timesteps do not extend beyond Lagrangian warmup")
    if completed_episode_capacity < required_episodes:
        raise ValueError("timesteps cannot satisfy completed-episode requirements")
    mechanics_training = replace(
        training,
        timesteps=timesteps,
        checkpoint_interval_steps=training.n_steps,
        max_checkpoints=rollout_count,
        lagrangian_probe_max_steps_per_episode=episode_decisions,
        behavior_cloning_teacher="oracle",
        behavior_cloning_validation_fraction=0.0,
    )
    mechanics_environment = replace(base.environment, episode_hours=episode_hours)
    resolved = replace(
        base,
        training=mechanics_training,
        environment=mechanics_environment,
    )
    if resolved.reward != base.reward:
        raise RuntimeError("mechanics smoke changed the reward objective")
    return resolved


def verify_lagrangian_mechanics_model(model: object) -> dict[str, object]:
    """Fail closed unless every maintained dual actuator actually updated."""

    schema = getattr(model, "lagrangian_schema", None)
    names = tuple(getattr(schema, "names", ()))
    history = getattr(model, "dual_report_history", None)
    if not names or not isinstance(history, list) or not history:
        raise RuntimeError("Lagrangian mechanics evidence is unavailable")
    updated_names: set[str] = set()
    for report_set in history:
        if not isinstance(report_set, Mapping) or tuple(report_set) != names:
            raise RuntimeError("Lagrangian dual report history is invalid")
        for name in names:
            report = report_set[name]
            if not isinstance(report, DualUpdateReport) or report.name != name:
                raise RuntimeError("Lagrangian dual report identity is invalid")
            if report.updated:
                updated_names.add(name)
    missing = tuple(name for name in names if name not in updated_names)
    if missing:
        raise RuntimeError("Lagrangian mechanics did not update: " + ", ".join(missing))
    controller = getattr(model, "lagrangian_controller", None)
    state_dict = getattr(controller, "state_dict", None)
    if not callable(state_dict):
        raise RuntimeError("Lagrangian controller state is unavailable")
    final_reports = history[-1]
    return {
        "schema_version": "lagrangian_mechanics_evidence_v1",
        "cost_names": list(names),
        "updated_cost_names": [name for name in names if name in updated_names],
        "dual_report_history_count": len(history),
        "controller_state": state_dict(),
        "final_dual_reports": {name: asdict(final_reports[name]) for name in names},
    }


__all__ = [
    "build_lagrangian_mechanics_config",
    "verify_lagrangian_mechanics_model",
]
