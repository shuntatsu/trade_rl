from pathlib import Path

import pytest

from trade_rl.rl.lagrangian import DualUpdateReport
from trade_rl.rl.training_run_config import TrainingRunConfig
from trade_rl.workflows.universal_lagrangian_mechanics import (
    build_lagrangian_mechanics_config,
    verify_lagrangian_mechanics_model,
)

CONFIG = Path("examples/binance-multitimeframe/universal-u6-lagrangian.json")


def test_mechanics_config_preserves_reward_and_exercises_dual_requirements() -> None:
    base = TrainingRunConfig.from_json(CONFIG)

    mechanics = build_lagrangian_mechanics_config(
        base,
        episode_hours=8.0,
        timesteps=1_024,
    )

    assert mechanics.reward == base.reward
    assert mechanics.environment.episode_hours == 8.0
    assert base.environment.episode_hours == 720.0
    assert mechanics.training.timesteps == 1_024
    assert mechanics.training.lagrangian_probe_max_steps_per_episode == 32
    assert 1_024 // 32 >= max(
        mechanics.training.lagrangian_minimum_completed_episodes
    )
    assert 1_024 // mechanics.training.n_steps > max(
        mechanics.training.lagrangian_warmup_rollouts
    )


def test_mechanics_config_rejects_smoke_that_cannot_complete_constraints() -> None:
    base = TrainingRunConfig.from_json(CONFIG)

    with pytest.raises(ValueError, match="warmup|completed-episode"):
        build_lagrangian_mechanics_config(
            base,
            episode_hours=8.0,
            timesteps=base.training.n_steps,
        )


def test_mechanics_evidence_requires_every_dual_to_update() -> None:
    names = ("drawdown", "turnover")

    def report(name: str, *, updated: bool) -> DualUpdateReport:
        return DualUpdateReport(
            name=name,
            raw_estimate=0.5 if updated else None,
            ema_estimate=0.5 if updated else None,
            budget=0.1,
            multiplier_before=0.0,
            multiplier_after=0.0004 if updated else 0.0,
            updated=updated,
            skip_reason=None if updated else "warmup",
            denominator=20 if updated else None,
            pending_numerator_before=10.0,
            pending_denominator_before=20,
            consumed_denominator=20 if updated else 0,
            censored_episode_count=0,
            constraint_residual=0.4 if updated else None,
            at_lower_bound=not updated,
            at_upper_cap=False,
            rollout_count=5,
            update_count=1 if updated else 0,
        )

    model = type(
        "Model",
        (),
        {
            "lagrangian_schema": type("Schema", (), {"names": names})(),
            "dual_report_history": [
                {name: report(name, updated=True) for name in names}
            ],
            "lagrangian_controller": type(
                "Controller",
                (),
                {"state_dict": lambda self: {"rollout_count": 5}},
            )(),
        },
    )()

    evidence = verify_lagrangian_mechanics_model(model)

    assert evidence["updated_cost_names"] == list(names)
    assert evidence["dual_report_history_count"] == 1

    model.dual_report_history[0]["turnover"] = report("turnover", updated=False)
    with pytest.raises(RuntimeError, match="turnover"):
        verify_lagrangian_mechanics_model(model)
