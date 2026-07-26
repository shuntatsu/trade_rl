from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv

from trade_rl.integrations.lagrangian_ppo import LagrangianPPO
from trade_rl.rl.cost_learning import canonical_cost_learning_schema
from trade_rl.rl.environment_constraints import (
    CONSTRAINT_COST_NAMES,
    ConstraintCostVector,
)
from trade_rl.rl.lagrangian import canonical_lagrangian_schema
from trade_rl.rl.lagrangian_diagnostics import ConstraintCorrelationDiagnostics
from trade_rl.rl.lagrangian_evidence import LagrangianRolloutEvidence
from trade_rl.rl.lagrangian_probe import (
    CanonicalActionProbeEvidence,
    CanonicalActionSemantic,
)


class _EvidenceEnvironment(gym.Env[np.ndarray, np.ndarray]):
    metadata = {"render_modes": []}

    def __init__(self) -> None:
        super().__init__()
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.step_index = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[np.ndarray, dict[str, object]]:
        del options
        super().reset(seed=seed)
        self.step_index = 0
        return np.zeros(3, dtype=np.float32), {}

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        del action
        self.step_index += 1
        terminated = self.step_index == 4
        costs = ConstraintCostVector(
            drawdown_excess=0.05 * self.step_index,
            drawdown_stop_event=float(terminated),
            margin_deficit_fraction=0.01 * self.step_index,
            forced_liquidation_event=0.0,
            gross_exposure_request_excess=0.02 * self.step_index,
            daily_turnover=0.1 * self.step_index,
            execution_cost_fraction=0.001 * self.step_index,
            funding_credit_fraction=0.0,
            transition_elapsed_hours=1.0,
        )
        return (
            np.asarray([self.step_index / 10.0, 0.25, -0.5], dtype=np.float32),
            0.1 * self.step_index,
            terminated,
            False,
            {
                "constraint_costs": costs,
                "transition_elapsed_hours": 1.0,
                "termination_reason": "evidence_episode_complete" if terminated else None,
            },
        )


def _schema():
    count = len(CONSTRAINT_COST_NAMES)
    return canonical_lagrangian_schema(
        names=CONSTRAINT_COST_NAMES,
        budgets=(0.0,) * count,
        dual_learning_rates=(0.1,) * count,
        ema_betas=(0.5,) * count,
        initial_multipliers=(2.0, *(0.0 for _ in range(count - 1))),
        max_multipliers=(10.0,) * count,
        warmup_rollouts=(0,) * count,
        update_interval_rollouts=(1,) * count,
        minimum_completed_episodes=(1,) * count,
    )


def _probe() -> CanonicalActionProbeEvidence:
    return CanonicalActionProbeEvidence(
        action_semantic=CanonicalActionSemantic.RESIDUAL_BASELINE,
        action=np.zeros(1, dtype=np.float32),
        estimates={name: 0.0 for name in CONSTRAINT_COST_NAMES},
        denominators={name: 1 for name in CONSTRAINT_COST_NAMES},
        budgets={name: 0.0 for name in CONSTRAINT_COST_NAMES},
        violated_costs=(),
        completed_episode_count=1,
        censored_episode_count=0,
        episode_count=1,
        max_steps_per_episode=4,
        warning=False,
    )


def test_finalized_rollout_records_raw_diagnostics_and_evidence() -> None:
    environment = DummyVecEnv([_EvidenceEnvironment])
    model = LagrangianPPO(
        "MlpPolicy",
        environment,
        seed=109,
        device="cpu",
        n_steps=4,
        batch_size=2,
        n_epochs=2,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        cost_schema=canonical_cost_learning_schema(),
        cost_learning_rate=5e-4,
        cost_n_epochs=1,
        cost_batch_size=2,
        cost_continuous_hidden_dims=(12,),
        cost_event_hidden_dims=(10,),
        lagrangian_schema=_schema(),
        canonical_action_probe_evidence=_probe(),
    )
    try:
        model.learn(total_timesteps=4)

        diagnostics = model.last_constraint_correlation_diagnostics
        assert isinstance(diagnostics, ConstraintCorrelationDiagnostics)
        raw_cost_advantages = model.cost_rollout_storage.advantages.swapaxes(
            0,
            1,
        ).reshape(4, len(CONSTRAINT_COST_NAMES))
        expected_penalty = raw_cost_advantages * np.asarray(
            [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        )[None, :]
        np.testing.assert_array_equal(
            diagnostics.penalty_contributions,
            expected_penalty,
        )
        assert diagnostics.penalty_contributions.flags.writeable is False

        assert len(model.dual_report_history) == 1
        assert tuple(model.dual_report_history[0]) == CONSTRAINT_COST_NAMES
        evidence = model.last_lagrangian_rollout_evidence
        assert isinstance(evidence, LagrangianRolloutEvidence)
        assert evidence.correlation_diagnostics is diagnostics
        assert evidence.dual_reports == tuple(
            model.last_dual_update_reports[name] for name in CONSTRAINT_COST_NAMES
        )
        assert evidence.completed_episode_count == 1
        assert evidence.censored_episode_count == 0
        assert evidence.probe_evidence.digest == model.canonical_action_probe_evidence.digest
        assert evidence.payload()["digest"] == evidence.digest
        assert evidence.payload()["constraints"]["drawdown_excess"][
            "at_upper_cap"
        ] is False
    finally:
        environment.close()


def test_rollout_diagnostics_do_not_modify_actor_or_cost_advantages() -> None:
    environment = DummyVecEnv([_EvidenceEnvironment])
    model = LagrangianPPO(
        "MlpPolicy",
        environment,
        seed=113,
        device="cpu",
        n_steps=4,
        batch_size=4,
        n_epochs=1,
        cost_schema=canonical_cost_learning_schema(),
        cost_continuous_hidden_dims=(12,),
        cost_event_hidden_dims=(10,),
        lagrangian_schema=_schema(),
        canonical_action_probe_evidence=_probe(),
    )
    captured: dict[str, np.ndarray] = {}
    original_actor_train = model._train_actor_with_lagrangian_advantages

    def actor_train() -> None:
        captured["reward_before"] = model.rollout_buffer.advantages.copy()
        captured["cost_before"] = model.cost_rollout_storage.advantages.copy()
        original_actor_train()
        np.testing.assert_array_equal(
            model.rollout_buffer.advantages,
            captured["reward_before"],
        )
        np.testing.assert_array_equal(
            model.cost_rollout_storage.advantages,
            captured["cost_before"],
        )

    model._train_actor_with_lagrangian_advantages = actor_train  # type: ignore[method-assign]
    try:
        model.learn(total_timesteps=4)

        np.testing.assert_array_equal(
            model.rollout_buffer.advantages,
            captured["reward_before"],
        )
        np.testing.assert_array_equal(
            model.cost_rollout_storage.advantages,
            captured["cost_before"],
        )
    finally:
        environment.close()
