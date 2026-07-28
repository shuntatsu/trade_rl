from __future__ import annotations

import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces

from trade_rl.artifacts.hashing import content_digest
from trade_rl.integrations.sb3_training import StableBaselines3Backend
from trade_rl.rl.actions import ActionMode, ActionSpec
from trade_rl.rl.environment_constraints import (
    CONSTRAINT_COST_NAMES,
    ConstraintCostVector,
)
from trade_rl.rl.lagrangian_probe import (
    CanonicalActionProbeEvidence,
    CanonicalActionSemantic,
)
from trade_rl.rl.training import ResidualTrainingConfig


class _ProbeTrainingEnvironment(gym.Env[np.ndarray, np.ndarray]):
    metadata = {"render_modes": []}
    environment_digest = "a" * 64
    initial_capital = 1_000.0
    decision_hours = 0.25
    action_names = (
        "target_weight:BTCUSDT",
        "target_weight:ETHUSDT",
        "target_weight:BNBUSDT",
    )
    action_spec = ActionSpec(
        mode=ActionMode.TARGET_WEIGHT,
        risk_tilt_enabled=False,
        target_weight_count=3,
    )
    action_spec_digest = content_digest({"names": action_names})
    alpha_artifact_digest = None
    factor_artifact_digest = None
    normalizer = None

    def __init__(
        self,
        *,
        events: list[str],
        cost_value: float,
    ) -> None:
        super().__init__()
        self.events = events
        self.cost_value = cost_value
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        self.events.append("environment_created")

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[np.ndarray, dict[str, object]]:
        del options
        super().reset(seed=seed)
        return np.zeros(3, dtype=np.float32), {}

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        self.events.append(f"probe_step:{action.tolist()}")
        costs = ConstraintCostVector(
            drawdown_excess=self.cost_value,
            drawdown_stop_event=0.0,
            margin_deficit_fraction=self.cost_value,
            forced_liquidation_event=0.0,
            gross_exposure_request_excess=self.cost_value,
            daily_turnover=self.cost_value,
            execution_cost_fraction=self.cost_value,
            funding_credit_fraction=0.0,
            transition_elapsed_hours=1.0,
        )
        return (
            np.zeros(3, dtype=np.float32),
            0.0,
            True,
            False,
            {
                "constraint_costs": costs,
                "transition_elapsed_hours": 1.0,
                "termination_reason": "probe_complete",
            },
        )

    def close(self) -> None:
        self.events.append("environment_closed")


class _FakeParameter:
    def numel(self) -> int:
        return 2


class _FakePolicy:
    action_distribution_name = "squashed_diag_gaussian"

    def parameters(self) -> tuple[_FakeParameter, ...]:
        return (_FakeParameter(),)


class _FakeCostCritic:
    architecture_digest = "c" * 64


class _FakeLagrangianPPO:
    device = "cpu"

    def __init__(
        self,
        policy: object,
        environment: object,
        *,
        events: list[str],
        **kwargs: object,
    ) -> None:
        self.policy = _FakePolicy()
        self.cost_critic = _FakeCostCritic()
        self.num_timesteps = 0
        self.policy_identifier = policy
        self.environment = environment
        self.kwargs = kwargs
        self.events = events
        self.events.append("model_constructed")

    def checkpoint_identity_payload(self) -> dict[str, object]:
        evidence = getattr(self, "canonical_action_probe_evidence", None)
        return {
            "algorithm": "lagrangian_ppo",
            "canonical_action_probe_digest": (
                evidence.digest
                if isinstance(evidence, CanonicalActionProbeEvidence)
                else None
            ),
        }

    def learn(self, **kwargs: object) -> None:
        self.num_timesteps = int(kwargs["total_timesteps"])

    def save(self, target: str) -> None:
        Path(f"{target}.zip").write_bytes(b"lagrangian-policy")


def _config() -> ResidualTrainingConfig:
    count = len(CONSTRAINT_COST_NAMES)
    return ResidualTrainingConfig(
        timesteps=4,
        gamma=1.0,
        seeds=(7,),
        algorithm="lagrangian_ppo",
        n_steps=4,
        n_envs=1,
        batch_size=4,
        n_epochs=1,
        observation_encoder=("flat_mlp"),
        device="cpu",
        lagrangian_budgets=(0.1,) * count,
        lagrangian_dual_learning_rates=(0.05,) * count,
        lagrangian_ema_betas=(0.9,) * count,
        lagrangian_initial_multipliers=(0.0,) * count,
        lagrangian_max_multipliers=(10.0,) * count,
        lagrangian_warmup_rollouts=(0,) * count,
        lagrangian_update_interval_rollouts=(1,) * count,
        lagrangian_minimum_completed_episodes=(1, 20, 1, 20, 1, 1, 1),
        lagrangian_probe_episodes=2,
        lagrangian_probe_max_steps_per_episode=4,
    )


def test_backend_runs_warning_only_probe_before_model_construction(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events: list[str] = []
    constructed: list[_FakeLagrangianPPO] = []

    def environment_factory() -> _ProbeTrainingEnvironment:
        return _ProbeTrainingEnvironment(events=events, cost_value=5.0)

    def build_model(
        policy: object,
        environment: object,
        **kwargs: object,
    ) -> _FakeLagrangianPPO:
        model = _FakeLagrangianPPO(
            policy,
            environment,
            events=events,
            **kwargs,
        )
        constructed.append(model)
        return model

    monkeypatch.setattr(
        "trade_rl.integrations.lagrangian_ppo.LagrangianPPO",
        build_model,
    )
    monkeypatch.setattr(
        "trade_rl.rl.checkpointing.build_checkpoint_callback",
        lambda **kwargs: object(),
    )

    result = StableBaselines3Backend(environment_factory).train(
        seed=7,
        config=_config(),
        output_path=tmp_path / "policy.zip",
    )

    assert result.actual_timesteps == 4
    assert len(constructed) == 1
    evidence = constructed[0].canonical_action_probe_evidence
    assert isinstance(evidence, CanonicalActionProbeEvidence)
    assert evidence.action_semantic is CanonicalActionSemantic.TARGET_WEIGHT_CASH
    assert evidence.action.tolist() == [0.0, 0.0, 0.0]
    assert evidence.completed_episode_count == 2
    assert evidence.warning is True
    assert evidence.violated_costs
    assert events.index("model_constructed") > max(
        index for index, event in enumerate(events) if event.startswith("probe_step:")
    )

    architecture = json.loads((tmp_path / "model-architecture.json").read_text())
    probe_payload = architecture["architecture"]["lagrangian_probe"]
    assert probe_payload["digest"] == evidence.digest
    assert probe_payload["payload"] == evidence.digest_payload()
    assert probe_payload["warning"] is True
    assert probe_payload["violated_costs"] == list(evidence.violated_costs)
