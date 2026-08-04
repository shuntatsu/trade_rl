from __future__ import annotations

import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces

from trade_rl.artifacts.hashing import content_digest
from trade_rl.integrations.sb3_training import (
    StableBaselines3Backend,
    _lagrangian_probe_worker_count,
    _teacher_worker_count,
)
from trade_rl.rl.environment_constraints import CONSTRAINT_COST_NAMES
from trade_rl.rl.lagrangian_probe import (
    CanonicalActionProbeEvidence,
    CanonicalActionSemantic,
)
from trade_rl.rl.training import ResidualTrainingConfig


def test_lagrangian_probe_workers_are_memory_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TRADE_RL_LAGRANGIAN_PROBE_WORKERS", raising=False)
    assert _lagrangian_probe_worker_count(8) == 1
    assert _lagrangian_probe_worker_count(2) == 1

    monkeypatch.setenv("TRADE_RL_LAGRANGIAN_PROBE_WORKERS", "3")
    assert _lagrangian_probe_worker_count(8) == 3
    monkeypatch.setenv("TRADE_RL_LAGRANGIAN_PROBE_WORKERS", "0")
    with pytest.raises(ValueError, match="must be positive"):
        _lagrangian_probe_worker_count(8)


def test_teacher_workers_are_independently_memory_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TRADE_RL_TEACHER_WORKERS", raising=False)
    assert _teacher_worker_count(8) == 8

    monkeypatch.setenv("TRADE_RL_TEACHER_WORKERS", "4")
    assert _teacher_worker_count(8) == 4
    assert _teacher_worker_count(4) == 4

    monkeypatch.setenv("TRADE_RL_TEACHER_WORKERS", "invalid")
    with pytest.raises(ValueError, match="must be an integer"):
        _teacher_worker_count(8)


class _LagrangianProbe(gym.Env[np.ndarray, np.ndarray]):
    metadata = {"render_modes": []}
    environment_digest = "e" * 64
    initial_capital = 1_000.0
    decision_hours = 0.25
    action_names = ("tilt",)
    action_spec_digest = content_digest({"names": action_names})
    alpha_artifact_digest = None
    factor_artifact_digest = None
    normalizer = None

    def __init__(self) -> None:
        super().__init__()
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.close_calls = 0

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
        del action
        return np.zeros(3, dtype=np.float32), 0.0, False, False, {}

    def close(self) -> None:
        self.close_calls += 1


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

    def __init__(self, policy: object, environment: object, **kwargs: object) -> None:
        self.policy = _FakePolicy()
        self.cost_critic = _FakeCostCritic()
        self.num_timesteps = 0
        self.policy_identifier = policy
        self.environment = environment
        self.kwargs = kwargs

    def checkpoint_identity_payload(self) -> dict[str, object]:
        schema = self.kwargs["lagrangian_schema"]
        evidence = getattr(self, "canonical_action_probe_evidence", None)
        return {
            "algorithm": "lagrangian_ppo",
            "lagrangian_schema_digest": schema.digest,
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
        lagrangian_probe_max_steps_per_episode=16,
    )


def _probe_evidence() -> CanonicalActionProbeEvidence:
    estimates = {name: 0.0 for name in CONSTRAINT_COST_NAMES}
    denominators = {name: 2 for name in CONSTRAINT_COST_NAMES}
    budgets = {name: 0.1 for name in CONSTRAINT_COST_NAMES}
    return CanonicalActionProbeEvidence(
        action_semantic=CanonicalActionSemantic.RESIDUAL_BASELINE,
        action=np.zeros(3, dtype=np.float32),
        estimates=estimates,
        denominators=denominators,
        budgets=budgets,
        violated_costs=(),
        completed_episode_count=2,
        censored_episode_count=0,
        episode_count=2,
        max_steps_per_episode=16,
        warning=False,
    )


def test_backend_constructs_lagrangian_ppo_with_full_schema(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    probe = _LagrangianProbe()
    evidence = _probe_evidence()
    constructed: list[_FakeLagrangianPPO] = []
    factory_calls = 0

    def environment_factory() -> _LagrangianProbe:
        nonlocal factory_calls
        factory_calls += 1
        return probe

    def run_probe(**kwargs: object) -> CanonicalActionProbeEvidence:
        # The full-market identity environment must not coexist with the
        # canonical probe environment under the training memory limit.
        assert factory_calls == 0
        assert kwargs["environment_factory"] is environment_factory
        return evidence

    def build_model(
        policy: object,
        environment: object,
        **kwargs: object,
    ) -> _FakeLagrangianPPO:
        model = _FakeLagrangianPPO(policy, environment, **kwargs)
        constructed.append(model)
        return model

    monkeypatch.setattr(
        "trade_rl.integrations.lagrangian_ppo.LagrangianPPO",
        build_model,
    )
    monkeypatch.setattr(
        "trade_rl.rl.lagrangian_probe.run_canonical_action_feasibility_probe",
        run_probe,
    )
    monkeypatch.setattr(
        "trade_rl.rl.checkpointing.build_checkpoint_callback",
        lambda **kwargs: object(),
    )

    config = _config()
    result = StableBaselines3Backend(environment_factory).train(
        seed=7,
        config=config,
        output_path=tmp_path / "policy.zip",
    )

    assert len(constructed) == 1
    cost_schema = constructed[0].kwargs["cost_schema"]
    schema = constructed[0].kwargs["lagrangian_schema"]
    assert cost_schema.names == CONSTRAINT_COST_NAMES
    assert constructed[0].kwargs["cost_learning_rate"] == pytest.approx(
        config.cost_learning_rate
    )
    assert constructed[0].kwargs["cost_n_epochs"] == config.cost_n_epochs
    assert constructed[0].canonical_action_probe_evidence is evidence
    assert schema.names == CONSTRAINT_COST_NAMES
    assert tuple(spec.minimum_completed_episodes for spec in schema.specs) == (
        1,
        20,
        1,
        20,
        1,
        1,
        1,
    )
    architecture = json.loads((tmp_path / "model-architecture.json").read_text())
    lagrangian = architecture["architecture"]["lagrangian"]
    assert lagrangian["actor_composition_mode"] == (
        "raw_lagrangian_then_sb3_normalize_v1"
    )
    assert lagrangian["completion_semantics"] == (
        "economic_time_limit_censored_shadow_v1"
    )
    assert lagrangian["schema"] == schema.digest_payload()
    assert lagrangian["schema_digest"] == schema.digest
    assert lagrangian["probe_episodes"] == 2
    assert lagrangian["probe_max_steps_per_episode"] == 16
    probe_identity = architecture["architecture"]["lagrangian_probe"]
    assert probe_identity["digest"] == evidence.digest
    assert probe_identity["payload"] == evidence.digest_payload()
    assert result.actual_timesteps == 4
    assert factory_calls == 1
    assert probe.close_calls == 1
