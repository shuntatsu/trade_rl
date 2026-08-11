from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from trade_rl.integrations import sb3_training
from trade_rl.rl.training import ResidualTrainingConfig


def _config(**overrides: object) -> ResidualTrainingConfig:
    values: dict[str, object] = {
        "timesteps": 128,
        "gamma": 1.0,
        "seeds": (7,),
        "n_steps": 8,
        "batch_size": 8,
    }
    values.update(overrides)
    return ResidualTrainingConfig(**values)  # type: ignore[arg-type]


def test_sb3_critic_warm_start_is_noop_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    called = False

    def unexpected(**kwargs: object) -> object:
        nonlocal called
        called = True
        return kwargs

    monkeypatch.setattr(sb3_training, "run_configured_critic_warm_start", unexpected)
    result = sb3_training._run_behavior_cloning_critic_warm_start_if_enabled(
        policy=object(),
        teacher_environment=object(),
        teacher_dataset=object(),
        episode_batch=None,
        episode_split=None,
        config=_config(),
        observation_provider=None,
        behavior_cloning_seed=7,
        output_root=Path("unused"),
    )

    assert result is None
    assert not called


def test_sb3_critic_warm_start_fails_closed_without_oracle_evidence() -> None:
    config = _config(
        behavior_cloning_epochs=2,
        behavior_cloning_critic_warm_start_steps=8,
        behavior_cloning_joint_warm_start_steps=4,
    )

    with pytest.raises(RuntimeError, match="Oracle episode evidence"):
        sb3_training._run_behavior_cloning_critic_warm_start_if_enabled(
            policy=object(),
            teacher_environment=object(),
            teacher_dataset=object(),
            episode_batch=None,
            episode_split=None,
            config=config,
            observation_provider=None,
            behavior_cloning_seed=7,
            output_root=Path("unused"),
        )


def test_sb3_critic_warm_start_forwards_exact_bc_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(
        behavior_cloning_epochs=2,
        behavior_cloning_critic_warm_start_steps=8,
        behavior_cloning_joint_warm_start_steps=4,
    )
    sentinel = object()
    captured: dict[str, object] = {}

    def configured(**kwargs: object) -> object:
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(sb3_training, "run_configured_critic_warm_start", configured)
    policy = object()
    environment = object()
    dataset = object()
    episode_batch = object()
    episode_split = object()
    provider = object()
    output_root = Path("artifact-root")

    result = sb3_training._run_behavior_cloning_critic_warm_start_if_enabled(
        policy=policy,
        teacher_environment=environment,
        teacher_dataset=dataset,
        episode_batch=episode_batch,
        episode_split=episode_split,
        config=config,
        observation_provider=provider,
        behavior_cloning_seed=13,
        output_root=output_root,
    )

    assert result is sentinel
    assert captured == {
        "policy": policy,
        "teacher_environment": environment,
        "teacher_dataset": dataset,
        "episode_batch": episode_batch,
        "split": episode_split,
        "config": config,
        "observation_provider": provider,
        "behavior_cloning_seed": 13,
        "output_root": output_root,
    }


def test_sb3_train_invokes_critic_warm_start_only_after_bc_gate() -> None:
    source = inspect.getsource(sb3_training.StableBaselines3Backend.train)
    gate_position = source.index("_enforce_behavior_cloning_gates")
    warm_start_position = source.index(
        "_run_behavior_cloning_critic_warm_start_if_enabled"
    )

    assert gate_position < warm_start_position
